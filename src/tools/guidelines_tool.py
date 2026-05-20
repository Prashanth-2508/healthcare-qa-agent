"""Clinical guidelines retrieval via RAG.

Pipeline:
  Ingest:  PDF → text extraction (pypdf) → sliding-window chunking
           → sentence-transformers embeddings → ChromaDB upsert
  Query:   embed query → ChromaDB cosine search → top-k chunks + metadata

Environment variables:
  GUIDELINES_PDF_DIR    Directory containing clinical guideline PDFs
                        (default: ./data/guidelines_pdfs)
  GUIDELINES_CHROMA_DIR Persistent ChromaDB directory
                        (default: ./data/chroma_guidelines)
  GUIDELINES_EMBED_MODEL sentence-transformers model name
                        (default: all-MiniLM-L6-v2)
  GUIDELINES_CHUNK_SIZE  Words per chunk  (default: 400)
  GUIDELINES_CHUNK_OVERLAP Overlap words  (default: 80)
  GUIDELINES_TOP_K       Chunks to retrieve per query (default: 3)
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.utils.logger import AgentStep, get_logger

_log = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_PDF_DIR: str = os.getenv("GUIDELINES_PDF_DIR", "./data/guidelines_pdfs")
_CHROMA_DIR: str = os.getenv("GUIDELINES_CHROMA_DIR", "./data/chroma_guidelines")
_EMBED_MODEL: str = os.getenv("GUIDELINES_EMBED_MODEL", "all-MiniLM-L6-v2")
_COLLECTION_NAME = "clinical_guidelines"
_CHUNK_SIZE: int = int(os.getenv("GUIDELINES_CHUNK_SIZE", "400"))
_CHUNK_OVERLAP: int = int(os.getenv("GUIDELINES_CHUNK_OVERLAP", "80"))
_TOP_K: int = int(os.getenv("GUIDELINES_TOP_K", "3"))

# Minimum cosine similarity to surface a chunk (0–1)
_MIN_SIMILARITY: float = float(os.getenv("GUIDELINES_MIN_SIMILARITY", "0.25"))


# ── PDF text extraction ───────────────────────────────────────────────────────

def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from every page of a PDF using pypdf."""
    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required for PDF extraction. Run: pip install pypdf"
        ) from exc

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


# ── Sliding-window word-level chunker ─────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_size: int = _CHUNK_SIZE,
    overlap: int = _CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping fixed-size word windows.

    Args:
        text: Input text (any length).
        chunk_size: Target window size in words.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        Non-empty chunk strings.
    """
    words = text.split()
    if not words:
        return []

    stride = max(1, chunk_size - overlap)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += stride

    return [c for c in chunks if c.strip()]


# ── Vector store (singleton) ──────────────────────────────────────────────────

class _GuidelinesVectorStore:
    """Manages sentence-transformer embeddings and a persistent ChromaDB collection.

    Pattern: lazy singleton initialised on first `.get()` call, protected by a
    module-level lock so concurrent agent threads share one instance safely.
    """

    _instance: _GuidelinesVectorStore | None = None
    _init_lock = threading.Lock()

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._embed_lock = threading.Lock()  # serialise encode() calls
        self._client, self._collection = self._init_chroma()
        self._model = self._init_embedder()

    def _init_chroma(self) -> tuple[Any, Any]:
        try:
            import chromadb  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required. Run: pip install chromadb"
            ) from exc

        chroma_path = Path(_CHROMA_DIR)
        chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        _log.info(
            "guidelines_chroma_init",
            dir=_CHROMA_DIR,
            existing_docs=collection.count(),
        )
        return client, collection

    def _init_embedder(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required. Run: pip install sentence-transformers"
            ) from exc

        model = SentenceTransformer(_EMBED_MODEL)
        _log.info("guidelines_embedder_init", model=_EMBED_MODEL)
        return model

    # ── Singleton accessor ────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> _GuidelinesVectorStore:
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> list[list[float]]:
        with self._embed_lock:
            vecs = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,  # required for cosine similarity
            )
        return vecs.tolist()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: Path, source_label: str | None = None) -> int:
        """Chunk, embed, and upsert one PDF. Returns number of chunks added."""
        label = source_label or pdf_path.stem
        _log.info("guidelines_ingest_start", file=pdf_path.name, label=label)

        raw_text = _extract_pdf_text(pdf_path)
        chunks = _chunk_text(raw_text)
        if not chunks:
            _log.warning("guidelines_pdf_no_text", file=pdf_path.name)
            return 0

        embeddings = self._embed(chunks)

        ids: list[str] = []
        metadatas: list[dict[str, str]] = []
        for idx, chunk in enumerate(chunks):
            # Deterministic, content-stable ID → safe to re-ingest
            chunk_id = hashlib.sha256(f"{label}::{idx}".encode()).hexdigest()[:24]
            ids.append(chunk_id)
            metadatas.append({
                "source": label,
                "file": pdf_path.name,
                "chunk_index": str(idx),
                "total_chunks": str(len(chunks)),
            })

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        _log.info("guidelines_ingest_done", label=label, chunks=len(chunks))
        return len(chunks)

    def ingest_directory(self, pdf_dir: Path | None = None) -> int:
        """Ingest all *.pdf files from pdf_dir. Returns total chunks added."""
        target = pdf_dir or Path(_PDF_DIR)
        if not target.exists():
            _log.warning("guidelines_pdf_dir_missing", path=str(target))
            return 0

        total = 0
        pdf_files = sorted(target.glob("*.pdf"))
        if not pdf_files:
            _log.warning("guidelines_pdf_dir_empty", path=str(target))
        for pdf_file in pdf_files:
            total += self.ingest_pdf(pdf_file)
        return total

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = _TOP_K) -> list[dict[str, Any]]:
        """Embed query and return top-k chunks above the similarity threshold.

        If the collection is empty, attempts a one-time auto-ingest from PDF_DIR.
        """
        if self._collection.count() == 0:
            ingested = self.ingest_directory()
            if ingested == 0:
                return []

        query_vec = self._embed([query_text])[0]
        n = min(top_k, self._collection.count())
        raw = self._collection.query(
            query_embeddings=[query_vec],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        docs: list[str] = (raw.get("documents") or [[]])[0]
        metas: list[dict[str, Any]] = (raw.get("metadatas") or [[]])[0]
        dists: list[float] = (raw.get("distances") or [[]])[0]

        results: list[dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            # ChromaDB cosine space returns distances in [0, 2]; convert to similarity.
            similarity = max(0.0, 1.0 - float(dist))
            if similarity < _MIN_SIMILARITY:
                continue
            results.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "file": meta.get("file", ""),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "total_chunks": int(meta.get("total_chunks", 0)),
                "similarity": round(similarity, 4),
            })

        return results


# ── Output formatting ─────────────────────────────────────────────────────────

def _format_chunks(chunks: list[dict[str, Any]], query: str) -> str:
    if not chunks:
        return (
            "No sufficiently relevant guideline passages found for this query.\n\n"
            f"**Setup**: Place clinical guideline PDFs in `{_PDF_DIR}` "
            "and they will be auto-indexed on first use. "
            "Lower `GUIDELINES_MIN_SIMILARITY` if results seem too sparse."
        )

    header = (
        f"## Clinical Guidelines — {len(chunks)} Relevant Passage(s)\n"
        f"**Query**: _{query}_\n"
    )
    blocks: list[str] = [header]
    for i, chunk in enumerate(chunks, 1):
        progress = (
            f"chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']}"
            if chunk["total_chunks"]
            else f"chunk {chunk['chunk_index'] + 1}"
        )
        blocks.append(
            f"### [{i}] {chunk['source']}  "
            f"*(similarity: {chunk['similarity']:.3f} | {progress})*\n"
            f"**File**: `{chunk['file']}`\n\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(blocks)


# ── Input schema ──────────────────────────────────────────────────────────────

class GuidelinesQueryInput(BaseModel):
    """JSON schema for the get_clinical_guidelines tool."""

    query: str = Field(
        description=(
            "Clinical question or topic to retrieve from indexed guideline PDFs. "
            "Examples: 'hypertension JNC 8 first-line agents', "
            "'HbA1c targets type 2 diabetes ADA 2024', "
            "'USPSTF colorectal cancer screening recommendations'."
        )
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of relevant passages to return (default 3, max 10).",
    )


# ── Tool ─────────────────────────────────────────────────────────────────────

@tool(args_schema=GuidelinesQueryInput)
def get_clinical_guidelines(query: str, top_k: int = 3) -> str:
    """Retrieve relevant passages from locally indexed clinical guideline PDFs using RAG.

    Uses sentence-transformers (all-MiniLM-L6-v2 by default) for dense semantic
    embeddings and a persistent ChromaDB vector store for approximate nearest-neighbour
    retrieval. PDFs are automatically chunked (sliding 400-word windows, 80-word overlap)
    and indexed on first use from the GUIDELINES_PDF_DIR directory.

    Each returned passage includes the source document name, chunk position,
    and cosine similarity score so the agent can cite specific guidelines.

    Use when the question asks about:
    - Treatment protocols, dosing algorithms, or clinical decision trees
    - Diagnostic criteria, staging systems, or risk calculators
    - Preventive care screening schedules (USPSTF, ACIP, etc.)
    - Drug contraindications, monitoring parameters, or titration schedules
    - Evidence-based guidelines from ADA, ACC/AHA, IDSA, NCCN, etc.

    Args:
        query: Clinical question or topic to search in the guideline knowledge base.
        top_k: Number of relevant passages to retrieve (default 3, max 10).

    Returns:
        Top-k guideline passages with source document, chunk position,
        similarity score, and raw passage text.

    Note:
        Place PDF guidelines in GUIDELINES_PDF_DIR (default: ./data/guidelines_pdfs).
        The store auto-indexes on first query. Call ingest_guidelines_pdfs() manually
        for bulk pre-loading before first use.
    """
    _log.info(
        "tool_call",
        step=AgentStep.TOOL_CALL.value,
        tool="get_clinical_guidelines",
        query=query[:120],
        top_k=top_k,
    )

    try:
        store = _GuidelinesVectorStore.get()
        chunks = store.query(query, top_k=max(1, min(top_k, 10)))
        result = _format_chunks(chunks, query)

        _log.info(
            "tool_result",
            step=AgentStep.TOOL_RESULT.value,
            tool="get_clinical_guidelines",
            chunks_returned=len(chunks),
        )
        return result

    except Exception as exc:
        _log.error("tool_error", tool="get_clinical_guidelines", error=str(exc))
        return f"Guidelines retrieval failed: {exc}"


# ── Public ingestion utility ──────────────────────────────────────────────────

def ingest_guidelines_pdfs(pdf_dir: str | Path | None = None) -> int:
    """Pre-load all PDFs from a directory into the vector store.

    Call this once at startup or whenever new guideline PDFs are added.
    Safe to call multiple times — existing chunks are upserted, not duplicated.

    Args:
        pdf_dir: Directory containing PDF files. Defaults to GUIDELINES_PDF_DIR env var.

    Returns:
        Total number of text chunks indexed.
    """
    store = _GuidelinesVectorStore.get()
    path = Path(pdf_dir) if pdf_dir else None
    total = store.ingest_directory(path)
    _log.info("guidelines_bulk_ingest_complete", total_chunks=total)
    return total
