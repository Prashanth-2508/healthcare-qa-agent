"""PubMed search via NCBI Entrez API.

Features:
- Natural language → PubMed Boolean query translation
- Thread-safe token-bucket rate limiter (3 req/s unauthenticated, 10/s with API key)
- Full abstract retrieval with PMID, title, authors, MeSH terms
- Custom ToolExecutionError on failure
- Pydantic input schema for LangGraph tool binding
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from src.tools.exceptions import ToolExecutionError
from src.utils.logger import AgentStep, get_logger

_log = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_EMAIL: str = os.getenv("NCBI_EMAIL", "user@example.com")
_API_KEY: str = os.getenv("NCBI_API_KEY", "")
_DEFAULT_MAX: int = int(os.getenv("PUBMED_MAX_RESULTS", "5"))

# NCBI rate-limit policy
_RATE_UNAUTH: float = 3.0   # requests/second without API key
_RATE_AUTH: float = 10.0    # requests/second with API key


# ── Rate limiter ──────────────────────────────────────────────────────────────

class _RateLimiter:
    """Thread-safe token-bucket rate limiter (one token per `1/rate` seconds)."""

    def __init__(self, rate: float) -> None:
        self._min_interval: float = 1.0 / rate
        self._lock = threading.Lock()
        self._last_call: float = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            gap = self._min_interval - elapsed
            if gap > 0:
                time.sleep(gap)
            self._last_call = time.monotonic()


# Singleton — re-created only if API key presence changes (rare).
_limiter: _RateLimiter | None = None
_limiter_lock = threading.Lock()


def _get_limiter() -> _RateLimiter:
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            rate = _RATE_AUTH if _API_KEY else _RATE_UNAUTH
            _limiter = _RateLimiter(rate)
            _log.info("pubmed_rate_limiter_init", rate=rate)
    return _limiter


# ── Natural language → PubMed query translation ───────────────────────────────

# Words that carry no medical signal and reduce search precision.
_STOPWORDS: frozenset[str] = frozenset({
    "what", "is", "the", "are", "how", "do", "does", "can", "could", "should",
    "would", "will", "tell", "me", "about", "explain", "describe", "show",
    "give", "list", "find", "search", "for", "of", "in", "on", "at", "to",
    "a", "an", "and", "or", "but", "with", "from", "by", "that", "this",
    "it", "its", "difference", "between", "vs", "versus", "compared",
    "comparison", "use", "used", "using", "based", "related", "according",
    "patients", "patient", "study", "studies",
})

# Publication-type markers that map to Entrez [pt] field qualifiers.
_PT_MARKERS: dict[str, str] = {
    "systematic review": "systematic review[pt]",
    "meta-analysis": "meta-analysis[pt]",
    "randomized controlled trial": "randomized controlled trial[pt]",
    "randomised controlled trial": "randomized controlled trial[pt]",
    "clinical trial": "clinical trial[pt]",
    "rct": "randomized controlled trial[pt]",
    "review": "review[pt]",
    "guideline": "guideline[pt]",
    "case report": "case reports[pt]",
    "cohort": "cohort studies[mh]",
}


def _translate_query(natural_query: str) -> str:
    """Convert a natural language question into a PubMed Boolean search string.

    Strategy:
      1. Detect and extract publication-type qualifiers.
      2. Preserve any existing Entrez field tags ([tiab], [mh], etc.).
      3. Strip stopwords from remaining tokens.
      4. Wrap bare multi-word phrases in quotes; tag single words with [tiab].
      5. Join all terms with AND.
    """
    lower = natural_query.lower()

    # 1. Detect PT qualifiers (longest match first to avoid partial clobber)
    pt_terms: list[str] = []
    remaining = lower
    for marker in sorted(_PT_MARKERS, key=len, reverse=True):
        if marker in remaining:
            pt_terms.append(_PT_MARKERS[marker])
            remaining = remaining.replace(marker, " ")

    # 2. Preserve already-tagged Entrez expressions
    tagged_pattern = re.compile(r'"[^"]+"\[\w+\]|\w+\[\w+\]')
    existing_tags = tagged_pattern.findall(natural_query)
    clean = tagged_pattern.sub(" ", remaining)

    # 3. Tokenise: keep quoted phrases intact, split the rest on word boundaries
    tokens = re.findall(r'"[^"]+"|\b\w[\w\'-]*\b', clean)
    kept: list[str] = []
    for tok in tokens:
        if tok.startswith('"'):
            # Already quoted phrase — add [tiab]
            kept.append(f"{tok}[tiab]")
        elif tok.lower() not in _STOPWORDS and len(tok) > 2 and not tok.isdigit():
            kept.append(f"{tok}[tiab]")

    all_parts = existing_tags + kept + pt_terms
    if not all_parts:
        # Fallback: treat the whole query as a phrase search
        return f'"{natural_query.strip()}"[tiab]'

    return " AND ".join(all_parts)


# ── Entrez helpers ────────────────────────────────────────────────────────────

def _configure_entrez() -> None:
    try:
        from Bio import Entrez  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ToolExecutionError(
            "search_pubmed",
            "biopython is not installed. Run: pip install biopython",
            exc,
        ) from exc
    Entrez.email = _EMAIL
    if _API_KEY:
        Entrez.api_key = _API_KEY


def _esearch(query: str, max_results: int) -> list[str]:
    """Return a list of PMIDs for the query."""
    from Bio import Entrez  # type: ignore[import-untyped]

    limiter = _get_limiter()
    limiter.wait()
    try:
        with Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance",
            usehistory="y",
        ) as handle:
            result = Entrez.read(handle)
    except Exception as exc:
        raise ToolExecutionError("search_pubmed", "ESearch request failed", exc) from exc

    return result.get("IdList", [])


def _efetch(pmids: list[str]) -> list[dict[str, Any]]:
    """Fetch MEDLINE records for a list of PMIDs and return structured dicts."""
    from Bio import Entrez, Medline  # type: ignore[import-untyped]

    limiter = _get_limiter()
    limiter.wait()
    try:
        with Entrez.efetch(
            db="pubmed",
            id=",".join(pmids),
            rettype="medline",
            retmode="text",
        ) as handle:
            records = list(Medline.parse(handle))
    except Exception as exc:
        raise ToolExecutionError("search_pubmed", "EFetch request failed", exc) from exc

    articles: list[dict[str, Any]] = []
    for rec in records:
        pmid = rec.get("PMID", "")
        abstract = (rec.get("AB") or "").strip() or "No abstract available."
        articles.append({
            "pmid": pmid,
            "title": rec.get("TI", "No title"),
            "authors": rec.get("AU", [])[:3],
            "journal": rec.get("TA", ""),
            "year": (rec.get("DP") or "")[:4],
            "abstract": abstract,
            "mesh_terms": rec.get("MH", [])[:6],
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return articles


def _format_articles(articles: list[dict[str, Any]]) -> str:
    if not articles:
        return "No PubMed articles found for the given query."

    blocks: list[str] = [f"## PubMed Results ({len(articles)} articles)\n"]
    for i, a in enumerate(articles, 1):
        authors = ", ".join(a["authors"]) if a["authors"] else "Unknown"
        mesh = ", ".join(a["mesh_terms"]) if a["mesh_terms"] else "—"
        blocks.append(
            f"### [{i}] {a['title']}\n"
            f"**Authors**: {authors}  \n"
            f"**Journal**: {a['journal']} ({a['year']})  \n"
            f"**PMID**: {a['pmid']} | **URL**: {a['url']}  \n"
            f"**MeSH**: {mesh}  \n\n"
            f"**Abstract**:  \n{a['abstract']}"
        )
    return "\n\n---\n\n".join(blocks)


# ── Input schema ──────────────────────────────────────────────────────────────

class PubMedSearchInput(BaseModel):
    """JSON schema for the search_pubmed tool."""

    natural_query: str = Field(
        description=(
            "Natural language medical question or topic to search on PubMed. "
            "Examples: 'What is the efficacy of metformin for type 2 diabetes?', "
            "'systematic reviews of GLP-1 agonists for obesity', "
            "'hypertension first-line treatment guidelines'."
        )
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of articles to return (1–20, default 5).",
    )

    @model_validator(mode="before")
    @classmethod
    def _remap_query_alias(cls, data: Any) -> Any:
        """Accept 'query' as an alias for 'natural_query' (gemma3 compat)."""
        if isinstance(data, dict) and "query" in data and "natural_query" not in data:
            data["natural_query"] = data.pop("query")
        return data


# ── Tool ─────────────────────────────────────────────────────────────────────

@tool(args_schema=PubMedSearchInput)
def search_pubmed(natural_query: str, max_results: int = 5) -> str:
    """Search PubMed for peer-reviewed medical literature using a natural language query.

    Automatically translates the question into a PubMed Boolean search string
    using MeSH-aware field tagging, fetches the top abstracts from NCBI Entrez,
    and returns PMID, title, authors, MeSH terms, and full abstract for each article.

    Respects NCBI rate limits: 3 req/s without NCBI_API_KEY, 10 req/s with it.
    Raises ToolExecutionError if the Entrez API call fails after exhausting retries.

    Args:
        natural_query: Natural language medical question or topic.
        max_results: Number of articles to return (default 5, max 20).

    Returns:
        Markdown-formatted list of articles with PMID, title, authors,
        journal, MeSH terms, and abstract text.

    Raises:
        ToolExecutionError: On Entrez API failure (network error, bad response, etc.).
    """
    _log.info(
        "tool_call",
        step=AgentStep.TOOL_CALL.value,
        tool="search_pubmed",
        query=natural_query[:120],
    )

    max_results = max(1, min(max_results, 20))

    try:
        _configure_entrez()
        pubmed_query = _translate_query(natural_query)
        _log.info(
            "pubmed_query_translated",
            original=natural_query[:80],
            translated=pubmed_query[:120],
        )

        pmids = _esearch(pubmed_query, max_results)
        if not pmids:
            return "No PubMed articles found for the given query."

        articles = _efetch(pmids)
        result = _format_articles(articles)

        _log.info(
            "tool_result",
            step=AgentStep.TOOL_RESULT.value,
            tool="search_pubmed",
            article_count=len(articles),
        )
        return result

    except ToolExecutionError:
        raise
    except Exception as exc:
        raise ToolExecutionError("search_pubmed", str(exc), exc) from exc
