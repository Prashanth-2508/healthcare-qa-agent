"""ICD-10-CM and CPT/HCPCS code lookup.

Data sources:
  ICD-10-CM  CMS FY2025 code description file (downloaded & cached automatically).
             URL: https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip
             Format: fixed-width text — first 7 chars = code, remainder = description.

  CPT        Proprietary (AMA license required). If a local CPT CSV is provided via
             CPT_CSV_PATH (columns: code, description), it is loaded and searched.
             Otherwise CPT queries return an explanatory notice.

Fuzzy matching via rapidfuzz.process.extract (WRatio scorer, configurable threshold).

Environment variables:
  CMS_ICD10_CACHE_DIR  Directory for downloaded/cached CMS files (default: ./data/cms)
  CPT_CSV_PATH         Optional path to a local CPT CSV file
  CODE_FUZZY_THRESHOLD Minimum fuzzy match score 0–100 (default: 55)
  CODE_MAX_RESULTS     Default number of results (default: 10)
"""
from __future__ import annotations

import csv
import io
import os
import threading
import zipfile
from pathlib import Path
from typing import Any

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.tools.exceptions import ToolExecutionError
from src.utils.logger import AgentStep, get_logger

_log = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_CACHE_DIR: str = os.getenv("CMS_ICD10_CACHE_DIR", "./data/cms")
_CPT_CSV_PATH: str = os.getenv("CPT_CSV_PATH", "")
_FUZZY_THRESHOLD: int = int(os.getenv("CODE_FUZZY_THRESHOLD", "55"))
_DEFAULT_MAX: int = int(os.getenv("CODE_MAX_RESULTS", "10"))
_DOWNLOAD_TIMEOUT: int = 60  # seconds for CMS zip download

# CMS FY2025 ICD-10-CM code description file
_CMS_ZIP_URL = (
    "https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip"
)
_CMS_FILENAME = "icd10cm_codes_2025.txt"
_CACHE_PATH = Path(_CACHE_DIR) / _CMS_FILENAME

_CPT_NOTICE = (
    "CPT® (Current Procedural Terminology) codes are proprietary to the American "
    "Medical Association and require a paid license for programmatic access.\n\n"
    "**Options**:\n"
    "1. Set `CPT_CSV_PATH` to a local CSV (columns: `code`, `description`) "
    "to enable CPT lookups.\n"
    "2. Use `code_type='hcpcs'` to search publicly available HCPCS Level II "
    "codes (durable medical equipment, drugs, supplies).\n"
    "3. Access CPT via the AMA's CodeManager at https://www.ama-assn.org/practice-management/cpt"
)


# ── CMS file download & parse ─────────────────────────────────────────────────

def _download_cms_icd10() -> Path:
    """Download the CMS ICD-10-CM zip and extract the code description file.

    Returns the path to the cached .txt file.
    """
    cache_dir = Path(_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _log.info("cms_icd10_download_start", url=_CMS_ZIP_URL)
    try:
        resp = requests.get(_CMS_ZIP_URL, timeout=_DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()
        zip_bytes = resp.content
    except Exception as exc:
        raise ToolExecutionError(
            "lookup_medical_code",
            f"Failed to download CMS ICD-10-CM file from {_CMS_ZIP_URL}",
            exc,
        ) from exc

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # The target file may be at the root or inside a subdirectory
            matched = next(
                (n for n in zf.namelist() if n.endswith(_CMS_FILENAME)),
                None,
            )
            if matched is None:
                # Fallback: any .txt file whose name contains "codes"
                matched = next(
                    (n for n in zf.namelist() if "codes" in n.lower() and n.endswith(".txt")),
                    None,
                )
            if matched is None:
                available = ", ".join(zf.namelist()[:10])
                raise ToolExecutionError(
                    "lookup_medical_code",
                    f"Could not find '{_CMS_FILENAME}' in the CMS zip. "
                    f"Available files: {available}",
                )
            with zf.open(matched) as src, open(_CACHE_PATH, "wb") as dst:
                dst.write(src.read())
    except ToolExecutionError:
        raise
    except Exception as exc:
        raise ToolExecutionError(
            "lookup_medical_code", "Failed to extract CMS zip archive", exc
        ) from exc

    _log.info("cms_icd10_download_complete", path=str(_CACHE_PATH))
    return _CACHE_PATH


def _parse_cms_icd10(path: Path) -> list[tuple[str, str]]:
    """Parse the CMS fixed-width ICD-10-CM text file.

    Format: each line = 7-char code (space-padded) + description.
    Returns list of (code, description) tuples.
    """
    records: list[tuple[str, str]] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if len(line) < 8:
                continue
            code = line[:7].strip()
            desc = line[7:].strip()
            if code and desc:
                records.append((code, desc))
    _log.info("cms_icd10_parsed", records=len(records))
    return records


def _parse_cpt_csv(path: Path) -> list[tuple[str, str]]:
    """Parse a local CPT CSV with columns 'code' and 'description'."""
    records: list[tuple[str, str]] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            code = (row.get("code") or row.get("Code") or "").strip()
            desc = (row.get("description") or row.get("Description") or "").strip()
            if code and desc:
                records.append((code, desc))
    _log.info("cpt_csv_parsed", path=str(path), records=len(records))
    return records


# ── Code database (singleton) ─────────────────────────────────────────────────

class _CodeDatabase:
    """Lazy-loaded, thread-safe in-memory code database.

    Loads ICD-10-CM from the CMS file and optionally CPT from a local CSV.
    All data is stored as parallel lists for fast rapidfuzz access.
    """

    _instance: _CodeDatabase | None = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self._icd10_codes: list[str] = []
        self._icd10_descs: list[str] = []
        self._cpt_codes: list[str] = []
        self._cpt_descs: list[str] = []
        self._load()

    def _load(self) -> None:
        # ICD-10-CM
        cache = Path(_CACHE_DIR) / _CMS_FILENAME
        if not cache.exists():
            cache = _download_cms_icd10()
        pairs = _parse_cms_icd10(cache)
        self._icd10_codes = [p[0] for p in pairs]
        self._icd10_descs = [p[1] for p in pairs]

        # CPT (optional)
        if _CPT_CSV_PATH and Path(_CPT_CSV_PATH).exists():
            cpt_pairs = _parse_cpt_csv(Path(_CPT_CSV_PATH))
            self._cpt_codes = [p[0] for p in cpt_pairs]
            self._cpt_descs = [p[1] for p in cpt_pairs]
            _log.info("cpt_loaded", count=len(self._cpt_codes))
        else:
            _log.info("cpt_not_configured", hint="Set CPT_CSV_PATH to enable CPT lookup")

    @classmethod
    def get(cls) -> _CodeDatabase:
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def icd10_count(self) -> int:
        return len(self._icd10_codes)

    def cpt_available(self) -> bool:
        return bool(self._cpt_codes)


# ── Fuzzy search ──────────────────────────────────────────────────────────────

def _fuzzy_search(
    query: str,
    codes: list[str],
    descriptions: list[str],
    limit: int,
    threshold: int,
) -> list[dict[str, Any]]:
    """Search descriptions using rapidfuzz WRatio scorer.

    Also performs an exact prefix match on the code itself (e.g. query "E11").

    Args:
        query: User search string (condition name, procedure, or code fragment).
        codes: Parallel list of code strings.
        descriptions: Parallel list of description strings.
        limit: Maximum results.
        threshold: Minimum WRatio score (0–100).

    Returns:
        List of dicts: {code, description, score, match_type}.
    """
    try:
        from rapidfuzz import fuzz, process  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ToolExecutionError(
            "lookup_medical_code",
            "rapidfuzz is required. Run: pip install rapidfuzz",
            exc,
        ) from exc

    results: dict[str, dict[str, Any]] = {}  # keyed by code to dedupe

    # 1. Exact / prefix code match (case-insensitive)
    upper_query = query.upper().replace(".", "").replace(" ", "")
    for code, desc in zip(codes, descriptions):
        norm_code = code.upper().replace(".", "")
        if norm_code.startswith(upper_query) or upper_query == norm_code:
            results[code] = {
                "code": code,
                "description": desc,
                "score": 100,
                "match_type": "code",
            }
        if len(results) >= limit:
            break

    # 2. Fuzzy description match (fills remaining slots)
    remaining = limit - len(results)
    if remaining > 0:
        # process.extract returns (match_string, score, index)
        hits = process.extract(
            query,
            descriptions,
            scorer=fuzz.WRatio,
            limit=remaining * 3,  # over-fetch then threshold-filter
            score_cutoff=threshold,
        )
        for desc_match, score, idx in hits:
            code = codes[idx]
            if code not in results:
                results[code] = {
                    "code": code,
                    "description": descriptions[idx],
                    "score": int(score),
                    "match_type": "fuzzy",
                }
            if len(results) >= limit:
                break

    # Sort: code matches first, then by descending fuzzy score
    sorted_results = sorted(
        results.values(),
        key=lambda r: (r["match_type"] != "code", -r["score"]),
    )
    return sorted_results[:limit]


# ── Output formatting ─────────────────────────────────────────────────────────

def _format_results(
    results: list[dict[str, Any]],
    code_type: str,
    query: str,
) -> str:
    label_map = {
        "icd10": "ICD-10-CM",
        "cpt": "CPT®",
        "hcpcs": "HCPCS Level II",
    }
    label = label_map.get(code_type, code_type.upper())

    if not results:
        return (
            f"No {label} codes found matching **{query}**.\n\n"
            "Try broader terms or check the spelling."
        )

    lines = [f"## {label} Code Lookup — \"{query}\"\n"]
    for r in results:
        match_badge = "🔤" if r["match_type"] == "code" else f"~{r['score']}%"
        lines.append(f"- **{r['code']}** — {r['description']}  *({match_badge})*")

    lines.append(
        f"\n*{len(results)} result(s). "
        "Verify codes against the official CMS tabular list before clinical use.*"
    )
    return "\n".join(lines)


# ── Input schemas ─────────────────────────────────────────────────────────────

class CodeLookupInput(BaseModel):
    """JSON schema for lookup_medical_code."""

    query: str = Field(
        description=(
            "Condition name, procedure description, or code fragment to search. "
            "Examples: 'type 2 diabetes with CKD', 'knee arthroplasty', 'E11', 'I10'."
        )
    )
    code_type: str = Field(
        default="icd10",
        description=(
            "Code set to search. Options: "
            "'icd10' (ICD-10-CM diagnosis codes, CMS dataset), "
            "'cpt' (CPT® procedure codes — requires local CPT_CSV_PATH), "
            "'hcpcs' (HCPCS Level II via NLM ClinicalTables API)."
        ),
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of matching codes to return (default 10).",
    )
    fuzzy_threshold: int = Field(
        default=_FUZZY_THRESHOLD,
        ge=0,
        le=100,
        description=(
            "Minimum rapidfuzz WRatio score (0–100) for a description match. "
            "Lower values return more results with looser matching (default 55)."
        ),
    )


class CodeDescribeInput(BaseModel):
    """JSON schema for describe_medical_code."""

    code: str = Field(
        description="Specific code to describe, e.g. 'E11.65', 'Z23', 'A4253'."
    )
    code_type: str = Field(
        default="icd10",
        description="Code set: 'icd10', 'cpt', or 'hcpcs'.",
    )


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool(args_schema=CodeLookupInput)
def lookup_medical_code(
    query: str,
    code_type: str = "icd10",
    max_results: int = 10,
    fuzzy_threshold: int = _FUZZY_THRESHOLD,
) -> str:
    """Look up ICD-10-CM diagnosis codes or CPT/HCPCS procedure codes by description.

    For ICD-10-CM: uses the CMS FY2025 public code description file, downloaded and
    cached locally on first use (~4 MB). Applies rapidfuzz WRatio fuzzy matching
    so misspellings and synonyms still return relevant codes. Also performs exact
    prefix matching on the code itself (e.g. query "E11" returns all E11.x codes).

    For CPT: requires a local CSV file at CPT_CSV_PATH (columns: code, description).
    For HCPCS: delegates to the NLM ClinicalTables API via the HCPCS endpoint.

    Use when the question involves:
    - Finding the correct ICD-10-CM code for a diagnosis or symptom
    - Validating or expanding a partial code (e.g. all E11.x variants)
    - Searching for procedure codes with tolerant spelling matching
    - Billing code documentation or coding audits

    Args:
        query: Condition name, procedure description, or code fragment.
        code_type: 'icd10' (default), 'cpt', or 'hcpcs'.
        max_results: Maximum codes to return (default 10, max 50).
        fuzzy_threshold: rapidfuzz WRatio cutoff 0–100 (default 55).

    Returns:
        Matched codes with descriptions, match type (exact code / fuzzy),
        and similarity score. Results are sorted: code prefix matches first,
        then by descending fuzzy score.

    Raises:
        ToolExecutionError: If the CMS file cannot be downloaded or rapidfuzz is missing.
    """
    _log.info(
        "tool_call",
        step=AgentStep.TOOL_CALL.value,
        tool="lookup_medical_code",
        query=query,
        code_type=code_type,
    )

    code_type = code_type.lower().strip()

    # ── ICD-10-CM ──────────────────────────────────────────────────────────────
    if code_type == "icd10":
        try:
            db = _CodeDatabase.get()
            results = _fuzzy_search(
                query,
                db._icd10_codes,
                db._icd10_descs,
                limit=max(1, min(max_results, 50)),
                threshold=fuzzy_threshold,
            )
            formatted = _format_results(results, "icd10", query)
            _log.info(
                "tool_result",
                step=AgentStep.TOOL_RESULT.value,
                tool="lookup_medical_code",
                code_type="icd10",
                count=len(results),
            )
            return formatted
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError("lookup_medical_code", str(exc), exc) from exc

    # ── CPT ────────────────────────────────────────────────────────────────────
    if code_type == "cpt":
        try:
            db = _CodeDatabase.get()
        except Exception as exc:
            raise ToolExecutionError("lookup_medical_code", str(exc), exc) from exc

        if not db.cpt_available():
            return _CPT_NOTICE

        try:
            results = _fuzzy_search(
                query,
                db._cpt_codes,
                db._cpt_descs,
                limit=max(1, min(max_results, 50)),
                threshold=fuzzy_threshold,
            )
            formatted = _format_results(results, "cpt", query)
            _log.info(
                "tool_result",
                step=AgentStep.TOOL_RESULT.value,
                tool="lookup_medical_code",
                code_type="cpt",
                count=len(results),
            )
            return formatted
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError("lookup_medical_code", str(exc), exc) from exc

    # ── HCPCS — delegate to NLM ClinicalTables (public, no license needed) ────
    if code_type == "hcpcs":
        try:
            import requests as _req

            url = "https://clinicaltables.nlm.nih.gov/api/hcpcs/v3/search"
            params = {
                "sf": "code,description",
                "df": "code,description",
                "terms": query,
                "maxList": max(1, min(max_results, 50)),
            }
            resp = _req.get(url, params=params, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            rows: list[list[str]] = (payload[3] if isinstance(payload, list) and len(payload) >= 4 else []) or []
            if not rows:
                return f"No HCPCS Level II codes found matching **{query}**."

            lines = [f"## HCPCS Level II — \"{query}\"\n"]
            for row in rows[:max_results]:
                code = row[0] if len(row) > 0 else "—"
                desc = row[1] if len(row) > 1 else "—"
                lines.append(f"- **{code}** — {desc}")
            lines.append(
                f"\n*{len(rows)} result(s) from NLM ClinicalTables. "
                "Verify against current CMS HCPCS release.*"
            )
            return "\n".join(lines)
        except Exception as exc:
            raise ToolExecutionError("lookup_medical_code", f"HCPCS lookup failed: {exc}", exc) from exc

    return (
        f"Unknown code_type '{code_type}'. "
        "Valid options: 'icd10', 'cpt', 'hcpcs'."
    )


@tool(args_schema=CodeDescribeInput)
def describe_medical_code(code: str, code_type: str = "icd10") -> str:
    """Return the official description for a specific ICD-10-CM, CPT, or HCPCS code.

    Use when you already have a code and need its full official description,
    or to verify that a code exists and is valid.

    Args:
        code: Specific code to look up (e.g. 'E11.65', 'Z23', 'A4253').
        code_type: Code set: 'icd10' (default), 'cpt', or 'hcpcs'.

    Returns:
        Code description, or a not-found message.

    Raises:
        ToolExecutionError: If the underlying data source is unavailable.
    """
    return lookup_medical_code.invoke(
        {
            "query": code.strip(),
            "code_type": code_type,
            "max_results": 5,
            "fuzzy_threshold": 90,  # tight threshold for exact-code lookups
        }
    )
