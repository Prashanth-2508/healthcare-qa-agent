"""Input validation, PII scrubbing, content filtering, and tool-output schema validation.

Public surface:
  GuardrailError          — raised when a query or output violates safety rules
  validate_query()        — token-length check, PII strip, diagnosis-flag detection
  validate_tool_output()  — ensures tool dicts conform to ToolOutputEnvelope schema
  run_all_guardrails()    — legacy entry-point used by agent.py pre-graph checks
  QueryInput              — Pydantic model for raw user input
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Custom exception ──────────────────────────────────────────────────────────

class GuardrailError(Exception):
    """Raised when a query or output violates a safety or compliance rule.

    Attributes:
        reason: Human-readable explanation shown to the user.
        rule:   Machine-readable rule identifier (for logging/metrics).
        safe_response: Optional pre-built fallback message for the agent.
    """

    def __init__(
        self,
        reason: str,
        rule: str = "unspecified",
        safe_response: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.rule = rule
        self.safe_response = safe_response or (
            "I'm unable to process this request due to a safety policy. "
            f"Reason: {reason}"
        )

    def __repr__(self) -> str:
        return f"GuardrailError(rule={self.rule!r}, reason={self.reason!r})"


# ── Pydantic input schemas ─────────────────────────────────────────────────────

class QueryInput(BaseModel):
    """Validated, normalised user query."""

    query: str = Field(..., min_length=5, max_length=8000)
    session_id: str = Field(default="default")

    @field_validator("query")
    @classmethod
    def strip_and_normalize(cls, v: str) -> str:
        return " ".join(v.strip().split())


class PubMedSearchInput(BaseModel):
    terms: str = Field(..., min_length=2, max_length=500)
    max_results: int = Field(default=5, ge=1, le=20)


class CodeLookupInput(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    code_type: str = Field(default="icd10")

    @field_validator("code_type")
    @classmethod
    def validate_code_type(cls, v: str) -> str:
        allowed = {"icd10", "cpt", "hcpcs"}
        v = v.lower()
        if v not in allowed:
            raise ValueError(f"code_type must be one of {allowed}")
        return v


# ── Tool-output Pydantic schemas ──────────────────────────────────────────────

class ToolOutputEnvelope(BaseModel):
    """Expected shape of every ToolResult dict produced by the act node."""

    tool: str = Field(..., min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    output: str = Field(..., min_length=0)
    success: bool
    duration_ms: float = Field(ge=0.0)
    error: str | None = None

    @model_validator(mode="after")
    def error_only_on_failure(self) -> "ToolOutputEnvelope":
        if self.success and self.error:
            # Tolerate: mark as failed if error text was set
            object.__setattr__(self, "success", False)
        return self


class SourceCitation(BaseModel):
    """One bibliographic citation in the structured agent response."""

    citation: str = Field(..., min_length=3)
    url: str = Field(default="")
    type: Literal["pubmed", "guideline", "cms", "other"] = "other"


class ICD10CodeRef(BaseModel):
    """ICD-10-CM code reference returned in the structured response."""

    code: str = Field(..., pattern=r"^[A-Z]\d{2}(\.\w+)?$")
    description: str = Field(..., min_length=3)
    usage_note: str = Field(default="")


class StructuredAgentResponse(BaseModel):
    """Schema for the JSON object the respond node must produce."""

    chain_of_thought: str = Field(..., min_length=10)
    answer: str = Field(..., min_length=10)
    sources: list[SourceCitation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low", "insufficient_evidence"]
    follow_up_questions: list[str] = Field(default_factory=list)
    icd10_codes: list[ICD10CodeRef] = Field(default_factory=list)


# ── Compiled PII patterns ─────────────────────────────────────────────────────

# SSN — XXX-XX-XXXX (with or without dashes)
_SSN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")

# Date-of-birth written out
_DOB_EXPLICIT = re.compile(
    r"\b(?:dob|date\s+of\s+birth|born(?:\s+on)?)[:\s]+\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b",
    re.IGNORECASE,
)

# ISO-format DOB inside a clinical context phrase
_DOB_ISO = re.compile(
    r"\b(?:dob|date\s+of\s+birth)[:\s]+\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)

# Medical Record Number — common representations
_MRN = re.compile(
    r"\b(?:mrn|medical\s+record\s+(?:number|no\.?|#))[:\s#]*\d{4,10}\b",
    re.IGNORECASE,
)

# NPI (10-digit national provider identifier)
_NPI = re.compile(r"\bNPI[:\s#]*\d{10}\b", re.IGNORECASE)

# DEA number — 2 letters + 7 digits
_DEA = re.compile(r"\b[A-Z]{2}\d{7}\b")

# Phone numbers — US formats
_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")

# Email address
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Patient name patterns (simple heuristic: "patient [Name]" or "for [First Last]")
_PATIENT_NAME = re.compile(
    r"\bpatient\s+(?:name\s*[:\-]?\s*)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)

_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_SSN, "[SSN REDACTED]"),
    (_DOB_EXPLICIT, "[DOB REDACTED]"),
    (_DOB_ISO, "[DOB REDACTED]"),
    (_MRN, "[MRN REDACTED]"),
    (_NPI, "[NPI REDACTED]"),
    (_DEA, "[DEA REDACTED]"),
    (_PHONE, "[PHONE REDACTED]"),
    (_EMAIL, "[EMAIL REDACTED]"),
]

# ── Token-length estimation ───────────────────────────────────────────────────

_CHARS_PER_TOKEN = 4        # conservative BPE estimate
_MAX_TOKENS = 2000
_MAX_CHARS = _MAX_TOKENS * _CHARS_PER_TOKEN  # 8 000 chars


def _estimate_tokens(text: str) -> int:
    """Rough token count: 1 token ≈ 4 characters (conservative for medical text)."""
    return max(len(text) // _CHARS_PER_TOKEN, len(text.split()))


# ── Disallowed content patterns ───────────────────────────────────────────────

# Patterns where users are directly asking for a personal clinical diagnosis.
_DIAGNOSIS_REQUEST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bdo\s+i\s+have\b",
        r"\bam\s+i\s+(?:having|suffering|experiencing|at\s+risk)\b",
        r"\bdiagnose\s+me\b",
        r"\bwhat(?:'s|\s+is)\s+wrong\s+with\s+me\b",
        r"\bis\s+(?:this|it)\s+(?:cancer|diabetes|covid|hiv|aids|a\s+heart\s+attack)\b",
        r"\bdo\s+i\s+(?:have|need)\s+(?:surgery|a\s+prescription|chemotherapy|dialysis)\b",
    ]
]

# Prescription-dosing requests lacking clinical context.
_DOSING_WITHOUT_CONTEXT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bhow\s+(?:much|many)\s+(?:mg|milligrams?|pills?|tablets?|doses?)\s+(?:should\s+i|do\s+i|can\s+i)\s+take\b",
        r"\bwhat(?:'s|\s+is)\s+(?:my|the\s+correct|the\s+right|the\s+maximum)\s+dose\b",
        r"\bprescribe\s+(?:me|for\s+me)\b",
        r"\bgive\s+me\s+(?:a\s+)?prescription\b",
        r"\bcan\s+you\s+write\s+(?:me\s+)?(?:a\s+)?(?:script|prescription|rx)\b",
        r"\bhow\s+(?:do\s+i|should\s+i)\s+(?:take|use|inject|administer)\s+\w+\s+(?:for|to\s+treat)\b",
    ]
]

# Controlled / high-risk substance requests
_CONTROLLED_SUBSTANCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:oxycodone|oxycontin|fentanyl|hydrocodone|morphine|xanax|valium|adderall|ritalin)\b.{0,80}\b(?:dose|how\s+much|prescription|get)\b",
        r"\bhow\s+to\s+(?:get|obtain|buy|order)\s+(?:opioids?|narcotics?|controlled\s+substances?)\b",
        r"\bwithout\s+(?:a\s+)?prescription\b.{0,40}\b(?:opioid|narcotic|controlled)\b",
    ]
]

# Emergency situations
_EMERGENCY_KEYWORDS: list[str] = [
    "chest pain", "heart attack", "cardiac arrest", "stroke", "can't breathe",
    "cannot breathe", "difficulty breathing", "suicidal", "overdose",
    "severe bleeding", "unconscious", "unresponsive", "anaphylaxis",
    "allergic reaction", "seizure", "choking",
]

# ── Legacy guardrail dataclass (kept for backward compatibility) ───────────────

@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""


# ── Core validation functions ─────────────────────────────────────────────────

def validate_query(query: str) -> str:
    """Validate and sanitise an incoming user query.

    Steps (in order):
    1. Token-length check — raises GuardrailError if > 2 000 tokens.
    2. PII stripping — replaces SSN / DOB / MRN / NPI / DEA / phone / email with
       redaction placeholders and returns the scrubbed text.
    3. Emergency detection — raises GuardrailError with 911 redirect.
    4. Diagnosis-request detection — raises GuardrailError with clinician referral.
    5. Dosing-without-context detection — raises GuardrailError.
    6. Controlled-substance detection — raises GuardrailError.

    Args:
        query: Raw user query string.

    Returns:
        Cleaned, PII-scrubbed query string.

    Raises:
        GuardrailError: On token overflow, emergency signal, or disallowed content.
    """
    # 1. Token length
    token_estimate = _estimate_tokens(query)
    if token_estimate > _MAX_TOKENS:
        raise GuardrailError(
            reason=(
                f"Your query is too long (~{token_estimate} tokens; limit is {_MAX_TOKENS}). "
                "Please shorten your question and try again."
            ),
            rule="token_limit_exceeded",
            safe_response=(
                "The submitted query exceeds the maximum allowed length. "
                f"Please reduce your query to under {_MAX_TOKENS} tokens and resubmit."
            ),
        )

    # 2. PII stripping (mutates a working copy; does NOT raise)
    scrubbed = query
    for pattern, placeholder in _PII_PATTERNS:
        scrubbed = pattern.sub(placeholder, scrubbed)

    lower = scrubbed.lower()

    # 3. Emergency detection
    for keyword in _EMERGENCY_KEYWORDS:
        if keyword in lower:
            raise GuardrailError(
                reason=(
                    f"Your query mentions '{keyword}', which may indicate a medical emergency. "
                    "Please call 911 or your local emergency services immediately."
                ),
                rule="emergency_detected",
                safe_response=(
                    "⚠️ **Medical Emergency Detected**\n\n"
                    "Your message contains language suggesting a potential medical emergency. "
                    "**Please call 911 or your local emergency number immediately.** "
                    "This tool cannot assist in emergency situations."
                ),
            )

    # 4. Diagnosis request
    for pattern in _DIAGNOSIS_REQUEST_PATTERNS:
        if pattern.search(scrubbed):
            raise GuardrailError(
                reason="Query appears to request a personal clinical diagnosis.",
                rule="diagnosis_request",
                safe_response=(
                    "I'm unable to provide a personal clinical diagnosis. "
                    "This tool supports **clinical documentation and coding professionals** "
                    "with evidence-based information.\n\n"
                    "For personal health concerns, please consult a licensed clinician. "
                    "I can answer general questions about conditions, treatments, and coding."
                ),
            )

    # 5. Dosing without clinical context
    for pattern in _DOSING_WITHOUT_CONTEXT_PATTERNS:
        if pattern.search(scrubbed):
            raise GuardrailError(
                reason="Query requests specific personal medication dosing without clinical context.",
                rule="dosing_without_context",
                safe_response=(
                    "I cannot provide personalised medication dosing instructions. "
                    "Dosing decisions depend on individual clinical factors that only a licensed "
                    "prescriber can evaluate.\n\n"
                    "I can provide general pharmacological information, drug class summaries, "
                    "or help with coding documentation. Would one of those be useful?"
                ),
            )

    # 6. Controlled substance
    for pattern in _CONTROLLED_SUBSTANCE_PATTERNS:
        if pattern.search(scrubbed):
            raise GuardrailError(
                reason="Query requests controlled-substance access or dosing without clinical authorisation.",
                rule="controlled_substance",
                safe_response=(
                    "I cannot assist with obtaining or dosing controlled substances. "
                    "Please consult a licensed prescribing clinician."
                ),
            )

    return scrubbed


def validate_tool_output(output: dict[str, Any]) -> dict[str, Any]:
    """Validate a raw ToolResult dict against the ToolOutputEnvelope schema.

    Raises:
        GuardrailError: If the output fails schema validation.

    Returns:
        The validated output dict (unchanged if valid).
    """
    from pydantic import ValidationError  # local import to keep top-level clean

    try:
        envelope = ToolOutputEnvelope.model_validate(output)
        return envelope.model_dump()
    except ValidationError as exc:
        tool_name = output.get("tool", "unknown")
        raise GuardrailError(
            reason=f"Tool '{tool_name}' returned an output that failed schema validation: {exc}",
            rule="tool_output_schema_violation",
            safe_response=(
                f"The tool '{tool_name}' returned an unexpected response format. "
                "The result has been discarded for safety. Please try a different query."
            ),
        ) from exc


def validate_structured_response(response: dict[str, Any]) -> StructuredAgentResponse:
    """Validate the structured JSON response from the respond node.

    Raises:
        GuardrailError: If required fields are missing or malformed.

    Returns:
        Parsed StructuredAgentResponse model.
    """
    from pydantic import ValidationError

    try:
        return StructuredAgentResponse.model_validate(response)
    except ValidationError as exc:
        raise GuardrailError(
            reason=f"Agent response failed schema validation: {exc}",
            rule="response_schema_violation",
            safe_response=(
                "The agent produced a response in an unexpected format. "
                "Please rephrase your question and try again."
            ),
        ) from exc


# ── Legacy guardrail functions (kept for backward compatibility) ───────────────

def _check_prescription_request_legacy(query: str) -> GuardrailResult:
    for pattern in _DOSING_WITHOUT_CONTEXT_PATTERNS:
        if pattern.search(query):
            return GuardrailResult(
                passed=False,
                reason=(
                    "I can provide general medical information but cannot "
                    "prescribe medications. Please consult a licensed physician."
                ),
            )
    return GuardrailResult(passed=True)


def check_prescription_request(query: str) -> GuardrailResult:
    return _check_prescription_request_legacy(query)


def check_emergency_situation(query: str) -> GuardrailResult:
    lower = query.lower()
    for keyword in _EMERGENCY_KEYWORDS:
        if keyword in lower:
            return GuardrailResult(
                passed=False,
                reason=(
                    f"Your query mentions '{keyword}', which may indicate a "
                    "medical emergency. Please call 911 or your local emergency "
                    "services immediately. Do not rely on this tool in emergencies."
                ),
            )
    return GuardrailResult(passed=True)


def check_pii(query: str) -> GuardrailResult:
    """Detect (but do not strip) PII for the legacy guardrail pipeline."""
    for pattern, _ in _PII_PATTERNS:
        if pattern.search(query):
            return GuardrailResult(
                passed=False,
                reason=(
                    "Query appears to contain personal identifiable information (PII). "
                    "Remove patient-identifying details before submitting."
                ),
            )
    return GuardrailResult(passed=True)


def run_all_guardrails(query: str) -> GuardrailResult:
    """Run legacy guardrails in sequence; returns the first failure or success.

    Used by agent.py before graph invocation.
    For new code, prefer validate_query() which also strips PII and raises GuardrailError.
    """
    for check in [check_emergency_situation, check_prescription_request, check_pii]:
        result = check(query)
        if not result.passed:
            return result
    return GuardrailResult(passed=True)
