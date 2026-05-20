"""Healthcare QA tool registry."""
from src.tools.code_lookup_tool import describe_medical_code, lookup_medical_code
from src.tools.exceptions import ToolExecutionError
from src.tools.guidelines_tool import get_clinical_guidelines, ingest_guidelines_pdfs
from src.tools.pubmed_tool import search_pubmed

ALL_TOOLS = [
    search_pubmed,
    get_clinical_guidelines,
    lookup_medical_code,
    describe_medical_code,
]

__all__ = [
    "ALL_TOOLS",
    "ToolExecutionError",
    "search_pubmed",
    "get_clinical_guidelines",
    "ingest_guidelines_pdfs",
    "lookup_medical_code",
    "describe_medical_code",
]
