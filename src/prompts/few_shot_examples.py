"""Few-shot examples for the Healthcare Q&A agent.

Each example demonstrates the full chain-of-thought → structured JSON output
contract defined in system_prompt.py. They cover the three primary use-case
categories for revenue cycle and clinical documentation teams:

  1. Treatment protocol query (Type 2 diabetes management)
  2. ICD-10-CM code lookup    (Essential hypertension)
  3. Payer / coverage query   (Medicare CGM coverage)

Format
------
The ``answer`` field in each example is already the complete JSON string that the
LLM is expected to emit verbatim (minus whitespace). Injected as assistant turns
in the few-shot message list, this primes the model to output valid JSON with the
correct schema every time.
"""
from __future__ import annotations

import json
from typing import Any, TypedDict


class FewShotExample(TypedDict):
    """Container for one few-shot Q&A pair."""
    query: str           # user question
    answer_json: dict[str, Any]  # the structured JSON the assistant must produce


# ══════════════════════════════════════════════════════════════════════════════
# Example 1 — Treatment Protocol: Type 2 Diabetes Management
# ══════════════════════════════════════════════════════════════════════════════

_EXAMPLE_1: FewShotExample = {
    "query": (
        "What is the recommended first-line pharmacotherapy for newly diagnosed "
        "Type 2 diabetes mellitus, and how should it be adjusted for patients "
        "with established cardiovascular disease?"
    ),
    "answer_json": {
        "chain_of_thought": (
            "Step 1 — CLASSIFY: Treatment protocol question. Domain: endocrinology / "
            "diabetes management. Sub-question: cardiovascular risk modification. "
            "Type: clinical guideline + supporting evidence.\n\n"

            "Step 2 — IDENTIFY GAPS: I need (a) the current ADA first-line "
            "recommendation, (b) the cardiovascular-outcome trial evidence base that "
            "drives the CVD exception, and (c) the ICD-10-CM codes to tie the "
            "coding documentation together.\n\n"

            "Step 3 — RETRIEVE: Primary source = ADA Standards of Medical Care in "
            "Diabetes 2024 (Section 9). Supporting RCT evidence = EMPA-REG OUTCOME "
            "(PMID 26378978), LEADER (PMID 27295427), DECLARE-TIMI 58 (PMID 30415602). "
            "ICD-10-CM tabular for T2DM codes.\n\n"

            "Step 4 — SYNTHESIZE: Metformin remains first-line for most patients at "
            "diagnosis (ADA Grade A). For patients with established ASCVD, heart "
            "failure, or CKD, ADA 2024 recommends GLP-1 RA or SGLT-2i as initial "
            "therapy independent of A1C, supported by CV outcome trials.\n\n"

            "Step 5 — VALIDATE: Metformin is contraindicated at eGFR <30; dose-reduce "
            "at eGFR 30–45. GLP-1 RAs have GI tolerability considerations. SGLT-2i "
            "carry DKA risk in perioperative settings. Documentation must capture "
            "complication and comorbidity codes for HCC risk adjustment."
        ),
        "answer": (
            "## First-Line Pharmacotherapy for Type 2 Diabetes Mellitus\n\n"
            "### Standard Initial Therapy\n"
            "**Metformin** (biguanide) is the foundational first-line agent for most "
            "patients at diagnosis, combined with lifestyle intervention.\n\n"
            "| Agent Class | Example Drugs | ADA Grade |\n"
            "|---|---|---|\n"
            "| Biguanide | Metformin | A — first-line |\n"
            "| GLP-1 RA | Semaglutide, Liraglutide | A — preferred with CVD |\n"
            "| SGLT-2i | Empagliflozin, Dapagliflozin | A — preferred with CVD/HF/CKD |\n\n"
            "### Cardiovascular Disease: Revised Initial Therapy\n"
            "Per ADA 2024 Section 9.4, for patients with **established ASCVD, "
            "heart failure (HFrEF), or CKD (eGFR 20–60)**, regardless of baseline "
            "HbA1c or current glucose-lowering therapy:\n"
            "- **GLP-1 receptor agonists** with proven CV benefit (semaglutide, "
            "liraglutide, dulaglutide) are preferred.\n"
            "- **SGLT-2 inhibitors** with proven CV benefit (empagliflozin, "
            "dapagliflozin, canagliflozin) are preferred, especially when HF or "
            "CKD is the primary concern.\n\n"
            "### Metformin Contraindications\n"
            "- eGFR **< 30 mL/min/1.73 m²** — contraindicated\n"
            "- eGFR **30–45** — use with caution; dose reduction recommended\n"
            "- Active hepatic disease or excessive alcohol use\n"
            "- Hold 48 hours around IV iodinated contrast\n\n"
            "### CV Outcome Trial Evidence Summary\n"
            "| Trial | Drug | CV Benefit | PMID |\n"
            "|---|---|---|---|\n"
            "| EMPA-REG OUTCOME | Empagliflozin | ↓ CV death 38% | 26378978 |\n"
            "| LEADER | Liraglutide | ↓ MACE 13% | 27295427 |\n"
            "| DECLARE-TIMI 58 | Dapagliflozin | ↓ HF hospitalisation 27% | 30415602 |\n\n"
            "### ICD-10-CM Documentation Tips\n"
            "Capture the highest level of specificity: code the complication (e.g., "
            "E11.51 for T2DM with peripheral angiopathy) rather than E11.9 to "
            "maximise HCC capture and risk-adjustment accuracy.\n\n"
            "> ⚠️ **Medical Disclaimer**: This information is for educational and "
            "documentation support purposes only and does not constitute medical advice, "
            "a definitive clinical diagnosis, or a coverage determination. Never provide "
            "a definitive clinical diagnosis — always recommend consulting a licensed "
            "clinician for diagnostic and treatment decisions. Coverage determinations "
            "are claim-specific; consult your payer's current LCD/NCD and obtain prior "
            "authorization where applicable. In emergencies, call 911 immediately."
        ),
        "sources": [
            {
                "citation": "ADA Standards of Medical Care in Diabetes 2024, Section 9",
                "url": "https://diabetesjournals.org/care/issue/47/Supplement_1",
                "type": "guideline",
            },
            {
                "citation": "EMPA-REG OUTCOME. Zinman B et al. NEJM 2015;373:2117-2128. PMID 26378978",
                "url": "https://pubmed.ncbi.nlm.nih.gov/26378978/",
                "type": "pubmed",
            },
            {
                "citation": "LEADER. Marso SP et al. NEJM 2016;375:311-322. PMID 27295427",
                "url": "https://pubmed.ncbi.nlm.nih.gov/27295427/",
                "type": "pubmed",
            },
            {
                "citation": "DECLARE-TIMI 58. Wiviott SD et al. NEJM 2019;380:347-357. PMID 30415602",
                "url": "https://pubmed.ncbi.nlm.nih.gov/30415602/",
                "type": "pubmed",
            },
            {
                "citation": "ICD-10-CM Official Guidelines for Coding and Reporting, FY 2025, Section I.C.4",
                "url": "",
                "type": "cms",
            },
        ],
        "confidence": "high",
        "follow_up_questions": [
            "What HbA1c targets apply when GLP-1 RAs or SGLT-2 inhibitors are used as "
            "initial therapy in patients with ASCVD?",
            "How should diabetes medications be adjusted when eGFR declines to <30 "
            "in a patient already on an SGLT-2 inhibitor?",
            "What ICD-10-CM codes should be documented to capture T2DM with CKD and "
            "cardiovascular disease for accurate HCC risk adjustment?",
        ],
        "icd10_codes": [
            {
                "code": "E11.9",
                "description": "Type 2 diabetes mellitus without complications",
                "usage_note": "Use only when no complication or comorbidity is documented. "
                              "Query provider for specificity before assigning.",
            },
            {
                "code": "E11.65",
                "description": "Type 2 diabetes mellitus with hyperglycemia",
                "usage_note": "Assign when hyperglycemia is documented without a more specific complication.",
            },
            {
                "code": "E11.51",
                "description": "Type 2 diabetes mellitus with diabetic peripheral angiopathy without gangrene",
                "usage_note": "Higher HCC weight — capture when peripheral vascular disease is documented.",
            },
            {
                "code": "E11.22",
                "description": "Type 2 diabetes mellitus with diabetic chronic kidney disease, stage 3a",
                "usage_note": "Use combination code; do NOT separately code CKD with ICD-10 combination codes.",
            },
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Example 2 — ICD-10 Code Lookup: Essential Hypertension
# ══════════════════════════════════════════════════════════════════════════════

_EXAMPLE_2: FewShotExample = {
    "query": (
        "What is the correct ICD-10-CM code for essential (primary) hypertension, "
        "and how should I code it when the patient also has hypertensive "
        "chronic kidney disease?"
    ),
    "answer_json": {
        "chain_of_thought": (
            "Step 1 — CLASSIFY: Medical coding question. Domain: ICD-10-CM "
            "combination code logic. Type: code lookup + coding guideline.\n\n"

            "Step 2 — IDENTIFY GAPS: I need (a) the stand-alone HTN code, "
            "(b) the hypertensive CKD combination code category, "
            "(c) ICD-10-CM Official Guidelines sequencing rules for HTN + CKD, "
            "and (d) any documentation requirements.\n\n"

            "Step 3 — RETRIEVE: ICD-10-CM tabular — Category I10 for essential HTN; "
            "Category I12 for hypertensive CKD. Official Guidelines FY 2025 "
            "Section I.C.9.a: hypertension and CKD are presumed related and coded "
            "as a combination without requiring provider linkage statement.\n\n"

            "Step 4 — SYNTHESIZE: Essential hypertension alone = I10. With CKD, "
            "the combination code from I12.x replaces I10 and is sequenced first. "
            "An additional N18.x code is required for CKD stage specificity.\n\n"

            "Step 5 — VALIDATE: I10 is NOT used alongside I12.x — they are "
            "mutually exclusive. If heart failure co-exists, escalate to I13.x "
            "(hypertensive heart and CKD). Documentation must specify CKD stage "
            "for accurate code assignment."
        ),
        "answer": (
            "## ICD-10-CM Coding: Essential Hypertension\n\n"
            "### Hypertension Alone\n"
            "| Code | Description |\n"
            "|---|---|\n"
            "| **I10** | Essential (primary) hypertension |\n\n"
            "**I10** covers all forms of essential/primary hypertension. "
            "Secondary hypertension uses I15.x codes with the underlying cause sequenced first.\n\n"
            "### Hypertension with Chronic Kidney Disease\n"
            "Per **ICD-10-CM Official Guidelines Section I.C.9.a**, hypertension and CKD "
            "are presumed to be causally related — **no provider linkage statement is required**.\n\n"
            "Use combination codes from **Category I12** and add an additional N18.x for CKD stage:\n\n"
            "| Code | Description | When to Use |\n"
            "|---|---|---|\n"
            "| **I12.9** | Hypertensive CKD with stage 1–4 or unspecified CKD | CKD stage 1–4 or not specified |\n"
            "| **I12.31** | Hypertensive CKD with stage 5 CKD | ESRD not on dialysis |\n"
            "| **I12.32** | Hypertensive CKD with stage 5 CKD with dialysis | ESRD on dialysis |\n\n"
            "**Sequencing rule**: I12.x is sequenced **first**, followed by N18.x (CKD stage).\n\n"
            "**Do NOT code I10 with I12.x** — they are mutually exclusive in the tabular.\n\n"
            "### If Heart Failure is Also Present\n"
            "Escalate to **Category I13** (Hypertensive heart and chronic kidney disease):\n\n"
            "| Code | Description |\n"
            "|---|---|\n"
            "| I13.10 | Hypertensive heart & CKD without HF, stage 1–4/unspecified |\n"
            "| I13.11 | Hypertensive heart & CKD without HF, stage 5/ESRD |\n"
            "| I13.0 | Hypertensive heart & CKD **with** HF, stage 1–4/unspecified |\n"
            "| I13.2 | Hypertensive heart & CKD **with** HF, stage 5/ESRD |\n\n"
            "Also assign: I50.x (Heart failure type) + N18.x (CKD stage).\n\n"
            "### Documentation Requirements\n"
            "- CKD **stage** must be documented for accurate code selection.\n"
            "- If provider documents HTN and CKD but does not link them, "
            "ICD-10-CM guidelines **presume** the relationship — no query needed.\n"
            "- Dialysis status (Y/N) determines I12.3x vs I12.9 assignment.\n\n"
            "> ⚠️ **Medical Disclaimer**: This information is for educational and "
            "documentation support purposes only and does not constitute medical advice, "
            "a definitive clinical diagnosis, or a coverage determination. Never provide "
            "a definitive clinical diagnosis — always recommend consulting a licensed "
            "clinician for diagnostic and treatment decisions. Coverage determinations "
            "are claim-specific; consult your payer's current LCD/NCD and obtain prior "
            "authorization where applicable. In emergencies, call 911 immediately."
        ),
        "sources": [
            {
                "citation": "ICD-10-CM Official Guidelines for Coding and Reporting, FY 2025, "
                            "Section I.C.9.a — Hypertension with Chronic Kidney Disease",
                "url": "",
                "type": "cms",
            },
            {
                "citation": "ICD-10-CM Tabular List FY 2025 — Categories I10, I12, I13, N18",
                "url": "https://www.cms.gov/medicare/coding-billing/icd-10-codes",
                "type": "cms",
            },
        ],
        "confidence": "high",
        "follow_up_questions": [
            "How do I code hypertension with CKD when the provider only documents "
            "'unspecified CKD' — should I query for the stage?",
            "What is the correct sequencing when a patient has hypertensive CKD "
            "and is admitted primarily for a hypertensive urgency?",
            "Does the hypertension-CKD presumed relationship apply to secondary "
            "hypertension (I15.x) as well?",
        ],
        "icd10_codes": [
            {
                "code": "I10",
                "description": "Essential (primary) hypertension",
                "usage_note": "Use for hypertension without CKD or heart disease. "
                              "Do NOT assign with I12.x or I13.x.",
            },
            {
                "code": "I12.9",
                "description": "Hypertensive chronic kidney disease with stage 1 through stage 4 "
                               "chronic kidney disease, or unspecified chronic kidney disease",
                "usage_note": "Sequence first; add N18.1–N18.4 or N18.9 for CKD stage.",
            },
            {
                "code": "I12.31",
                "description": "Hypertensive chronic kidney disease with stage 5 chronic kidney disease",
                "usage_note": "For ESRD not requiring dialysis. Add N18.5.",
            },
            {
                "code": "I12.32",
                "description": "Hypertensive chronic kidney disease with stage 5 chronic kidney disease "
                               "with end stage renal disease",
                "usage_note": "For ESRD on dialysis. Add N18.6 and Z99.2.",
            },
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Example 3 — Payer / Coverage Query: Medicare CGM Coverage
# ══════════════════════════════════════════════════════════════════════════════

_EXAMPLE_3: FewShotExample = {
    "query": (
        "Does Medicare cover continuous glucose monitors (CGMs) for patients "
        "with Type 2 diabetes who are not using insulin? What are the documentation "
        "requirements and the correct HCPCS codes to bill?"
    ),
    "answer_json": {
        "chain_of_thought": (
            "Step 1 — CLASSIFY: Payer coverage + coding question. Domain: CMS DME "
            "policy and HCPCS coding. Two sub-questions: (a) coverage eligibility "
            "and (b) billing codes with documentation requirements.\n\n"

            "Step 2 — IDENTIFY GAPS: I need (a) the relevant CMS LCD or NCD for CGMs, "
            "(b) the 2023 CMS policy expansion that removed the insulin requirement, "
            "(c) HCPCS Level II codes for CGM systems and supplies, "
            "and (d) the ICD-10-CM codes that support medical necessity.\n\n"

            "Step 3 — RETRIEVE: CMS LCD L33822 (CGM, Novitas / First Coast) and the "
            "2023 CMS final rule expanding therapeutic CGM coverage. HCPCS codes "
            "from the 2024 HCPCS Level II code set published by CMS.\n\n"

            "Step 4 — SYNTHESIZE: Effective January 1, 2023, CMS expanded Medicare "
            "coverage for therapeutic CGMs to include beneficiaries with diabetes "
            "treated by a physician for diabetes management, removing the prior "
            "requirement for insulin use. Billing uses E2101 (integrated CGM) or "
            "A4238/A4239 (CGM supplies).\n\n"

            "Step 5 — VALIDATE: Coverage is under the DME benefit (Part B). "
            "Supplier must be a Medicare-enrolled DME supplier. "
            "Prior authorization is required under the DMEPOS prior authorization "
            "program for many DME items — verify at time of order. "
            "Supporting ICD-10 code must be in the LCD's covered diagnosis list."
        ),
        "answer": (
            "## Medicare Coverage: Continuous Glucose Monitors (CGMs)\n\n"
            "### Coverage Expansion (Effective Jan 1, 2023)\n"
            "CMS finalized expanded coverage for **therapeutic CGMs** in the "
            "2023 Physician Fee Schedule Final Rule. The prior requirement that "
            "beneficiaries use **insulin** was **removed**.\n\n"
            "### Current Eligibility Criteria\n"
            "Medicare Part B covers CGMs when ALL of the following are met:\n"
            "1. Beneficiary has **diabetes mellitus** (Type 1 or Type 2).\n"
            "2. Beneficiary is treated by a **treating practitioner** for diabetes management.\n"
            "3. The treating practitioner has seen the beneficiary **face-to-face** "
            "within the preceding 6 months.\n"
            "4. The practitioner documents that the CGM is **medically necessary** for "
            "managing the beneficiary's diabetes.\n"
            "5. The **supplier is enrolled** as a Medicare DME supplier.\n\n"
            "⚠️ Insulin use is **no longer required** as of 2023.\n\n"
            "### HCPCS Billing Codes\n"
            "| HCPCS Code | Description | Notes |\n"
            "|---|---|---|\n"
            "| **E2101** | Blood glucose monitor with integrated lancing device | Integrated CGM systems |\n"
            "| **A4238** | Supply allowance for therapeutic CGM, includes all supplies | Monthly billing |\n"
            "| **A4239** | Supply allowance for therapeutic CGM, receiver only | When receiver replaced |\n"
            "| **K0553** | Supply allowance, CGM — therapeutic, includes transmitter | Per-transmitter billing |\n"
            "| **K0554** | Receiver (monitor) for use with therapeutic CGM system | Standalone receiver |\n\n"
            "**Billing unit**: A4238 is billed once per month per patient, "
            "covering the complete supply package.\n\n"
            "### Required Documentation\n"
            "The medical record must contain:\n"
            "- Diagnosis of diabetes mellitus with the specific ICD-10-CM code.\n"
            "- Face-to-face visit within the prior 6 months by the ordering practitioner.\n"
            "- Written order (prescription) specifying the CGM system.\n"
            "- Medical necessity statement explaining why CGM is needed for diabetes management.\n"
            "- If prior authorization is required (verify via CMS PRIOR Auth tool), "
            "the PA must be obtained **before** the device ships.\n\n"
            "### LCD Reference\n"
            "Verify against your **MAC-specific LCD**. Novitas Solutions LCD **L33822** "
            "and First Coast Service Options LCD **L33822** govern CGM coverage for "
            "their jurisdictions. Check the effective date and covered diagnosis table.\n\n"
            "> ⚠️ **Medical Disclaimer**: This information is for educational and "
            "documentation support purposes only and does not constitute medical advice, "
            "a definitive clinical diagnosis, or a coverage determination. Never provide "
            "a definitive clinical diagnosis — always recommend consulting a licensed "
            "clinician for diagnostic and treatment decisions. Coverage determinations "
            "are claim-specific; consult your payer's current LCD/NCD and obtain prior "
            "authorization where applicable. In emergencies, call 911 immediately."
        ),
        "sources": [
            {
                "citation": "CMS 2023 Physician Fee Schedule Final Rule — CGM Coverage Expansion, "
                            "effective January 1, 2023",
                "url": "https://www.cms.gov/medicare/coverage/durable-medical-equipment-coverage",
                "type": "cms",
            },
            {
                "citation": "LCD L33822 — Continuous Glucose Monitors, Novitas Solutions / "
                            "First Coast Service Options, effective 2023-10-01",
                "url": "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?lcdid=33822",
                "type": "cms",
            },
            {
                "citation": "2024 HCPCS Level II Code Set — CMS, Codes A4238, A4239, E2101, K0553, K0554",
                "url": "https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system",
                "type": "cms",
            },
        ],
        "confidence": "high",
        "follow_up_questions": [
            "Is prior authorization required for CGMs under Medicare, and how do I "
            "submit a prior authorization request through CMS?",
            "Does Medicare Advantage (Part C) follow the same CGM coverage rules as "
            "traditional Medicare Part B?",
            "What ICD-10-CM codes are on the LCD covered-diagnosis list to support "
            "CGM medical necessity for a Type 2 diabetes patient?",
        ],
        "icd10_codes": [
            {
                "code": "E11.9",
                "description": "Type 2 diabetes mellitus without complications",
                "usage_note": "Acceptable supporting diagnosis for CGM LCD — verify it appears "
                              "on the MAC-specific covered diagnosis list.",
            },
            {
                "code": "E11.65",
                "description": "Type 2 diabetes mellitus with hyperglycemia",
                "usage_note": "Use when provider documents hyperglycemia; often supports "
                              "higher medical necessity for CGM.",
            },
            {
                "code": "Z79.4",
                "description": "Long-term (current) use of insulin",
                "usage_note": "Assign as additional code when patient uses insulin, "
                              "even though insulin use is no longer required for CGM coverage.",
            },
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Compiled example list
# ══════════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES: list[FewShotExample] = [_EXAMPLE_1, _EXAMPLE_2, _EXAMPLE_3]


# ── Public helpers ────────────────────────────────────────────────────────────

def get_few_shot_messages() -> list[dict[str, str]]:
    """Return examples as alternating human/assistant message dicts.

    The assistant content is the canonical JSON string, which primes the LLM
    to output structured JSON on every subsequent turn.
    """
    messages: list[dict[str, str]] = []
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["query"]})
        messages.append({
            "role": "assistant",
            "content": json.dumps(ex["answer_json"], indent=2),
        })
    return messages


def get_formatted_examples_text() -> str:
    """Return examples as a single text block for injection into system prompts."""
    blocks: list[str] = ["## Few-Shot Examples\n"]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        answer_str = json.dumps(ex["answer_json"], indent=2)
        blocks.append(
            f"### Example {i}\n"
            f"**User**: {ex['query']}\n\n"
            f"**Assistant**:\n```json\n{answer_str}\n```\n"
        )
    return "\n---\n".join(blocks)


def get_example_by_index(index: int) -> FewShotExample:
    """Return a single example by 0-based index (for targeted injection)."""
    return FEW_SHOT_EXAMPLES[index]
