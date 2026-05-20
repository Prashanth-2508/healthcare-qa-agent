## Healthcare Q&A Agent — Evaluation Results

**Date**: May 20, 2026  
**Scoring method**: Heuristic (keyword + citation)  
**Pass threshold**: 3.5 / 5.0  
**Scenarios**: 5 | **Passed**: 1 | **Failed**: 4  
**Mean overall score**: 3.14 / 5.0  
**Mean latency**: 26.58s

| # | Scenario | Accuracy | Completeness | Citations | **Overall** | Pass | Notes |
|---|----------|:--------:|:------------:|:---------:|:-----------:|:----:|-------|
| TC-001 | ADA 2024 T2DM First-Line Treatment | 1 | 3 | 4 | **2.7** | ❌ | Heuristic: keyword_recall=0.40, citation=yes, rouge1=0.06 |
| TC-002 | Essential Hypertension ICD-10 Code | 1 | 2 | 4 | **2.3** | ❌ | Heuristic: keyword_recall=0.25, citation=yes, rouge1=0.05 |
| TC-003 | Medicare Part B CGM Coverage 2024 | 3 | 3 | 4 | **3.3** | ❌ | Heuristic: keyword_recall=0.60, citation=yes, rouge1=0.08 |
| TC-004 | PubMed Evidence: SGLT2 Inhibitors and Heart Failure | 5 | 5 | 4 | **4.7** | ✅ | Heuristic: keyword_recall=0.92, citation=yes, rouge1=0.07 |
| TC-005 | CPT Code: New Patient Office Visit, Moderate Complexity | 1 | 3 | 4 | **2.7** | ❌ | Heuristic: keyword_recall=0.42, citation=yes, rouge1=0.04 |

---

_Scores are 1–5 (5 = excellent). Pass threshold = 3.5._  
_Traces saved to_ `logs\eval_traces/`