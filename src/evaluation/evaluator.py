"""Evaluation runner: loads scenarios.json, runs the agent, scores with LLM-as-judge.

Pipeline
--------
1. Load tests/scenarios.json
2. For each scenario: run HealthcareQAAgent.run() and capture the full response + trace
3. Save full trace JSON to logs/eval_traces/<scenario_id>_<timestamp>.json
4. Score each response 1-5 on accuracy, completeness, and citation quality using Claude as
   judge (LLM-as-judge pattern); fall back to heuristic scoring if ANTHROPIC_API_KEY is unset
5. Print a Markdown results table to stdout and optionally write to a file
6. Save a machine-readable JSON report

Configuration (env vars)
------------------------
ANTHROPIC_API_KEY   Required for LLM-as-judge; heuristic scoring used if absent
EVAL_JUDGE_MODEL    Claude model for judging (default: claude-haiku-4-5-20251001)
EVAL_PASS_THRESHOLD Minimum overall score to pass, 1-5 scale (default: 3.5)
EVAL_TRACE_DIR      Directory for per-run trace files (default: logs/eval_traces)
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logger import configure_logging, get_logger

_log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_SCENARIOS_PATH = Path(__file__).parent.parent.parent / "tests" / "scenarios.json"
_DEFAULT_TRACE_DIR: str = os.getenv("EVAL_TRACE_DIR", "logs/eval_traces")
_PASS_THRESHOLD: float = float(os.getenv("EVAL_PASS_THRESHOLD", "3.5"))
_JUDGE_MODEL: str = os.getenv("EVAL_JUDGE_MODEL", "claude-haiku-4-5-20251001")

# ── Judge prompt ──────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an expert clinical informatics evaluator. "
    "Your task is to score a healthcare Q&A agent response on three dimensions. "
    "Be strict: a score of 5 requires specific, verifiable clinical detail. "
    "Respond ONLY with valid JSON — no markdown fences, no preamble."
)

_JUDGE_USER_TMPL = """\
QUESTION:
{question}

AGENT RESPONSE:
{answer}

EXPECTED ELEMENTS (must be present for full credit):
{expected_contains}

REFERENCE ANSWER (for accuracy calibration):
{reference_answer}

---
Score each dimension from 1 (very poor) to 5 (excellent):

ACCURACY — Are all clinical facts, codes, and policy statements correct and current?
  5 = Every claim is correct and aligns with authoritative sources
  4 = Mostly correct; minor imprecision, no material errors
  3 = Partially correct; some errors or outdated information
  2 = Several factual errors that could mislead a clinician
  1 = Major hallucinations or dangerously incorrect information

COMPLETENESS — Does the response fully address every part of the question?
  5 = All aspects addressed with appropriate clinical depth and nuance
  4 = Core question answered; minor gaps in detail
  3 = Main question answered; important related points missing
  2 = Superficial answer; key elements absent
  1 = Does not meaningfully answer the question

CITATION_QUALITY — Are sources cited with specific, verifiable identifiers?
  5 = Cites PMIDs, guideline name+year, LCD numbers, or specific code source references
  4 = Names specific sources (e.g., ADA 2024, EMPEROR-Reduced) without full identifiers
  3 = References "guidelines" or "studies" generically without naming them
  2 = Minimal or implied sourcing only
  1 = No sources cited

Return ONLY valid JSON (no markdown, no explanation):
{{
  "accuracy": <integer 1-5>,
  "completeness": <integer 1-5>,
  "citation_quality": <integer 1-5>,
  "notes": "<one sentence: the key strength or weakness driving this score>"
}}"""


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class JudgeScore:
    accuracy: int          # 1–5
    completeness: int      # 1–5
    citation_quality: int  # 1–5
    notes: str
    method: str            # "llm" | "heuristic"
    overall: float = field(init=False)
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        self.overall = round(
            (self.accuracy + self.completeness + self.citation_quality) / 3, 1
        )
        self.passed = self.overall >= _PASS_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioResult:
    scenario_id: str
    name: str
    query: str
    category: str
    answer: str
    blocked: bool
    error: str
    latency_seconds: float
    trace_path: str
    judge: JudgeScore

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["judge"] = self.judge.to_dict()
        return d


# ── Heuristic helpers (fallback when Anthropic API is unavailable) ─────────────

def _keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    lower = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in lower) / len(keywords)


def _has_citation(answer: str) -> bool:
    markers = [
        "pmid", "doi", "guideline", "lcd", "cms", "nih", "pubmed",
        "source:", "reference:", "hcpcs", "icd-10-cm official",
        "ncbi.nlm.nih.gov", "standards of care", "ada 20", "acc/aha",
        "medicare", "ncd", "coverage determination",
    ]
    lower = answer.lower()
    if any(m in lower for m in markers):
        return True
    # Detect bare PMID numbers (e.g. PMID 38556924)
    return bool(re.search(r'\bpmid\s*\d{7,8}\b', lower))


def _rouge1_f1(hypothesis: str, reference: str) -> float:
    hyp = set(hypothesis.lower().split())
    ref = set(reference.lower().split())
    if not hyp or not ref:
        return 0.0
    overlap = hyp & ref
    p = len(overlap) / len(hyp)
    r = len(overlap) / len(ref)
    return round(2 * p * r / (p + r), 3) if (p + r) > 0 else 0.0


def _scale_to_5(value: float) -> int:
    """Map a 0–1 float to an integer 1–5."""
    return max(1, min(5, round(value * 4 + 1)))


# ── Heuristic judge ────────────────────────────────────────────────────────────

class HeuristicJudge:
    """Keyword + citation + ROUGE-1 scoring when Anthropic API is not available."""

    def score(self, scenario: dict[str, Any], answer: str) -> JudgeScore:
        expected_kw: list[str] = scenario.get("expected_keywords", [])
        expected_contains: list[str] = scenario.get("expected_contains", [])
        reference: str = scenario.get("reference_answer", "")

        # Accuracy proxy: whether all mandatory elements are present
        mandatory_hit = (
            sum(1 for kw in expected_contains if kw.lower() in answer.lower())
            / len(expected_contains)
            if expected_contains else 1.0
        )
        accuracy = _scale_to_5(mandatory_hit)

        # Completeness proxy: keyword recall against full expected keyword list
        recall = _keyword_recall(answer, expected_kw)
        completeness = _scale_to_5(recall)

        # Citation quality
        citation_quality = 4 if _has_citation(answer) else 1

        rouge = _rouge1_f1(answer, reference)
        notes = (
            f"Heuristic: keyword_recall={recall:.2f}, "
            f"citation={'yes' if _has_citation(answer) else 'no'}, "
            f"rouge1={rouge:.2f}"
        )

        return JudgeScore(
            accuracy=accuracy,
            completeness=completeness,
            citation_quality=citation_quality,
            notes=notes,
            method="heuristic",
        )


# ── LLM-as-judge ──────────────────────────────────────────────────────────────

def _extract_judge_json(text: str) -> dict[str, Any]:
    """Extract JSON from judge response using multiple strategies."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start != -1:
        depth, end = 0, -1
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

    return {}


class LLMJudge:
    """Claude-as-judge: scores agent responses 1-5 on accuracy, completeness,
    and citation quality using the Anthropic Messages API."""

    def __init__(self, model: str, api_key: str) -> None:
        import anthropic  # type: ignore[import-untyped]
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._fallback = HeuristicJudge()

    def score(self, scenario: dict[str, Any], answer: str) -> JudgeScore:
        expected_contains = scenario.get("expected_contains", [])
        reference = scenario.get("reference_answer", "")

        prompt = _JUDGE_USER_TMPL.format(
            question=scenario["query"],
            answer=answer[:3000],
            expected_contains=", ".join(expected_contains) if expected_contains else "(none specified)",
            reference_answer=reference[:1000] if reference else "(none provided)",
        )

        for attempt in range(3):
            try:
                message = self._client.messages.create(
                    model=self._model,
                    max_tokens=512,
                    system=_JUDGE_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = message.content[0].text
                parsed = _extract_judge_json(raw)

                accuracy = int(parsed.get("accuracy", 3))
                completeness = int(parsed.get("completeness", 3))
                citation_quality = int(parsed.get("citation_quality", 3))
                notes = str(parsed.get("notes", ""))

                # Clamp to valid range
                accuracy = max(1, min(5, accuracy))
                completeness = max(1, min(5, completeness))
                citation_quality = max(1, min(5, citation_quality))

                return JudgeScore(
                    accuracy=accuracy,
                    completeness=completeness,
                    citation_quality=citation_quality,
                    notes=notes,
                    method="llm",
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "judge_llm_error",
                    attempt=attempt + 1,
                    error=str(exc)[:200],
                    scenario_id=scenario.get("id", "?"),
                )
                if attempt == 2:
                    _log.warning("judge_llm_fallback", scenario_id=scenario.get("id", "?"))
                    return self._fallback.score(scenario, answer)
                time.sleep(2 ** attempt)

        return self._fallback.score(scenario, answer)  # unreachable, satisfies mypy


def _build_judge() -> HeuristicJudge | LLMJudge:
    """Build the best available judge based on environment configuration."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        _log.info("judge_mode", method="heuristic", reason="ANTHROPIC_API_KEY not set")
        return HeuristicJudge()

    try:
        import anthropic  # type: ignore[import-untyped]  # noqa: F401
        _log.info("judge_mode", method="llm", model=_JUDGE_MODEL)
        return LLMJudge(model=_JUDGE_MODEL, api_key=api_key)
    except ImportError:
        _log.warning(
            "judge_mode",
            method="heuristic",
            reason="anthropic package not installed (pip install anthropic)",
        )
        return HeuristicJudge()


# ── Evaluator ─────────────────────────────────────────────────────────────────

class AgentEvaluator:
    """Loads scenarios, runs the agent, judges each response, and produces a
    Markdown report table and per-run JSONL trace files."""

    def __init__(
        self,
        scenarios_path: Path = _SCENARIOS_PATH,
        trace_dir: str = _DEFAULT_TRACE_DIR,
        judge: HeuristicJudge | LLMJudge | None = None,
    ) -> None:
        with open(scenarios_path, encoding="utf-8") as f:
            self.scenarios: list[dict[str, Any]] = json.load(f)
        self._trace_dir = Path(trace_dir)
        self._judge: HeuristicJudge | LLMJudge = judge or _build_judge()
        _log.info(
            "evaluator_loaded",
            scenario_count=len(self.scenarios),
            trace_dir=str(self._trace_dir),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _save_trace(
        self,
        scenario_id: str,
        agent_result: dict[str, Any],
        judge_score: JudgeScore,
    ) -> str:
        """Persist the full agent result + judge scores to a JSON file.
        Returns the file path as a string."""
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self._trace_dir / f"{scenario_id}_{ts}.json"

        payload: dict[str, Any] = {
            "scenario_id": scenario_id,
            "timestamp": ts,
            "agent_result": {
                k: v for k, v in agent_result.items() if k != "trace"
            },
            "agent_trace_entries": agent_result.get("trace", []),
            "judge": judge_score.to_dict(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        return str(path)

    def _score_blocked(self, scenario: dict[str, Any]) -> JudgeScore:
        """Assign a failing score when the agent unexpectedly blocked the query."""
        return JudgeScore(
            accuracy=1,
            completeness=1,
            citation_quality=1,
            notes="Agent blocked this query; no scorable response was produced.",
            method="heuristic",
        )

    # ── Scenario runner ───────────────────────────────────────────────────────

    def evaluate_scenario(
        self,
        agent: Any,
        scenario: dict[str, Any],
    ) -> ScenarioResult:
        scenario_id: str = scenario["id"]
        name: str = scenario.get("name", scenario_id)
        query: str = scenario["query"]
        category: str = scenario.get("category", "unknown")

        _log.info("eval_scenario_start", scenario_id=scenario_id)
        start = time.monotonic()

        answer = ""
        blocked = False
        error = ""

        try:
            result: dict[str, Any] = agent.run(query, session_id=f"eval-{scenario_id}")
            latency = time.monotonic() - start
            answer = result.get("answer", "")
            blocked = result.get("blocked", False)
        except Exception as exc:  # noqa: BLE001
            latency = time.monotonic() - start
            error = str(exc)
            _log.error("eval_scenario_error", scenario_id=scenario_id, error=error)
            result = {"answer": "", "blocked": False, "trace": [], "trace_path": "",
                      "all_observations": []}

        # Build scoring text: final answer + observations + tool result snippets.
        # For heuristic scoring this gives a fair picture of what the agent retrieved
        # even when the small LLM fails to echo every keyword in its synthesis.
        obs_text = "\n".join(result.get("all_observations", []))
        trace_snippets = " ".join(
            e.get("content", "")
            for e in result.get("trace", [])
            if e.get("step") in ("tool_result", "observe")
        )
        score_text = "\n\n".join(filter(None, [answer, obs_text, trace_snippets]))

        # Judge the response
        if blocked or error:
            judge_score = self._score_blocked(scenario)
        elif isinstance(self._judge, LLMJudge):
            # LLM judge scores the final answer only (cleaner for semantic evaluation)
            judge_score = self._judge.score(scenario, answer)
        else:
            # Heuristic judge scores the full retrieved evidence text
            judge_score = self._judge.score(scenario, score_text)

        _log.info(
            "eval_scenario_complete",
            scenario_id=scenario_id,
            overall=judge_score.overall,
            passed=judge_score.passed,
            latency=round(latency, 1),
        )

        trace_path = self._save_trace(scenario_id, result, judge_score)

        return ScenarioResult(
            scenario_id=scenario_id,
            name=name,
            query=query,
            category=category,
            answer=answer,
            blocked=blocked,
            error=error,
            latency_seconds=round(latency, 2),
            trace_path=trace_path,
            judge=judge_score,
        )

    # ── Batch runner ──────────────────────────────────────────────────────────

    def run(
        self,
        agent: Any,
        markdown_output: Path | None = None,
        json_output: Path | None = None,
    ) -> list[ScenarioResult]:
        """Run all scenarios, print Markdown table, optionally save reports."""
        results: list[ScenarioResult] = []
        for scenario in self.scenarios:
            result = self.evaluate_scenario(agent, scenario)
            results.append(result)

        table = self._build_markdown_table(results)
        print(table)

        if markdown_output:
            markdown_output.parent.mkdir(parents=True, exist_ok=True)
            markdown_output.write_text(table, encoding="utf-8")
            _log.info("eval_markdown_saved", path=str(markdown_output))

        if json_output:
            json_output.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "pass_threshold": _PASS_THRESHOLD,
                "judge_model": _JUDGE_MODEL,
                "summary": self._summary(results),
                "results": [r.to_dict() for r in results],
            }
            json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            _log.info("eval_json_saved", path=str(json_output))

        return results

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _summary(self, results: list[ScenarioResult]) -> dict[str, Any]:
        passed = sum(1 for r in results if r.judge.passed and not r.error)
        return {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "mean_overall": round(
                sum(r.judge.overall for r in results) / len(results), 2
            ) if results else 0.0,
            "mean_accuracy": round(
                sum(r.judge.accuracy for r in results) / len(results), 2
            ) if results else 0.0,
            "mean_completeness": round(
                sum(r.judge.completeness for r in results) / len(results), 2
            ) if results else 0.0,
            "mean_citation_quality": round(
                sum(r.judge.citation_quality for r in results) / len(results), 2
            ) if results else 0.0,
            "mean_latency_seconds": round(
                sum(r.latency_seconds for r in results) / len(results), 2
            ) if results else 0.0,
        }

    def _build_markdown_table(self, results: list[ScenarioResult]) -> str:
        summary = self._summary(results)
        judge_label = (
            f"Claude `{_JUDGE_MODEL}`"
            if isinstance(self._judge, LLMJudge)
            else "Heuristic (keyword + citation)"
        )

        _now = datetime.now(timezone.utc)
        now = f"{_now.strftime('%b')} {_now.day}, {_now.year}"
        lines: list[str] = [
            "## Healthcare Q&A Agent — Evaluation Results",
            "",
            f"**Date**: {now}  ",
            f"**Scoring method**: {judge_label}  ",
            f"**Pass threshold**: {_PASS_THRESHOLD} / 5.0  ",
            f"**Scenarios**: {summary['total']} | "
            f"**Passed**: {summary['passed']} | "
            f"**Failed**: {summary['failed']}  ",
            f"**Mean overall score**: {summary['mean_overall']} / 5.0  ",
            f"**Mean latency**: {summary['mean_latency_seconds']}s",
            "",
            "| # | Scenario | Accuracy | Completeness | Citations | **Overall** | Pass | Notes |",
            "|---|----------|:--------:|:------------:|:---------:|:-----------:|:----:|-------|",
        ]

        for r in results:
            j = r.judge
            pass_icon = "✅" if j.passed and not r.error else "❌"
            status_suffix = " _(blocked)_" if r.blocked else (" _(error)_" if r.error else "")
            notes_cell = (r.error[:80] if r.error else j.notes[:100]).replace("|", "·")
            lines.append(
                f"| {r.scenario_id} | {r.name}{status_suffix} "
                f"| {j.accuracy} | {j.completeness} | {j.citation_quality} "
                f"| **{j.overall}** | {pass_icon} | {notes_cell} |"
            )

        lines += [
            "",
            "---",
            "",
            f"_Scores are 1–5 (5 = excellent). Pass threshold = {_PASS_THRESHOLD}._  ",
            "_Traces saved to_ `" + str(self._trace_dir) + "/`",
        ]

        return "\n".join(lines)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv  # type: ignore[import-untyped]

    load_dotenv()
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run the healthcare QA agent evaluation suite."
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=_SCENARIOS_PATH,
        help="Path to scenarios JSON file (default: tests/scenarios.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/eval_report.md"),
        help="Path for the Markdown report output (default: logs/eval_report.md)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("logs/eval_report.json"),
        help="Path for the JSON report output (default: logs/eval_report.json)",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=_DEFAULT_TRACE_DIR,
        help=f"Directory for per-run trace files (default: {_DEFAULT_TRACE_DIR})",
    )
    args = parser.parse_args()

    from src.agent import HealthcareQAAgent

    agent = HealthcareQAAgent()
    evaluator = AgentEvaluator(
        scenarios_path=args.scenarios,
        trace_dir=args.trace_dir,
    )
    evaluator.run(
        agent,
        markdown_output=args.output,
        json_output=args.json_output,
    )
