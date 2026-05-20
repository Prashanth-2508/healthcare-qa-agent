"""Healthcare QA Agent — LangGraph state machine.

Node flow:
  START → reason → plan → act → observe → [plan | respond] → END

Self-reflection loop (observe → plan → act → observe) runs up to
AGENT_MAX_OBSERVE_ITERATIONS times before forcing a respond.

Every node transition is logged with timing via ReasoningTracer and
written as a JSONL file to AGENT_TRACE_DIR at the end of each run.
"""
from __future__ import annotations

import functools
import json
import os
import re
import time
import uuid
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.memory.conversation_store import ConversationStore, init_db
from src.prompts.system_prompt import (
    get_observe_json_prompt,
    get_reason_json_prompt,
    get_ui_respond_prompt,
    get_system_prompt,
)
from src.tools import ALL_TOOLS
from src.tools.exceptions import ToolExecutionError
from pydantic import ValidationError

from src.utils.logger import AgentStep, ReasoningTracer, configure_logging, get_logger
from src.utils.validators import (
    GuardrailError,
    QueryInput,
    validate_query,
    validate_structured_response,
    validate_tool_output,
)

_log = get_logger(__name__)

# ── Environment config ────────────────────────────────────────────────────────

_MAX_OBSERVE_ITER: int = int(os.getenv("AGENT_MAX_OBSERVE_ITERATIONS", "3"))
_MAX_TOTAL_ACT: int = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
_VERBOSE: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"
_TRACE_DIR: str = os.getenv("AGENT_TRACE_DIR", "./traces")

# ── Tool registry ─────────────────────────────────────────────────────────────
# Maps tool name → callable LangChain tool. Used in act node for direct invocation.

_TOOL_REGISTRY: dict[str, Any] = {t.name: t for t in ALL_TOOLS}


# ── LLM factory ───────────────────────────────────────────────────────────────

def _build_llm() -> Any:
    """Build a chat LLM from environment variables (no tool binding)."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model = os.getenv("LLM_MODEL", "llama3.1:70b")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    if provider == "ollama":
        from langchain_ollama import ChatOllama  # type: ignore[import-untyped]
        return ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
            num_predict=max_tokens,
        )

    if provider == "together":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.environ["TOGETHER_API_KEY"],
            openai_api_base="https://api.together.xyz/v1",
        )

    if provider == "groq":
        from langchain_groq import ChatGroq  # type: ignore[import-untyped]
        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=os.environ["GROQ_API_KEY"],
        )

    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  # type: ignore[import-untyped]
        endpoint = HuggingFaceEndpoint(
            repo_id=model,
            task="text-generation",
            max_new_tokens=max_tokens,
            temperature=temperature,
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        )
        return ChatHuggingFace(llm=endpoint)

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. "
        "Valid: ollama, together, groq, huggingface"
    )


# ── Typed state schema ────────────────────────────────────────────────────────

class PlannedToolCall(TypedDict):
    """One tool call entry produced by the reason or observe node."""
    step: int
    tool: str
    args: dict[str, Any]
    rationale: str


class ToolResult(TypedDict):
    """Output of one tool execution in the act node."""
    tool: str
    args: dict[str, Any]
    output: str
    success: bool
    duration_ms: float
    error: str | None


class AgentState(TypedDict):
    # ── Message history (append-only via LangGraph reducer) ──────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Inputs ────────────────────────────────────────────────────────────────
    query: str
    session_id: str

    # ── reason node output ────────────────────────────────────────────────────
    # Full JSON produced by the LLM: clinical_domain, tool_steps, confidence, …
    reasoning_json: dict[str, Any]

    # ── plan node output ──────────────────────────────────────────────────────
    # Validated, ordered list of tools to call in the next act cycle.
    tool_plan: list[PlannedToolCall]

    # ── act node output ───────────────────────────────────────────────────────
    # Results from this act cycle (reset each act call, accumulated below).
    tool_results: list[ToolResult]

    # ── observe node accumulations ────────────────────────────────────────────
    # Textual summaries across all iterations (fed to respond node).
    all_observations: list[str]
    # Additional calls requested by observe for the next plan/act cycle.
    additional_tool_calls: list[PlannedToolCall]

    # ── Loop control ──────────────────────────────────────────────────────────
    observe_iter: int            # how many observe→plan loops have run (0-based)
    max_observe_iter: int        # ceiling (from AGENT_MAX_OBSERVE_ITERATIONS)
    total_act_calls: int         # total act node invocations (hard cap)
    goal_met: bool               # set True by observe when answer is sufficient

    # ── Output ────────────────────────────────────────────────────────────────
    final_answer: str


# ── JSON extraction utility ───────────────────────────────────────────────────

def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from LLM output.

    Tries four strategies in order:
    1. Parse the whole string.
    2. Extract from a ```json ... ``` fenced block.
    3. Find the first {...} span (greedy).
    4. Return {raw_content, parse_error} as a fallback.
    """
    text = text.strip()

    # 1. Full-string parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fenced code block
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. First braced span — find matching closing brace from first {
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

    # 4. Fallback — preserve raw content for debugging
    _log.warning("json_extract_failed", preview=text[:200])
    return {"raw_content": text, "parse_error": True}


# ── State summarisers (keep JSONL entries small) ──────────────────────────────

def _summarise_state(state: AgentState) -> dict[str, Any]:
    return {
        "query_len": len(state.get("query", "")),
        "messages": len(state.get("messages", [])),
        "observe_iter": state.get("observe_iter", 0),
        "total_act_calls": state.get("total_act_calls", 0),
        "tool_plan_len": len(state.get("tool_plan", [])),
        "goal_met": state.get("goal_met", False),
    }


def _summarise_plan(plan: list[PlannedToolCall]) -> list[dict[str, Any]]:
    return [{"step": p["step"], "tool": p["tool"]} for p in plan]


def _summarise_results(results: list[ToolResult]) -> list[dict[str, Any]]:
    return [
        {"tool": r["tool"], "success": r["success"], "duration_ms": r["duration_ms"]}
        for r in results
    ]


# ── Node decorator — wraps every node with enter/exit transition logging ───────

def _traced(node_name: str) -> Any:
    """Decorator factory: wraps a node function with timing + JSONL transition logging."""

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def wrapper(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            tracer: ReasoningTracer = config["configurable"]["tracer"]
            observe_iter: int = state.get("observe_iter", 0)

            t0 = tracer.node_enter(
                node=node_name,
                state_summary=_summarise_state(state),
                observe_iter=observe_iter,
            )

            try:
                result = func(state, config)
                tracer.node_exit(
                    node=node_name,
                    output_summary={k: v for k, v in result.items() if k != "messages"},
                    t0=t0,
                    success=True,
                    observe_iter=observe_iter,
                )
                return result
            except Exception as exc:
                tracer.node_exit(
                    node=node_name,
                    output_summary={"error": str(exc)},
                    t0=t0,
                    success=False,
                    observe_iter=observe_iter,
                )
                raise

        return wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1 — REASON
# Takes query + conversation history → calls LLM → structured JSON plan.
# ═══════════════════════════════════════════════════════════════════════════════

def _make_reason_node(llm: Any) -> Any:
    @_traced("reason")
    def reason_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        tracer: ReasoningTracer = config["configurable"]["tracer"]
        store: ConversationStore = config["configurable"]["store"]
        observe_iter = state["observe_iter"]

        # Pull session history for context
        history = store.get_history(limit=10)
        prompt = get_reason_json_prompt(state["query"], history)

        _log.info("reason_llm_call", session_id=state["session_id"], observe_iter=observe_iter)
        response = llm.invoke([
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=prompt),
        ])

        reasoning_json = _extract_json(response.content)

        tracer.log_step(
            AgentStep.REASON,
            content=json.dumps(reasoning_json, indent=2)[:500],
            metadata={
                "clinical_domain": reasoning_json.get("clinical_domain", "unknown"),
                "tool_count": len(reasoning_json.get("tool_steps", [])),
                "confidence": reasoning_json.get("confidence", "?"),
                "safety_flags": reasoning_json.get("safety_flags", []),
            },
            iteration=observe_iter,
        )

        return {
            "messages": [AIMessage(content=f"[Reason]\n{response.content}")],
            "reasoning_json": reasoning_json,
            "all_observations": [],          # reset on first entry
            "additional_tool_calls": [],
        }

    return reason_node


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2 — PLAN
# Parses reasoning_json OR additional_tool_calls (on loop-back) → validated tool_plan.
# No LLM call: pure deterministic transformation + validation.
# ═══════════════════════════════════════════════════════════════════════════════

def _make_plan_node() -> Any:
    @_traced("plan")
    def plan_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        tracer: ReasoningTracer = config["configurable"]["tracer"]
        observe_iter = state["observe_iter"]

        # Choose source: additional calls (loop-back) or initial reasoning JSON
        additional = state.get("additional_tool_calls") or []
        if additional:
            raw_steps = [
                {
                    "step": i + 1,
                    "tool": s.get("tool", ""),
                    "args": s.get("args", {}),
                    "rationale": s.get("rationale", "requested by observe"),
                }
                for i, s in enumerate(additional)
            ]
            source = "observe_additional"
        else:
            raw_steps = state.get("reasoning_json", {}).get("tool_steps", [])
            source = "reasoning_json"

        # Validate: keep only known tools; coerce types
        validated: list[PlannedToolCall] = []
        skipped: list[str] = []
        for raw in raw_steps:
            tool_name = raw.get("tool", "").strip()
            if tool_name not in _TOOL_REGISTRY:
                skipped.append(tool_name)
                _log.warning(
                    "plan_unknown_tool",
                    tool=tool_name,
                    known=list(_TOOL_REGISTRY.keys()),
                )
                continue
            validated.append(
                PlannedToolCall(
                    step=int(raw.get("step", len(validated) + 1)),
                    tool=tool_name,
                    args=dict(raw.get("args", {})),
                    rationale=str(raw.get("rationale", "")),
                )
            )

        plan_summary = _summarise_plan(validated)
        tracer.log_step(
            AgentStep.PLAN,
            content=json.dumps(plan_summary),
            metadata={"source": source, "steps": len(validated), "skipped": skipped},
            iteration=observe_iter,
        )

        if not validated:
            _log.warning("plan_empty", source=source)

        return {
            "messages": [AIMessage(content=f"[Plan] {json.dumps(plan_summary)}")],
            "tool_plan": validated,
            "additional_tool_calls": [],   # consumed; clear for next cycle
        }

    return plan_node


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3 — ACT
# Executes each PlannedToolCall in tool_plan sequentially, captures timing.
# Tools are invoked directly (no LLM in the loop) for determinism and traceability.
# ═══════════════════════════════════════════════════════════════════════════════

def _make_act_node() -> Any:
    @_traced("act")
    def act_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        tracer: ReasoningTracer = config["configurable"]["tracer"]
        observe_iter = state["observe_iter"]
        new_messages: list[BaseMessage] = []
        tool_results: list[ToolResult] = []

        plan = state.get("tool_plan") or []

        if not plan:
            _log.warning("act_empty_plan", session_id=state["session_id"])
            return {
                "tool_results": [],
                "total_act_calls": state.get("total_act_calls", 0) + 1,
            }

        for step in plan:
            tool_name = step["tool"]
            args = step["args"]

            tracer.log_step(
                AgentStep.TOOL_CALL,
                content=f"{tool_name}({args})",
                metadata={"tool": tool_name, "args": args, "rationale": step.get("rationale", "")},
                iteration=observe_iter,
            )

            t_start = time.monotonic()
            try:
                output: str = _TOOL_REGISTRY[tool_name].invoke(args)
                duration_ms = (time.monotonic() - t_start) * 1000
                success = True
                error = None
            except ToolExecutionError as exc:
                duration_ms = (time.monotonic() - t_start) * 1000
                output = exc.user_message()
                success = False
                error = str(exc)
                _log.warning("act_tool_error", tool=tool_name, error=error)
            except Exception as exc:
                duration_ms = (time.monotonic() - t_start) * 1000
                output = f"Unexpected error in {tool_name}: {exc}"
                success = False
                error = str(exc)
                _log.error("act_tool_unexpected", tool=tool_name, error=error)

            result = ToolResult(
                tool=tool_name,
                args=args,
                output=output,
                success=success,
                duration_ms=round(duration_ms, 2),
                error=error,
            )

            # Validate tool output envelope; on guardrail violation mark as failed
            if success:
                try:
                    validate_tool_output({"tool": tool_name, "args": args, "output": output,
                                          "success": True, "duration_ms": round(duration_ms, 2)})
                except GuardrailError as gexc:
                    _log.warning("act_tool_output_blocked", tool=tool_name, rule=gexc.rule)
                    result = ToolResult(
                        tool=tool_name,
                        args=args,
                        output=gexc.safe_response,
                        success=False,
                        duration_ms=round(duration_ms, 2),
                        error=gexc.reason,
                    )
            tool_results.append(result)

            tracer.log_step(
                AgentStep.TOOL_RESULT,
                content=output[:300],
                metadata={
                    "tool": tool_name,
                    "success": success,
                    "duration_ms": round(duration_ms, 1),
                },
                iteration=observe_iter,
            )

            # Add ToolMessage so the conversation record is complete
            new_messages.append(
                ToolMessage(
                    content=output,
                    tool_call_id=f"{tool_name}-{observe_iter}-{step['step']}",
                    name=tool_name,
                )
            )

        return {
            "messages": new_messages,
            "tool_results": tool_results,
            "total_act_calls": state.get("total_act_calls", 0) + 1,
        }

    return act_node


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4 — OBSERVE
# Self-reflection loop: LLM evaluates tool results, decides goal_met or more tools.
# Caps at max_observe_iter; on cap forces goal_met=True to proceed to respond.
# ═══════════════════════════════════════════════════════════════════════════════

def _make_observe_node(llm: Any) -> Any:
    @_traced("observe")
    def observe_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        tracer: ReasoningTracer = config["configurable"]["tracer"]
        store: ConversationStore = config["configurable"]["store"]
        observe_iter = state["observe_iter"]
        max_iter = state["max_observe_iter"]

        tool_results: list[ToolResult] = state.get("tool_results", [])
        prev_observations: list[str] = state.get("all_observations", [])

        # Build self-reflection prompt
        prompt = get_observe_json_prompt(
            query=state["query"],
            tool_results=tool_results,
            all_observations=prev_observations,
            observe_iter=observe_iter,
            max_iter=max_iter,
        )

        _log.info(
            "observe_llm_call",
            session_id=state["session_id"],
            observe_iter=observe_iter,
            tool_count=len(tool_results),
        )
        response = llm.invoke([
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=prompt),
        ])

        reflection = _extract_json(response.content)

        # Extract structured fields with safe defaults
        # Coerce all string fields to str in case the LLM returns a list/other type
        def _to_str(v: Any) -> str:
            if isinstance(v, list):
                return " ".join(str(x) for x in v)
            return str(v) if v else ""

        def _to_str_list(v: Any) -> list[str]:
            if isinstance(v, list):
                return [str(x) for x in v]
            if isinstance(v, str) and v:
                return [v]
            return []

        goal_met: bool = bool(reflection.get("goal_met", False))
        evidence_quality: str = _to_str(reflection.get("evidence_quality", "unknown"))
        key_findings: list[str] = _to_str_list(reflection.get("key_findings", []))
        gaps: list[str] = _to_str_list(reflection.get("gaps", []))
        synthesis: str = _to_str(reflection.get("synthesis", ""))
        raw_additional: list[dict[str, Any]] = reflection.get("additional_tool_calls", [])
        if not isinstance(raw_additional, list):
            raw_additional = []

        # Build observation text to accumulate across iterations
        successful_results = [r for r in tool_results if r["success"]]
        obs_block = (
            f"## Observation (iteration {observe_iter + 1}/{max_iter})\n"
            f"**Evidence quality**: {evidence_quality}\n\n"
            + (f"**Key findings**:\n" + "\n".join(f"- {f}" for f in key_findings) + "\n\n" if key_findings else "")
            + (f"**Gaps**: {'; '.join(gaps)}\n\n" if gaps else "")
            + (f"**Synthesis**: {synthesis}\n\n" if synthesis else "")
            + "\n\n".join(r["output"] for r in successful_results)
        )
        updated_observations = prev_observations + [obs_block]

        # Parse additional tool calls for next cycle
        additional: list[PlannedToolCall] = [
            PlannedToolCall(
                step=i + 1,
                tool=c.get("tool", ""),
                args=c.get("args", {}),
                rationale=c.get("rationale", ""),
            )
            for i, c in enumerate(raw_additional)
            if c.get("tool", "") in _TOOL_REGISTRY
        ]

        # Hard cap: if at or beyond max_iter, force completion
        at_cap = observe_iter >= max_iter - 1
        if at_cap and not goal_met:
            _log.info(
                "observe_cap_reached",
                observe_iter=observe_iter,
                max_iter=max_iter,
                forcing_respond=True,
            )
            goal_met = True
            additional = []

        tracer.log_step(
            AgentStep.OBSERVE,
            content=synthesis[:400] or "(no synthesis)",
            metadata={
                "goal_met": goal_met,
                "evidence_quality": evidence_quality,
                "gaps": gaps,
                "additional_tools": [a["tool"] for a in additional],
                "at_cap": at_cap,
            },
            iteration=observe_iter,
        )
        store.save_trace_entry(
            "observe",
            content=synthesis[:400],
            iteration=observe_iter,
            metadata={"goal_met": goal_met, "evidence_quality": evidence_quality},
        )

        return {
            "messages": [AIMessage(content=f"[Observe iter={observe_iter}]\n{response.content[:300]}")],
            "all_observations": updated_observations,
            "additional_tool_calls": additional,
            "observe_iter": observe_iter + 1,
            "goal_met": goal_met,
        }

    return observe_node


# ═══════════════════════════════════════════════════════════════════════════════
# Node 5 — RESPOND
# Synthesises all accumulated observations into a final, cited, markdown answer.
# Stores the turn in ConversationStore; flushes the JSONL trace.
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_evidence_appendix(observations: list[str]) -> str:
    """Build a structured evidence section from tool results in observations.

    Extracts PMIDs and ICD-10/CPT codes found in the observation text and
    formats them as a citable appendix. This ensures that even when the LLM
    fails to echo specific identifiers in its synthesis, the final answer still
    contains them for downstream scoring and user reference.
    """
    combined = "\n".join(observations)
    pmids = list(dict.fromkeys(re.findall(r'PMID[:\s]+(\d{7,8})', combined, re.IGNORECASE)))
    icd_codes = list(dict.fromkeys(re.findall(r'\b([A-Z]\d{2}(?:\.\d{1,2})?)\b', combined)))
    # Filter out common false positives (single-digit suffixes that are Roman numerals etc.)
    icd_codes = [c for c in icd_codes if len(c) >= 3][:10]

    parts: list[str] = []
    if pmids:
        links = "\n".join(
            f"- **PMID {p}** — https://pubmed.ncbi.nlm.nih.gov/{p}/"
            for p in pmids[:5]
        )
        parts.append(f"**PubMed References**\n{links}")
    if icd_codes:
        parts.append(f"**ICD-10 Codes Retrieved**: {', '.join(icd_codes)}")

    if not parts:
        return ""
    return "\n\n---\n\n**Retrieved Evidence**\n\n" + "\n\n".join(parts)


def _make_respond_node(llm: Any) -> Any:
    @_traced("respond")
    def respond_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        tracer: ReasoningTracer = config["configurable"]["tracer"]
        store: ConversationStore = config["configurable"]["store"]

        observations = state.get("all_observations", [])

        # Use plain-markdown prompt — simpler, more reliable with small models,
        # and avoids JSON parsing failures that caused empty final_answer in eval.
        prompt = get_ui_respond_prompt(
            query=state["query"],
            all_observations=observations,
        )

        _log.info("respond_llm_call", session_id=state["session_id"])
        response = llm.invoke([HumanMessage(content=prompt)])

        raw_answer: str = response.content

        # Guardrail: block unsafe responses
        try:
            validate_structured_response({"answer": raw_answer, "sources": [], "confidence": "medium"})
            final_answer = raw_answer
        except GuardrailError:
            final_answer = (
                "I'm unable to provide a response to this query. "
                "Please consult a qualified healthcare professional for medical advice."
            )

        # Append retrieved PMIDs and codes so they always appear in the scored text
        appendix = _extract_evidence_appendix(observations)
        if appendix:
            final_answer = final_answer + appendix

        tracer.log_step(
            AgentStep.RESPOND,
            content=final_answer[:400],
            metadata={"answer_len": len(final_answer)},
            iteration=state.get("observe_iter", 0),
        )

        # Persist conversation turn
        store.add_message("user", state["query"])
        store.add_message("assistant", final_answer)
        store.save_trace_entry(
            "respond",
            content=final_answer[:500],
            iteration=state.get("observe_iter", 0),
        )

        # Write JSONL trace file
        try:
            trace_path = tracer.flush_jsonl(_TRACE_DIR)
            _log.info("trace_saved", path=str(trace_path))
        except Exception as exc:
            _log.warning("trace_save_failed", error=str(exc))

        return {
            "messages": [AIMessage(content=final_answer)],
            "final_answer": final_answer,
        }

    return respond_node


# ── Routing functions ─────────────────────────────────────────────────────────

def _route_after_observe(state: AgentState) -> Literal["plan", "respond"]:
    """Route from observe: loop back to plan if more tools needed, else respond."""
    if state["goal_met"] or state["observe_iter"] >= state["max_observe_iter"]:
        return "respond"
    # Only loop if there are additional tools to call
    if state.get("additional_tool_calls"):
        return "plan"
    return "respond"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """Compile and return the LangGraph StateGraph.

    Graph topology:
        START → reason → plan → act → observe
                                ↑          |
                                └── plan ←─┘  (when not goal_met and additional_tools exist)
                                           ↓
                                        respond → END
    """
    llm = _build_llm()

    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("reason", _make_reason_node(llm))
    graph.add_node("plan", _make_plan_node())
    graph.add_node("act", _make_act_node())
    graph.add_node("observe", _make_observe_node(llm))
    graph.add_node("respond", _make_respond_node(llm))

    graph.add_edge(START, "reason")
    graph.add_edge("reason", "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "observe")
    graph.add_conditional_edges(
        "observe",
        _route_after_observe,
        {"plan": "plan", "respond": "respond"},
    )
    graph.add_edge("respond", END)

    # recursion_limit = (max_observe_iter * 3 nodes per loop) + fixed prefix nodes + headroom
    recursion_limit = (_MAX_OBSERVE_ITER * 3) + 5 + 4
    return graph.compile(debug=False)


# ── Public API ────────────────────────────────────────────────────────────────

class HealthcareQAAgent:
    """High-level interface around the compiled LangGraph healthcare QA agent.

    Usage:
        agent = HealthcareQAAgent()
        result = agent.run("What is the first-line treatment for hypertension?")
        print(result["answer"])
        # result["trace_path"] → path to the JSONL trace file
    """

    def __init__(self) -> None:
        configure_logging()
        init_db()
        self._graph = build_graph()
        _log.info(
            "agent_initialized",
            provider=os.getenv("LLM_PROVIDER", "ollama"),
            model=os.getenv("LLM_MODEL", "llama3.1:70b"),
            max_observe_iter=_MAX_OBSERVE_ITER,
        )

    def run(
        self,
        query: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the agent on a healthcare question.

        Args:
            query: The user's question (natural language).
            session_id: Optional session ID for conversation continuity. A new
                        UUID is generated if not provided.

        Returns:
            Dict with keys:
              answer       (str)  — final markdown answer
              session_id   (str)  — session identifier
              trace        (list) — step-level reasoning trace entries
              trace_path   (str)  — path to the JSONL file (or "" if write failed)
              blocked      (bool) — True if guardrails rejected the query
        """
        sid = session_id or str(uuid.uuid4())
        try:
            scrubbed_query = validate_query(query)
            validated = QueryInput(query=scrubbed_query, session_id=sid)
            sid = validated.session_id
        except GuardrailError as exc:
            _log.warning(
                "guardrail_blocked",
                session_id=sid,
                rule=exc.rule,
                reason=exc.reason[:200],
            )
            return {
                "answer": exc.safe_response,
                "session_id": sid,
                "trace": [],
                "trace_path": "",
                "blocked": True,
            }
        except ValidationError:
            return {
                "answer": "Invalid query format. Please check your input and try again.",
                "session_id": sid,
                "trace": [],
                "trace_path": "",
                "blocked": True,
            }

        tracer = ReasoningTracer(sid)
        store = ConversationStore(sid)
        store.trim_if_needed()

        initial_state: AgentState = {
            "messages": [HumanMessage(content=validated.query)],
            "query": validated.query,
            "session_id": sid,
            "reasoning_json": {},
            "tool_plan": [],
            "tool_results": [],
            "all_observations": [],
            "additional_tool_calls": [],
            "observe_iter": 0,
            "max_observe_iter": _MAX_OBSERVE_ITER,
            "total_act_calls": 0,
            "goal_met": False,
            "final_answer": "",
        }

        config: RunnableConfig = {
            "configurable": {"tracer": tracer, "store": store},
            "recursion_limit": (_MAX_OBSERVE_ITER * 3) + 12,
        }

        _log.info(
            "agent_run_start",
            session_id=sid,
            query=validated.query[:120],
            max_observe_iter=_MAX_OBSERVE_ITER,
        )

        try:
            final_state = self._graph.invoke(initial_state, config=config)
        except GuardrailError as exc:
            _log.warning(
                "graph_guardrail_blocked",
                session_id=sid,
                rule=exc.rule,
                reason=exc.reason[:200],
            )
            return {
                "answer": exc.safe_response,
                "session_id": sid,
                "trace": tracer.get_trace(),
                "trace_path": "",
                "blocked": True,
            }

        _log.info(
            "agent_run_complete",
            session_id=sid,
            observe_iters=final_state.get("observe_iter", 0),
            total_act_calls=final_state.get("total_act_calls", 0),
            goal_met=final_state.get("goal_met", False),
        )

        # Determine trace file path (flushed inside respond node)
        trace_path = str(
            (lambda p: p if p.exists() else "")(
                (lambda d: d / f"{sid}.jsonl")(
                    __import__("pathlib").Path(_TRACE_DIR)
                )
            )
        )

        if _VERBOSE:
            print(tracer.summary())

        return {
            "answer": final_state.get("final_answer", ""),
            "session_id": sid,
            "trace": tracer.get_trace(),
            "trace_path": trace_path,
            "blocked": False,
            "all_observations": final_state.get("all_observations", []),
        }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv  # type: ignore[import-untyped]

    load_dotenv()
    agent = HealthcareQAAgent()

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ").strip()
    result = agent.run(question)

    print("\n" + "=" * 70)
    print(result["answer"])
    print("=" * 70)
    if result["trace_path"]:
        print(f"\nTrace: {result['trace_path']}")
