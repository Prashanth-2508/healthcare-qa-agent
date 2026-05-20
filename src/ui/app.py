"""Streamlit chat UI for the Healthcare Q&A Agent — user-friendly redesign.

Run:
    streamlit run src/ui/app.py
"""
from __future__ import annotations

import os
import re
import uuid
from typing import Any, Generator

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

load_dotenv()

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Healthcare Q&A Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Healthcare Q&A Agent** — AI-powered clinical decision support. "
            "For informational and educational purposes only. Not medical advice."
        ),
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Answer card */
.answer-card {
    background: linear-gradient(135deg, #f0f7ff 0%, #e8f4f8 100%);
    border-left: 4px solid #1a73e8;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
}
/* Step badge */
.step-badge {
    display: inline-block;
    background: #e8f0fe;
    color: #1967d2;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.8em;
    font-weight: 600;
    margin-right: 6px;
}
/* Section label */
.section-label {
    font-size: 0.75em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #5f6368;
    margin-bottom: 4px;
}
/* Example question button */
.example-btn {
    background: #f8f9fa;
    border: 1px solid #dadce0;
    border-radius: 8px;
    padding: 10px 14px;
    cursor: pointer;
    width: 100%;
    text-align: left;
    margin-bottom: 6px;
}
/* Disclaimer banner */
.disclaimer {
    background: #fff8e1;
    border-left: 3px solid #f9a825;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 0.85em;
    color: #5d4037;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

_PMID_RE = re.compile(r"\bPMID[:\s]+(\d{7,8})\b", re.IGNORECASE)

# Plain-English labels for each agent step shown during processing
_STEP_LABELS: dict[str, dict[str, str]] = {
    "reason": {
        "icon": "🧠",
        "label": "Understanding your question",
        "desc": "The AI is figuring out what kind of question this is and which sources to check.",
    },
    "plan": {
        "icon": "📋",
        "label": "Building a research plan",
        "desc": "Deciding which medical databases and tools to look up.",
    },
    "act": {
        "icon": "🔍",
        "label": "Searching medical sources",
        "desc": "Looking up PubMed research, clinical guidelines, and medical codes.",
    },
    "observe": {
        "icon": "📊",
        "label": "Reviewing the evidence",
        "desc": "Evaluating the quality of information found and checking for gaps.",
    },
}

_EXAMPLE_QUESTIONS = [
    "What is the ICD-10 code for essential hypertension?",
    "What are first-line treatments for Type 2 diabetes per ADA guidelines?",
    "Does Medicare Part B cover continuous glucose monitors in 2024?",
    "Summarize PubMed evidence on SGLT2 inhibitors and heart failure",
    "What CPT code applies to a new patient office visit, moderate complexity?",
    "What is the ICD-10 code for Type 2 diabetes with diabetic retinopathy?",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _linkify_pmids(text: str) -> str:
    return _PMID_RE.sub(
        lambda m: f"[PMID {m.group(1)}](https://pubmed.ncbi.nlm.nih.gov/{m.group(1)}/)",
        text,
    )


def _tool_friendly_name(tool_name: str) -> str:
    return {
        "search_pubmed": "PubMed Research Database",
        "get_clinical_guidelines": "Clinical Guidelines Library",
        "lookup_medical_code": "ICD-10 / HCPCS Code Database",
        "describe_medical_code": "Medical Code Lookup",
    }.get(tool_name, tool_name)


def _node_progress_text(node_name: str, output: dict[str, Any]) -> str:
    """Plain-English progress summary for non-technical users."""
    meta = _STEP_LABELS.get(node_name, {})
    if node_name == "reason":
        domain = output.get("reasoning_json", {}).get("clinical_domain", "")
        n = len(output.get("reasoning_json", {}).get("tool_steps", []))
        detail = f"Topic: **{domain or 'clinical'}** · {n} source(s) to check"
        return f"{meta['icon']} **{meta['label']}** — {detail}"

    if node_name == "plan":
        plan = output.get("tool_plan", [])
        sources = ", ".join(_tool_friendly_name(p["tool"]) for p in plan)
        return f"{meta['icon']} **{meta['label']}** — Will check: {sources or 'none'}"

    if node_name == "act":
        results = output.get("tool_results", [])
        parts = []
        for r in results:
            status = "found results" if r["success"] else "no results"
            parts.append(f"{_tool_friendly_name(r['tool'])}: {status}")
        detail = " · ".join(parts) if parts else "no sources checked"
        return f"{meta['icon']} **{meta['label']}** — {detail}"

    if node_name == "observe":
        goal = output.get("goal_met", False)
        quality = ""
        obs = output.get("all_observations", [])
        if obs:
            # Try to extract evidence quality from observation text
            match = re.search(r"Evidence quality\*\*: (\w+)", obs[-1])
            if match:
                quality = f" · Evidence quality: **{match.group(1)}**"
        detail = ("Enough evidence found — writing answer" if goal else "Checking if more info is needed") + quality
        return f"{meta['icon']} **{meta['label']}** — {detail}"

    return f"• {node_name}"


# ── Cached graph + LLM factory ────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI model — this takes about 10 seconds on first load...")
def _build_processing_graph(
    temperature: float,
    max_pubmed: int,
    enable_icd10: bool,
) -> tuple[Any, Any]:
    from langgraph.graph import END, START, StateGraph

    from src.agent import (
        AgentState,
        _build_llm,
        _make_observe_node,
        _make_plan_node,
        _make_reason_node,
        _route_after_observe,
    )

    _MAX_OBSERVE_ITER = 1
    from src.tools import ALL_TOOLS

    os.environ["LLM_TEMPERATURE"] = str(temperature)
    os.environ["PUBMED_MAX_RESULTS"] = str(max_pubmed)

    llm = _build_llm()

    _ICD10_TOOLS = {"lookup_medical_code", "describe_medical_code"}
    active_tools = [t for t in ALL_TOOLS if enable_icd10 or t.name not in _ICD10_TOOLS]
    tool_registry: dict[str, Any] = {t.name: t for t in active_tools}

    def _ui_act_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        import time
        from langchain_core.messages import ToolMessage
        from src.tools.exceptions import ToolExecutionError
        from src.utils.logger import AgentStep
        from src.utils.validators import GuardrailError, validate_tool_output

        tracer = config["configurable"]["tracer"]
        observe_iter = state.get("observe_iter", 0)
        plan = state.get("tool_plan") or []
        new_messages: list[Any] = []
        tool_results: list[dict[str, Any]] = []

        if not plan:
            return {"tool_results": [], "total_act_calls": state.get("total_act_calls", 0) + 1}

        for step in plan:
            tool_name = step["tool"]
            args = step["args"]
            tracer.log_step(AgentStep.TOOL_CALL, content=f"{tool_name}({args})",
                            metadata={"tool": tool_name, "args": args}, iteration=observe_iter)

            if tool_name not in tool_registry:
                tool_results.append({"tool": tool_name, "args": args,
                                     "output": f"Tool '{tool_name}' is disabled.",
                                     "success": False, "duration_ms": 0.0, "error": "tool_disabled"})
                continue

            t0 = time.monotonic()
            try:
                output: str = tool_registry[tool_name].invoke(args)
                duration_ms = (time.monotonic() - t0) * 1000
                success, error = True, None
            except ToolExecutionError as exc:
                duration_ms = (time.monotonic() - t0) * 1000
                output = exc.user_message()
                success, error = False, str(exc)
            except Exception as exc:  # noqa: BLE001
                duration_ms = (time.monotonic() - t0) * 1000
                output = f"Unexpected error in {tool_name}: {exc}"
                success, error = False, str(exc)

            result = {"tool": tool_name, "args": args, "output": output,
                      "success": success, "duration_ms": round(duration_ms, 2), "error": error}

            if success:
                try:
                    validate_tool_output({"tool": tool_name, "args": args, "output": output,
                                          "success": True, "duration_ms": round(duration_ms, 2)})
                except GuardrailError as gexc:
                    result["output"] = gexc.safe_response
                    result["success"] = False
                    result["error"] = gexc.reason

            tool_results.append(result)
            tracer.log_step(AgentStep.TOOL_RESULT, content=output[:300],
                            metadata={"tool": tool_name, "success": success}, iteration=observe_iter)
            new_messages.append(ToolMessage(content=output,
                                            tool_call_id=f"{tool_name}-{observe_iter}-{step['step']}",
                                            name=tool_name))

        return {"messages": new_messages, "tool_results": tool_results,
                "total_act_calls": state.get("total_act_calls", 0) + 1}

    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("reason", _make_reason_node(llm))
    graph.add_node("plan",   _make_plan_node())
    graph.add_node("act",    _ui_act_node)
    graph.add_node("observe", _make_observe_node(llm))
    graph.add_edge(START, "reason")
    graph.add_edge("reason", "plan")
    graph.add_edge("plan",   "act")
    graph.add_edge("act",    "observe")
    graph.add_conditional_edges("observe", _route_after_observe, {"plan": "plan", "respond": END})

    return graph.compile(), llm


# ── Token-streaming respond ───────────────────────────────────────────────────

def _stream_response(llm: Any, query: str, observations: list[str]) -> Generator[str, None, None]:
    from src.prompts.system_prompt import get_ui_respond_prompt
    prompt = get_ui_respond_prompt(query=query, all_observations=observations)
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content:
            yield content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    yield part["text"]


# ── Trace formatter (human-readable) ─────────────────────────────────────────

def _render_trace(trace_data: list[dict]) -> None:
    """Render the reasoning trace in a human-readable way."""
    if not trace_data:
        st.caption("No trace data recorded for this question.")
        return

    st.caption("This shows the step-by-step process the AI used to answer your question.")

    for entry in trace_data:
        step = entry.get("step", "")
        content = entry.get("content", "")
        ts = entry.get("ts", "")

        step_meta = {
            "reason":       ("🧠", "Understanding the question", "#e8f0fe"),
            "plan":         ("📋", "Building research plan",     "#e6f4ea"),
            "tool_call":    ("🔍", "Searching a source",         "#fce8e6"),
            "tool_result":  ("📄", "Source returned data",       "#f0f4c3"),
            "observe":      ("📊", "Reviewing evidence",         "#fce4ec"),
            "respond":      ("✍️", "Writing the answer",          "#e8eaf6"),
        }.get(step, ("•", step.replace("_", " ").title(), "#f5f5f5"))

        icon, label, bg = step_meta
        # Truncate long content for display
        display = content[:300] + ("…" if len(content) > 300 else "")
        st.markdown(
            f"""<div style="background:{bg}; border-radius:6px; padding:8px 12px; margin:4px 0;">
            <span style="font-weight:700;">{icon} {label}</span>
            <span style="font-size:0.75em; color:#888; margin-left:8px;">{ts[11:19] if ts else ''}</span>
            <div style="margin-top:4px; font-size:0.87em; color:#444;">{display}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ── Core execution ────────────────────────────────────────────────────────────

def _run_and_stream(processing_graph: Any, llm: Any, query: str) -> None:
    from src.agent import AgentState
    from src.memory.conversation_store import ConversationStore
    from src.utils.logger import ReasoningTracer
    from src.utils.validators import GuardrailError

    _MAX_OBSERVE_ITER = 1
    sid: str = st.session_state["session_id"]
    tracer = ReasoningTracer(sid)
    store = ConversationStore(sid)

    initial_state: AgentState = {  # type: ignore[misc]
        "messages": [HumanMessage(content=query)],
        "query": query,
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
    config = {
        "configurable": {"tracer": tracer, "store": store},
        "recursion_limit": (_MAX_OBSERVE_ITER * 3) + 12,
    }

    # ── Phase 1: Research phase ───────────────────────────────────────────────
    all_observations: list[str] = []
    tool_results_collected: list[dict] = []

    st.markdown('<p class="section-label">Research Phase</p>', unsafe_allow_html=True)
    with st.status("Researching your question...", expanded=True) as progress:
        try:
            for updates in processing_graph.stream(initial_state, config, stream_mode="updates"):
                for node_name, node_output in updates.items():
                    if node_name.startswith("__"):
                        continue
                    progress.write(_node_progress_text(node_name, node_output))
                    if node_name == "observe":
                        all_observations = node_output.get("all_observations", all_observations)
                    if node_name == "act":
                        tool_results_collected.extend(node_output.get("tool_results", []))
        except GuardrailError as exc:
            progress.update(label="Blocked by safety check", state="error")
            st.warning(f"**Safety check triggered**: {exc.safe_response}")
            st.session_state["messages"].append(
                {"role": "assistant", "content": exc.safe_response, "trace": [],
                 "tool_results": [], "pmids": []}
            )
            return
        except Exception as exc:  # noqa: BLE001
            progress.update(label="Something went wrong", state="error")
            st.error(f"**Error during research**: {exc}")
            return

        progress.update(label="Research complete — writing answer...", state="running")

    # ── Phase 2: Answer streaming ─────────────────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:12px;">Answer</p>', unsafe_allow_html=True)
    response_container = st.empty()
    tokens: list[str] = []
    for token in _stream_response(llm, query, all_observations):
        tokens.append(token)
        response_container.markdown("".join(tokens) + " ▌")
    response_text = "".join(tokens)
    response_container.markdown(_linkify_pmids(response_text))
    progress.update(label="Done", state="complete", expanded=False)

    # Persist
    store.add_message("user", query)
    store.add_message("assistant", response_text)
    trace_dir = os.getenv("AGENT_TRACE_DIR", "./traces")
    try:
        tracer.flush_jsonl(trace_dir)
    except Exception:  # noqa: BLE001
        pass

    # ── Sources used ──────────────────────────────────────────────────────────
    pmids = list(dict.fromkeys(_PMID_RE.findall(response_text)))
    successful_tools = [r for r in tool_results_collected if r.get("success")]

    if pmids or successful_tools:
        st.markdown('<p class="section-label" style="margin-top:12px;">Sources Used</p>',
                    unsafe_allow_html=True)
        with st.expander(
            f"View {len(successful_tools)} source(s) searched"
            + (f" · {len(pmids)} PubMed article(s) found" if pmids else ""),
            expanded=False,
        ):
            if successful_tools:
                st.markdown("**Databases searched:**")
                for r in successful_tools:
                    st.markdown(f"- {_tool_friendly_name(r['tool'])} ✅ ({r['duration_ms']:.0f} ms)")

            if pmids:
                st.markdown("**PubMed articles cited:**")
                for pmid in pmids:
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    st.markdown(f"- [PMID {pmid} — View on PubMed]({url})")

    # ── How the agent worked ──────────────────────────────────────────────────
    trace_data = tracer.get_trace()
    st.markdown('<p class="section-label" style="margin-top:12px;">How the Agent Worked</p>',
                unsafe_allow_html=True)
    with st.expander("See step-by-step reasoning", expanded=False):
        _render_trace(trace_data)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": response_text,
        "trace": trace_data,
        "tool_results": successful_tools,
        "pmids": pmids,
    })


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> tuple[float, int, bool]:
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/caduceus.png", width=60)
        st.markdown("## Healthcare Q&A Agent")
        st.caption("AI-powered clinical decision support")

        st.divider()

        # Model status
        provider = os.getenv("LLM_PROVIDER", "ollama").upper()
        model = os.getenv("LLM_MODEL", "unknown")
        st.markdown("### AI Model")
        st.info(f"**{provider}** · `{model}`", icon="🤖")

        st.divider()

        st.markdown("### Settings")
        st.caption("Adjust how the agent searches and responds.")

        temperature = st.slider(
            "Response Style",
            min_value=0.0, max_value=1.0, value=0.1, step=0.05,
            help="0 = More precise and factual. 1 = More creative but less predictable. "
                 "Keep at 0.1 for clinical questions.",
            format="%.2f",
        )
        # Show a friendly label instead of a raw number
        style_label = "Precise & Factual" if temperature <= 0.2 else ("Balanced" if temperature <= 0.5 else "Creative")
        st.caption(f"Current style: **{style_label}**")

        max_pubmed = st.number_input(
            "Max Research Articles",
            min_value=1, max_value=10, value=3,
            help="How many PubMed research articles to retrieve per question. "
                 "More articles = slower but more thorough.",
        )

        enable_icd10 = st.checkbox(
            "Include Medical Code Lookup",
            value=True,
            help="Enable ICD-10 and HCPCS medical code lookups. "
                 "Uncheck for faster responses to non-coding questions.",
        )

        st.divider()

        st.markdown("### What can I ask?")
        st.caption(
            "This agent specializes in:\n"
            "- **Medical codes** — ICD-10, CPT, HCPCS codes\n"
            "- **Treatment guidelines** — ADA, ACC/AHA, USPSTF protocols\n"
            "- **Insurance coverage** — Medicare/Medicaid LCD/NCD policies\n"
            "- **Research evidence** — PubMed clinical trial summaries"
        )

        st.divider()

        if st.button("Start a new conversation", use_container_width=True, type="secondary"):
            st.session_state["messages"] = []
            st.session_state["session_id"] = str(uuid.uuid4())
            st.rerun()

        st.divider()
        st.markdown(
            '<div class="disclaimer">'
            "⚠️ <b>For educational purposes only.</b><br>"
            "Not medical advice. Always consult a licensed clinician.<br>"
            "In emergencies call <b>911</b>."
            "</div>",
            unsafe_allow_html=True,
        )

    return temperature, max_pubmed, enable_icd10


# ── Welcome screen (shown when no messages yet) ───────────────────────────────

def _render_welcome() -> str | None:
    """Render welcome screen with example questions. Returns a question if clicked."""
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size: 3em;">🏥</div>
        <h2 style="margin: 8px 0 4px 0;">Healthcare Q&A Agent</h2>
        <p style="color: #5f6368; font-size: 1.05em;">
            Ask clinical questions and get evidence-based answers grounded in<br>
            <b>PubMed research</b>, <b>clinical guidelines</b>, and <b>medical code databases</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # How it works explainer
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 🧠 Understands\nReads your question and figures out what kind of clinical answer is needed.")
    with col2:
        st.markdown("### 🔍 Researches\nSearches PubMed, clinical guidelines, and ICD-10 databases in real time.")
    with col3:
        st.markdown("### 📊 Evaluates\nChecks if the evidence found is good enough to answer your question reliably.")
    with col4:
        st.markdown("### ✍️ Answers\nWrites a clear, cited answer with the sources it used.")

    st.markdown("---")
    st.markdown("### Try one of these example questions:")

    # Two-column grid of example buttons
    cols = st.columns(2)
    for i, q in enumerate(_EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"example_{i}", use_container_width=True):
            return q

    st.markdown("")
    st.markdown(
        '<div class="disclaimer" style="max-width:600px; margin:0 auto;">'
        "⚠️ <b>Important:</b> Answers are for <b>educational and documentation support only</b>. "
        "This is not medical advice and does not replace clinical judgment. "
        "Always verify codes against official CMS publications."
        "</div>",
        unsafe_allow_html=True,
    )
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from src.memory.conversation_store import init_db
    from src.utils.logger import configure_logging
    from src.utils.validators import GuardrailError, validate_query

    configure_logging()
    init_db()

    # Session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    # Sidebar
    temperature, max_pubmed, enable_icd10 = _render_sidebar()

    # Header (compact when messages exist)
    if not st.session_state["messages"]:
        # Full welcome screen — may return a pre-clicked example question
        pre_filled = _render_welcome()
    else:
        st.markdown("## 🏥 Healthcare Q&A Agent")
        st.caption(
            "Evidence-based answers from **PubMed**, **clinical guidelines**, and **ICD-10 / HCPCS** databases. "
            "Not medical advice."
        )
        pre_filled = None

    # Render conversation history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(f"**Your question:** {msg['content']}")
            else:
                # Answer section
                st.markdown('<p class="section-label">Answer</p>', unsafe_allow_html=True)
                st.markdown(_linkify_pmids(msg["content"]))

                # Sources
                pmids = msg.get("pmids", [])
                tools = msg.get("tool_results", [])
                if pmids or tools:
                    st.markdown('<p class="section-label" style="margin-top:10px;">Sources Used</p>',
                                unsafe_allow_html=True)
                    with st.expander(
                        f"{len(tools)} source(s) searched"
                        + (f" · {len(pmids)} PubMed article(s)" if pmids else ""),
                        expanded=False,
                    ):
                        for r in tools:
                            st.markdown(f"- {_tool_friendly_name(r['tool'])} ✅")
                        for pmid in pmids:
                            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                            st.markdown(f"- [PMID {pmid}]({url})")

                # Reasoning trace
                if msg.get("trace"):
                    st.markdown('<p class="section-label" style="margin-top:10px;">How the Agent Worked</p>',
                                unsafe_allow_html=True)
                    with st.expander("See step-by-step reasoning", expanded=False):
                        _render_trace(msg["trace"])

    # Chat input
    user_input: str | None = st.chat_input(
        "Type your clinical question here, e.g. 'What is the ICD-10 code for heart failure?'"
    ) or pre_filled

    if not user_input:
        return

    # Validate
    try:
        clean_input = validate_query(user_input)
    except GuardrailError as exc:
        with st.chat_message("assistant"):
            st.warning(f"**Safety check:** {exc.safe_response}")
        st.session_state["messages"].append(
            {"role": "assistant", "content": exc.safe_response,
             "trace": [], "tool_results": [], "pmids": []}
        )
        return

    # Show user message
    st.session_state["messages"].append({"role": "user", "content": clean_input})
    with st.chat_message("user"):
        st.markdown(f"**Your question:** {clean_input}")

    # Load model
    try:
        processing_graph, llm = _build_processing_graph(
            temperature=temperature, max_pubmed=max_pubmed, enable_icd10=enable_icd10,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"Could not load the AI model. Check that Ollama is running and the model is downloaded.  \n`{exc}`"
        )
        return

    # Run agent
    with st.chat_message("assistant"):
        _run_and_stream(processing_graph, llm, clean_input)


if __name__ == "__main__":
    main()
