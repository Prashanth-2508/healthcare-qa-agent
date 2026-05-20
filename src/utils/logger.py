"""Step-level reasoning trace logger with node-transition timing and JSONL output."""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, Processor

# ── Agent step enum ───────────────────────────────────────────────────────────

class AgentStep(str, Enum):
    REASON = "reason"
    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    RESPOND = "respond"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    EVALUATION = "evaluation"


# ── structlog configuration ───────────────────────────────────────────────────

_STEP_ICONS: dict[str, str] = {
    AgentStep.REASON: "[REASON]",
    AgentStep.PLAN: "[PLAN]",
    AgentStep.ACT: "[ACT]",
    AgentStep.OBSERVE: "[OBSERVE]",
    AgentStep.RESPOND: "[RESPOND]",
    AgentStep.TOOL_CALL: "[TOOL]",
    AgentStep.TOOL_RESULT: "[RESULT]",
    AgentStep.MEMORY_READ: "[MEM-R]",
    AgentStep.MEMORY_WRITE: "[MEM-W]",
    AgentStep.EVALUATION: "[EVAL]",
}


def _add_step_icon(logger: Any, method: str, event_dict: EventDict) -> EventDict:
    step = event_dict.get("step")
    if step and step in _STEP_ICONS:
        event_dict["step"] = f"{_STEP_ICONS[step]} {step}"
    return event_dict


def _add_logger_name(logger: Any, method: str, event_dict: EventDict) -> EventDict:
    """Safe logger-name injector that works with both PrintLogger and stdlib Logger."""
    name = getattr(logger, "name", None) or getattr(logger, "_name", None)
    if name:
        event_dict["logger"] = name
    return event_dict


def _build_processors(log_format: str) -> list[Processor]:
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        _add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
    ]
    if log_format == "json":
        return shared + [structlog.processors.JSONRenderer()]
    return shared + [
        _add_step_icon,
        structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
    ]


def configure_logging() -> None:
    """Call once at application startup."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()
    structlog.configure(
        processors=_build_processors(log_format),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level, 20)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    return structlog.get_logger(name)


# ── JSONL trace event schema ──────────────────────────────────────────────────
#
# Every line in the JSONL file is one of two event shapes:
#
#   enter event:
#   { "event": "node_enter", "ts": "<ISO>", "session_id": "...",
#     "node": "reason", "observe_iter": 0,
#     "state_summary": {<trimmed key fields>} }
#
#   exit event:
#   { "event": "node_exit",  "ts": "<ISO>", "session_id": "...",
#     "node": "reason", "observe_iter": 0,
#     "duration_ms": 234.5, "success": true,
#     "output_summary": {<trimmed key fields>} }
#
# Step events (existing ReasoningTracer.log_step):
#   { "event": "step", "ts": "<ISO>", "session_id": "...",
#     "step": "reason", "observe_iter": 0, "content": "..." }


# ── ReasoningTracer ───────────────────────────────────────────────────────────

class ReasoningTracer:
    """Records every agent step and node transition for post-hoc inspection.

    Two complementary outputs:
    - In-memory trace list (returned in the run() result)
    - JSONL file: one JSON object per line, written via flush_jsonl()
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        # Human-readable step trace (existing interface)
        self.trace: list[dict[str, Any]] = []
        # Full JSONL event stream
        self._events: list[dict[str, Any]] = []
        # Pending enter timestamps: node → monotonic start time
        self._pending: dict[str, float] = {}
        self._log = get_logger("tracer")

    # ── Step-level logging (existing interface) ───────────────────────────────

    def log_step(
        self,
        step: AgentStep,
        content: str,
        metadata: dict[str, Any] | None = None,
        iteration: int = 0,
    ) -> None:
        """Record a named reasoning step with optional metadata."""
        ts = _iso_now()
        entry: dict[str, Any] = {
            "session_id": self.session_id,
            "observe_iter": iteration,
            "step": step.value,
            "content": content,
            "ts": ts,
            **(metadata or {}),
        }
        self.trace.append(entry)

        jsonl_event: dict[str, Any] = {
            "event": "step",
            "ts": ts,
            "session_id": self.session_id,
            "step": step.value,
            "observe_iter": iteration,
            "content": content[:500],
            **(metadata or {}),
        }
        self._events.append(jsonl_event)

        self._log.info(
            "agent_step",
            step=step.value,
            session_id=self.session_id,
            observe_iter=iteration,
            content=content[:200],
        )

    # ── Node-transition logging ───────────────────────────────────────────────

    def node_enter(
        self,
        node: str,
        state_summary: dict[str, Any],
        observe_iter: int = 0,
    ) -> float:
        """Record a node entry event; returns the monotonic start time."""
        ts = _iso_now()
        t0 = time.monotonic()
        self._pending[node] = t0

        event: dict[str, Any] = {
            "event": "node_enter",
            "ts": ts,
            "session_id": self.session_id,
            "node": node,
            "observe_iter": observe_iter,
            "state_summary": state_summary,
        }
        self._events.append(event)
        self._log.debug(
            "node_enter",
            node=node,
            session_id=self.session_id,
            observe_iter=observe_iter,
        )
        return t0

    def node_exit(
        self,
        node: str,
        output_summary: dict[str, Any],
        t0: float,
        success: bool = True,
        observe_iter: int = 0,
    ) -> None:
        """Record a node exit event with wall-clock duration."""
        duration_ms = (time.monotonic() - t0) * 1000
        ts = _iso_now()
        self._pending.pop(node, None)

        event: dict[str, Any] = {
            "event": "node_exit",
            "ts": ts,
            "session_id": self.session_id,
            "node": node,
            "observe_iter": observe_iter,
            "duration_ms": round(duration_ms, 2),
            "success": success,
            "output_summary": output_summary,
        }
        self._events.append(event)
        self._log.info(
            "node_exit",
            node=node,
            session_id=self.session_id,
            observe_iter=observe_iter,
            duration_ms=round(duration_ms, 1),
            success=success,
        )

    # ── JSONL output ──────────────────────────────────────────────────────────

    def flush_jsonl(self, trace_dir: str | Path | None = None) -> Path:
        """Write all accumulated events to a JSONL file.

        Args:
            trace_dir: Directory for trace files (default: AGENT_TRACE_DIR env var,
                       or ./traces/).

        Returns:
            Path to the written JSONL file.
        """
        out_dir = Path(
            trace_dir
            or os.getenv("AGENT_TRACE_DIR", "./traces")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.session_id}.jsonl"

        with open(out_path, "w", encoding="utf-8") as fh:
            for event in self._events:
                fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

        self._log.info(
            "jsonl_trace_written",
            path=str(out_path),
            event_count=len(self._events),
        )
        return out_path

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_trace(self) -> list[dict[str, Any]]:
        return list(self.trace)

    def get_events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def summary(self) -> str:
        """Human-readable step summary for console output."""
        lines = [f"=== Reasoning Trace [{self.session_id}] ==="]
        for e in self._events:
            if e["event"] == "node_enter":
                lines.append(
                    f"  → ENTER {e['node'].upper()} "
                    f"[iter={e['observe_iter']}] @ {e['ts']}"
                )
            elif e["event"] == "node_exit":
                status = "✓" if e.get("success") else "✗"
                lines.append(
                    f"  ← EXIT  {e['node'].upper()} "
                    f"[iter={e['observe_iter']}] "
                    f"{status} {e['duration_ms']:.0f}ms"
                )
            elif e["event"] == "step":
                snippet = e.get("content", "")[:100]
                lines.append(
                    f"     [{e['step'].upper()}] {snippet}"
                )
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
