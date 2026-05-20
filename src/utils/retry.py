"""Retry decorators for LLM and tool calls.

Decorators
----------
with_anthropic_retry    Anthropic SDK errors (APIStatusError, APITimeoutError,
                        RateLimitError). Max 3 retries, 1 s initial / ×2 / 30 s max.
with_llm_retry          Generic LLM/HTTP transient errors (network, timeout).
with_tool_retry         Tool-call errors with configurable attempt count.
"""
from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import (
    RetryCallState,
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logger import get_logger

_log = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ── Anthropic error types ─────────────────────────────────────────────────────
# Imported lazily so the package is optional; falls back to empty tuple if not installed.

def _load_anthropic_errors() -> tuple[type[Exception], ...]:
    try:
        import anthropic  # type: ignore[import-untyped]
        return (
            anthropic.APIStatusError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
        )
    except ImportError:
        return ()


_ANTHROPIC_ERRORS: tuple[type[Exception], ...] = _load_anthropic_errors()


# ── Generic retryable transient errors ───────────────────────────────────────

_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

try:
    import httpx  # type: ignore[import-untyped]
    _TRANSIENT_ERRORS = _TRANSIENT_ERRORS + (httpx.TimeoutException, httpx.ConnectError)
except ImportError:
    pass

try:
    import requests.exceptions as _req_exc
    _TRANSIENT_ERRORS = _TRANSIENT_ERRORS + (_req_exc.Timeout, _req_exc.ConnectionError)
except ImportError:
    pass


# ── Structured before-sleep callback ─────────────────────────────────────────

def _make_before_sleep(label: str) -> Callable[[RetryCallState], None]:
    """Return a before_sleep callback that emits structured log lines."""

    def _before_sleep(rs: RetryCallState) -> None:
        exc = rs.outcome.exception() if rs.outcome else None
        exc_type = type(exc).__name__ if exc else "unknown"
        next_action = rs.next_action
        delay = getattr(next_action, "sleep", 0.0) if next_action else 0.0

        _log.warning(
            "retry_attempt",
            label=label,
            attempt=rs.attempt_number,
            exc_type=exc_type,
            exc_message=str(exc)[:200] if exc else None,
            delay_seconds=round(delay, 2),
        )

    return _before_sleep


# ── Decorator: Anthropic API ──────────────────────────────────────────────────

_ANTHROPIC_MAX_ATTEMPTS: int = int(os.getenv("ANTHROPIC_RETRY_MAX_ATTEMPTS", "3"))
_ANTHROPIC_MIN_WAIT: float = float(os.getenv("ANTHROPIC_RETRY_MIN_WAIT", "1.0"))
_ANTHROPIC_MAX_WAIT: float = float(os.getenv("ANTHROPIC_RETRY_MAX_WAIT", "30.0"))
_ANTHROPIC_MULTIPLIER: float = float(os.getenv("ANTHROPIC_RETRY_MULTIPLIER", "2.0"))


def with_anthropic_retry(func: F) -> F:
    """Retry an Anthropic SDK call with exponential backoff.

    Catches:
      - anthropic.RateLimitError   (HTTP 429)
      - anthropic.APITimeoutError  (request/connect timeout)
      - anthropic.APIStatusError   (5xx server errors)

    Policy (all configurable via env vars):
      Max attempts : ANTHROPIC_RETRY_MAX_ATTEMPTS  (default 3)
      Initial wait : ANTHROPIC_RETRY_MIN_WAIT      (default 1 s)
      Backoff      : ANTHROPIC_RETRY_MULTIPLIER    (default ×2)
      Max wait     : ANTHROPIC_RETRY_MAX_WAIT      (default 30 s)

    Each retry is logged with: label, attempt number, exc_type, exc_message, delay_seconds.

    Raises:
        The last exception after all attempts are exhausted.
    """
    if not _ANTHROPIC_ERRORS:
        # anthropic package not installed — return function unmodified
        return func

    # Separate RateLimitError so we can add a jitter-friendly minimum wait
    try:
        import anthropic  # type: ignore[import-untyped]
        _rate_limit_cls: type[Exception] = anthropic.RateLimitError
    except ImportError:
        _rate_limit_cls = Exception  # unreachable branch

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        label = f"anthropic:{func.__qualname__}"

        decorated = retry(
            reraise=True,
            retry=retry_if_exception_type(_ANTHROPIC_ERRORS),
            stop=stop_after_attempt(_ANTHROPIC_MAX_ATTEMPTS),
            wait=wait_exponential(
                multiplier=_ANTHROPIC_MULTIPLIER,
                min=_ANTHROPIC_MIN_WAIT,
                max=_ANTHROPIC_MAX_WAIT,
            ),
            before_sleep=_make_before_sleep(label),
        )(func)

        try:
            return decorated(*args, **kwargs)
        except _ANTHROPIC_ERRORS as exc:
            _log.error(
                "retry_exhausted",
                label=label,
                exc_type=type(exc).__name__,
                exc_message=str(exc)[:300],
                max_attempts=_ANTHROPIC_MAX_ATTEMPTS,
            )
            raise

    return wrapper  # type: ignore[return-value]


# ── Decorator: generic LLM / HTTP ─────────────────────────────────────────────

def _get_llm_retry_config() -> dict[str, Any]:
    return {
        "max_attempts": int(os.getenv("LLM_RETRY_MAX_ATTEMPTS", "4")),
        "min_wait": float(os.getenv("LLM_RETRY_MIN_WAIT", "1.0")),
        "max_wait": float(os.getenv("LLM_RETRY_MAX_WAIT", "30.0")),
        "multiplier": float(os.getenv("LLM_RETRY_MULTIPLIER", "2.0")),
    }


def with_llm_retry(func: F) -> F:
    """Retry on generic transient LLM/HTTP errors (network failures, timeouts).

    Uses the same exponential-backoff policy as with_anthropic_retry but targets
    connection-level errors rather than Anthropic-specific API errors.
    """
    cfg = _get_llm_retry_config()
    label = f"llm:{func.__qualname__}"

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        decorated = retry(
            reraise=True,
            retry=retry_if_exception_type(_TRANSIENT_ERRORS),
            stop=stop_after_attempt(cfg["max_attempts"]),
            wait=wait_exponential(
                multiplier=cfg["multiplier"],
                min=cfg["min_wait"],
                max=cfg["max_wait"],
            ),
            before_sleep=_make_before_sleep(label),
        )(func)
        return decorated(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


# ── Decorator: tool calls ─────────────────────────────────────────────────────

def with_tool_retry(
    max_attempts: int = 3,
    reraise: bool = False,
) -> Callable[[F], F]:
    """Parametrised retry decorator for tool calls.

    Args:
        max_attempts: Maximum retry attempts (default 3).
        reraise:      Re-raise the final exception if True; otherwise return an
                      error string (useful for non-critical tools).

    Returns:
        The wrapped function.
    """

    def decorator(func: F) -> F:
        label = f"tool:{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            decorated = retry(
                reraise=reraise,
                retry=retry_if_exception_type((Exception,)),
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1.0, min=0.5, max=10.0),
                before_sleep=_make_before_sleep(label),
            )(func)
            try:
                return decorated(*args, **kwargs)
            except RetryError as exc:
                _log.error(
                    "retry_exhausted",
                    label=label,
                    max_attempts=max_attempts,
                    error=str(exc),
                )
                return f"Tool '{func.__name__}' failed after {max_attempts} attempts."

        return wrapper  # type: ignore[return-value]

    return decorator
