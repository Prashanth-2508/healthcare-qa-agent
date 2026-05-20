"""Shared exception types for all healthcare QA tools."""
from __future__ import annotations


class ToolExecutionError(Exception):
    """Raised when a tool fails to execute after all retries.

    Attributes:
        tool_name: The name of the tool that failed.
        reason: Human-readable description of the failure.
        original: The underlying exception, if any.
    """

    def __init__(
        self,
        tool_name: str,
        reason: str,
        original: Exception | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.reason = reason
        self.original = original
        super().__init__(str(self))

    def __str__(self) -> str:
        base = f"[{self.tool_name}] {self.reason}"
        if self.original:
            return f"{base} — caused by {type(self.original).__name__}: {self.original}"
        return base

    def user_message(self) -> str:
        """Safe, non-technical message suitable for returning to the agent."""
        return (
            f"The tool '{self.tool_name}' could not complete the request. "
            f"Reason: {self.reason}"
        )
