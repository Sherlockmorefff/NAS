"""Candidate-vs-infrastructure evaluation error classification."""

from __future__ import annotations

from typing import Any, Mapping


class CandidateInvalidError(ValueError):
    """A decoded candidate violates architecture/search constraints."""


class InfrastructureEvaluationError(RuntimeError):
    """The evaluation environment failed independently of candidate quality."""

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.context = dict(context or {})

    def with_context(self, **fields: Any) -> "InfrastructureEvaluationError":
        context = {**self.context, **fields}
        return InfrastructureEvaluationError(str(self), context=context)


def is_infrastructure_failure(exc: BaseException) -> bool:
    if isinstance(exc, (InfrastructureEvaluationError, MemoryError)):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    markers = (
        "out of memory",
        "cuda error",
        "cuda driver",
        "cudnn error",
        "cublas error",
        "device-side assert",
        "illegal memory access",
        "system memory",
        "cannot allocate memory",
        "not enough memory",
        "deterministic implementation",
        "deterministic algorithms",
        "deterministic_algorithms",
    )
    return any(marker in message for marker in markers)
