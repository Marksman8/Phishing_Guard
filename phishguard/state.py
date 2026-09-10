from datetime import datetime, timezone
from typing import Any, TypedDict

from .redact import redact


class PhishGuardState(TypedDict, total=False):
    # --- input ---
    case_id: str
    raw_email: str

    # --- node 1: analyzer ---
    parsed: dict[str, Any]
    injection_detected: bool
    injection_matches: list[dict[str, Any]]

    # --- node 2: retriever ---
    retrieved: list[dict[str, Any]]
    max_similarity: float
    mean_top3_similarity: float
    max_legitimate_similarity: float
    phishing_margin: float
    retrieval_evidence: float
    retrieval_available: bool

    # --- node 3: tool verification ---
    tool_results: dict[str, Any]
    tools_available_count: int
    tools_attempted_count: int

    # --- node 4: risk ---
    content_signals: dict[str, Any]
    risk_score: float
    risk_breakdown: dict[str, Any]
    confidence: float
    confidence_breakdown: dict[str, Any]

    # --- node 5: decision ---
    classification: str
    recommendation: str
    citations: list[dict[str, Any]]
    escalation_ticket: str | None

    # --- cross-cutting ---
    trace: list[dict[str, Any]]
    errors: list[str]


def new_state(case_id: str, raw_email: str) -> PhishGuardState:
    return PhishGuardState(
        case_id=case_id,
        raw_email=raw_email,
        injection_detected=False,
        injection_matches=[],
        retrieved=[],
        max_similarity=0.0,
        mean_top3_similarity=0.0,
        max_legitimate_similarity=0.0,
        phishing_margin=0.0,
        retrieval_evidence=0.0,
        retrieval_available=False,
        tool_results={},
        tools_available_count=0,
        tools_attempted_count=0,
        content_signals={},
        citations=[],
        escalation_ticket=None,
        trace=[],
        errors=[],
    )


def trace_entry(
    node: str,
    inputs_seen: dict[str, Any] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    output: dict[str, Any] | None = None,
    reason: str = "",
) -> dict[str, Any]:
    """Build one redacted audit-trail record. Redaction happens here so no node
    can bypass it on the way to disk."""
    return redact(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "node": node,
            "inputs_seen": inputs_seen or {},
            "tool_calls": tool_calls or [],
            "output": output or {},
            "reason": reason,
        }
    )
