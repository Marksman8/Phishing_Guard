"""LangGraph wiring.

Nodes are added as each phase lands; the compiled graph is always runnable.
Trace accumulation is explicit (each node returns the full list) rather than
reducer-based, so the audit trail has a single, readable write path.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, StateGraph

from . import config
from .nodes.analyzer import analyzer_node
from .nodes.decision import decision_node
from .nodes.retriever import retriever_node
from .nodes.risk import risk_node
from .nodes.verifier import verifier_node
from .state import PhishGuardState, new_state


def build_graph():
    graph = StateGraph(PhishGuardState)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("evidence_retriever", retriever_node)
    graph.add_node("tool_verification", verifier_node)
    graph.add_node("risk_agent", risk_node)
    graph.add_node("decision_agent", decision_node)
    graph.set_entry_point("analyzer")
    graph.add_edge("analyzer", "evidence_retriever")
    graph.add_edge("evidence_retriever", "tool_verification")
    graph.add_edge("tool_verification", "risk_agent")
    graph.add_edge("risk_agent", "decision_agent")
    graph.add_edge("decision_agent", END)
    return graph.compile()


_COMPILED = None


def get_graph():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED


def new_case_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def persist_trace(state: PhishGuardState) -> str:
    """Write the redacted audit trail. The raw email body is never included."""
    config.ensure_dirs()
    path = config.TRACE_DIR / f"{state['case_id']}.json"
    payload = {
        "case_id": state["case_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classification": state.get("classification"),
        "risk_score": state.get("risk_score"),
        "confidence": state.get("confidence"),
        "risk_breakdown": state.get("risk_breakdown"),
        "confidence_breakdown": state.get("confidence_breakdown"),
        "injection_detected": state.get("injection_detected"),
        "escalation_ticket": state.get("escalation_ticket"),
        "errors": state.get("errors", []),
        "trace": state.get("trace", []),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def run_case(raw_email: str, case_id: str | None = None) -> dict[str, Any]:
    case_id = case_id or new_case_id()
    result = get_graph().invoke(new_state(case_id, raw_email))
    result["trace_path"] = persist_trace(result)
    return result
