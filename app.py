import json
from pathlib import Path

import streamlit as st

from phishguard import config
from phishguard.graph import run_case
from phishguard.llm import fallback_reason, get_provider
from phishguard.vectorstore import collection_stats

st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="wide")

SAMPLE_DIR = config.ROOT / "samples"


def load_samples() -> dict[str, str]:
    if not SAMPLE_DIR.exists():
        return {}
    return {
        p.stem.replace("_", " ").title(): p.read_text(encoding="utf-8")
        for p in sorted(SAMPLE_DIR.glob("*.eml"))
    }


def render_parsed(parsed: dict) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Sender**")
        st.write(
            {
                "display name": parsed["sender_display_name"] or "—",
                "address": parsed["sender_address"] or "—",
                "domain": parsed["sender_domain"] or "—",
                "reply-to": parsed["reply_to"] or "—",
                "reply-to mismatch": parsed["reply_to_mismatch"],
            }
        )
        st.markdown("**Subject**")
        st.write(parsed["subject"] or "— (none)")
        if not parsed["has_headers"]:
            st.warning(
                "No headers in this paste — SPF/DKIM/DMARC cannot be verified. "
                "This lowers confidence rather than counting as a pass."
            )
    with right:
        st.markdown(f"**URLs ({len(parsed['urls'])})**")
        if parsed["urls"]:
            st.dataframe(
                [
                    {
                        "display text": u["display_text"][:50],
                        "real href": u["href"][:50],
                        "mismatch": "⚠️" if u["display_mismatch"] else "—",
                    }
                    for u in parsed["urls"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("None found.")
        st.markdown(f"**Attachments ({len(parsed['attachments'])})**")
        if parsed["attachments"]:
            st.dataframe(
                [
                    {
                        "filename": a["filename"],
                        "risky type": "⚠️" if a["risky_type"] else "—",
                    }
                    for a in parsed["attachments"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("None mentioned.")


VERDICT_STYLE = {
    "HIGH RISK": ("🛑", "error"),
    "SUSPICIOUS": ("⚠️", "warning"),
    "SAFE": ("✅", "success"),
    "NEEDS HUMAN REVIEW": ("🧑‍⚖️", "info"),
}


def render_verdict(result: dict) -> None:
    classification = result.get("classification", "—")
    icon, style = VERDICT_STYLE.get(classification, ("❔", "info"))
    risk = result.get("risk_score", 0.0)
    confidence = result.get("confidence", 0.0)

    getattr(st, style)(f"### {icon} {classification}\n\n{result.get('recommendation', '')}")

    cols = st.columns(3)
    cols[0].metric("Risk score", f"{risk:.2f}")
    cols[1].metric("Confidence", f"{confidence:.2f}")
    cols[2].metric(
        "Escalation ticket", result.get("escalation_ticket") or "—"
    )
    st.progress(min(max(risk, 0.0), 1.0), text=f"risk {risk:.2f}")

    if classification == "NEEDS HUMAN REVIEW":
        st.info(
            f"**Why this escalated:** confidence {confidence:.2f} is below the "
            f"{config.CONFIDENCE_ESCALATION_THRESHOLD} threshold, so there was not enough "
            f"evidence to stand behind *any* verdict — including the risk score of "
            f"{risk:.2f}. Risk and confidence are separate measures: a high-risk guess "
            "on thin evidence is exactly the case that must reach a human rather than "
            "be reported as a finding."
        )

    st.markdown("**Cited evidence** — every bullet traces to a corpus chunk or a tool result")
    citations = result.get("citations", [])
    if not citations:
        st.caption("No citable evidence was produced.")
    for bullet in citations:
        with st.container(border=True):
            st.markdown(f"{bullet['text']} &nbsp;`[{bullet['citation']}]`", unsafe_allow_html=True)
            st.caption(f"Source {bullet['citation']}: {bullet['evidence'][:260]}")


def render_scoring(result: dict) -> None:
    st.subheader("How the score was computed")
    breakdown = result.get("risk_breakdown", {})
    components = breakdown.get("components", {})
    st.caption(
        "The risk score is deterministic arithmetic over the evidence, not a number "
        "asserted by a language model. The same evidence always yields the same score."
    )
    st.dataframe(
        [
            {
                "component": name,
                "value": round(c["value"], 3),
                "weight": c["weight"],
                "contribution": round(c["contribution"], 4),
                "basis": c["detail"][:120],
            }
            for name, c in components.items()
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.code(breakdown.get("formula", ""), language="text")
    if breakdown.get("injection_floor_applied"):
        st.error(
            f"**Injection floor applied.** The weighted sum was "
            f"{breakdown['weighted_sum']:.3f}, raised to {breakdown['injection_floor']} "
            "because a prompt-injection attempt was detected. Legitimate mail does not "
            "try to reprogram a classifier, so this alone keeps the case out of SAFE."
        )

    st.markdown("**Confidence — how much evidence there was to reason about**")
    st.caption(
        "Components whose preconditions do not exist are marked N/A and dropped from "
        "the weighted average rather than scored zero. An email with no links has "
        "nothing for WHOIS to check; that is an absence of attack surface, not a "
        "failure to gather evidence."
    )
    st.dataframe(
        [
            {
                "component": name,
                "value": "N/A" if not c.get("applicable") else round(c["value"], 3),
                "weight": c["weight"] if c.get("applicable") else "—",
                "basis": c["detail"][:120],
            }
            for name, c in result.get("confidence_breakdown", {}).items()
        ],
        use_container_width=True,
        hide_index=True,
    )

    signals = result.get("content_signals", {})
    if signals:
        st.markdown("**LLM content signals** — discrete 0/1 flags, each justified")
        st.dataframe(
            [
                {
                    "signal": name,
                    "flag": entry["flag"],
                    "justification": entry["justification"][:140],
                }
                for name, entry in signals.items()
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_injection(result: dict) -> None:
    matches = result.get("injection_matches", [])
    if not result.get("injection_detected"):
        st.success(
            "**No prompt-injection indicators.** The body was still passed to every "
            "model as fenced, labeled data rather than as instructions."
        )
        return

    st.error(
        f"**Prompt injection detected — {len(matches)} indicator(s).** "
        "The content was deliberately *not* sanitised: an attempt to reprogram the "
        "classifier is itself strong evidence of phishing, and is carried forward as "
        "a positive signal."
    )
    st.dataframe(
        [
            {
                "technique": m["technique"],
                "matched span": m["matched_span"],
                "offset": m["offset"],
                "severity": m["severity"],
                "why it matters": m["why"],
            }
            for m in matches
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_retrieval(result: dict) -> None:
    st.subheader("Retrieved evidence (RAG)")
    if not result.get("retrieval_available"):
        st.warning(
            "**Retrieval unavailable** — the Chroma index is missing or empty. "
            "Recorded as unavailable, which lowers confidence; it is *not* read as "
            "'no phishing patterns matched'. Run `python scripts/ingest.py`."
        )
        return

    cols = st.columns(4)
    cols[0].metric("Nearest phishing", f"{result['max_similarity']:.3f}")
    cols[1].metric("Nearest legitimate", f"{result['max_legitimate_similarity']:.3f}")
    cols[2].metric("Margin", f"{result['phishing_margin']:+.3f}")
    cols[3].metric("Calibrated evidence", f"{result['retrieval_evidence']:.3f}")

    st.caption(
        "MiniLM cosine similarity over English prose sits in a narrow band, so raw "
        "similarity alone would give every email a floor of risk. The margin — how "
        "much closer the text is to phishing than to legitimate mail — is what "
        "separates a real shipping notice from a fake one."
    )

    st.dataframe(
        [
            {
                "chunk": c["chunk_id"],
                "label": c["label"],
                "pattern": c["pattern"],
                "similarity": round(c["similarity"], 3),
                "excerpt": c["text"][:110] + ("…" if len(c["text"]) > 110 else ""),
            }
            for c in result.get("retrieved", [])
        ],
        use_container_width=True,
        hide_index=True,
    )


STATUS_ICON = {"ok": "✅", "not_registered": "✅", "unavailable": "❓"}


def render_tools(result: dict) -> None:
    tools = result.get("tool_results", {})
    calls = tools.get("calls", [])
    st.subheader("Tool verification (FastMCP)")
    if not calls:
        st.caption("No tool calls were made.")
        return

    cols = st.columns(3)
    cols[0].metric("Returned real data", f"{tools['available']}/{tools['attempted']}")
    cols[1].metric("Unavailable", len(tools.get("unavailable_refs", [])))
    cols[2].metric("Positive signals", len(tools.get("signals", [])))

    st.dataframe(
        [
            {
                "ref": c["ref"],
                "tool": c["tool"],
                "target": str(list(c["args"].values())[0])[:44],
                "status": f"{STATUS_ICON.get(c['status'], '❓')} {c['status']}",
                "signal": c.get("signal") or "—",
                "detail": c.get("detail", "")[:130],
            }
            for c in calls
        ],
        use_container_width=True,
        hide_index=True,
    )

    if tools.get("unavailable_refs"):
        st.info(
            f"**{len(tools['unavailable_refs'])} check(s) could not be completed** "
            f"({', '.join(tools['unavailable_refs'])}). These lower confidence. "
            "An unavailable check is never treated as a check that passed."
        )


def render_trace(result: dict) -> None:
    st.subheader("Audit trail")
    trace = result.get("trace", [])
    st.caption(
        f"Case `{result['case_id']}` — {len(trace)} node step(s). "
        f"Persisted to `{Path(result.get('trace_path', '')).name}`. "
        "Addresses and credential-looking strings are redacted before anything is written."
    )
    for i, entry in enumerate(trace, start=1):
        with st.expander(f"{i}. {entry['node']} — {entry['timestamp']}"):
            st.markdown(f"**Reason / branch:** {entry['reason']}")
            st.markdown("**Inputs seen**")
            st.json(entry["inputs_seen"], expanded=False)
            if entry["tool_calls"]:
                st.markdown("**Tool calls**")
                st.json(entry["tool_calls"], expanded=False)
            st.markdown("**Output**")
            st.json(entry["output"], expanded=False)

    with st.expander("Raw trace JSON (as persisted)"):
        st.code(json.dumps(trace, indent=2, ensure_ascii=False), language="json")


def main() -> None:
    st.title("🛡️ PhishGuard")
    st.caption(
        "Multi-agent phishing analysis — cited evidence, a deterministic risk score, "
        "and a full audit trail."
    )

    samples = load_samples()

    with st.sidebar:
        st.header("Session")
        provider = get_provider()
        reason = fallback_reason()
        if reason:
            st.warning(f"LLM fallback active — {reason}")
        st.metric("LLM provider", provider.describe())
        st.caption(f"Configured in .env: `{config.PROVIDER}`")
        stats = collection_stats()
        if stats["available"]:
            st.metric("Pattern corpus", f"{stats['count']} chunks")
        else:
            st.error("Chroma index missing — run `python scripts/ingest.py`")
        st.divider()
        if samples:
            st.header("Samples")
            choice = st.selectbox("Load a sample", ["—"] + list(samples))
            if choice != "—" and st.button("Load", use_container_width=True):
                st.session_state["raw"] = samples[choice]
                st.session_state.pop("result", None)
                st.rerun()
        st.divider()
        if st.button("Clear session", type="secondary", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.caption(
            "Pasted email text is held in memory only. Clearing the session removes it; "
            "only the redacted trace file remains on disk."
        )

    raw = st.text_area(
        "Paste a raw email (headers + body, or just the body)",
        value=st.session_state.get("raw", ""),
        height=260,
        placeholder="From: ...\nSubject: ...\n\nBody text...",
    )

    if st.button("Analyze", type="primary"):
        if not raw.strip():
            st.error("Paste an email first.")
            return
        st.session_state["raw"] = raw
        with st.spinner("Running agent graph…"):
            st.session_state["result"] = run_case(raw)

    result = st.session_state.get("result")
    if not result:
        return

    st.divider()
    render_verdict(result)
    st.divider()
    render_scoring(result)
    st.divider()
    render_injection(result)
    st.divider()
    st.subheader("Parsed email")
    render_parsed(result["parsed"])
    st.divider()
    render_retrieval(result)
    st.divider()
    render_tools(result)
    st.divider()
    render_trace(result)


if __name__ == "__main__":
    main()
