"""Node 1 — Analyzer.

Deterministic by design: no LLM call happens here. Parsing and injection
detection must not themselves be influenced by the content being parsed, so the
first thing that ever touches untrusted input is plain code, not a model.
"""

from ..injection import detect_injection, injection_summary
from ..parsing import parse_email
from ..state import PhishGuardState, trace_entry

NODE = "analyzer"


def analyzer_node(state: PhishGuardState) -> dict:
    raw = state.get("raw_email", "")
    parsed = parse_email(raw)
    findings = detect_injection(raw)
    detected = bool(findings)

    reason = (
        "Injection indicators found; preserved verbatim and carried forward as a "
        "positive phishing signal. Content was NOT sanitised, because a "
        "reprogramming attempt is itself strong evidence."
        if detected
        else "No injection indicators. Body will be passed to downstream models as "
        "fenced, labeled data only."
    )

    return {
        "parsed": parsed,
        "injection_detected": detected,
        "injection_matches": findings,
        "trace": state.get("trace", [])
        + [
            trace_entry(
                node=NODE,
                inputs_seen={
                    "raw_email_chars": len(raw),
                    "headers_present": parsed["has_headers"],
                },
                output={
                    "sender_address": parsed["sender_address"] or None,
                    "sender_display_name": parsed["sender_display_name"] or None,
                    "sender_domain": parsed["sender_domain"] or None,
                    "reply_to": parsed["reply_to"] or None,
                    "reply_to_mismatch": parsed["reply_to_mismatch"],
                    "subject": parsed["subject"] or None,
                    "url_count": len(parsed["urls"]),
                    "display_mismatch_count": sum(
                        1 for u in parsed["urls"] if u["display_mismatch"]
                    ),
                    "link_domains": parsed["link_domains"],
                    "attachments": [a["filename"] for a in parsed["attachments"]],
                    "risky_attachments": [
                        a["filename"] for a in parsed["attachments"] if a["risky_type"]
                    ],
                    "injection_detected": detected,
                    "injection_summary": injection_summary(findings),
                    "injection_matches": [
                        {
                            "technique": f["technique"],
                            "matched_span": f["matched_span"],
                            "offset": f["offset"],
                            "severity": f["severity"],
                        }
                        for f in findings
                    ],
                },
                reason=reason,
            )
        ],
    }
