"""Node 3 — Tool Verification via FastMCP.

Which tools run is decided from the parsed email, not from anything the email
asks for. Every call is recorded with a citation ref (T1, T2, …) so the decision
node can point at a specific tool result instead of asserting a conclusion.

An "unavailable" result is counted separately from a clean result. The two must
never collapse: "we checked and it was fine" and "we could not check" lead to
different confidence, and therefore sometimes to different routing.
"""

from typing import Any

from ..mcp_client import call_tools
from ..parsing import registrable_domain
from ..state import PhishGuardState, trace_entry

NODE = "tool_verification"

MAX_WHOIS = 3
MAX_RESOLVE = 3
MAX_HOMOGLYPH = 4

# Signals that count as positive evidence of phishing, with their weight within
# the tool component of the risk formula. Kept here so the set of recognised
# signals is visible in one place.
SIGNAL_WEIGHTS = {
    "auth_failure": 1.00,
    "homoglyph_lookalike": 0.90,
    "display_href_mismatch": 0.85,
    "unregistered_domain": 0.70,
    "newly_registered_domain": 0.70,
    "redirects_offsite": 0.60,
    "suspicious_tld": 0.35,
}


def _dedupe_by_registrable(hosts: list[str]) -> list[str]:
    """Collapse hosts sharing a registrable domain; www.x.com and x.com are one
    WHOIS record, so checking both burns a network call for no new evidence."""
    seen: set[str] = set()
    kept: list[str] = []
    for host in hosts:
        if not host:
            continue
        key = registrable_domain(host)
        if key in seen:
            continue
        seen.add(key)
        kept.append(host)
    return kept


def _plan(parsed: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Decide the tool calls for this email."""
    calls: list[tuple[str, dict[str, Any]]] = []

    sender_domain = parsed.get("sender_domain") or ""
    reply_domain = parsed.get("reply_to_domain") or ""
    link_domains = list(parsed.get("link_domains") or [])
    urls = list(parsed.get("urls") or [])

    # Header authentication, once, always attempted so its absence is recorded.
    calls.append(("check_auth_headers", {"raw_headers": parsed.get("raw_headers", "")}))

    # Homoglyph checks keep full hosts: a brand hidden in a subdomain is exactly
    # what that tool looks for, so collapsing to the registrable domain would
    # discard the evidence.
    homoglyph_targets: list[str] = []
    for domain in [sender_domain, reply_domain, *link_domains]:
        domain = domain[4:] if domain.startswith("www.") else domain
        if domain and domain not in homoglyph_targets:
            homoglyph_targets.append(domain)
    for domain in homoglyph_targets[:MAX_HOMOGLYPH]:
        calls.append(("check_homoglyph", {"domain": domain}))

    for domain in _dedupe_by_registrable([sender_domain, *link_domains])[:MAX_WHOIS]:
        calls.append(("whois_lookup", {"domain": domain}))

    # Mismatched links first — they carry the most signal per call.
    ordered_urls = sorted(urls, key=lambda u: not u.get("display_mismatch"))
    for url in ordered_urls[:MAX_RESOLVE]:
        calls.append(
            (
                "resolve_url",
                {"url": url["href"], "display_text": url.get("display_text", "")},
            )
        )

    return calls


def verifier_node(state: PhishGuardState) -> dict:
    parsed = state.get("parsed", {})
    planned = _plan(parsed)
    raw_results = call_tools(planned)

    calls: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    available = 0

    for index, ((name, args), result) in enumerate(zip(planned, raw_results), start=1):
        ref = f"T{index}"
        status = result.get("status", "unavailable")
        if status in ("ok", "not_registered"):
            available += 1

        # Arguments are summarised; raw_headers would otherwise bloat the trace.
        arg_summary = {
            k: (f"<{len(v)} chars>" if k == "raw_headers" else v)
            for k, v in args.items()
        }

        record = {
            "ref": ref,
            "tool": name,
            "args": arg_summary,
            "status": status,
            "detail": result.get("detail", ""),
            "signal": result.get("signal"),
        }
        for key in (
            "age_days", "registrar", "newly_registered", "domain", "is_lookalike",
            "known_brand_domain", "findings", "spf", "dkim", "dmarc",
            "failed_mechanisms", "passed_mechanisms", "final_host", "hop_count",
            "display_mismatch", "redirects_offsite", "href_host", "stated_host",
        ):
            if key in result:
                record[key] = result[key]
        calls.append(record)

        if result.get("signal") in SIGNAL_WEIGHTS:
            signals.append(
                {
                    "signal": result["signal"],
                    "ref": ref,
                    "tool": name,
                    "weight": SIGNAL_WEIGHTS[result["signal"]],
                    "detail": result.get("detail", ""),
                }
            )

    attempted = len(planned)
    unavailable = [c["ref"] for c in calls if c["status"] == "unavailable"]

    reason = (
        f"{available}/{attempted} tool call(s) returned real data. "
        f"{len(signals)} positive signal(s): "
        f"{', '.join(sorted({s['signal'] for s in signals})) or 'none'}."
    )
    if unavailable:
        reason += (
            f" Unavailable: {', '.join(unavailable)} — these lower confidence and are "
            "not counted as passing."
        )

    return {
        "tool_results": {
            "calls": calls,
            "signals": signals,
            "attempted": attempted,
            "available": available,
            "unavailable_refs": unavailable,
        },
        "tools_available_count": available,
        "tools_attempted_count": attempted,
        "trace": state.get("trace", [])
        + [
            trace_entry(
                node=NODE,
                inputs_seen={
                    "sender_domain": parsed.get("sender_domain") or None,
                    "reply_to_domain": parsed.get("reply_to_domain") or None,
                    "link_domains": parsed.get("link_domains") or [],
                    "headers_present": parsed.get("has_headers", False),
                    "planned_calls": [n for n, _ in planned],
                },
                tool_calls=calls,
                output={
                    "tools_attempted": attempted,
                    "tools_returning_data": available,
                    "tools_unavailable": unavailable,
                    "positive_signals": signals,
                },
                reason=reason,
            )
        ],
    }
