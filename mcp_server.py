"""PhishGuard verification tools, exposed as a real FastMCP server over stdio.

    python mcp_server.py          # run standalone (stdio)

Contract every tool obeys: return a dict with a "status" field of
  "ok"          — real data was obtained
  "unavailable" — the check could not be performed (network, timeout, no input)
and never raise, and never guess. "unavailable" must lower confidence upstream;
it is never equivalent to "passed". A tool that cannot tell the truth says so.
"""

import json
import re
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastmcp import FastMCP

mcp = FastMCP("phishguard-tools")

ROOT = Path(__file__).resolve().parent
ESCALATION_LOG = ROOT / "escalations.jsonl"

# Public WHOIS servers rate-limit aggressively, so the same domain can answer on
# one run and time out on the next. Caching real answers keeps a demo reproducible
# and cuts load on the registry. Failures are never cached — a transient timeout
# must stay retryable, and must never be replayed as though it were a result.
WHOIS_CACHE = ROOT / ".whois_cache.json"
WHOIS_CACHE_TTL_S = 7 * 24 * 3600


def _cache_read(domain: str) -> dict[str, Any] | None:
    try:
        cache = json.loads(WHOIS_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entry = cache.get(domain)
    if not entry or time.time() - entry.get("_cached_at", 0) > WHOIS_CACHE_TTL_S:
        return None
    payload = {k: v for k, v in entry.items() if k != "_cached_at"}
    payload["from_cache"] = True
    return payload


def _cache_write(domain: str, payload: dict[str, Any]) -> None:
    try:
        cache = json.loads(WHOIS_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cache = {}
    cache[domain] = {**payload, "_cached_at": time.time()}
    try:
        WHOIS_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError:
        pass

NETWORK_TIMEOUT = 6.0
NEW_DOMAIN_DAYS = 90

IMPERSONATED_BRANDS = [
    "paypal", "microsoft", "office365", "outlook", "apple", "icloud", "google",
    "gmail", "amazon", "netflix", "facebook", "instagram", "whatsapp", "linkedin",
    "dropbox", "docusign", "adobe", "coinbase", "binance", "chase", "wellsfargo",
    "bankofamerica", "hsbc", "barclays", "santander", "dhl", "fedex", "ups",
    "royalmail", "usps", "hmrc", "irs", "steam", "spotify", "zoom", "slack",
]

# Tokens attackers append to a brand to build a plausible-looking domain.
LURE_SUFFIXES = [
    "secure", "security", "login", "signin", "verify", "verification", "account",
    "accounts", "support", "helpdesk", "service", "services", "alert", "alerts",
    "update", "billing", "payment", "confirm", "recovery", "team", "center",
    "centre", "online", "portal", "auth", "id", "mail", "web", "app", "customer",
]

# Visually confusable substitutions, applied to fold a lookalike onto its target.
CONFUSABLES = {
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "b", "7": "t",
    "8": "b", "9": "g", "$": "s", "@": "a", "!": "i", "|": "l",
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ѕ": "s", "ј": "j", "ԁ": "d", "ɡ": "g", "ı": "i",
}

MULTI_CONFUSABLES = [("rn", "m"), ("vv", "w"), ("cl", "d"), ("nn", "m")]

# Registrable domains the impersonated brands genuinely own. Without this,
# microsoftonline.com reads as "brand + lure word" and is flagged as a fake.
KNOWN_BRAND_DOMAINS = {
    "paypal.com", "microsoft.com", "microsoftonline.com", "office.com",
    "office365.com", "outlook.com", "live.com", "msn.com", "sharepoint.com",
    "apple.com", "icloud.com", "google.com", "gmail.com", "googlemail.com",
    "youtube.com", "amazon.com", "amazonses.com", "aws.amazon.com", "netflix.com",
    "facebook.com", "instagram.com", "whatsapp.com", "linkedin.com", "dropbox.com",
    "docusign.com", "docusign.net", "adobe.com", "coinbase.com", "binance.com",
    "chase.com", "wellsfargo.com", "bankofamerica.com", "hsbc.com", "hsbc.co.uk",
    "barclays.co.uk", "santander.co.uk", "dhl.com", "fedex.com", "ups.com",
    "royalmail.com", "usps.com", "gov.uk", "irs.gov", "steampowered.com",
    "spotify.com", "zoom.us", "slack.com", "github.com", "salesforce.com",
}

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "click", "link", "zip", "mov",
    "country", "kim", "work", "support", "review", "loan", "date", "racing",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def _normalise_host(host: str) -> str:
    host = (host or "").strip().lower().rstrip(".")
    if host.startswith("http://") or host.startswith("https://"):
        host = urlparse(host).hostname or ""
    if "/" in host:
        host = host.split("/", 1)[0]
    if "@" in host:
        host = host.split("@", 1)[-1]
    return host


def _registrable(host: str) -> str:
    parts = [p for p in host.split(".") if p]
    if len(parts) < 2:
        return host
    # Good enough for a prototype: treat two-part ccTLD suffixes explicitly.
    two_part = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "co.nz", "co.jp", "com.br"}
    if len(parts) >= 3 and ".".join(parts[-2:]) in two_part:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _fold(text: str) -> str:
    folded = text.lower()
    for src, dst in MULTI_CONFUSABLES:
        folded = folded.replace(src, dst)
    folded = "".join(CONFUSABLES.get(ch, ch) for ch in folded)
    return re.sub(r"[^a-z0-9]", "", folded)


@mcp.tool()
def whois_lookup(domain: str) -> dict[str, Any]:
    """Look up registration age and registrar for a domain.

    Returns status "ok" with age_days and registrar, "not_registered" when the
    domain has no registration record, or "unavailable" when WHOIS could not be
    reached. Domains younger than 90 days are flagged as newly registered.
    """
    host = _normalise_host(domain)
    if not host or "." not in host:
        return {
            "tool": "whois_lookup", "status": "unavailable", "domain": host,
            "detail": "No usable domain was supplied.", "checked_at": _now(),
        }

    registrable = _registrable(host)
    started = time.time()

    cached = _cache_read(registrable)
    if cached:
        return cached

    try:
        socket.setdefaulttimeout(NETWORK_TIMEOUT)
        import whois  # python-whois

        record = whois.whois(registrable)
    except Exception as exc:
        message = str(exc).lower()
        if "no match" in message or "not found" in message or "no entries" in message:
            payload = {
                "tool": "whois_lookup", "status": "not_registered",
                "domain": registrable, "is_registered": False,
                "detail": "WHOIS returned no registration record for this domain.",
                "signal": "unregistered_domain", "checked_at": _now(),
            }
            _cache_write(registrable, payload)
            return payload
        return {
            "tool": "whois_lookup", "status": "unavailable", "domain": registrable,
            "detail": f"WHOIS lookup failed: {type(exc).__name__}.",
            "elapsed_s": round(time.time() - started, 2), "checked_at": _now(),
        }
    finally:
        socket.setdefaulttimeout(None)

    created = record.get("creation_date") if record else None
    if isinstance(created, list):
        created = next((d for d in created if d), None)

    if not record or not created:
        return {
            "tool": "whois_lookup", "status": "unavailable", "domain": registrable,
            "detail": "WHOIS responded but returned no creation date.",
            "registrar": (record.get("registrar") if record else None),
            "checked_at": _now(),
        }

    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - created).days

    payload = {
        "tool": "whois_lookup", "status": "ok", "domain": registrable,
        "is_registered": True,
        "created": created.date().isoformat(),
        "age_days": age_days,
        "registrar": record.get("registrar") or "unknown",
        "newly_registered": age_days < NEW_DOMAIN_DAYS,
        "signal": "newly_registered_domain" if age_days < NEW_DOMAIN_DAYS else None,
        "detail": (
            f"Registered {age_days} days ago — under the {NEW_DOMAIN_DAYS}-day threshold."
            if age_days < NEW_DOMAIN_DAYS
            else f"Registered {age_days} days ago, which is not unusually recent."
        ),
        "elapsed_s": round(time.time() - started, 2),
        "checked_at": _now(),
    }
    _cache_write(registrable, payload)
    return payload


@mcp.tool()
def resolve_url(url: str, display_text: str = "") -> dict[str, Any]:
    """Follow a URL's redirect chain and report the final destination host.

    Returns hop count and final host, and flags a mismatch between the link text
    shown to the reader and where the link actually leads. Status is "unavailable"
    if the host cannot be reached — the mismatch check still runs, since it needs
    no network.
    """
    import requests

    target = (url or "").strip()
    if not target:
        return {
            "tool": "resolve_url", "status": "unavailable", "url": "",
            "detail": "No URL supplied.", "checked_at": _now(),
        }
    if "://" not in target:
        target = f"http://{target}"

    stated_host = _normalise_host(display_text) if display_text else ""
    real_host = _normalise_host(urlparse(target).hostname or "")
    display_mismatch = bool(
        stated_host and _registrable(stated_host) != _registrable(real_host)
    )

    base = {
        "tool": "resolve_url", "url": target, "stated_host": stated_host or None,
        "href_host": real_host,
        "display_mismatch": display_mismatch,
        "signal": "display_href_mismatch" if display_mismatch else None,
        "checked_at": _now(),
    }

    started = time.time()
    try:
        response = requests.get(
            target, timeout=NETWORK_TIMEOUT, allow_redirects=True,
            headers={"User-Agent": "PhishGuard/1.0 (+security-analysis)"},
            stream=True,
        )
        hops = [r.headers.get("Location", "") for r in response.history]
        final_host = _normalise_host(urlparse(response.url).hostname or "")
        response.close()
        base.update(
            {
                "status": "ok",
                "final_url": response.url,
                "final_host": final_host,
                "hop_count": len(response.history),
                "http_status": response.status_code,
                "redirect_chain": hops,
                "redirects_offsite": bool(
                    final_host and _registrable(final_host) != _registrable(real_host)
                ),
                "elapsed_s": round(time.time() - started, 2),
            }
        )
        if base["redirects_offsite"]:
            base["signal"] = "redirects_offsite"
        base["detail"] = (
            f"Resolved through {len(response.history)} redirect(s) to {final_host}."
        )
        return base
    except Exception as exc:
        base.update(
            {
                "status": "unavailable",
                "detail": (
                    f"Could not reach the host ({type(exc).__name__}). The "
                    "display-vs-href comparison below needed no network and is still valid."
                ),
                "elapsed_s": round(time.time() - started, 2),
            }
        )
        return base


@mcp.tool()
def check_homoglyph(domain: str) -> dict[str, Any]:
    """Detect lookalike domains impersonating commonly-targeted brands.

    Covers punycode, visually confusable substitutions (0 for o, rn for m),
    brand-plus-lure constructions, and a brand placed in a subdomain of an
    unrelated registrable domain. Entirely local, so it is always available.
    """
    host = _normalise_host(domain)
    if not host or "." not in host:
        return {
            "tool": "check_homoglyph", "status": "unavailable", "domain": host,
            "detail": "No usable domain was supplied.", "checked_at": _now(),
        }

    registrable = _registrable(host)
    label = registrable.rsplit(".", 1)[0]
    tld = registrable.rsplit(".", 1)[-1]
    folded = _fold(label)
    plain = re.sub(r"[^a-z0-9]", "", label.lower())
    findings: list[dict[str, Any]] = []

    known_brand_domain = registrable in KNOWN_BRAND_DOMAINS
    if known_brand_domain:
        return {
            "tool": "check_homoglyph", "status": "ok", "domain": registrable,
            "full_host": host, "folded_label": folded,
            "is_lookalike": False, "known_brand_domain": True, "findings": [],
            "signal": None,
            "detail": f"'{registrable}' is a known domain genuinely operated by the brand.",
            "checked_at": _now(),
        }

    if "xn--" in host:
        findings.append(
            {
                "kind": "punycode",
                "detail": f"'{host}' uses punycode, which can render as non-Latin lookalike text.",
            }
        )

    # Folding a lookalike can make it identical to the brand (paypa1 -> paypal).
    # That collision IS the finding, so it is checked before anything else.
    if folded in IMPERSONATED_BRANDS and plain != folded:
        findings.append(
            {
                "kind": "confusable_substitution",
                "brand": folded,
                "detail": (
                    f"'{label}' is not '{folded}', but becomes identical to it once "
                    f"visually confusable characters are folded."
                ),
            }
        )

    for brand in IMPERSONATED_BRANDS:
        if folded == brand:
            continue

        if brand in folded:
            extra = folded.replace(brand, "", 1)
            lure = next((s for s in LURE_SUFFIXES if s in extra), None)
            findings.append(
                {
                    "kind": "brand_with_additions",
                    "brand": brand,
                    "detail": (
                        f"Contains the brand '{brand}' plus '{extra}'"
                        + (f", including the lure word '{lure}'" if lure else "")
                        + f" — '{registrable}' is not the brand's own domain."
                    ),
                }
            )
            continue

        distance = _levenshtein(folded, brand)
        if 0 < distance <= (1 if len(brand) <= 6 else 2):
            findings.append(
                {
                    "kind": "typosquat",
                    "brand": brand,
                    "edit_distance": distance,
                    "detail": (
                        f"'{label}' folds to '{folded}', {distance} edit(s) from '{brand}'."
                    ),
                }
            )

    # A brand in the subdomain of an unrelated registrable domain.
    subdomain = host[: -len(registrable)].rstrip(".") if host.endswith(registrable) else ""
    if subdomain:
        folded_subdomain = _fold(subdomain)
        for brand in IMPERSONATED_BRANDS:
            if brand in folded_subdomain and brand not in folded:
                findings.append(
                    {
                        "kind": "brand_in_subdomain",
                        "brand": brand,
                        "detail": (
                            f"'{brand}' appears in the subdomain '{subdomain}', but the "
                            f"registrable domain is '{registrable}', which the brand does not own."
                        ),
                    }
                )
                break

    if tld in SUSPICIOUS_TLDS:
        findings.append(
            {
                "kind": "suspicious_tld",
                "detail": f"'.{tld}' is disproportionately common in abuse reporting.",
            }
        )

    # Deduplicate by (kind, brand), keeping the first.
    unique: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for finding in findings:
        key = (finding["kind"], finding.get("brand"))
        if key not in seen:
            seen.add(key)
            unique.append(finding)

    strong = [f for f in unique if f["kind"] != "suspicious_tld"]
    # A high-abuse TLD on its own is weak evidence, but it should not disappear:
    # it is reported under its own lower-weighted signal rather than as a lookalike.
    if strong:
        signal = "homoglyph_lookalike"
    elif unique:
        signal = "suspicious_tld"
    else:
        signal = None

    return {
        "tool": "check_homoglyph", "status": "ok", "domain": registrable,
        "full_host": host, "folded_label": folded,
        "is_lookalike": bool(strong),
        "known_brand_domain": False,
        "findings": unique,
        "signal": signal,
        "detail": (
            "; ".join(f["detail"] for f in unique)
            if unique
            else f"No brand-impersonation pattern found for '{registrable}'."
        ),
        "checked_at": _now(),
    }


@mcp.tool()
def check_auth_headers(raw_headers: str) -> dict[str, Any]:
    """Parse SPF, DKIM and DMARC results from raw email headers.

    Returns "unavailable" when no headers were supplied or no authentication
    results are present. Absence of a result is never reported as a pass.
    """
    headers = raw_headers or ""
    if not headers.strip():
        return {
            "tool": "check_auth_headers", "status": "unavailable",
            "spf": None, "dkim": None, "dmarc": None,
            "detail": (
                "No headers were provided — SPF, DKIM and DMARC cannot be evaluated. "
                "This is unknown, not a pass."
            ),
            "checked_at": _now(),
        }

    flat = re.sub(r"\n[ \t]+", " ", headers)
    results: dict[str, str | None] = {"spf": None, "dkim": None, "dmarc": None}

    for mechanism in results:
        match = re.search(rf"\b{mechanism}\s*=\s*([a-z]+)", flat, re.IGNORECASE)
        if match:
            results[mechanism] = match.group(1).lower()

    if results["spf"] is None:
        received_spf = re.search(r"^Received-SPF:\s*([a-z]+)", flat, re.IGNORECASE | re.MULTILINE)
        if received_spf:
            results["spf"] = received_spf.group(1).lower()

    if results["dkim"] is None and re.search(r"^DKIM-Signature:", flat, re.IGNORECASE | re.MULTILINE):
        results["dkim"] = "present_unverified"

    if all(value is None for value in results.values()):
        return {
            "tool": "check_auth_headers", "status": "unavailable", **results,
            "detail": (
                "Headers were supplied but contain no Authentication-Results, "
                "Received-SPF or DKIM-Signature field. Treated as unknown, not a pass."
            ),
            "checked_at": _now(),
        }

    failed = [
        name for name, value in results.items()
        if value in ("fail", "softfail", "permerror", "temperror", "none", "neutral")
    ]
    passed = [name for name, value in results.items() if value == "pass"]

    return {
        "tool": "check_auth_headers", "status": "ok", **results,
        "failed_mechanisms": failed,
        "passed_mechanisms": passed,
        "any_failure": bool(failed),
        "all_passed": bool(passed) and not failed and len(passed) >= 2,
        "signal": "auth_failure" if failed else None,
        "detail": (
            f"Failed or absent: {', '.join(failed)}."
            if failed
            else f"Passed: {', '.join(passed)}."
        ),
        "checked_at": _now(),
    }


@mcp.tool()
def escalate_to_human(case_summary: str, trace: str = "") -> dict[str, Any]:
    """Open a human-review ticket and append the full case to escalations.jsonl.

    `trace` should be a JSON string of the audit trail. Returns the ticket id.
    Content is expected to be pre-redacted by the caller.
    """
    ticket_id = f"PG-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    try:
        parsed_trace = json.loads(trace) if trace else []
    except json.JSONDecodeError:
        parsed_trace = [{"note": "trace was not valid JSON", "raw_length": len(trace)}]

    record = {
        "ticket_id": ticket_id,
        "opened_utc": _now(),
        "status": "awaiting_human_review",
        "case_summary": case_summary,
        "trace": parsed_trace,
    }
    try:
        with ESCALATION_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        written = True
        detail = f"Ticket {ticket_id} written to {ESCALATION_LOG.name}."
    except Exception as exc:
        written = False
        detail = f"Ticket {ticket_id} created but could not be persisted: {type(exc).__name__}."

    return {
        "tool": "escalate_to_human", "status": "ok", "ticket_id": ticket_id,
        "persisted": written, "queue": "human_review",
        "detail": detail, "checked_at": _now(),
    }


if __name__ == "__main__":
    # Banner off: stdio transport shares the stream with the protocol.
    mcp.run(show_banner=False)
