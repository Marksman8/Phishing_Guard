"""Raw-paste email parsing.

Accepts a full RFC-822 paste (headers + blank line + body) or a bare body. The
distinction matters downstream: absent headers make SPF/DKIM/DMARC unverifiable,
which must lower confidence rather than be treated as a pass.
"""

import re
from email import message_from_string
from email.utils import parseaddr
from typing import Any
from urllib.parse import urlparse

_HEADER_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9\-]{1,40}:\s", re.MULTILINE)
_ANCHOR = re.compile(
    r"<a\b[^>]*?href\s*=\s*[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL
)
_MARKDOWN_LINK = re.compile(r"\[([^\]]{1,200})\]\((https?://[^\s)]+)\)")
_BARE_URL = re.compile(r"(?<![\"'(=])\b((?:https?://|www\.)[^\s<>\"'\]),]+)", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_ATTACHMENT = re.compile(
    r"\b([\w\-.]{1,60}\.(?:zip|rar|7z|exe|scr|js|vbs|jar|iso|img|docm|xlsm|pptm|doc|docx|xls|xlsx|pdf|html|htm|lnk))\b",
    re.IGNORECASE,
)
_RISKY_EXT = {
    "exe", "scr", "js", "vbs", "jar", "iso", "img", "lnk",
    "docm", "xlsm", "pptm", "zip", "rar", "7z", "html", "htm",
}


def looks_like_headers(text: str) -> bool:
    head = text.lstrip().split("\n\n", 1)[0]
    return len(_HEADER_LINE.findall(head)) >= 2


def _host(url: str) -> str:
    candidate = url if "://" in url else f"http://{url}"
    try:
        return (urlparse(candidate).hostname or "").lower()
    except ValueError:
        return ""


def registrable_domain(host: str) -> str:
    """Best-effort registrable domain. Handles the common two-part ccTLD suffixes;
    a prototype stand-in for a full public-suffix list."""
    parts = [p for p in (host or "").lower().split(".") if p]
    if len(parts) < 2:
        return host or ""
    two_part = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "co.nz", "co.jp", "com.br"}
    if len(parts) >= 3 and ".".join(parts[-2:]) in two_part:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


_registrable = registrable_domain


def extract_urls(body: str) -> list[dict[str, Any]]:
    """Return each URL with its displayed text and real href, flagging mismatches."""
    urls: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(href: str, display: str) -> None:
        href = href.strip()
        display = _TAG.sub("", display or "").strip()
        if not href:
            return
        key = (href, display)
        if key in seen:
            return
        seen.add(key)
        href_host = _host(href)
        display_host = _host(display) if re.search(r"[a-z0-9\-]+\.[a-z]{2,}", display, re.I) else ""
        mismatch = bool(
            display_host and _registrable(display_host) != _registrable(href_host)
        )
        urls.append(
            {
                "href": href,
                "display_text": display or href,
                "href_host": href_host,
                "display_host": display_host,
                "display_mismatch": mismatch,
            }
        )

    for match in _ANCHOR.finditer(body):
        add(match.group(1), match.group(2))
    for match in _MARKDOWN_LINK.finditer(body):
        add(match.group(2), match.group(1))

    stripped = _ANCHOR.sub(" ", body)
    stripped = _MARKDOWN_LINK.sub(" ", stripped)
    for match in _BARE_URL.finditer(_TAG.sub(" ", stripped)):
        add(match.group(1).rstrip(".,);:"), "")

    return urls


def extract_attachments(text: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _ATTACHMENT.finditer(text):
        name = match.group(1).strip()
        # A preceding path/host separator means this is part of a URL, not a filename.
        preceding = text[match.start() - 1] if match.start() else " "
        if name.lower() in seen or preceding in "/@.":
            continue
        seen.add(name.lower())
        ext = name.rsplit(".", 1)[-1].lower()
        found.append({"filename": name, "extension": ext, "risky_type": ext in _RISKY_EXT})
    return found


def parse_email(raw: str) -> dict[str, Any]:
    raw = (raw or "").replace("\r\n", "\n").strip()
    has_headers = looks_like_headers(raw)

    if has_headers:
        message = message_from_string(raw)
        header_block = raw.split("\n\n", 1)[0]
        if message.is_multipart():
            parts = [
                p.get_payload(decode=False)
                for p in message.walk()
                if p.get_content_type() in ("text/plain", "text/html")
            ]
            body = "\n".join(str(p) for p in parts if p)
        else:
            body = str(message.get_payload() or "")
        from_header = message.get("From", "")
        display_name, sender_address = parseaddr(from_header)
        reply_to = parseaddr(message.get("Reply-To", ""))[1]
        subject = message.get("Subject", "")
        return_path = parseaddr(message.get("Return-Path", ""))[1]
    else:
        message = None
        header_block = ""
        body = raw
        display_name = sender_address = reply_to = subject = return_path = ""

    sender_domain = sender_address.split("@")[-1].lower() if "@" in sender_address else ""
    reply_to_domain = reply_to.split("@")[-1].lower() if "@" in reply_to else ""

    urls = extract_urls(body)
    link_domains = sorted({u["href_host"] for u in urls if u["href_host"]})

    return {
        "has_headers": has_headers,
        "raw_headers": header_block,
        "sender_display_name": display_name,
        "sender_address": sender_address,
        "sender_domain": sender_domain,
        "reply_to": reply_to,
        "reply_to_domain": reply_to_domain,
        "reply_to_mismatch": bool(
            reply_to_domain and sender_domain and reply_to_domain != sender_domain
        ),
        "return_path": return_path,
        "subject": subject,
        "body": body.strip(),
        "body_preview": body.strip()[:400],
        "urls": urls,
        "link_domains": link_domains,
        "attachments": extract_attachments(raw),
    }
