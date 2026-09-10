"""Redaction applied to anything leaving memory for a log or trace file.

Domains survive redaction on purpose: they are the analytic payload (homoglyph
checks, WHOIS age). The local-part of an address is what gets masked.
"""

import re
from typing import Any

_EMAIL = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")

_CREDENTIAL_PATTERNS = [
    (re.compile(r"(?i)\b(password|passwd|pwd|passcode)\b\s*[:=]\s*\S+"), "password: [REDACTED]"),
    (re.compile(r"(?i)\b(api[_-]?key|secret|token|bearer)\b\s*[:=]\s*\S+"), "credential: [REDACTED]"),
    (re.compile(r"(?i)\b(otp|pin|verification code|2fa)\b\s*[:=]?\s*\b\d{4,8}\b"), "otp: [REDACTED]"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[REDACTED_CARD_NUMBER]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED_GOOGLE_KEY]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "[REDACTED_API_KEY]"),
]


def _mask_email(match: re.Match) -> str:
    local, domain = match.group(1), match.group(2)
    head = local[0] if local else "?"
    return f"{head}{'*' * max(len(local) - 1, 3)}@{domain}"


def redact_text(text: str) -> str:
    if not text:
        return text
    out = _EMAIL.sub(_mask_email, text)
    for pattern, replacement in _CREDENTIAL_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def redact(obj: Any) -> Any:
    """Recursively redact strings inside dicts/lists so whole trace entries are safe."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    return obj
