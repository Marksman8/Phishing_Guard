"""Prompt-injection detection over untrusted email content.

Design decision: detection never sanitises. Stripping an injection attempt would
destroy the single most diagnostic signal in the message — legitimate mail does
not try to reprogram a classifier. The matched span is preserved verbatim in the
trace and fed forward as a positive phishing signal.
"""

import base64
import binascii
import re
import unicodedata
from typing import Any

# Zero-width and bidirectional-override characters used to hide text from humans
# while remaining visible to a tokenizer.
_INVISIBLE = {
    "​": "ZERO WIDTH SPACE",
    "‌": "ZERO WIDTH NON-JOINER",
    "‍": "ZERO WIDTH JOINER",
    "⁠": "WORD JOINER",
    "﻿": "ZERO WIDTH NO-BREAK SPACE",
    "­": "SOFT HYPHEN",
    "‪": "LEFT-TO-RIGHT EMBEDDING",
    "‫": "RIGHT-TO-LEFT EMBEDDING",
    "‭": "LEFT-TO-RIGHT OVERRIDE",
    "‮": "RIGHT-TO-LEFT OVERRIDE",
    "⁦": "LEFT-TO-RIGHT ISOLATE",
    "⁧": "RIGHT-TO-LEFT ISOLATE",
}

# Patterns are matched against whitespace-collapsed text (see _collapse), so a
# single literal space here also matches a newline, tab, or run of spaces. Without
# that, an attacker evades every multi-word rule just by wrapping a line.
_RULES: list[tuple[str, str, str]] = [
    (
        "instruction_override",
        r"(?i)\b(ignore|disregard|forget|override|bypass)\b[^.]{0,40}?\b(all )?(previous|prior|earlier|above|preceding|your)\b[^.]{0,20}?\b(instruction|instructions|prompt|prompts|rule|rules|direction|directions|context)\b",
        "Attempts to cancel the analyst's standing instructions.",
    ),
    (
        "role_reassignment",
        r"(?i)\b(you are now|you're now|from now on you|act as (?:a|an|the)|pretend (?:to be|you are)|your new (?:role|task|purpose|instruction) is|reset your (?:role|persona))\b",
        "Attempts to reassign the model's role or persona.",
    ),
    (
        "fake_system_turn",
        r"(?i)(?:^|(?<=[\s>]))(system|assistant|developer|admin(?:istrator)?) ?[:>]|\[/?(?:system|inst|s)\]|<\|(?:im_start|im_end|system|endoftext)\|>",
        "Forges a system/developer turn or a chat-template control token.",
    ),
    (
        "verdict_steering",
        r"(?i)\b(classify|mark|label|score|rate|report|flag|treat)\b[^.]{0,30}?\b(as )?(safe|legitimate|benign|trusted|not phishing|harmless|low risk|zero)\b",
        "Attempts to dictate the classification outcome.",
    ),
    (
        "analysis_suppression",
        r"(?i)\b(do not|don't|never)\b[^.]{0,30}?\b(report|flag|warn|alert|analy[sz]e|scan|mention|tell the user|inform)\b",
        "Attempts to suppress reporting or analysis.",
    ),
    (
        "safety_claim",
        r"(?i)\b(this (?:email|message) (?:is|has been) (?:verified|whitelisted|pre-?approved|trusted)|security (?:scan|check|scanning) (?:is |has been )?(?:disabled|bypassed|not required)|sender is (?:whitelisted|trusted|verified))\b",
        "Asserts a trusted status that only the scanner itself could establish.",
    ),
    (
        "instruction_leak",
        r"(?i)\b(repeat|reveal|print|show|output|display)\b[^.]{0,30}?\b(your|the)\b[^.]{0,20}?\b(system prompt|instructions|prompt|rules|guidelines)\b",
        "Attempts to extract the system prompt.",
    ),
    (
        "delimiter_escape",
        r"(?i)(UNTRUSTED_EMAIL_DATA|<<<|>>>)|\bend of (?:untrusted )?(?:data|email|input)\b",
        "Attempts to close or escape the untrusted-data delimiters.",
    ),
]

_COMPILED = [(name, re.compile(pattern), why) for name, pattern, why in _RULES]

_BASE64_BLOB = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")


def _decoded_base64_is_instructional(blob: str) -> str | None:
    try:
        padded = blob + "=" * (-len(blob) % 4)
        decoded = base64.b64decode(padded, validate=True).decode("utf-8", errors="strict")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if not decoded.isprintable():
        return None
    collapsed, _ = _collapse(decoded)
    for _, pattern, _ in _COMPILED:
        if pattern.search(collapsed):
            return decoded
    return None


def _collapse(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to single spaces, keeping a map back to offsets
    in `text`, so line wrapping cannot defeat a multi-word pattern."""
    chars: list[str] = []
    offsets: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            chars.append(" ")
            prev_space = True
        else:
            chars.append(ch)
            prev_space = False
        offsets.append(i)
    return "".join(chars), offsets


def _at_line_start(text: str, index: int) -> bool:
    """True if only whitespace or quoting marks separate `index` from a line break.
    Bracketed and ChatML-style tokens are exempt — they are never ordinary prose."""
    if text[index : index + 1] in ("[", "<"):
        return True
    cursor = index - 1
    while cursor >= 0 and text[cursor] in " \t>|-*":
        cursor -= 1
    return cursor < 0 or text[cursor] == "\n"


def _snippet(text: str, start: int, end: int, pad: int = 40) -> str:
    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(text) else ""
    return f"{prefix}{text[lo:hi]}{suffix}".replace("\n", " ⏎ ")


def detect_injection(text: str) -> list[dict[str, Any]]:
    """Return every injection indicator found, each with its exact matched span."""
    if not text:
        return []

    findings: list[dict[str, Any]] = []
    # Offsets below refer to this normalized form, not the raw paste.
    normalized = unicodedata.normalize("NFKC", text)
    collapsed, offsets = _collapse(normalized)

    for name, pattern, why in _COMPILED:
        for match in pattern.finditer(collapsed):
            start = offsets[match.start()] if match.start() < len(offsets) else 0
            end = offsets[match.end() - 1] + 1 if match.end() <= len(offsets) else start
            if name == "fake_system_turn" and not _at_line_start(normalized, start):
                # "the system: overview" is prose; only a turn marker beginning a
                # line is plausibly an attempt to forge a conversation turn.
                continue
            findings.append(
                {
                    "technique": name,
                    "why": why,
                    "matched_span": normalized[start:end].strip()[:200],
                    "offset": start,
                    "context": _snippet(normalized, start, end),
                    "severity": "high",
                }
            )

    for char, char_name in _INVISIBLE.items():
        count = text.count(char)
        if count:
            index = text.find(char)
            findings.append(
                {
                    "technique": "invisible_unicode",
                    "why": f"Contains {count} {char_name} character(s), used to hide text from a human reader.",
                    "matched_span": f"U+{ord(char):04X} ({char_name}) x{count}",
                    "offset": index,
                    "context": _snippet(text.replace(char, "␀"), index, index + 1),
                    "severity": "high" if char in ("‮", "‭") else "medium",
                }
            )

    for match in _BASE64_BLOB.finditer(text):
        decoded = _decoded_base64_is_instructional(match.group(0))
        if decoded:
            findings.append(
                {
                    "technique": "encoded_payload",
                    "why": "A base64 blob decodes to text containing model-directed instructions.",
                    "matched_span": match.group(0)[:80],
                    "offset": match.start(),
                    "context": f"decodes to: {decoded[:200]}",
                    "severity": "high",
                }
            )

    findings.sort(key=lambda f: f["offset"])
    return findings


def injection_summary(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "No prompt-injection indicators found."
    techniques = sorted({f["technique"] for f in findings})
    return f"{len(findings)} indicator(s) across technique(s): {', '.join(techniques)}."
