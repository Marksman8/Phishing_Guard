"""Offline deterministic provider.

Not a mock of an LLM — it is a transparent keyword heuristic that satisfies the
same contract, so the graph still completes end-to-end when no model is reachable.
The UI always shows which provider produced a result, so this never masquerades
as model output.
"""

import json
import re

from .base import LLMProvider, LLMResult, build_prompt

_SIGNAL_RULES = {
    "urgency_pressure": [
        r"\b(urgent|immediately|within \d+ hours?|act now|expires? (today|soon)|final (notice|warning)|last chance|as soon as possible)\b",
    ],
    "credential_request": [
        r"\b(verify your (account|identity|password)|confirm your (password|credentials|login)|sign in to (confirm|verify)|update your payment|re-?enter your password|login details)\b",
    ],
    "sender_content_mismatch": [
        r"\b(dear (customer|user|sir/madam)|valued customer)\b",
    ],
    "threat_framing": [
        r"\b(suspend|suspended|terminate|closed permanently|legal action|unauthorized (access|login)|will be (deleted|locked|disabled)|penalt(y|ies))\b",
    ],
}


class EchoProvider(LLMProvider):
    name = "echo"
    model = "offline-heuristic"

    def complete(self, system: str, data_fields: dict[str, str] | None = None) -> LLMResult:
        _, user_prompt = build_prompt(system, data_fields)
        corpus = user_prompt.lower()

        if "content_signals" in system:
            signals = {}
            for signal, patterns in _SIGNAL_RULES.items():
                hit = next(
                    (m.group(0) for p in patterns for m in [re.search(p, corpus)] if m), None
                )
                signals[signal] = {
                    "flag": 1 if hit else 0,
                    "justification": (
                        f"Offline heuristic matched the phrase '{hit}'."
                        if hit
                        else "No matching phrasing found by the offline heuristic."
                    ),
                }
            return LLMResult(
                text=json.dumps({"content_signals": signals}),
                provider=self.name,
                model=self.model,
            )

        return LLMResult(
            text=(
                "Offline mode is active, so no language model drafted this summary. "
                "The verdict, risk score and evidence citations shown above are produced "
                "by the deterministic scoring formula and the verification tools, and are "
                "unaffected by the absence of a model."
            ),
            provider=self.name,
            model=self.model,
        )
