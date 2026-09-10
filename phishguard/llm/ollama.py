import requests

from .. import config
from .base import LLMProvider, LLMResult, build_prompt


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def complete(self, system: str, data_fields: dict[str, str] | None = None) -> LLMResult:
        system_prompt, user_prompt = build_prompt(system, data_fields)
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "options": {"temperature": 0.1},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=config.LLM_TIMEOUT,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            return LLMResult(text=content.strip(), provider=self.name, model=self.model)
        except Exception as exc:
            return LLMResult(
                text="", ok=False, error=f"{type(exc).__name__}: {exc}",
                provider=self.name, model=self.model,
            )
