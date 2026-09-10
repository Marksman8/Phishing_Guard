from .. import config
from .base import LLMProvider
from .echo import EchoProvider

_FALLBACK_REASON: str | None = None


def get_provider(name: str | None = None) -> LLMProvider:
    """Build the configured provider, falling back to the offline heuristic.

    A fallback is always visible via `fallback_reason()` — it is reported in the
    UI and lowers confidence rather than passing silently.
    """
    global _FALLBACK_REASON
    _FALLBACK_REASON = None
    choice = (name or config.PROVIDER).strip().lower()

    try:
        if choice == "gemini":
            from .gemini import GeminiProvider

            return GeminiProvider(config.GOOGLE_API_KEY, config.GEMINI_MODEL)
        if choice == "ollama":
            from .ollama import OllamaProvider

            return OllamaProvider(config.OLLAMA_HOST, config.OLLAMA_MODEL)
        if choice == "echo":
            return EchoProvider()
        _FALLBACK_REASON = f"Unknown provider '{choice}'."
    except Exception as exc:
        _FALLBACK_REASON = f"Could not initialise '{choice}': {type(exc).__name__}: {exc}"

    return EchoProvider()


def fallback_reason() -> str | None:
    return _FALLBACK_REASON
