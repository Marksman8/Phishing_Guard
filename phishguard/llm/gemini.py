from .. import config
from .base import LLMProvider, LLMResult, build_prompt


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is empty — set it in .env")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = model
        self._genai = genai

    def complete(self, system: str, data_fields: dict[str, str] | None = None) -> LLMResult:
        system_prompt, user_prompt = build_prompt(system, data_fields)
        try:
            model = self._genai.GenerativeModel(
                model_name=self.model, system_instruction=system_prompt
            )
            response = model.generate_content(
                user_prompt,
                generation_config={"temperature": 0.1},
                request_options={"timeout": config.LLM_TIMEOUT},
            )
            return LLMResult(
                text=(response.text or "").strip(), provider=self.name, model=self.model
            )
        except Exception as exc:
            return LLMResult(
                text="", ok=False, error=f"{type(exc).__name__}: {exc}",
                provider=self.name, model=self.model,
            )
