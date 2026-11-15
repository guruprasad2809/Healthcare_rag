"""Gemini client (google-genai SDK)."""
from __future__ import annotations

import time


class LLMError(RuntimeError):
    pass


class GeminiLLM:
    def __init__(self, api_key: str, model: str, temperature: float = 0.1, max_output_tokens: int = 2048, max_retries: int = 3):
        self.api_key, self.model = api_key, model
        self.temperature, self.max_output_tokens, self.max_retries = temperature, max_output_tokens, max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, system: str = "") -> str:
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system or None, temperature=self.temperature,
                                             max_output_tokens=self.max_output_tokens)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
                return (response.text or "").strip()
            except LLMError:
                raise
            except Exception as exc:  # rate limits / transient network errors: back off and retry
                last_error = exc
                time.sleep(2 ** attempt)
        raise LLMError(f"Gemini request failed after {self.max_retries} attempts: {last_error}")
