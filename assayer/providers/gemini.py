import time

import litellm

from assayer.config import get_api_key
from assayer.models import ModelResult
from assayer.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    def __init__(self, model: str) -> None:
        self.model = model

    async def run(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResult:
        api_key = get_api_key("GEMINI_API_KEY")
        if not api_key:
            return ModelResult(
                model=self.model,
                output="",
                tokens_input=0,
                tokens_output=0,
                latency_seconds=0.0,
                cost_usd=0.0,
                error="GEMINI_API_KEY is not set",
            )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": f"gemini/{self.model}",
            "messages": messages,
            "api_key": api_key,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        start = time.perf_counter()
        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:  # noqa: BLE001 - any provider failure becomes a result error
            return ModelResult(
                model=self.model,
                output="",
                tokens_input=0,
                tokens_output=0,
                latency_seconds=time.perf_counter() - start,
                cost_usd=0.0,
                error=str(exc),
            )

        latency = time.perf_counter() - start
        usage = response.usage
        cost = litellm.completion_cost(completion_response=response)

        return ModelResult(
            model=self.model,
            output=response.choices[0].message.content or "",
            tokens_input=usage.prompt_tokens,
            tokens_output=usage.completion_tokens,
            latency_seconds=latency,
            cost_usd=cost,
        )
