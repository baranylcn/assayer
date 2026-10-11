import time

import httpx

from assayer.models import ModelResult
from assayer.providers.base import BaseProvider

_BASE_URL = "http://localhost:11434"


async def is_running() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(_BASE_URL)
            return response.status_code == 200
    except httpx.ConnectError:
        return False


class OllamaProvider(BaseProvider):
    def __init__(self, model: str) -> None:
        self.model = model.removeprefix("ollama/")

    async def run(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict = {"model": self.model, "messages": messages, "stream": False}
        if temperature is not None:
            body.setdefault("options", {})["temperature"] = temperature
        if max_tokens is not None:
            body.setdefault("options", {})["num_predict"] = max_tokens

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{_BASE_URL}/api/chat", json=body)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError:
            return ModelResult(
                model=f"ollama/{self.model}",
                output="",
                tokens_input=0,
                tokens_output=0,
                latency_seconds=time.perf_counter() - start,
                cost_usd=0.0,
                error="Ollama is not running at localhost:11434",
            )
        except Exception as exc:  # noqa: BLE001 - any provider failure becomes a result error
            return ModelResult(
                model=f"ollama/{self.model}",
                output="",
                tokens_input=0,
                tokens_output=0,
                latency_seconds=time.perf_counter() - start,
                cost_usd=0.0,
                error=str(exc),
            )

        return ModelResult(
            model=f"ollama/{self.model}",
            output=data["message"]["content"],
            tokens_input=data.get("prompt_eval_count", 0),
            tokens_output=data.get("eval_count", 0),
            latency_seconds=time.perf_counter() - start,
            cost_usd=0.0,
        )
