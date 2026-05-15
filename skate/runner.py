import asyncio

from skate.models import ModelResult
from skate.providers.anthropic import AnthropicProvider
from skate.providers.base import BaseProvider
from skate.providers.gemini import GeminiProvider
from skate.providers.ollama import OllamaProvider
from skate.providers.openai import OpenAIProvider


_TIMEOUT = 30.0


def _make_provider(model: str) -> BaseProvider:
    if model.startswith("ollama/"):
        return OllamaProvider(model)
    if model.startswith("claude-"):
        return AnthropicProvider(model)
    if model.startswith("gemini-"):
        return GeminiProvider(model)
    return OpenAIProvider(model)


async def _run_one(
    model: str,
    prompt: str,
    system: str | None,
    temperature: float | None,
    max_tokens: int | None,
) -> ModelResult:
    provider = _make_provider(model)
    try:
        return await asyncio.wait_for(
            provider.run(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            timeout=_TIMEOUT,
        )
    except TimeoutError:
        return ModelResult(
            model=model,
            output="",
            tokens_input=0,
            tokens_output=0,
            latency_seconds=30.0,
            cost_usd=0.0,
            error="Request timed out after 30 seconds",
        )


async def run_all(
    prompt: str,
    models: list[str],
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> list[ModelResult]:
    tasks = [
        _run_one(model, prompt, system, temperature, max_tokens)
        for model in models
    ]
    return list(await asyncio.gather(*tasks))
