import asyncio

from assayer.models import ModelResult
from assayer.providers.anthropic import AnthropicProvider
from assayer.providers.base import BaseProvider
from assayer.providers.gemini import GeminiProvider
from assayer.providers.ollama import OllamaProvider
from assayer.providers.openai import OpenAIProvider


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
    timeout: float = 30.0,
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
            timeout=timeout,
        )
    except TimeoutError:
        return ModelResult(
            model=model,
            output="",
            tokens_input=0,
            tokens_output=0,
            latency_seconds=timeout,
            cost_usd=0.0,
            error=f"Request timed out after {timeout} seconds",
        )


async def run_all(
    prompt: str,
    models: list[str],
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 30.0,
) -> list[ModelResult]:
    tasks = [
        _run_one(model, prompt, system, temperature, max_tokens, timeout)
        for model in models
    ]
    return list(await asyncio.gather(*tasks))
