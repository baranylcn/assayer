import asyncio

import pytest

from assayer.models import ModelResult
from assayer.providers.anthropic import AnthropicProvider
from assayer.providers.gemini import GeminiProvider
from assayer.providers.ollama import OllamaProvider
from assayer.providers.openai import OpenAIProvider
from assayer.runner import _make_provider, _run_one, run_all


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-4o", OpenAIProvider),
        ("gpt-4o-mini", OpenAIProvider),
        ("claude-sonnet-4-5", AnthropicProvider),
        ("claude-haiku-4-5-20251001", AnthropicProvider),
        ("gemini-1.5-pro", GeminiProvider),
        ("ollama/llama3", OllamaProvider),
    ],
)
def test_make_provider_routing(model, expected):
    assert isinstance(_make_provider(model), expected)


async def test_run_all_returns_one_result_per_model(monkeypatch):
    async def _fake_run(self, prompt, **kwargs):
        return ModelResult(
            model=self.model,
            output="hello",
            tokens_input=5,
            tokens_output=3,
            latency_seconds=0.1,
            cost_usd=0.0,
        )

    monkeypatch.setattr(OpenAIProvider, "run", _fake_run)
    monkeypatch.setattr(AnthropicProvider, "run", _fake_run)

    results = await run_all("test prompt", ["gpt-4o-mini", "claude-haiku-4-5-20251001"])

    assert len(results) == 2
    assert all(r.output == "hello" for r in results)
    assert all(r.error is None for r in results)


async def test_run_all_forwards_kwargs(monkeypatch):
    received: dict = {}

    async def _capture_run(self, prompt, **kwargs):
        received.update(kwargs)
        return ModelResult(
            model=self.model,
            output="ok",
            tokens_input=0,
            tokens_output=0,
            latency_seconds=0.0,
            cost_usd=0.0,
        )

    monkeypatch.setattr(OpenAIProvider, "run", _capture_run)

    await run_all("prompt", ["gpt-4o-mini"], system="sys", temperature=0.5, max_tokens=100)

    assert received["system"] == "sys"
    assert received["temperature"] == 0.5
    assert received["max_tokens"] == 100


async def test_run_all_includes_partial_failures(monkeypatch):
    call_count = 0

    async def _sometimes_fail(self, prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResult(
                model=self.model,
                output="ok",
                tokens_input=5,
                tokens_output=3,
                latency_seconds=0.1,
                cost_usd=0.0,
            )
        return ModelResult(
            model=self.model,
            output="",
            tokens_input=0,
            tokens_output=0,
            latency_seconds=0.0,
            cost_usd=0.0,
            error="API error",
        )

    monkeypatch.setattr(OpenAIProvider, "run", _sometimes_fail)

    results = await run_all("test", ["gpt-4o-mini", "gpt-4o"])

    assert len(results) == 2
    assert sum(1 for r in results if r.error is None) == 1
    assert sum(1 for r in results if r.error is not None) == 1


async def test_run_one_records_timeout(monkeypatch):
    async def _slow_run(self, prompt, **kwargs):
        await asyncio.sleep(60)
        return ModelResult(
            model=self.model,
            output="",
            tokens_input=0,
            tokens_output=0,
            latency_seconds=60.0,
            cost_usd=0.0,
        )

    monkeypatch.setattr(OpenAIProvider, "run", _slow_run)

    result = await _run_one("gpt-4o-mini", "test", None, None, None, timeout=0.05)

    assert result.error is not None
    assert "timed out" in result.error
