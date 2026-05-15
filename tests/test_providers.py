import os
from pathlib import Path

import httpx
import pytest

from skate.providers.anthropic import AnthropicProvider
from skate.providers.gemini import GeminiProvider
from skate.providers.ollama import OllamaProvider, is_running
from skate.providers.openai import OpenAIProvider

_NONEXISTENT_CONFIG = Path("/nonexistent")


@pytest.mark.parametrize(
    "provider_cls, env_var, model",
    [
        (OpenAIProvider, "OPENAI_API_KEY", "gpt-4o-mini"),
        (AnthropicProvider, "ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001"),
        (GeminiProvider, "GEMINI_API_KEY", "gemini-1.5-flash"),
    ],
)
async def test_missing_api_key(provider_cls, env_var, model, monkeypatch):
    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr("skate.config._CONFIG_PATH", _NONEXISTENT_CONFIG)

    result = await provider_cls(model).run("hello")

    assert result.error == f"{env_var} is not set"
    assert result.output == ""


async def test_ollama_not_running(monkeypatch):
    class _FailClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("refused")

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr("skate.providers.ollama.httpx.AsyncClient", _FailClient)

    result = await OllamaProvider("ollama/llama3").run("hello")

    assert result.error is not None
    assert "11434" in result.error or "Ollama" in result.error
    assert result.cost_usd == 0.0


@pytest.mark.integration
async def test_openai_integration():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    result = await OpenAIProvider("gpt-4o-mini").run("Say hello in one word.")

    assert result.error is None
    assert len(result.output) > 0
    assert result.tokens_output > 0


@pytest.mark.integration
async def test_anthropic_integration():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    result = await AnthropicProvider("claude-haiku-4-5-20251001").run("Say hello in one word.")

    assert result.error is None
    assert len(result.output) > 0
    assert result.tokens_output > 0


@pytest.mark.integration
async def test_gemini_integration():
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    result = await GeminiProvider("gemini-1.5-flash").run("Say hello in one word.")

    assert result.error is None
    assert len(result.output) > 0
    assert result.tokens_output > 0


@pytest.mark.integration
async def test_ollama_integration():
    if not await is_running():
        pytest.skip("Ollama is not running")

    result = await OllamaProvider("ollama/llama3").run("Say hello in one word.")

    assert result.error is None
    assert len(result.output) > 0
    assert result.cost_usd == 0.0
