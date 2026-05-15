import json
from unittest.mock import patch

import pytest

from skate.judge import JudgeResult, _build_prompt, _parse_response, run_judge
from skate.models import ModelResult


def _result(model: str, output: str, error: str | None = None) -> ModelResult:
    return ModelResult(
        model=model,
        output=output,
        tokens_input=10,
        tokens_output=20,
        latency_seconds=0.5,
        cost_usd=0.0,
        error=error,
    )


def test_parse_response_plain_json():
    payload = json.dumps({
        "winner": "gpt-4o",
        "reasoning": "Clearer answer.",
        "scores": {"gpt-4o": {"clarity": 5}, "claude": {"clarity": 4}},
    })
    result = _parse_response(payload)
    assert result.winner == "gpt-4o"
    assert result.reasoning == "Clearer answer."
    assert result.scores["gpt-4o"]["clarity"] == 5


def test_parse_response_strips_code_fences():
    payload = "```json\n" + '{"winner": "claude", "reasoning": "Better.", "scores": {}}\n' + "```"
    result = _parse_response(payload)
    assert result.winner == "claude"


def test_parse_response_strips_plain_fences():
    payload = "```\n" + '{"winner": "gpt-4o", "reasoning": "Good.", "scores": {}}\n' + "```"
    result = _parse_response(payload)
    assert result.winner == "gpt-4o"


def test_build_prompt_contains_prompt_and_outputs():
    results = [_result("gpt-4o", "Four"), _result("claude", "4")]
    prompt = _build_prompt("What is 2+2?", results, criteria=None)

    assert "What is 2+2?" in prompt
    assert "gpt-4o" in prompt
    assert "Four" in prompt
    assert "claude" in prompt
    assert "4" in prompt


def test_build_prompt_without_criteria_omits_criteria_section():
    results = [_result("gpt-4o", "Four"), _result("claude", "4")]
    prompt = _build_prompt("q", results, criteria=None)
    assert "Evaluate on these criteria" not in prompt


def test_build_prompt_with_criteria_includes_criteria():
    results = [_result("gpt-4o", "Four"), _result("claude", "4")]
    prompt = _build_prompt("q", results, criteria=["clarity", "brevity"])
    assert "clarity" in prompt
    assert "brevity" in prompt
    assert "Evaluate on these criteria" in prompt


def test_build_prompt_skips_errored_results():
    results = [_result("gpt-4o", "Four"), _result("claude", "", error="API error")]
    prompt = _build_prompt("q", results, criteria=None)
    assert "gpt-4o" in prompt
    assert "claude" not in prompt


async def test_run_judge_success():
    mock_output = json.dumps({
        "winner": "gpt-4o",
        "reasoning": "More concise.",
        "scores": {},
    })

    async def _fake_run(self, prompt, **kwargs):
        return ModelResult(
            model="gpt-4o",
            output=mock_output,
            tokens_input=100,
            tokens_output=50,
            latency_seconds=1.0,
            cost_usd=0.001,
        )

    with patch("skate.providers.openai.OpenAIProvider.run", _fake_run):
        judge_result = await run_judge(
            "What is 2+2?",
            [_result("gpt-4o", "4"), _result("claude", "Four")],
            "gpt-4o",
        )

    assert isinstance(judge_result, JudgeResult)
    assert judge_result.winner == "gpt-4o"


async def test_run_judge_with_criteria():
    mock_output = json.dumps({
        "winner": "gpt-4o",
        "reasoning": "More concise.",
        "scores": {"gpt-4o": {"clarity": 5, "brevity": 4}},
    })

    async def _fake_run(self, prompt, **kwargs):
        return ModelResult(
            model="gpt-4o",
            output=mock_output,
            tokens_input=100,
            tokens_output=50,
            latency_seconds=1.0,
            cost_usd=0.001,
        )

    with patch("skate.providers.openai.OpenAIProvider.run", _fake_run):
        judge_result = await run_judge(
            "What is 2+2?",
            [_result("gpt-4o", "4"), _result("claude", "Four")],
            "gpt-4o",
            criteria=["clarity", "brevity"],
        )

    assert isinstance(judge_result, JudgeResult)
    assert judge_result.scores["gpt-4o"]["clarity"] == 5
    assert judge_result.scores["gpt-4o"]["brevity"] == 4


async def test_run_judge_skips_on_fewer_than_two_valid(capsys):
    results = [_result("gpt-4o", "hello"), _result("claude", "", error="API error")]
    judge_result = await run_judge("prompt", results, "gpt-4o")
    assert judge_result is None
    assert "skipped" in capsys.readouterr().err


async def test_run_judge_handles_provider_error(capsys):
    async def _fail(self, prompt, **kwargs):
        return ModelResult(
            model="gpt-4o",
            output="",
            tokens_input=0,
            tokens_output=0,
            latency_seconds=0.0,
            cost_usd=0.0,
            error="Rate limit",
        )

    with patch("skate.providers.openai.OpenAIProvider.run", _fail):
        judge_result = await run_judge(
            "prompt", [_result("gpt-4o", "a"), _result("claude", "b")], "gpt-4o"
        )

    assert judge_result is None
    assert "failed" in capsys.readouterr().err


async def test_run_judge_handles_bad_json(capsys):
    async def _bad_json(self, prompt, **kwargs):
        return ModelResult(
            model="gpt-4o",
            output="not json",
            tokens_input=0,
            tokens_output=0,
            latency_seconds=0.0,
            cost_usd=0.0,
        )

    with patch("skate.providers.openai.OpenAIProvider.run", _bad_json):
        judge_result = await run_judge(
            "prompt", [_result("gpt-4o", "a"), _result("claude", "b")], "gpt-4o"
        )

    assert judge_result is None
    assert "parsed" in capsys.readouterr().err
