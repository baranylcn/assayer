from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from assayer.cli.main import cli
from assayer.judge import JudgeResult
from assayer.models import ModelResult


def _result(model: str = "gpt-4o", error: str | None = None) -> ModelResult:
    return ModelResult(
        model=model,
        output="test output" if not error else "",
        tokens_input=10,
        tokens_output=20,
        latency_seconds=0.5,
        cost_usd=0.001,
        error=error,
    )


# ---------------------------------------------------------------------------
# run — input validation
# ---------------------------------------------------------------------------


def test_run_no_prompt_exits_with_error():
    result = CliRunner().invoke(cli, ["run", "--models", "gpt-4o"])
    assert result.exit_code != 0
    assert "prompt" in result.output.lower()


def test_run_var_bad_format_exits_with_error():
    result = CliRunner().invoke(
        cli, ["run", "hello", "--models", "gpt-4o", "--var", "BADFORMAT"]
    )
    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output


def test_run_var_missing_template_key_exits_with_error():
    result = CliRunner().invoke(
        cli,
        ["run", "hello {missing}", "--models", "gpt-4o", "--var", "other=value"],
    )
    assert result.exit_code != 0
    assert "missing" in result.output


def test_run_warns_when_prompt_and_prompt_file_are_both_supplied(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt from file")

    result = CliRunner().invoke(
        cli,
        [
            "run",
            "inline prompt",
            "--prompt-file",
            str(prompt_file),
            "--models",
            "gpt-4o",
            "--var",
            "BADFORMAT",
        ],
    )

    assert (
        "Warning: --prompt-file takes precedence; the inline prompt is ignored."
        in result.output
    )


# ---------------------------------------------------------------------------
# run — successful flow
# ---------------------------------------------------------------------------


def test_run_basic_calls_run_all():
    with (
        patch("assayer.runner.run_all", new_callable=AsyncMock) as mock_run_all,
        patch("assayer.renderer.render_run"),
    ):
        mock_run_all.return_value = [_result("gpt-4o"), _result("claude-sonnet-4-5")]
        result = CliRunner().invoke(
            cli, ["run", "hello", "--models", "gpt-4o,claude-sonnet-4-5"]
        )

    assert result.exit_code == 0
    mock_run_all.assert_called_once()
    args, kwargs = mock_run_all.call_args
    assert args[0] == "hello"
    assert args[1] == ["gpt-4o", "claude-sonnet-4-5"]


def test_run_prompt_file_reads_file(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt from file")

    with (
        patch("assayer.runner.run_all", new_callable=AsyncMock) as mock_run_all,
        patch("assayer.renderer.render_run"),
    ):
        mock_run_all.return_value = [_result()]
        result = CliRunner().invoke(
            cli,
            ["run", "--prompt-file", str(prompt_file), "--models", "gpt-4o"],
        )

    assert result.exit_code == 0
    args, _ = mock_run_all.call_args
    assert args[0] == "prompt from file"


def test_run_var_substitutes_into_prompt():
    with (
        patch("assayer.runner.run_all", new_callable=AsyncMock) as mock_run_all,
        patch("assayer.renderer.render_run"),
    ):
        mock_run_all.return_value = [_result()]
        CliRunner().invoke(
            cli,
            ["run", "translate {text}", "--models", "gpt-4o", "--var", "text=hello"],
        )

    args, _ = mock_run_all.call_args
    assert args[0] == "translate hello"


def test_run_passes_system_temperature_max_tokens():
    with (
        patch("assayer.runner.run_all", new_callable=AsyncMock) as mock_run_all,
        patch("assayer.renderer.render_run"),
    ):
        mock_run_all.return_value = [_result()]
        CliRunner().invoke(
            cli,
            [
                "run", "hello", "--models", "gpt-4o",
                "--system", "You are a poet.",
                "--temperature", "0.7",
                "--max-tokens", "256",
            ],
        )

    _, kwargs = mock_run_all.call_args
    assert kwargs["system"] == "You are a poet."
    assert kwargs["temperature"] == pytest.approx(0.7)
    assert kwargs["max_tokens"] == 256


def test_run_passes_timeout_to_run_all():
    with (
        patch("assayer.runner.run_all", new_callable=AsyncMock) as mock_run_all,
        patch("assayer.renderer.render_run"),
    ):
        mock_run_all.return_value = [_result()]
        CliRunner().invoke(
            cli, ["run", "hello", "--models", "gpt-4o", "--timeout", "60"]
        )

    _, kwargs = mock_run_all.call_args
    assert kwargs["timeout"] == pytest.approx(60.0)


def test_run_score_calls_compute_similarity():
    with (
        patch("assayer.runner.run_all", new_callable=AsyncMock) as mock_run_all,
        patch("assayer.scorer.compute_similarity", return_value={}) as mock_score,
        patch("assayer.renderer.render_run"),
    ):
        mock_run_all.return_value = [_result("gpt-4o"), _result("claude-sonnet-4-5")]
        result = CliRunner().invoke(
            cli,
            ["run", "hello", "--models", "gpt-4o,claude-sonnet-4-5", "--score"],
        )

    assert result.exit_code == 0
    mock_score.assert_called_once()


def test_run_without_score_skips_compute_similarity():
    with (
        patch("assayer.runner.run_all", new_callable=AsyncMock) as mock_run_all,
        patch("assayer.scorer.compute_similarity") as mock_score,
        patch("assayer.renderer.render_run"),
    ):
        mock_run_all.return_value = [_result()]
        CliRunner().invoke(cli, ["run", "hello", "--models", "gpt-4o"])

    mock_score.assert_not_called()


def test_run_judge_calls_run_judge():
    judge_result = JudgeResult(winner="gpt-4o", reasoning="Better.", scores={})

    with (
        patch("assayer.runner.run_all", new_callable=AsyncMock) as mock_run_all,
        patch("assayer.judge.run_judge", new_callable=AsyncMock) as mock_judge,
        patch("assayer.renderer.render_run"),
    ):
        mock_run_all.return_value = [_result("gpt-4o"), _result("claude-sonnet-4-5")]
        mock_judge.return_value = judge_result
        result = CliRunner().invoke(
            cli,
            ["run", "hello", "--models", "gpt-4o,claude-sonnet-4-5", "--judge", "gpt-4o"],
        )

    assert result.exit_code == 0
    mock_judge.assert_called_once()


def test_run_without_judge_skips_run_judge():
    with (
        patch("assayer.runner.run_all", new_callable=AsyncMock) as mock_run_all,
        patch("assayer.judge.run_judge", new_callable=AsyncMock) as mock_judge,
        patch("assayer.renderer.render_run"),
    ):
        mock_run_all.return_value = [_result()]
        CliRunner().invoke(cli, ["run", "hello", "--models", "gpt-4o"])

    mock_judge.assert_not_called()


def test_run_output_calls_export(tmp_path):
    output_path = str(tmp_path / "results.json")

    with (
        patch("assayer.runner.run_all", new_callable=AsyncMock) as mock_run_all,
        patch("assayer.renderer.render_run"),
        patch("assayer.exporter.export") as mock_export,
    ):
        mock_run_all.return_value = [_result()]
        result = CliRunner().invoke(
            cli,
            ["run", "hello", "--models", "gpt-4o", "--output", output_path],
        )

    assert result.exit_code == 0
    mock_export.assert_called_once_with([_result()], output_path)
    assert "Results saved" in result.output


def test_run_without_output_skips_export():
    with (
        patch("assayer.runner.run_all", new_callable=AsyncMock) as mock_run_all,
        patch("assayer.renderer.render_run"),
        patch("assayer.exporter.export") as mock_export,
    ):
        mock_run_all.return_value = [_result()]
        CliRunner().invoke(cli, ["run", "hello", "--models", "gpt-4o"])

    mock_export.assert_not_called()


# ---------------------------------------------------------------------------
# models list
# ---------------------------------------------------------------------------


def test_models_list_shows_all_providers():
    result = CliRunner().invoke(cli, ["models", "list"])
    assert result.exit_code == 0
    for provider in ("openai", "anthropic", "gemini", "ollama"):
        assert provider in result.output


def test_models_list_includes_known_model_identifiers():
    result = CliRunner().invoke(cli, ["models", "list"])
    assert "gpt-4o" in result.output
    assert "claude-sonnet" in result.output
    assert "gemini-2" in result.output
    assert "ollama/" in result.output


# ---------------------------------------------------------------------------
# models check
# ---------------------------------------------------------------------------


def test_models_check_shows_all_api_key_names():
    result = CliRunner().invoke(cli, ["models", "check"])
    assert result.exit_code == 0
    assert "OPENAI_API_KEY" in result.output
    assert "ANTHROPIC_API_KEY" in result.output
    assert "GEMINI_API_KEY" in result.output


def test_models_check_ollama_not_running():
    with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
        result = CliRunner().invoke(cli, ["models", "check", "ollama"])
    assert "not running" in result.output.lower()


def test_models_check_ollama_running_lists_models():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "models": [{"name": "llama3"}, {"name": "mistral"}]
    }
    mock_response.raise_for_status.return_value = None

    with patch("httpx.get", return_value=mock_response):
        result = CliRunner().invoke(cli, ["models", "check", "ollama"])

    assert result.exit_code == 0
    assert "running" in result.output.lower()
    assert "llama3" in result.output
    assert "mistral" in result.output


def test_models_check_ollama_running_no_models():
    mock_response = MagicMock()
    mock_response.json.return_value = {"models": []}
    mock_response.raise_for_status.return_value = None

    with patch("httpx.get", return_value=mock_response):
        result = CliRunner().invoke(cli, ["models", "check", "ollama"])

    assert result.exit_code == 0
    assert "No local models" in result.output


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_show_exits_cleanly():
    result = CliRunner().invoke(cli, ["config", "show"])
    assert result.exit_code == 0


def test_config_show_lists_key_names():
    result = CliRunner().invoke(cli, ["config", "show"])
    assert "OPENAI_API_KEY" in result.output
    assert "ANTHROPIC_API_KEY" in result.output


def test_config_set_calls_set_api_key():
    with patch("assayer.cli.main.set_api_key") as mock_set:
        result = CliRunner().invoke(
            cli, ["config", "set", "OPENAI_API_KEY", "sk-test"]
        )

    assert result.exit_code == 0
    mock_set.assert_called_once_with("OPENAI_API_KEY", "sk-test")
    assert "saved" in result.output.lower()
