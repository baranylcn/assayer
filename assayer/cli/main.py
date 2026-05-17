import asyncio
import logging
import sys

import click
import httpx

logging.getLogger("LiteLLM").addFilter(
    type("_F", (logging.Filter,), {"filter": lambda self, r: r.levelno >= logging.ERROR})()
)

from assayer.config import get_api_key, set_api_key, show_config

_KNOWN_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-mini"],
    "anthropic": ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5-20251001"],
    "gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "ollama": ["ollama/llama3", "ollama/mistral", "ollama/phi3"],
}


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("prompt", required=False)
@click.option("--models", required=True, help="Comma-separated model identifiers.")
@click.option(
    "--prompt-file",
    type=click.Path(exists=True),
    help="Path to a .txt prompt file.",
)
@click.option(
    "--var",
    multiple=True,
    metavar="KEY=VALUE",
    help="Template variables, repeatable.",
)
@click.option("--system", default=None, help="System prompt applied to all models.")
@click.option("--temperature", type=float, default=None, help="Sampling temperature.")
@click.option("--max-tokens", type=int, default=None, help="Max output tokens.")
@click.option("--output", default=None, help="Save results to file (.json or .csv).")
@click.option("--score", is_flag=True, default=False, help="Show similarity matrix.")
@click.option("--judge", default=None, help="Model to use as judge.")
@click.option(
    "--judge-criteria", default=None, help="Comma-separated evaluation criteria."
)
def run(
    prompt: str | None,
    models: str,
    prompt_file: str | None,
    var: tuple[str, ...],
    system: str | None,
    temperature: float | None,
    max_tokens: int | None,
    output: str | None,
    score: bool,
    judge: str | None,
    judge_criteria: str | None,
) -> None:
    if prompt_file:
        with open(prompt_file) as f:
            prompt_text = f.read().strip()
    elif prompt:
        prompt_text = prompt
    else:
        click.echo("Provide a prompt or --prompt-file.", err=True)
        sys.exit(1)

    if var:
        variables: dict[str, str] = {}
        for item in var:
            if "=" not in item:
                click.echo(
                    f"Invalid --var format: {item!r}. Expected KEY=VALUE.", err=True
                )
                sys.exit(1)
            key, _, value = item.partition("=")
            variables[key.strip()] = value
        try:
            prompt_text = prompt_text.format_map(variables)
        except KeyError as exc:
            click.echo(f"Missing template variable: {exc}", err=True)
            sys.exit(1)

    from assayer.exporter import export
    from assayer.judge import run_judge
    from assayer.renderer import render_run
    from assayer.runner import run_all
    from assayer.scorer import compute_similarity

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    results = asyncio.run(
        run_all(
            prompt_text,
            model_list,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )
    similarity = compute_similarity(results) if score else None

    criteria = (
        [c.strip() for c in judge_criteria.split(",")] if judge_criteria else None
    )
    judge_result = (
        asyncio.run(run_judge(prompt_text, results, judge, criteria)) if judge else None
    )

    render_run(prompt_text, results, similarity=similarity, judge_result=judge_result)

    if output:
        export(results, output)
        click.echo(f"Results saved to {output}")


@cli.group()
def models_cmd() -> None:
    pass


cli.add_command(models_cmd, name="models")


@models_cmd.command(name="list")
def models_list() -> None:
    for provider, names in _KNOWN_MODELS.items():
        click.echo(f"\n{provider}")
        for name in names:
            click.echo(f"  {name}")


@models_cmd.command(name="check")
@click.argument("provider", required=False)
def models_check(provider: str | None) -> None:
    if provider == "ollama":
        _check_ollama()
        return

    keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    for name, env_var in keys.items():
        value = get_api_key(env_var)
        status = "set" if value else "not set"
        symbol = "+" if value else "-"
        click.echo(f"  [{symbol}] {name}: {env_var} {status}")


def _check_ollama() -> None:
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        response.raise_for_status()
        data = response.json()
        local_models: list[str] = [m["name"] for m in data.get("models", [])]
        click.echo("Ollama is running.")
        if local_models:
            click.echo("Local models:")
            for m in local_models:
                click.echo(f"  ollama/{m}")
        else:
            click.echo("No local models found.")
    except httpx.ConnectError:
        click.echo("Ollama is not running at localhost:11434.", err=True)
    except Exception as exc:
        click.echo(f"Ollama check failed: {exc}", err=True)


@cli.group()
def config() -> None:
    pass


@config.command(name="set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    set_api_key(key, value)
    click.echo(f"{key} saved.")


@config.command(name="show")
def config_show() -> None:
    data = show_config()
    for key, value in data.items():
        if value:
            masked = value[:8] + "..." if len(value) > 8 else value
            click.echo(f"  {key}: {masked}")
        else:
            click.echo(f"  {key}: not set")
