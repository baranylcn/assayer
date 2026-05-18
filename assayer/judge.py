from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import click

from assayer.models import ModelResult
from assayer.runner import _make_provider

_PROMPT_TEMPLATE = """\
You are evaluating outputs from multiple language models for the same prompt.

Original prompt: {prompt}

Outputs:
{outputs}
{criteria_section}
Return JSON only:
{{
  "winner": "model name",
  "reasoning": "one paragraph",
  "scores": {{"model_name": {{"criterion": score_1_to_5}}}}
}}"""


@dataclass
class JudgeResult:
    winner: str
    reasoning: str
    scores: dict[str, dict[str, float]] = field(default_factory=dict)


def _build_prompt(
    prompt: str, results: list[ModelResult], criteria: list[str] | None
) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    outputs_block = "\n".join(
        f"Model {labels[i]} ({r.model}): {r.output}"
        for i, r in enumerate(results)
        if not r.error
    )

    if criteria:
        criteria_section = f"\nEvaluate on these criteria: {', '.join(criteria)}\n"
    else:
        criteria_section = ""

    return _PROMPT_TEMPLATE.format(
        prompt=prompt,
        outputs=outputs_block,
        criteria_section=criteria_section,
    )


def _parse_response(text: str) -> JudgeResult:
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    data = json.loads(stripped)
    return JudgeResult(
        winner=data["winner"],
        reasoning=data["reasoning"],
        scores=data.get("scores", {}),
    )


async def run_judge(
    prompt: str,
    results: list[ModelResult],
    judge_model: str,
    criteria: list[str] | None = None,
) -> JudgeResult | None:
    valid = [r for r in results if not r.error and r.output]
    if len(valid) < 2:
        click.echo("Judge skipped: fewer than 2 successful results.", err=True)
        return None

    judge_prompt = _build_prompt(prompt, valid, criteria)
    provider = _make_provider(judge_model)

    try:
        result = await provider.run(judge_prompt)
    except Exception as exc:
        click.echo(f"Judge call failed: {exc}", err=True)
        return None

    if result.error:
        click.echo(f"Judge call failed: {result.error}", err=True)
        return None

    try:
        return _parse_response(result.output)
    except Exception as exc:
        click.echo(f"Judge response could not be parsed: {exc}", err=True)
        return None
