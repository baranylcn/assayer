import csv
import json
from pathlib import Path

from assayer.models import ModelResult

_FIELDS = [
    "model",
    "output",
    "tokens_input",
    "tokens_output",
    "latency_seconds",
    "cost_usd",
    "error",
]


def _to_dict(result: ModelResult) -> dict:
    return {
        "model": result.model,
        "output": result.output,
        "tokens_input": result.tokens_input,
        "tokens_output": result.tokens_output,
        "latency_seconds": result.latency_seconds,
        "cost_usd": result.cost_usd,
        "error": result.error,
    }


def export(results: list[ModelResult], path: str) -> None:
    dest = Path(path)
    records = [_to_dict(r) for r in results]

    if dest.suffix.lower() == ".csv":
        with dest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()
            if records:
                writer.writerows(records)
    else:
        dest.write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
