import csv
import json

import pytest

from skate.exporter import export
from skate.models import ModelResult

_EXPECTED_FIELDS = {"model", "output", "tokens_input", "tokens_output", "latency_seconds", "cost_usd", "error"}


def _results() -> list[ModelResult]:
    return [
        ModelResult(
            model="gpt-4o",
            output="Hello",
            tokens_input=5,
            tokens_output=2,
            latency_seconds=0.3,
            cost_usd=0.001,
        ),
        ModelResult(
            model="claude",
            output="Hi there",
            tokens_input=3,
            tokens_output=4,
            latency_seconds=0.5,
            cost_usd=0.0,
        ),
    ]


def test_export_json_structure(tmp_path):
    path = tmp_path / "results.json"
    export(_results(), str(path))

    data = json.loads(path.read_text(encoding="utf-8"))

    assert len(data) == 2
    assert set(data[0].keys()) == _EXPECTED_FIELDS


def test_export_json_values(tmp_path):
    path = tmp_path / "results.json"
    export(_results(), str(path))

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data[0]["model"] == "gpt-4o"
    assert data[0]["output"] == "Hello"
    assert data[0]["cost_usd"] == pytest.approx(0.001)
    assert data[1]["model"] == "claude"


def test_export_json_with_error_result(tmp_path):
    results = [
        ModelResult(
            model="gpt-4o",
            output="",
            tokens_input=0,
            tokens_output=0,
            latency_seconds=0.0,
            cost_usd=0.0,
            error="OPENAI_API_KEY is not set",
        )
    ]
    path = tmp_path / "results.json"
    export(results, str(path))

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data[0]["error"] == "OPENAI_API_KEY is not set"
    assert data[0]["output"] == ""


def test_export_csv_rows(tmp_path):
    path = tmp_path / "results.csv"
    export(_results(), str(path))

    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["model"] == "gpt-4o"
    assert rows[1]["model"] == "claude"


def test_export_csv_has_all_headers(tmp_path):
    path = tmp_path / "results.csv"
    export(_results(), str(path))

    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert set(reader.fieldnames) == _EXPECTED_FIELDS


def test_export_csv_case_insensitive_extension(tmp_path):
    path = tmp_path / "results.CSV"
    export(_results(), str(path))

    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2


def test_export_unknown_extension_writes_json(tmp_path):
    path = tmp_path / "results.txt"
    export(_results(), str(path))

    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 2
