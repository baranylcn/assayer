from unittest.mock import patch

import pytest

from assayer.models import ModelResult
from assayer.scorer import compute_similarity, readability_stats


def _result(model: str, output: str, error: str | None = None) -> ModelResult:
    return ModelResult(
        model=model,
        output=output,
        tokens_input=0,
        tokens_output=0,
        latency_seconds=0.0,
        cost_usd=0.0,
        error=error,
    )


def test_compute_similarity_two_models():
    results = [_result("gpt-4o", "Hello world"), _result("claude", "Hello world")]
    with patch("assayer.scorer._get_model") as mock_get:
        import numpy as np

        vec = np.ones(384, dtype="float32")
        mock_get.return_value.encode.return_value = np.stack([vec, vec])
        similarity = compute_similarity(results)

    assert ("gpt-4o", "claude") in similarity
    score = similarity[("gpt-4o", "claude")]
    assert 0.0 <= score <= 1.0


def test_compute_similarity_three_models():
    results = [
        _result("gpt-4o", "Hello"),
        _result("claude", "Hello"),
        _result("gemini", "Hello"),
    ]
    with patch("assayer.scorer._get_model") as mock_get:
        import numpy as np

        vec = np.ones(384, dtype="float32")
        mock_get.return_value.encode.return_value = np.stack([vec, vec, vec])
        similarity = compute_similarity(results)

    assert ("gpt-4o", "claude") in similarity
    assert ("gpt-4o", "gemini") in similarity
    assert ("claude", "gemini") in similarity
    assert len(similarity) == 3


def test_compute_similarity_skips_errors():
    results = [
        _result("gpt-4o", "Hello"),
        _result("claude", "", error="API error"),
    ]
    similarity = compute_similarity(results)
    assert similarity == {}


def test_compute_similarity_single_model():
    results = [_result("gpt-4o", "Hello world")]
    similarity = compute_similarity(results)
    assert similarity == {}


def test_readability_stats_basic():
    stats = readability_stats("Hello world. How are you? I am fine.")
    assert stats["word_count"] == 8
    assert stats["sentence_count"] == 3
    assert stats["avg_sentence_length"] == pytest.approx(8 / 3)


@pytest.mark.parametrize(
    ("text", "sentence_count"),
    [
        ("Dr. Smith scored 3.5. Well done.", 2),
        ("Visit example.com for details.", 1),
        ("The price is $3.99. Cheap!", 2),
        ("Mr. Jones paid at 4.30 p.m. Done?", 2),
    ],
)
def test_readability_stats_ignores_non_boundary_periods(text, sentence_count):
    stats = readability_stats(text)
    assert stats["sentence_count"] == sentence_count


def test_readability_stats_empty():
    stats = readability_stats("")
    assert stats["word_count"] == 0
    assert stats["sentence_count"] == 1
    assert stats["avg_sentence_length"] == 0.0
