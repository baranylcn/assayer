from __future__ import annotations

import re

from assayer.models import ModelResult

_model = None

_NON_BOUNDARY_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st"}


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for --score. "
                "Install it with: pip install 'assayer[score]'"
            )
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def compute_similarity(results: list[ModelResult]) -> dict[tuple[str, str], float]:
    valid = [r for r in results if not r.error and r.output]
    if len(valid) < 2:
        return {}

    import numpy as np

    embeddings = _get_model().encode([r.output for r in valid], convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.where(norms == 0, 1, norms)
    similarity: dict[tuple[str, str], float] = {}

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            score = float(np.dot(normalized[i], normalized[j]))
            score = max(-1.0, min(1.0, score))
            similarity[(valid[i].model, valid[j].model)] = score

    return similarity


def readability_stats(text: str) -> dict[str, float]:
    words = text.split()
    word_count = len(words)
    sentence_count = _count_sentences(text)
    return {
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "avg_sentence_length": word_count / sentence_count,
    }


def _count_sentences(text: str) -> int:
    count = 0
    start = 0

    for match in re.finditer(r"[.!?]+", text):
        punct_start, punct_end = match.span()
        if punct_end < len(text) and not text[punct_end].isspace():
            continue
        if _is_non_boundary_period(text, punct_start, punct_end):
            continue

        if text[start:punct_end].strip():
            count += 1
        start = punct_end

    if text[start:].strip():
        count += 1

    return count or 1


def _is_non_boundary_period(text: str, punct_start: int, punct_end: int) -> bool:
    if text[punct_start] != ".":
        return False
    if (
        punct_start > 0
        and punct_start + 1 < len(text)
        and text[punct_start - 1].isdigit()
        and text[punct_start + 1].isdigit()
    ):
        return True

    token_match = re.search(r"([A-Za-z]+)\.$", text[:punct_end])
    if not token_match:
        return False

    return token_match.group(1).lower() in _NON_BOUNDARY_ABBREVIATIONS
