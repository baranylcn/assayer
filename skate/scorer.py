from __future__ import annotations

from skate.models import ModelResult

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for --score. "
                "Install it with: pip install 'skate[score]'"
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
            similarity[(valid[i].model, valid[j].model)] = score

    return similarity


def readability_stats(text: str) -> dict[str, float]:
    sentences = [
        s
        for s in text.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]
    words = text.split()
    word_count = len(words)
    sentence_count = len(sentences) or 1
    return {
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "avg_sentence_length": word_count / sentence_count,
    }
