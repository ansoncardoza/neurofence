"""Semantic similarity backends.

Two implementations of the same `SimilarityBackend` protocol:

- `TfidfSimilarityBackend` (default): TF-IDF + cosine similarity, fit
  on-the-fly per comparison pair. Lightweight, deterministic, requires no
  network access or downloaded model -- appropriate as NeuroFence's
  default so behavioral comparison works offline and reproducibly out of
  the box. Weaker than a real sentence embedding at catching paraphrases
  with little vocabulary overlap.
- `SentenceTransformerSimilarityBackend` (optional): real sentence
  embeddings via the `sentence-transformers` package. Stronger semantic
  matching, at the cost of a one-time model download and materially more
  compute per comparison. Opt in explicitly when network/model access is
  available and paraphrase-level sensitivity matters.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityBackend(Protocol):
    def similarity(self, text_a: str, text_b: str) -> float:
        """Return a similarity score in [0, 1]; 1.0 means semantically identical."""
        ...


class TfidfSimilarityBackend:
    def similarity(self, text_a: str, text_b: str) -> float:
        a, b = text_a.strip(), text_b.strip()

        if a == b:
            return 1.0
        if not a or not b:
            # One side has no content at all -- treat as maximally dissimilar
            # rather than raising, since "empty output" is itself a valid
            # (and behaviorally interesting) model response.
            return 0.0

        try:
            vectors = TfidfVectorizer().fit_transform([a, b])
        except ValueError:
            # Both strings are non-empty but reduce to an empty vocabulary
            # after tokenization (e.g. pure punctuation/whitespace/symbols).
            return 1.0 if a == b else 0.0

        sim = float(cosine_similarity(vectors[0], vectors[1])[0, 0])
        return float(np.clip(sim, 0.0, 1.0))


class SentenceTransformerSimilarityBackend:
    """Lazily loads a sentence-transformers model on first use.

    Requires the optional `sentence-transformers` package (and, on first
    use, network access to download the model) -- not imported at module
    load time so NeuroFence's default path has no such dependency.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _get_model(self):  # noqa: ANN202
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "SentenceTransformerSimilarityBackend requires the "
                    "'sentence-transformers' package. Install the 'ml' extra "
                    "or use TfidfSimilarityBackend instead."
                ) from e
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def similarity(self, text_a: str, text_b: str) -> float:
        a, b = text_a.strip(), text_b.strip()
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0

        model = self._get_model()
        embeddings = model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
        sim = float(np.dot(embeddings[0], embeddings[1]))
        return float(np.clip(sim, 0.0, 1.0))
