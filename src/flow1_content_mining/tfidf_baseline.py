"""TF-IDF baseline for lyric search and comparison."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


class TfidfBaseline:
    def __init__(self, **vectorizer_options: object) -> None:
        defaults: dict[str, object] = {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": (1, 2),
            "max_features": 100_000,
        }
        defaults.update(vectorizer_options)
        self.vectorizer = TfidfVectorizer(**defaults)
        self.matrix: sparse.csr_matrix | None = None
        self.song_ids: list[str] = []

    def fit(self, song_ids: Sequence[str], lyrics: Sequence[str]) -> sparse.csr_matrix:
        if len(song_ids) != len(lyrics):
            raise ValueError("song_ids and lyrics must have the same length")
        if len(lyrics) == 0:
            raise ValueError("At least one song is required")
        self.song_ids = [str(song_id) for song_id in song_ids]
        self.matrix = self.vectorizer.fit_transform(lyrics).tocsr()
        return self.matrix

    def search(self, query: str, top_k: int = 10) -> list[dict[str, float | str]]:
        if self.matrix is None:
            raise RuntimeError("Call fit() before search().")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {"song_id": self.song_ids[i], "score": float(scores[i])}
            for i in order
            if scores[i] > 0
        ]

