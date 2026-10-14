"""SBERT embeddings, cosine search, and pairwise song similarity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SbertEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers with `pip install sentence-transformers`."
            ) from exc
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if len(texts) == 0:
            raise ValueError("At least one text is required")
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > batch_size,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def search(
        self, query: str, song_ids: Sequence[str], embeddings: np.ndarray, top_k: int = 10
    ) -> list[dict[str, float | str]]:
        if len(song_ids) != len(embeddings):
            raise ValueError("song_ids and embeddings must have the same length")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_vector = self.encode([query])
        scores = cosine_similarity(query_vector, embeddings).ravel()
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {"song_id": str(song_ids[i]), "score": float(scores[i])} for i in order
        ]

    @staticmethod
    def pairwise_similarity(embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        return cosine_similarity(embeddings).astype(np.float32)
