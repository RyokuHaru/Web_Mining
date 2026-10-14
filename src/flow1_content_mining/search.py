"""Semantic search and metadata filters used by the Flow 1 app surface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.flow1_content_mining.sbert_encoder import SbertEncoder


def filter_songs(
    songs: Sequence[Mapping[str, Any]],
    *,
    artist: str | None = None,
    writer: str | None = None,
    genre: str | None = None,
) -> list[Mapping[str, Any]]:
    """Filter feature rows by artist, writer, and genre (case-insensitive)."""
    filters = {"artist": artist, "writers": writer, "genres": genre}
    results = []
    for song in songs:
        matches = True
        for field, query in filters.items():
            if query and query.casefold() not in str(song.get(field, "")).casefold():
                matches = False
                break
        if matches:
            results.append(song)
    return results


def semantic_search(
    encoder: SbertEncoder,
    query: str,
    song_ids: Sequence[str],
    embeddings: np.ndarray,
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 10,
    artist: str | None = None,
    writer: str | None = None,
    genre: str | None = None,
) -> list[dict[str, Any]]:
    """Search lyric meaning, optionally restricting results by extracted fields."""
    if len(song_ids) != len(embeddings) or len(song_ids) != len(feature_rows):
        raise ValueError("song_ids, embeddings, and feature_rows must align")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    allowed_ids = {
        str(row["song_id"])
        for row in filter_songs(
            feature_rows, artist=artist, writer=writer, genre=genre
        )
    }
    selected_indices = [
        index for index, song_id in enumerate(song_ids) if str(song_id) in allowed_ids
    ]
    if not selected_indices:
        return []

    candidate_ids = [str(song_ids[index]) for index in selected_indices]
    candidate_vectors = embeddings[selected_indices]
    scores = cosine_similarity(encoder.encode([query]), candidate_vectors)[0]
    order = np.argsort(-np.asarray(scores), kind="stable")[:top_k]
    rows_by_id = {str(row["song_id"]): row for row in feature_rows}
    return [
        {
            **dict(rows_by_id[candidate_ids[index]]),
            "score": float(scores[index]),
        }
        for index in order
    ]
