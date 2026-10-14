"""Run Flow 1 over a JSON or CSV collection of songs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from config import settings
from src.flow1_content_mining.ner_extractor import NERExtractor
from src.flow1_content_mining.sbert_encoder import SbertEncoder
from src.flow1_content_mining.tfidf_baseline import TfidfBaseline

DEFAULT_GENRE_TERMS = (
    "alternative", "blues", "classical", "country", "dance", "disco",
    "edm", "electronic", "folk", "funk", "gospel", "hip hop", "hip-hop",
    "indie", "jazz", "k-pop", "metal", "pop", "punk", "rap", "reggae",
    "r&b", "rock", "soul", "techno", "trap",
)


def load_songs(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if path.suffix.casefold() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("songs", [payload])
    elif path.suffix.casefold() == ".csv":
        payload = pd.read_csv(path).fillna("").to_dict(orient="records")
    else:
        raise ValueError("Input must be a .json or .csv file")
    if not isinstance(payload, list) or not payload:
        raise ValueError("Input must contain a non-empty list of songs")

    songs = []
    for row_number, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Song at row {row_number} must be a JSON object")
        song_id = str(item.get("song_id", "")).strip()
        lyrics = str(item.get("lyrics", "")).strip()
        if not song_id or not lyrics:
            raise ValueError(
                f"Song at row {row_number} must include non-empty song_id and lyrics"
            )
        songs.append({**item, "song_id": song_id, "lyrics": lyrics})
    ids = [song["song_id"] for song in songs]
    if len(ids) != len(set(ids)):
        raise ValueError("song_id values must be unique")
    return songs


def run(input_path: Path, output_dir: Path) -> None:
    songs = load_songs(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = [song["song_id"] for song in songs]
    lyrics = [song["lyrics"] for song in songs]

    extractor = NERExtractor(
        model_name=settings.spacy_model, genre_terms=DEFAULT_GENRE_TERMS
    )
    feature_rows = []
    vector_records = []
    for song in songs:
        entities = extractor.extract(song["lyrics"], song)
        feature_rows.append(
            {
                "song_id": song["song_id"],
                "title": song.get("title", ""),
                "artist": "; ".join(entities["artist"]),
                "writers": "; ".join(entities["writers"]),
                "genres": "; ".join(entities["genres"]),
                "entity_candidates": "; ".join(entities["entity_candidates"]),
            }
        )
        record = {
            "song_id": song["song_id"],
            "title": song.get("title", ""),
            "artist": entities["artist"],
            "writers": entities["writers"],
            "genres": entities["genres"],
            "entity_candidates": entities["entity_candidates"],
        }
        for field in (
            "album",
            "release_date",
            "source",
            "source_url",
            "video_id",
            "youtube_video_id",
            "spotify_id",
            "genius_id",
            "track_url",
        ):
            if song.get(field) not in (None, ""):
                record[field] = song[field]
        vector_records.append(record)
    pd.DataFrame(feature_rows).to_csv(
        output_dir / "song_features.csv", index=False, encoding="utf-8-sig"
    )

    tfidf = TfidfBaseline()
    tfidf_matrix = tfidf.fit(ids, lyrics)
    save_npz(output_dir / "tfidf_matrix.npz", tfidf_matrix)
    _write_json(output_dir / "tfidf_song_ids.json", ids)

    encoder = SbertEncoder(model_name=settings.sbert_model)
    embeddings = encoder.encode(lyrics)
    np.save(output_dir / "semantic_embeddings.npy", embeddings)
    _write_json(output_dir / "semantic_song_ids.json", ids)
    _write_vectors(
        output_dir / "semantic_vectors.jsonl",
        vector_records,
        embeddings,
        settings.sbert_model,
    )

    similarity = encoder.pairwise_similarity(embeddings)
    pd.DataFrame(similarity, index=ids, columns=ids).rename_axis("song_id").to_csv(
        output_dir / "semantic_similarity.csv", encoding="utf-8-sig"
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_vectors(
    path: Path,
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    model_name: str,
) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record, vector in zip(records, embeddings):
            row = {
                **record,
                "vector_model": model_name,
                "vector_dimension": int(embeddings.shape[1]),
                "semantic_vector": vector.tolist(),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Music Flow 1 pipeline")
    parser.add_argument("--input", type=Path, required=True, help="JSON or CSV songs")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
