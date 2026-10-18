"""Analyze YouTube comments and aggregate emotion scores by song."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config import settings
from src.flow2_social_mining.sentiment_analyzer import RobertaSentimentAnalyzer

logger = logging.getLogger(__name__)


def load_comments(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Comments input does not exist: {path}")
    if path.suffix.casefold() == ".jsonl":
        payload = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                payload.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record in {path} at line {line_number}, "
                    f"column {exc.colno}: {exc.msg}"
                ) from exc
    elif path.suffix.casefold() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload = payload.get("comments", []) if isinstance(payload, dict) else payload
    else:
        raise ValueError("Comments input must be .jsonl or .json")
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError("Comments input must contain a list of comment objects")
    return [row for row in payload if str(row.get("text") or "").strip()]


def _load_json_list(path: Path | None, key: str) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get(key, [])
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError(f"{path} must contain a list of {key}")
    return payload


def _link_comments(
    comments: list[dict[str, Any]], videos: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    video_song_ids: dict[str, str] = {}
    for video in videos:
        video_id = str(video.get("video_id") or "").strip()
        song_id = str(video.get("song_id") or "").strip()
        if video_id and song_id:
            previous = video_song_ids.setdefault(video_id, song_id)
            if previous != song_id:
                raise ValueError(f"Video {video_id} is linked to multiple song IDs")

    linked = []
    for row in comments:
        comment = dict(row)
        video_id = str(comment.get("video_id") or "").strip()
        song_id = str(comment.get("song_id") or "").strip()
        mapped_song_id = video_song_ids.get(video_id, "")
        if song_id and mapped_song_id and song_id != mapped_song_id:
            raise ValueError(
                f"Comment for video {video_id} has song_id {song_id}, "
                f"but video metadata maps it to {mapped_song_id}"
            )
        comment["song_id"] = song_id or mapped_song_id or None
        linked.append(comment)
    return linked


def _summarize_by_song(
    comments: list[dict[str, Any]],
    songs: list[dict[str, Any]],
    videos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    song_catalog: dict[str, dict[str, Any]] = {}
    for song in songs:
        song_id = str(song.get("song_id") or "").strip()
        if not song_id:
            raise ValueError("Each song record must include song_id")
        if song_id in song_catalog:
            raise ValueError(f"Duplicate song_id in song catalog: {song_id}")
        song_catalog[song_id] = song

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comments:
        song_id = str(row.get("song_id") or "").strip()
        if song_id:
            grouped[song_id].append(row)

    video_ids_by_song: dict[str, list[str]] = defaultdict(list)
    video_by_id: dict[str, dict[str, Any]] = {}
    for video in videos:
        video_id = str(video.get("video_id") or "").strip()
        song_id = str(video.get("song_id") or "").strip()
        if video_id:
            video_by_id[video_id] = video
        if song_id and video_id and video_id not in video_ids_by_song[song_id]:
            video_ids_by_song[song_id].append(video_id)
    for rows in grouped.values():
        for row in rows:
            video_id = str(row.get("video_id") or "").strip()
            song_id = str(row.get("song_id") or "").strip()
            if video_id and song_id and video_id not in video_ids_by_song[song_id]:
                video_ids_by_song[song_id].append(video_id)

    all_ids = list(song_catalog)
    if not all_ids:
        all_ids = list(dict.fromkeys([*video_ids_by_song, *grouped]))

    summaries = []
    for song_id in all_ids:
        rows = grouped.get(song_id, [])
        song = song_catalog.get(song_id, {})
        video_ids = video_ids_by_song.get(song_id, [])
        valences = [
            float(row["valence"])
            for row in rows
            if row.get("valence") is not None
        ]
        sentiment_counts = Counter(
            str(row.get("sentiment", "unknown")) for row in rows
        )
        video_titles = [
            video_by_id[video_id].get("title", "")
            for video_id in video_ids
            if video_id in video_by_id
        ]
        summaries.append(
            {
                "song_id": song_id,
                "title": song.get("title") or (video_titles[0] if video_titles else ""),
                "artist": song.get("artist", ""),
                "video_count": len(video_ids),
                "video_ids": video_ids,
                "comment_count": len(rows),
                "social_valence_score": (
                    sum(valences) / len(valences) if valences else None
                ),
                "sentiment_counts": dict(sentiment_counts),
                "sentiment_model": settings.sentiment_model,
            }
        )
    return summaries


def run(
    input_path: Path,
    output_dir: Path,
    songs_path: Path | None = None,
    videos_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    comments = load_comments(input_path)
    songs = _load_json_list(songs_path, "songs")
    videos = _load_json_list(videos_path, "videos")
    comments = _link_comments(comments, videos)

    known_song_ids = {
        str(song.get("song_id")) for song in songs if song.get("song_id")
    }
    unknown_ids = sorted(
        {
            str(row["song_id"])
            for row in comments
            if row.get("song_id") and known_song_ids and str(row["song_id"]) not in known_song_ids
        }
    )
    if unknown_ids:
        raise ValueError(
            "YouTube mapping references song_id values missing from Flow 1 input: "
            + ", ".join(unknown_ids[:10])
        )

    if comments:
        analyzer = RobertaSentimentAnalyzer(settings.sentiment_model)
        scored = analyzer.analyze(comments)
        by_video = analyzer.summarize(scored, videos)
    else:
        scored = []
        by_video = [
            {
                "video_id": str(video.get("video_id") or ""),
                "song_id": video.get("song_id"),
                "title": video.get("title", ""),
                "comment_count": 0,
                "social_valence_score": None,
                "sentiment_counts": {},
                "sentiment_model": settings.sentiment_model,
            }
            for video in videos
        ]

    by_song = _summarize_by_song(scored, songs, videos)
    unlinked_count = sum(1 for row in scored if not row.get("song_id"))
    if unlinked_count:
        logger.warning(
            "%d comments are not linked to a song; they appear only in video-level output",
            unlinked_count,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "youtube_comments_scored.jsonl"
    video_summary_path = output_dir / "youtube_video_valence.json"
    song_summary_path = output_dir / "social_valence.json"
    scored_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored),
        encoding="utf-8",
    )
    video_summary_path.write_text(
        json.dumps(by_video, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    song_summary_path.write_text(
        json.dumps(by_song, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return scored_path, video_summary_path, song_summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flow 2 on YouTube comments")
    parser.add_argument(
        "--input", type=Path, default=Path("data/raw/youtube_comments.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--songs-input", type=Path, default=Path("data/raw/songs.json"))
    parser.add_argument(
        "--videos-input", type=Path, default=Path("data/raw/youtube_videos.json")
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    scored_path, video_path, song_path = run(
        args.input,
        args.output_dir,
        args.songs_input if args.songs_input.exists() else None,
        args.videos_input if args.videos_input.exists() else None,
    )
    print(f"Saved scored comments to {scored_path}")
    print(f"Saved video sentiment to {video_path}")
    print(f"Saved song sentiment to {song_path}")


if __name__ == "__main__":
    main()
