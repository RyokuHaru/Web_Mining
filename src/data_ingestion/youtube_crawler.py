"""Collect YouTube metadata and comments linked to Flow 1 song IDs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


class YouTubeCrawler:
    def __init__(self, max_comments: int | None = 500, sort_by: str = "recent") -> None:
        if max_comments is not None and max_comments < 1:
            raise ValueError("max_comments must be positive or None")
        if sort_by not in {"recent", "popular"}:
            raise ValueError("sort_by must be 'recent' or 'popular'")

        try:
            from youtube_comment_downloader import (
                SORT_BY_POPULAR,
                SORT_BY_RECENT,
                YoutubeCommentDownloader,
            )
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError(
                "Install YouTube dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.downloader = YoutubeCommentDownloader()
        self.yt_dlp = yt_dlp
        self.sort_order = SORT_BY_RECENT if sort_by == "recent" else SORT_BY_POPULAR
        self.max_comments = max_comments

    def get_video(self, url: str) -> dict[str, Any]:
        video_url, video_id = _normalize_url(url)
        options = {"quiet": True, "no_warnings": True, "skip_download": True}
        with self.yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(video_url, download=False)

        if not info or info.get("_type") == "playlist":
            raise ValueError(f"Expected one YouTube video: {url}")
        return {
            "video_id": str(info.get("id") or video_id),
            "title": info.get("title") or "",
            "description": info.get("description") or "",
            "channel": info.get("channel") or info.get("uploader") or "",
            "channel_id": info.get("channel_id") or info.get("uploader_id") or "",
            "published_at": info.get("upload_date") or "",
            "duration_seconds": info.get("duration"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "tags": info.get("tags") or [],
            "url": video_url,
        }

    def get_comments(
        self,
        url: str,
        video_id: str | None = None,
        song_id: str | None = None,
    ) -> list[dict[str, Any]]:
        video_url, parsed_id = _normalize_url(url)
        video_id = video_id or parsed_id
        rows = self.downloader.get_comments_from_url(
            video_url, sort_by=self.sort_order
        )
        if self.max_comments is not None:
            rows = islice(rows, self.max_comments)
        return [
            _normalize_comment(row, video_id, song_id)
            for row in rows
            if str(row.get("text", "")).strip()
        ]

    def crawl_video(self, url: str, song_id: str | None = None) -> dict[str, Any]:
        video_url, video_id = _normalize_url(url)
        try:
            video = self.get_video(video_url)
        except Exception as exc:
            video = {
                "video_id": video_id,
                "title": "",
                "description": "",
                "channel": "",
                "channel_id": "",
                "published_at": "",
                "duration_seconds": None,
                "view_count": None,
                "like_count": None,
                "tags": [],
                "url": video_url,
                "metadata_error": _brief_error(exc),
            }
            logger.warning(
                "Could not fetch metadata for video %s; trying comments anyway: %s",
                video_id,
                _brief_error(exc),
            )
        video["song_id"] = song_id
        comments = self.get_comments(video["url"], video["video_id"], song_id)
        return {"video": video, "comments": comments}

    def crawl_to_files(
        self, inputs: Iterable[Mapping[str, Any] | str], output_dir: Path
    ) -> tuple[Path, Path, Path]:
        requests = [_normalize_video_input(item) for item in inputs]
        if not requests:
            raise ValueError("Provide at least one YouTube video mapping")

        video_song_ids: dict[str, str | None] = {}
        for item in requests:
            _, video_id = _normalize_url(item["url"])
            song_id = item["song_id"]
            if video_id in video_song_ids:
                previous = video_song_ids[video_id]
                if previous != song_id:
                    raise ValueError(
                        f"YouTube video {video_id} is mapped to multiple song IDs"
                    )
            video_song_ids[video_id] = song_id

        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / "youtube_videos.json"
        comments_path = output_dir / "youtube_comments.jsonl"
        errors_path = output_dir / "youtube_crawl_errors.json"

        videos = _load_json_records(video_path)
        comments = _load_jsonl_records(comments_path)
        previous_errors = _load_json_records(errors_path)
        requested_ids = {
            _normalize_url(item["url"])[1] for item in requests
        }
        errors: list[dict[str, str | None]] = [
            row for row in previous_errors if row.get("video_id") not in requested_ids
        ]

        comment_counts: dict[str, int] = {}
        for row in comments:
            video_id = str(row.get("video_id") or "")
            if video_id and str(row.get("text") or "").strip():
                comment_counts[video_id] = comment_counts.get(video_id, 0) + 1
        completed = set(comment_counts)
        already_done = completed & requested_ids
        logger.info(
            "Found existing comments for %d of %d mapped videos; retrying the rest",
            len(already_done),
            len(requested_ids),
        )

        video_index = {
            str(row.get("video_id")): index
            for index, row in enumerate(videos)
            if row.get("video_id") not in (None, "")
        }
        for item in requests:
            _, video_id = _normalize_url(item["url"])
            if video_id in completed:
                logger.info(
                    "Skipping %s; already have %d comments",
                    video_id,
                    comment_counts.get(video_id, 0),
                )
                continue
            completed.add(video_id)
            try:
                result = self.crawl_video(item["url"], item["song_id"])
            except Exception as exc:
                logger.warning("Could not crawl video %s: %s", video_id, _brief_error(exc))
                errors.append(
                    {
                        "video_id": video_id,
                        "song_id": item["song_id"],
                        "url": item["url"],
                        "error": _brief_error(exc),
                    }
                )
                continue
            if video_id in video_index:
                videos[video_index[video_id]] = result["video"]
            else:
                video_index[video_id] = len(videos)
                videos.append(result["video"])
            comments.extend(result["comments"])
            comment_counts[video_id] = len(result["comments"])
            logger.info(
                "Collected %d comments for %s",
                len(result["comments"]),
                result["video"]["title"] or video_id,
            )

        video_path.write_text(
            json.dumps(videos, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        comments_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in comments),
            encoding="utf-8",
        )
        errors_path.write_text(
            json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not videos and not comments:
            raise RuntimeError(
                f"Could not crawl any video. Review errors in {errors_path}."
            )
        return video_path, comments_path, errors_path


def load_video_inputs(path: Path) -> list[dict[str, Any]]:
    """Load video-to-song mappings from CSV, JSON, JSONL, or a URL text file."""
    if not path.is_file():
        raise FileNotFoundError(f"Video mapping file does not exist: {path}")

    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    elif suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("videos", []) if isinstance(payload, dict) else payload
    elif suffix == ".txt":
        rows = [
            {"url": line.strip()}
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raise ValueError("Video mapping input must be .csv, .json, .jsonl, or .txt")

    if not isinstance(rows, list):
        raise ValueError("Video mapping file must contain a list of video records")
    return [_normalize_video_input(row) for row in rows]


def validate_song_ids(inputs: list[dict[str, Any]], songs_path: Path) -> None:
    """Reject mapping IDs that do not exist in the Flow 1 song catalog."""
    mapped_ids = {str(row["song_id"]) for row in inputs if row.get("song_id")}
    if not mapped_ids or not songs_path.is_file():
        return
    payload = json.loads(songs_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("songs", [])
    if not isinstance(payload, list):
        raise ValueError(f"Song catalog must contain a list: {songs_path}")
    known_ids = {
        str(row.get("song_id"))
        for row in payload
        if isinstance(row, dict) and row.get("song_id") not in (None, "")
    }
    unknown_ids = sorted(mapped_ids - known_ids)
    if unknown_ids:
        raise ValueError(
            "Video map contains song_id values absent from Flow 1 songs.json: "
            + ", ".join(unknown_ids[:10])
        )


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a JSON list of records in {path}")
    return payload


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL record in {path} at line {line_number}, "
                f"column {exc.colno}: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        records.append(row)
    return records


def _normalize_video_input(row: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(row, str):
        row = {"url": row}
    if not isinstance(row, Mapping):
        raise ValueError("Each video mapping must be a URL or an object")
    url = str(row.get("url") or row.get("youtube_url") or "").strip()
    if not url:
        raise ValueError("Each video mapping must include a url")
    _normalize_url(url)
    song_id_value = row.get("song_id")
    song_id = str(song_id_value).strip() if song_id_value not in (None, "") else None
    return {"url": url, "song_id": song_id}


def _normalize_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").casefold()
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host == "youtube.com" or host.endswith(".youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            match = re.match(r"/(?:embed|shorts|live)/([^/]+)", parsed.path)
            video_id = match.group(1) if match else ""
    else:
        video_id = ""

    if not video_id or not re.fullmatch(r"[\w-]+", video_id):
        raise ValueError(f"Invalid YouTube video URL: {url}")
    return f"https://www.youtube.com/watch?v={video_id}", video_id


def _normalize_comment(
    row: dict[str, Any], video_id: str, song_id: str | None = None
) -> dict[str, Any]:
    comment_id = row.get("cid")
    return {
        "song_id": song_id,
        "video_id": video_id,
        "comment_id": str(comment_id) if comment_id is not None else None,
        "text": str(row.get("text") or "").strip(),
        "author": row.get("author", ""),
        "channel": row.get("channel", ""),
        "published_at": row.get("time", ""),
        "likes": row.get("votes", ""),
        "reply_count": row.get("replies", 0),
        "is_reply": bool(row.get("reply", False)),
    }


def _brief_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:240]}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect YouTube metadata and comments for Flow 2"
    )
    parser.add_argument(
        "--urls-file",
        type=Path,
        default=Path("data/raw/youtube_video_map.csv"),
        help="CSV with song_id,youtube_url; .txt URL lists are also accepted",
    )
    parser.add_argument("--songs-file", type=Path, default=Path("data/raw/songs.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-comments", type=int, default=500)
    parser.add_argument("--sort-by", choices=("recent", "popular"), default="recent")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    inputs = load_video_inputs(args.urls_file)
    validate_song_ids(inputs, args.songs_file)
    crawler = YouTubeCrawler(args.max_comments, args.sort_by)
    video_path, comments_path, errors_path = crawler.crawl_to_files(
        inputs, args.output_dir
    )
    print(f"Saved video metadata to {video_path}")
    print(f"Saved comments to {comments_path}")
    print(f"Saved crawl errors to {errors_path}")
    videos = json.loads(video_path.read_text(encoding="utf-8"))
    metadata_missing = sum(1 for video in videos if video.get("metadata_error"))
    print(
        f"Crawled {len(videos)} videos; metadata unavailable {metadata_missing}; "
        f"failed {len(json.loads(errors_path.read_text(encoding='utf-8')))}"
    )


if __name__ == "__main__":
    main()
