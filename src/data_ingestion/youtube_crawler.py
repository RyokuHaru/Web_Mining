"""Collect YouTube video metadata and comments."""

from __future__ import annotations

import argparse
import json
import re
from itertools import islice
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse


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

    def get_comments(self, url: str, video_id: str | None = None) -> list[dict[str, Any]]:
        video_url, parsed_id = _normalize_url(url)
        video_id = video_id or parsed_id
        rows = self.downloader.get_comments_from_url(
            video_url, sort_by=self.sort_order
        )
        if self.max_comments is not None:
            rows = islice(rows, self.max_comments)
        return [_normalize_comment(row, video_id) for row in rows]

    def crawl_video(self, url: str) -> dict[str, Any]:
        video = self.get_video(url)
        comments = self.get_comments(video["url"], video["video_id"])
        return {"video": video, "comments": comments}

    def crawl_to_files(self, urls: Iterable[str], output_dir: Path) -> tuple[Path, Path]:
        urls = list(urls)
        if not urls:
            raise ValueError("Provide at least one YouTube URL")
        videos: list[dict[str, Any]] = []
        comments: list[dict[str, Any]] = []
        for url in urls:
            result = self.crawl_video(url)
            videos.append(result["video"])
            comments.extend(result["comments"])

        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / "youtube_videos.json"
        comments_path = output_dir / "youtube_comments.jsonl"
        video_path.write_text(
            json.dumps(videos, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        comments_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in comments),
            encoding="utf-8",
        )
        return video_path, comments_path


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


def _normalize_comment(row: dict[str, Any], video_id: str) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "comment_id": row.get("cid"),
        "text": row.get("text", ""),
        "author": row.get("author", ""),
        "channel": row.get("channel", ""),
        "published_at": row.get("time", ""),
        "likes": row.get("votes", ""),
        "reply_count": row.get("replies", 0),
        "is_reply": bool(row.get("reply", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect YouTube video metadata and comments"
    )
    parser.add_argument("--urls-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-comments", type=int, default=500)
    parser.add_argument("--sort-by", choices=("recent", "popular"), default="recent")
    args = parser.parse_args()
    urls = [
        line.strip()
        for line in args.urls_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    crawler = YouTubeCrawler(args.max_comments, args.sort_by)
    video_path, comments_path = crawler.crawl_to_files(urls, args.output_dir)
    print(f"Saved video metadata to {video_path}")
    print(f"Saved comments to {comments_path}")


if __name__ == "__main__":
    main()
