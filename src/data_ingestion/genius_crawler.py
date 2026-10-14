"""Collect song lyrics and metadata from Genius."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

RANDOM_SEARCH_TERMS = (
    "love", "heart", "night", "life", "time", "dream", "home", "money",
    "dance", "lonely", "summer", "rain", "fire", "goodbye", "light", "baby",
    "world", "friend", "party", "alone", "forever", "pop", "rock", "rap",
    "hip hop", "country", "jazz", "indie", "r&b", "metal", "reggae", "soul",
    "punk", "folk", "disco", "blues", "electronic", "k-pop",
)


class GeniusLyricsCrawler:
    def __init__(self, access_token: str, sleep_time: float = 0.8) -> None:
        if not access_token:
            raise ValueError("A Genius access token is required")
        try:
            import lyricsgenius
        except ImportError as exc:
            raise RuntimeError("Install lyricsgenius to use the Genius connector") from exc
        self.client = lyricsgenius.Genius(
            access_token,
            skip_non_songs=True,
            remove_section_headers=True,
            sleep_time=sleep_time,
            retries=2,
        )

    def get_song(self, title: str, artist: str) -> dict[str, Any] | None:
        """Return one song in the input shape expected by Flow 1."""
        result = self.client.search_song(title, artist)
        if result is None or not result.lyrics:
            return None
        return {
            "song_id": str(result.id),
            "genius_id": str(result.id),
            "title": result.title,
            "artist": result.artist,
            "lyrics": result.lyrics,
            "source": "Genius",
            "source_url": result.url,
        }

    def crawl_random_songs(
        self,
        count: int = 100,
        seed: int | None = None,
        max_searches: int = 30,
    ) -> list[dict[str, Any]]:
        if count < 1:
            raise ValueError("count must be positive")
        if max_searches < 1:
            raise ValueError("max_searches must be positive")

        rng = random.Random(seed)
        terms = list(RANDOM_SEARCH_TERMS)
        rng.shuffle(terms)
        candidates: dict[str, dict[str, Any]] = {}

        for term in terms[:max_searches]:
            page = rng.randint(1, 20)
            try:
                response = self.client.search_songs(term, per_page=50, page=page)
            except Exception as exc:
                logger.warning("Genius search failed for %r: %s", term, exc)
                continue

            for hit in _search_hits(response):
                song = hit.get("result", hit)
                song_id = song.get("id")
                if (
                    not song_id
                    or not song.get("url")
                    or song.get("lyrics_state") != "complete"
                    or song.get("instrumental")
                ):
                    continue
                candidates.setdefault(str(song_id), song)
            if len(candidates) >= count * 4:
                break

        candidate_rows = list(candidates.values())
        rng.shuffle(candidate_rows)
        songs: list[dict[str, Any]] = []
        for candidate in candidate_rows:
            if len(songs) >= count:
                break
            try:
                lyrics = self.client.lyrics(song_url=candidate["url"])
            except Exception as exc:
                logger.warning(
                    "Could not fetch Genius lyrics for %s: %s",
                    candidate.get("url", candidate.get("id")),
                    exc,
                )
                continue
            if not lyrics or not lyrics.strip():
                continue

            song_id = str(candidate["id"])
            primary_artist = candidate.get("primary_artist") or {}
            album = candidate.get("album") or {}
            primary_tag = candidate.get("primary_tag") or {}
            record: dict[str, Any] = {
                "song_id": song_id,
                "genius_id": song_id,
                "title": candidate.get("title") or candidate.get("full_title", ""),
                "artist": primary_artist.get("name")
                or candidate.get("artist_names", ""),
                "lyrics": lyrics,
                "source": "Genius",
                "source_url": candidate["url"],
            }
            if album:
                record["album"] = album.get("name") or album.get("full_title", "")
            release_date = candidate.get("release_date_for_display")
            if release_date:
                record["release_date"] = release_date
            if primary_tag.get("name"):
                record["genres"] = [primary_tag["name"]]
            songs.append(record)
            logger.info("Fetched %d/%d: %s", len(songs), count, record["title"])

        if len(songs) < count:
            logger.warning("Fetched %d of %d requested songs", len(songs), count)
        return songs


def _search_hits(response: dict[str, Any]) -> list[dict[str, Any]]:
    hits = response.get("hits")
    if isinstance(hits, list):
        return hits
    return [
        hit
        for section in response.get("sections", [])
        for hit in section.get("hits", [])
    ]


def crawl_random_to_file(
    count: int,
    output_path: Path,
    seed: int | None = None,
    max_searches: int = 30,
) -> list[dict[str, Any]]:
    if not settings.genius_access_token:
        raise RuntimeError(
            "Set GENIUS_ACCESS_TOKEN in your local .env file before crawling"
        )
    crawler = GeniusLyricsCrawler(settings.genius_access_token)
    songs = crawler.crawl_random_songs(count, seed, max_searches)
    if not songs:
        raise RuntimeError("Genius returned no songs with accessible lyrics")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(songs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return songs


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect random Genius songs")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-searches", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("data/raw/songs.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    songs = crawl_random_to_file(
        args.count, args.output, args.seed, args.max_searches
    )
    print(f"Saved {len(songs)} songs to {args.output}")


if __name__ == "__main__":
    main()
