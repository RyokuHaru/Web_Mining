"""Sample Genius songs and fetch matching lyrics from LRCLIB."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

from config import settings
from src.data_ingestion.lrclib_client import LRCLibClient, _similarity

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
            raise ValueError("A Genius access token is required for song search")
        try:
            import lyricsgenius
        except ImportError as exc:
            raise RuntimeError("Install lyricsgenius to use the Genius connector") from exc
        self.client = lyricsgenius.API(
            access_token,
            sleep_time=sleep_time,
            retries=2,
        )
        self.lyrics_client = LRCLibClient()

    def get_song(self, title: str, artist: str) -> dict[str, Any] | None:
        """Find a Genius listing and its matching LRCLIB lyrics."""
        response = self.client.search_songs(f"{artist} {title}", per_page=5)
        candidates = [hit.get("result", hit) for hit in _search_hits(response)]
        candidate = next(
            (
                song
                for song in candidates
                if _similarity(title, song.get("title", "")) >= 0.86
                and _similarity(artist, song.get("artist_names", "")) >= 0.70
            ),
            None,
        )
        if candidate is None:
            return None
        return self._build_record(candidate)

    def crawl_random_songs(
        self,
        count: int = 100,
        seed: int | None = None,
        max_searches: int = 200,
        exclude_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if count < 1:
            raise ValueError("count must be positive")
        if max_searches < 1:
            raise ValueError("max_searches must be positive")

        rng = random.Random(seed)
        exclude_ids = exclude_ids or set()
        terms = list(RANDOM_SEARCH_TERMS)
        rng.shuffle(terms)
        candidates: dict[str, dict[str, Any]] = {}

        for search_number in range(max_searches):
            term = terms[search_number % len(terms)]
            page = rng.randint(1, 20)
            try:
                response = self.client.search_songs(term, per_page=5, page=page)
            except Exception as exc:
                logger.warning("Genius search failed for %r: %s", term, _format_error(exc))
                if _error_status(exc) in {401, 403, 429}:
                    break
                continue

            for hit in _search_hits(response):
                song = hit.get("result", hit)
                song_id = song.get("id")
                if (
                    not song_id
                    or str(song_id) in exclude_ids
                    or not song.get("url")
                    or song.get("lyrics_state") != "complete"
                    or song.get("instrumental")
                ):
                    continue
                candidates.setdefault(str(song_id), song)
            if len(candidates) >= count * 5:
                break

        candidate_rows = list(candidates.values())
        rng.shuffle(candidate_rows)
        songs: list[dict[str, Any]] = []
        consecutive_service_errors = 0
        for candidate in candidate_rows:
            if len(songs) >= count:
                break
            try:
                record = self._build_record(candidate)
            except RuntimeError as exc:
                if "HTTP 503" in str(exc) or "HTTP 429" in str(exc):
                    consecutive_service_errors += 1
                    logger.warning(
                        "Skipping LRCLIB lookup (%d consecutive service errors): %s",
                        consecutive_service_errors,
                        exc,
                    )
                    if consecutive_service_errors >= 3:
                        logger.warning("LRCLIB appears unavailable; stopping lookups")
                        break
                    continue
                logger.warning("LRCLIB lookup failed: %s", exc)
                continue
            except Exception as exc:
                logger.warning("LRCLIB lookup failed for %s: %s", candidate.get("url"), exc)
                continue
            consecutive_service_errors = 0
            if record is None:
                logger.info("No close LRCLIB match: %s", candidate.get("full_title"))
                continue
            songs.append(record)
            logger.info("Fetched %d/%d: %s", len(songs), count, record["title"])

        if len(songs) < count:
            logger.warning("Fetched %d of %d requested songs", len(songs), count)
        return songs

    def _build_record(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        title = candidate.get("title") or candidate.get("full_title", "")
        primary_artist = candidate.get("primary_artist") or {}
        artist = primary_artist.get("name") or candidate.get("artist_names", "")
        album = candidate.get("album") or {}
        album_name = album.get("name") or album.get("full_title", "")
        match = self.lyrics_client.find_lyrics(title, artist, album_name)
        if match is None:
            return None

        song_id = str(candidate["id"])
        primary_tag = candidate.get("primary_tag") or {}
        record: dict[str, Any] = {
            "song_id": song_id,
            "genius_id": song_id,
            "title": title,
            "artist": artist,
            "lyrics": match["lyrics"],
            "source": "Genius",
            "metadata_source": "Genius",
            "source_url": candidate["url"],
            "genius_url": candidate["url"],
            "lyrics_source": "LRCLIB",
            "lyrics_source_id": str(match.get("id", "")),
            "lyrics_source_url": f"https://lrclib.net/api/get/{match.get('id', '')}",
            "lyrics_match_score": match["match_score"],
        }
        if album_name:
            record["album"] = album_name
        release_date = candidate.get("release_date_for_display")
        if release_date:
            record["release_date"] = release_date
        if primary_tag.get("name"):
            record["genres"] = [primary_tag["name"]]
        return record


def _search_hits(response: dict[str, Any]) -> list[dict[str, Any]]:
    hits = response.get("hits")
    if isinstance(hits, list):
        return hits
    return [
        hit
        for section in response.get("sections", [])
        for hit in section.get("hits", [])
    ]


def _error_status(exc: Exception) -> int | None:
    match = re.search(r"(?:status code:|HTTPError:)\s*(\d{3})", str(exc))
    if not match:
        match = re.search(r"\b(401|403|429)\b", str(exc))
    return int(match.group(1)) if match else None


def _format_error(exc: Exception) -> str:
    message = str(exc).split("Response body:", 1)[0].strip()
    return f"{type(exc).__name__}: {message[:180]}"


def crawl_random_to_file(
    count: int,
    output_path: Path,
    seed: int | None = None,
    max_searches: int = 200,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    if not settings.genius_access_token:
        raise RuntimeError(
            "Set GENIUS_ACCESS_TOKEN in your local .env file to search Genius"
        )
    existing: list[dict[str, Any]] = []
    if output_path.exists():
        try:
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = [
                    row
                    for row in loaded
                    if isinstance(row, dict)
                    and row.get("song_id")
                    and isinstance(row.get("lyrics"), str)
                    and row["lyrics"].strip()
                ]
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read existing dataset; starting a new sample")

    unique_existing = {str(row["song_id"]): row for row in existing}
    songs = list(unique_existing.values())[:count]
    remaining = count - len(songs)
    if remaining > 0:
        crawler = GeniusLyricsCrawler(settings.genius_access_token)
        fetched = crawler.crawl_random_songs(
            remaining,
            seed,
            max_searches,
            exclude_ids=set(unique_existing),
        )
        for row in fetched:
            unique_existing.setdefault(str(row["song_id"]), row)
        songs = list(unique_existing.values())[:count]
    if not songs:
        raise RuntimeError(
            "No songs matched between Genius and LRCLIB. Check network access and LRCLIB coverage."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(songs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return songs


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect random Genius songs")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-searches", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("data/raw/songs.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    songs = crawl_random_to_file(
        args.count, args.output, args.seed, args.max_searches
    )
    print(f"Saved {len(songs)} songs to {args.output}")


if __name__ == "__main__":
    main()
