"""Find Genius song metadata and match lyrics from LRCLIB."""

from __future__ import annotations

from typing import Any

from src.data_ingestion.lrclib_client import LRCLibClient, _normalize, _similarity


class AmbiguousSongMatch(ValueError):
    """More than one Genius listing matches the requested song equally well."""


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

    def find_song(
        self, title: str, artist: str, search_query: str = ""
    ) -> dict[str, Any] | None:
        """Return the best Genius hit matching both the requested title and artist."""
        response = self.client.search_songs(
            search_query.strip() or f"{artist} {title}", per_page=10
        )
        candidates = [hit.get("result", hit) for hit in _search_hits(response)]
        scored = []
        for song in candidates:
            if not isinstance(song, dict) or not song.get("id") or not song.get("url"):
                continue
            found_artist = (song.get("primary_artist") or {}).get("name") or song.get(
                "artist_names", ""
            )
            title_score = _similarity(title, song.get("title", ""))
            artist_score = _similarity(artist, found_artist)
            score = title_score * 0.65 + artist_score * 0.35
            if title_score >= 0.86 and artist_score >= 0.70 and score >= 0.85:
                scored.append((score, song))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored:
            return None
        if (
            len(scored) > 1
            and scored[0][0] - scored[1][0] < 0.03
            and str(scored[0][1]["id"]) != str(scored[1][1]["id"])
            and (
                _normalize(scored[0][1].get("title", ""))
                != _normalize(scored[1][1].get("title", ""))
                or _normalize((scored[0][1].get("primary_artist") or {}).get("name", ""))
                != _normalize((scored[1][1].get("primary_artist") or {}).get("name", ""))
            )
        ):
            raise AmbiguousSongMatch(
                f"Conflicting Genius IDs {scored[0][1]['id']} and {scored[1][1]['id']}"
            )
        return {**scored[0][1], "_match_score": round(scored[0][0], 3)}

    def get_song(self, title: str, artist: str) -> dict[str, Any] | None:
        """Find a Genius listing and its matching LRCLIB lyrics."""
        candidate = self.find_song(title, artist)
        return self.build_record(candidate) if candidate is not None else None

    def build_record(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        title = candidate.get("title") or candidate.get("full_title", "")
        primary_artist = candidate.get("primary_artist") or {}
        artist = primary_artist.get("name") or candidate.get("artist_names", "")
        album = candidate.get("album") or {}
        album_name = album.get("name") or album.get("full_title", "")
        match = self.lyrics_client.find_lyrics(title, artist, album_name)
        if match is None:
            return None

        genius_id = str(candidate["id"])
        genius_score = candidate.get("_match_score", 1.0)
        primary_tag = candidate.get("primary_tag") or {}
        record: dict[str, Any] = {
            "genius_id": genius_id,
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
            "genius_match_score": genius_score,
            "match_confidence": min(genius_score, match["match_score"]),
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
