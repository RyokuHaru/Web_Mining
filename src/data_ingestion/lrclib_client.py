"""Look up lyrics from LRCLIB by track metadata."""

from __future__ import annotations

import re
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from typing import Any

import requests

class LRCLibClient:
    BASE_URL = "https://lrclib.net/api"

    def __init__(
        self, delay: float = 0.35, timeout: float = 20, max_retries: int = 3
    ) -> None:
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MusicFlow/1.0 (academic project)"})
        self._last_request = 0.0

    def find_lyrics(
        self, title: str, artist: str, album: str = ""
    ) -> dict[str, Any] | None:
        query = " ".join(part for part in (artist, title) if part)
        response = self._get(f"{self.BASE_URL}/search", {"q": query})
        matches = response.json()
        if not isinstance(matches, list):
            return None

        best: dict[str, Any] | None = None
        best_score = 0.0
        for match in matches:
            if not isinstance(match, dict):
                continue
            title_score = _similarity(title, match.get("trackName", ""))
            artist_score = _similarity(artist, match.get("artistName", ""))
            if title_score < 0.86 or artist_score < 0.70:
                continue
            score = title_score * 0.65 + artist_score * 0.35
            if album and match.get("albumName"):
                score = (
                    title_score * 0.62
                    + artist_score * 0.33
                    + _similarity(album, match["albumName"]) * 0.05
                )
            if score > best_score:
                best = match
                best_score = score

        if best is None:
            return None

        lyrics = (best.get("plainLyrics") or "").strip()
        if not lyrics:
            lyrics = _plain_from_synced(best.get("syncedLyrics") or "")
        if not lyrics:
            return None

        return {**best, "lyrics": lyrics, "match_score": round(best_score, 3)}

    def _get(self, url: str, params: dict[str, str]) -> requests.Response:
        for attempt in range(self.max_retries + 1):
            wait = self.delay - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)

            response = self.session.get(url, params=params, timeout=self.timeout)
            self._last_request = time.monotonic()
            if response.status_code in (429, 503):
                retry_after = response.headers.get("Retry-After")
                if attempt < self.max_retries:
                    time.sleep(_retry_delay(retry_after))
                    continue
                raise RuntimeError(
                    f"LRCLIB returned HTTP {response.status_code} after "
                    f"{self.max_retries} retries; Retry-After: {retry_after or 'unspecified'}"
                )
            if response.status_code in (401, 403):
                raise RuntimeError(f"LRCLIB returned HTTP {response.status_code}")
            response.raise_for_status()
            return response
        raise RuntimeError("LRCLIB request failed after retries")


def _retry_delay(value: str | None) -> float:
    if not value:
        return 1.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 1.0


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"\b(feat(?:uring)?|ft)\b.*$", "", normalized, flags=re.I)
    return " ".join(re.findall(r"[a-z0-9]+", normalized.casefold()))


def _similarity(left: str, right: str) -> float:
    left_normalized = _normalize(left)
    right_normalized = _normalize(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def _plain_from_synced(synced: str) -> str:
    lines = []
    for line in synced.splitlines():
        if re.fullmatch(r"\s*\[[a-z]{2,8}:.*\]\s*", line, flags=re.I):
            continue
        text = re.sub(r"\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]", "", line).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)
