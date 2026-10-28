"""Collect Genius metadata and LRCLIB lyrics for a fixed song manifest."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from config import settings
from src.data_ingestion.genius_crawler import AmbiguousSongMatch, GeniusLyricsCrawler
from src.data_ingestion.lrclib_client import AmbiguousLyricsMatch, _similarity


logger = logging.getLogger(__name__)
REQUIRED_FIELDS = ("track_id", "artist_name", "track_name")


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(field not in reader.fieldnames for field in REQUIRED_FIELDS):
            raise ValueError(f"Manifest must contain: {', '.join(REQUIRED_FIELDS)}")
        rows = []
        ids = set()
        for line_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"Unexpected extra CSV columns at line {line_number}")
            row = {key: (value or "").strip() for key, value in raw.items()}
            if any(not row[field] for field in REQUIRED_FIELDS):
                raise ValueError(f"Missing track_id, artist_name or track_name at line {line_number}")
            if row["track_id"] in ids:
                raise ValueError(f"Duplicate track_id at line {line_number}: {row['track_id']}")
            ids.add(row["track_id"])
            rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def _existing_songs(path: Path, manifest: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Existing songs file must contain a JSON list: {path}")
    expected = {row["track_id"]: row for row in manifest}
    cached = {}
    for song in payload:
        if not isinstance(song, dict):
            continue
        song_id = str(song.get("song_id", ""))
        row = expected.get(song_id)
        if (
            row
            and song.get("track_id") == song_id
            and song.get("manifest_artist") == row["artist_name"]
            and song.get("manifest_title") == row["track_name"]
            and song.get("match_policy_version") == 2
            and isinstance(song.get("lyrics"), str)
            and song["lyrics"].strip()
        ):
            genius_score = song.get("genius_match_score") or (
                _similarity(row["track_name"], song.get("title", "")) * 0.65
                + _similarity(row["artist_name"], song.get("artist", "")) * 0.35
            )
            lyrics_score = song.get("lyrics_match_score", 0)
            if min(float(genius_score), float(lyrics_score)) >= 0.85:
                cached[song_id] = {
                    **song, "genius_match_score": round(float(genius_score), 3),
                    "match_confidence": round(min(float(genius_score), float(lyrics_score)), 3),
                    "status": "success", "lyrics_status": "success",
                }
    return cached


def _existing_report(path: Path, manifest: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    expected = {row["track_id"]: row for row in manifest}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            report["track_id"]: report
            for report in csv.DictReader(stream)
            if report.get("track_id") in expected
            and report.get("artist_name") == expected[report["track_id"]]["artist_name"]
            and report.get("track_name") == expected[report["track_id"]]["track_name"]
        }


def _attempt_count(report: dict[str, str]) -> int:
    try:
        return max(0, int(report["attempt_count"]))
    except (KeyError, TypeError, ValueError):
        return int(report.get("crawl_status", "pending") != "pending")


def _saved_report_row(
    row: dict[str, str], prior: dict[str, str], max_attempts: int
) -> dict[str, Any]:
    result = _report_row(
        row, prior.get("crawl_status") or prior.get("status", "pending"),
        error=prior.get("error", ""), attempts=_attempt_count(prior),
        max_attempts=max_attempts,
    )
    for field in ("genius_id", "genius_url", "genius_title", "genius_artist",
                  "lyrics_source_id", "lyrics_match_score", "genius_match_score",
                  "match_confidence"):
        result[field] = prior.get(field, "")
    return result


def _report_row(
    row: dict[str, str], status: str, candidate: dict[str, Any] | None = None,
    song: dict[str, Any] | None = None, error: str = "", attempts: int = 0,
    max_attempts: int = 3,
) -> dict[str, Any]:
    candidate = candidate or {}
    song = song or {}
    artist = (candidate.get("primary_artist") or {}).get("name") or candidate.get(
        "artist_names", ""
    )
    standardized = {
        "matched": "success", "lyrics_not_found": "not_found",
        "genius_not_found": "not_found", "ambiguous_genius": "ambiguous",
        "ambiguous_lyrics": "ambiguous", "error": "error",
        "success": "success", "not_found": "not_found", "ambiguous": "ambiguous",
    }.get(status, "pending")
    return {
        **row,
        "lyrics_status": standardized,
        "status": standardized,
        "crawl_status": status,
        "attempt_count": attempts,
        "retry_exhausted": standardized == "error" and attempts >= max_attempts,
        "genius_id": song.get("genius_id") or candidate.get("id", ""),
        "genius_url": song.get("genius_url") or candidate.get("url", ""),
        "genius_title": song.get("title") or candidate.get("title", ""),
        "genius_artist": song.get("artist") or artist,
        "lyrics_source_id": song.get("lyrics_source_id", ""),
        "lyrics_match_score": song.get("lyrics_match_score", ""),
        "genius_match_score": song.get("genius_match_score") or candidate.get("_match_score", ""),
        "match_confidence": song.get("match_confidence", ""),
        "error": error,
    }


def _write_outputs(
    output_path: Path, report_path: Path, songs: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    song_temp = output_path.with_suffix(output_path.suffix + ".tmp")
    report_temp = report_path.with_suffix(report_path.suffix + ".tmp")
    song_temp.write_text(json.dumps(songs, ensure_ascii=False, indent=2), encoding="utf-8")
    song_temp.replace(output_path)
    with report_temp.open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    report_temp.replace(report_path)


def crawl_manifest_to_file(
    input_path: Path,
    output_path: Path = Path("data/raw/songs.json"),
    report_path: Path = Path("data/processed/lyrics_pilot_results.csv"),
    *,
    limit: int | None = None,
    max_attempts: int = 3,
    retry_failed: bool = False,
    retry_ambiguous: bool = False,
    crawler: GeniusLyricsCrawler | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process manifest rows in order, resuming only verified lyric matches."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    paths = (input_path.resolve(), output_path.resolve(), report_path.resolve())
    if len(set(paths)) != len(paths):
        raise ValueError("Manifest, songs and report paths must differ")
    manifest = load_manifest(input_path)
    cached = _existing_songs(output_path, manifest)
    previous_report = _existing_report(report_path, manifest)
    needs_lookup = any(
        row["track_id"] not in cached
        and (
            retry_failed
            or retry_ambiguous and previous_report.get(row["track_id"], {}).get("status") == "ambiguous"
            or previous_report.get(row["track_id"], {}).get("status")
            not in {"not_found", "ambiguous"}
            and (
                previous_report.get(row["track_id"], {}).get("status") == "success"
                or _attempt_count(previous_report.get(row["track_id"], {})) < max_attempts
            )
        )
        for row in manifest
    )
    if crawler is None and not settings.genius_access_token and needs_lookup:
        raise RuntimeError("Set GENIUS_ACCESS_TOKEN in .env to search Genius")
    songs_by_id = dict(cached)
    results = []
    service_error = ""
    consecutive_service_errors = 0
    processed = 0

    def complete_report() -> list[dict[str, Any]]:
        snapshot = list(results)
        completed_ids = {result["track_id"] for result in results}
        for remaining in manifest:
            if remaining["track_id"] in completed_ids:
                continue
            prior = previous_report.get(remaining["track_id"], {})
            prior_status = prior.get("status") or {
                "genius_not_found": "not_found", "lyrics_not_found": "not_found",
            }.get(prior.get("crawl_status"), "pending")
            if prior_status in {"not_found", "ambiguous", "error"}:
                snapshot.append(_saved_report_row(remaining, prior, max_attempts))
            else:
                snapshot.append(_report_row(
                    remaining, "pending", attempts=_attempt_count(prior),
                    max_attempts=max_attempts,
                ))
        return snapshot

    def append_result(result: dict[str, Any], *, checkpoint: bool = False) -> None:
        results.append(result)
        if checkpoint:
            ordered = [songs_by_id[item["track_id"]] for item in manifest
                       if item["track_id"] in songs_by_id]
            _write_outputs(output_path, report_path, ordered, complete_report())

    try:
        for row in manifest:
            song_id = row["track_id"]
            prior = previous_report.get(song_id, {})
            attempts = _attempt_count(prior)
            if song_id in cached:
                append_result(_report_row(
                    row, "matched", song=cached[song_id],
                    attempts=max(1, attempts), max_attempts=max_attempts,
                ))
                continue
            previous_status = prior.get("status")
            if not previous_status:
                previous_status = {
                    "genius_not_found": "not_found", "lyrics_not_found": "not_found",
                    "ambiguous_genius": "ambiguous", "ambiguous_lyrics": "ambiguous",
                }.get(prior.get("crawl_status"), "error" if prior.get("crawl_status") == "error" else "pending")
            if (
                previous_status == "not_found" and not retry_failed
                or previous_status == "ambiguous" and not (retry_failed or retry_ambiguous)
                or previous_status == "error" and attempts >= max_attempts and not retry_failed
            ):
                append_result(_saved_report_row(row, prior, max_attempts))
                continue
            if service_error or (limit is not None and processed >= limit):
                append_result(_report_row(
                    row, "pending", error=service_error, attempts=attempts,
                    max_attempts=max_attempts,
                ))
                continue
            processed += 1
            attempts += 1
            if crawler is None:
                crawler = GeniusLyricsCrawler(settings.genius_access_token)
            candidate = None
            try:
                candidate = crawler.find_song(
                    row["track_name"], row["artist_name"], row.get("search_query", "")
                )
                if candidate is None:
                    consecutive_service_errors = 0
                    append_result(_report_row(
                        row, "genius_not_found", attempts=attempts,
                        max_attempts=max_attempts,
                    ), checkpoint=True)
                    continue
                record = crawler.build_record(candidate)
                consecutive_service_errors = 0
                if record is None:
                    append_result(_report_row(
                        row, "lyrics_not_found", candidate, attempts=attempts,
                        max_attempts=max_attempts,
                    ), checkpoint=True)
                    continue
                record.update({
                    "song_id": song_id,
                    "track_id": song_id,
                    "status": "success",
                    "lyrics_status": "success",
                    "match_policy_version": 2,
                    "manifest_artist": row["artist_name"],
                    "manifest_title": row["track_name"],
                    **{key: value for key, value in row.items()
                       if key not in {"track_id", "artist_name", "track_name",
                                      "lyrics_status", "content_status", "notes"}},
                })
                songs_by_id[song_id] = record
                append_result(_report_row(
                    row, "matched", candidate, record, attempts=attempts,
                    max_attempts=max_attempts,
                ), checkpoint=True)
                logger.info("Matched %s: %s - %s", song_id, row["artist_name"], row["track_name"])
            except (AmbiguousSongMatch, AmbiguousLyricsMatch) as exc:
                consecutive_service_errors = 0
                kind = "ambiguous_genius" if isinstance(exc, AmbiguousSongMatch) else "ambiguous_lyrics"
                append_result(_report_row(
                    row, kind, candidate, error=str(exc), attempts=attempts,
                    max_attempts=max_attempts,
                ), checkpoint=True)
                logger.warning("Ambiguous %s: %s", song_id, exc)
            except Exception as exc:
                message = f"{type(exc).__name__}: {str(exc)[:180]}"
                logger.warning("Failed %s: %s", song_id, message)
                append_result(_report_row(
                    row, "error", candidate, error=message, attempts=attempts,
                    max_attempts=max_attempts,
                ), checkpoint=True)
                if re.search(r"\b(401|403|429)\b", str(exc)):
                    service_error = f"Stopped after service error at {song_id}"
                elif re.search(r"\b503\b", str(exc)):
                    consecutive_service_errors += 1
                    if consecutive_service_errors >= 3:
                        service_error = "Stopped after three consecutive LRCLIB HTTP 503 errors"
                else:
                    consecutive_service_errors = 0
    finally:
        results = complete_report()
        ordered_songs = [songs_by_id[row["track_id"]] for row in manifest
                         if row["track_id"] in songs_by_id]
        _write_outputs(output_path, report_path, ordered_songs, results)
    return ordered_songs, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect lyrics for songs in a CSV manifest")
    parser.add_argument("--input", type=Path, default=Path("data/raw/lyrics_pilot_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/songs.json"))
    parser.add_argument("--report", type=Path, default=Path("data/processed/lyrics_pilot_results.csv"))
    parser.add_argument("--limit", type=int, help="Attempt at most this many uncached rows")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Maximum attempts for API errors (default: 3)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Explicitly retry terminal statuses and exhausted errors")
    parser.add_argument("--retry-ambiguous", action="store_true",
                        help="Recheck ambiguous rows without retrying not_found rows")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    songs, results = crawl_manifest_to_file(
        args.input, args.output, args.report, limit=args.limit,
        max_attempts=args.max_attempts, retry_failed=args.retry_failed,
        retry_ambiguous=args.retry_ambiguous,
    )
    print(f"Saved {len(songs)} songs with lyrics to {args.output}")
    print(f"Saved {len(results)} manifest statuses to {args.report}")


if __name__ == "__main__":
    main()
