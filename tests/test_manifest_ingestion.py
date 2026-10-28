"""Manifest ingestion checks without live Genius or LRCLIB calls."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data_ingestion.genius_crawler import AmbiguousSongMatch, GeniusLyricsCrawler
from src.data_ingestion.lrclib_client import AmbiguousLyricsMatch, LRCLibClient
from src.data_ingestion.lyrics_pilot_matcher import crawl_manifest_to_file, load_manifest


class FakeCrawler:
    def __init__(self):
        self.calls = []

    def find_song(self, title, artist, search_query):
        self.calls.append((title, artist, search_query))
        if title == "Missing Genius":
            return None
        return {
            "id": len(self.calls), "title": title, "url": "https://genius.com/song",
            "primary_artist": {"name": artist},
        }

    def build_record(self, candidate):
        if candidate["title"] == "Missing Lyrics":
            return None
        return {
            "genius_id": str(candidate["id"]), "genius_url": candidate["url"],
            "title": candidate["title"],
            "artist": candidate["primary_artist"]["name"],
            "lyrics": "A real lyric text", "lyrics_source_id": "123",
            "lyrics_match_score": 1.0,
        }


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "pilot_id", "track_id", "artist_name", "track_name", "search_query",
            "lyrics_status", "content_status",
        ))
        writer.writeheader()
        writer.writerows(rows)


class ManifestIngestionTests(unittest.TestCase):
    def test_processes_only_manifest_rows_and_resumes_valid_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            songs_path = root / "songs.json"
            report_path = root / "report.csv"
            rows = [
                {"pilot_id": "P1", "track_id": "TRK_1", "artist_name": "Artist",
                 "track_name": "Found", "search_query": "Artist - Found",
                 "lyrics_status": "pending", "content_status": "pending"},
                {"pilot_id": "P2", "track_id": "TRK_2", "artist_name": "Artist",
                 "track_name": "Missing Genius", "search_query": "",
                 "lyrics_status": "pending", "content_status": "pending"},
                {"pilot_id": "P3", "track_id": "TRK_3", "artist_name": "Artist",
                 "track_name": "Missing Lyrics", "search_query": "",
                 "lyrics_status": "pending", "content_status": "pending"},
            ]
            write_manifest(manifest, rows)
            songs_path.write_text(json.dumps([{"song_id": "random", "lyrics": "old"}]))
            crawler = FakeCrawler()
            songs, report = crawl_manifest_to_file(
                manifest, songs_path, report_path, crawler=crawler
            )
            self.assertEqual([song["song_id"] for song in songs], ["TRK_1"])
            self.assertEqual(songs[0]["genius_id"], "1")
            self.assertEqual(songs[0]["manifest_title"], "Found")
            self.assertEqual([row["crawl_status"] for row in report], [
                "matched", "genius_not_found", "lyrics_not_found",
            ])
            self.assertEqual([row["status"] for row in report], [
                "success", "not_found", "not_found",
            ])
            self.assertEqual(songs[0]["status"], "success")
            with report_path.open(encoding="utf-8-sig") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
            self.assertEqual(len(json.loads(songs_path.read_text())), 1)

            next_crawler = FakeCrawler()
            songs, report = crawl_manifest_to_file(
                manifest, songs_path, report_path, crawler=next_crawler
            )
            self.assertEqual(len(songs), 1)
            self.assertEqual(len(next_crawler.calls), 0)
            self.assertEqual(report[0]["crawl_status"], "matched")

    def test_rejects_duplicate_track_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            row = {"track_id": "T1", "artist_name": "A", "track_name": "B"}
            write_manifest(path, [row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate track_id"):
                load_manifest(path)

    def test_one_lrclib_503_does_not_stop_other_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            write_manifest(manifest, [
                {"track_id": "T1", "artist_name": "A", "track_name": "First"},
                {"track_id": "T2", "artist_name": "A", "track_name": "Second"},
            ])

            class RecoveringCrawler(FakeCrawler):
                def build_record(self, candidate):
                    if candidate["title"] == "First":
                        raise RuntimeError("LRCLIB returned HTTP 503")
                    return super().build_record(candidate)

            songs, report = crawl_manifest_to_file(
                manifest, root / "songs.json", root / "report.csv",
                crawler=RecoveringCrawler(),
            )
            self.assertEqual([row["crawl_status"] for row in report], ["error", "matched"])
            self.assertEqual([song["song_id"] for song in songs], ["T2"])

    def test_error_is_not_retried_after_attempt_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            songs_path = root / "songs.json"
            report_path = root / "report.csv"
            write_manifest(manifest, [
                {"track_id": "T1", "artist_name": "A", "track_name": "First"},
            ])

            class FailingCrawler(FakeCrawler):
                def find_song(self, title, artist, search_query):
                    self.calls.append(title)
                    raise RuntimeError("LRCLIB returned HTTP 503")

            crawler = FailingCrawler()
            for expected_attempts in (1, 2):
                _, report = crawl_manifest_to_file(
                    manifest, songs_path, report_path, crawler=crawler,
                    max_attempts=2,
                )
                self.assertEqual(report[0]["status"], "error")
                self.assertEqual(report[0]["attempt_count"], expected_attempts)
            _, report = crawl_manifest_to_file(
                manifest, songs_path, report_path, crawler=crawler,
                max_attempts=2,
            )
            self.assertEqual(len(crawler.calls), 2)
            self.assertTrue(report[0]["retry_exhausted"])

    def test_completed_row_is_checkpointed_before_next_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            songs_path = root / "songs.json"
            report_path = root / "report.csv"
            write_manifest(manifest, [
                {"track_id": "T1", "artist_name": "A", "track_name": "First"},
                {"track_id": "T2", "artist_name": "A", "track_name": "Second"},
            ])
            observed = []

            class InspectingCrawler(FakeCrawler):
                def find_song(self, title, artist, search_query):
                    if title == "Second":
                        with report_path.open(encoding="utf-8-sig") as stream:
                            statuses = [row["status"] for row in csv.DictReader(stream)]
                        song_ids = [row["song_id"] for row in json.loads(songs_path.read_text())]
                        observed.append((statuses, song_ids))
                    return super().find_song(title, artist, search_query)

            crawl_manifest_to_file(
                manifest, songs_path, report_path, crawler=InspectingCrawler()
            )
            self.assertEqual(observed, [(["success", "pending"], ["T1"])])

    def test_genius_search_rejects_wrong_artist_and_uses_best_match(self):
        crawler = GeniusLyricsCrawler.__new__(GeniusLyricsCrawler)

        class API:
            def search_songs(self, query, per_page):
                self.query = query
                return {"hits": [
                    {"result": {"id": 1, "title": "Live Forever",
                                "primary_artist": {"name": "Other Band"}, "url": "u1"}},
                    {"result": {"id": 2, "title": "Live Forever",
                                "primary_artist": {"name": "Oasis"}, "url": "u2"}},
                ]}

        crawler.client = API()
        self.assertEqual(crawler.find_song("Live Forever", "Oasis", "Oasis - Live Forever")["id"], 2)
        self.assertEqual(crawler.client.query, "Oasis - Live Forever")

    def test_conflicting_genius_hits_are_ambiguous(self):
        crawler = GeniusLyricsCrawler.__new__(GeniusLyricsCrawler)

        class API:
            def search_songs(self, query, per_page):
                return {"hits": [
                    {"result": {"id": identifier, "title": title,
                                "primary_artist": {"name": "Artist"}, "url": f"u{identifier}"}}
                    for identifier, title in ((1, "Same Songs"), (2, "Same Songz"))
                ]}

        crawler.client = API()
        with self.assertRaises(AmbiguousSongMatch):
            crawler.find_song("Same Song", "Artist")

    def test_conflicting_lrclib_lyrics_are_ambiguous(self):
        client = LRCLibClient.__new__(LRCLibClient)

        class Response:
            def json(self):
                return [
                    {"id": 1, "trackName": "Same Song", "artistName": "Artist",
                     "plainLyrics": "First lyric"},
                    {"id": 2, "trackName": "Same Song", "artistName": "Artist",
                     "plainLyrics": "Other lyric"},
                ]

        client._get = lambda url, params: Response()
        with self.assertRaises(AmbiguousLyricsMatch):
            client.find_lyrics("Same Song", "Artist")

    def test_minor_transcription_difference_is_not_ambiguous(self):
        client = LRCLibClient.__new__(LRCLibClient)

        class Response:
            def json(self):
                return [
                    {"id": 1, "trackName": "Same Song", "artistName": "Artist",
                     "plainLyrics": "This is the same lyric, line one and line two"},
                    {"id": 2, "trackName": "Same Song", "artistName": "Artist",
                     "plainLyrics": "This is the same lyric line one and line two"},
                ]

        client._get = lambda url, params: Response()
        self.assertEqual(client.find_lyrics("Same Song", "Artist")["id"], 1)

    def test_shorter_copy_of_same_lyrics_is_not_ambiguous(self):
        client = LRCLibClient.__new__(LRCLibClient)

        class Response:
            def json(self):
                return [
                    {"id": 1, "trackName": "Same Song", "artistName": "Artist",
                     "plainLyrics": "One two three four. Five six seven eight. Nine ten."},
                    {"id": 2, "trackName": "Same Song", "artistName": "Artist",
                     "plainLyrics": "One two three four. Five six seven eight."},
                ]

        client._get = lambda url, params: Response()
        self.assertEqual(client.find_lyrics("Same Song", "Artist")["id"], 1)


if __name__ == "__main__":
    unittest.main()
