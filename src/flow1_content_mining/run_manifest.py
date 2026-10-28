"""Crawl songs from the pilot manifest and run the Flow 1 pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data_ingestion.lyrics_pilot_matcher import crawl_manifest_to_file
from src.flow1_content_mining.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl the CSV manifest and create Flow 1 features"
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw/lyrics_pilot_manifest.csv"))
    parser.add_argument("--raw-output", type=Path, default=Path("data/raw/songs.json"))
    parser.add_argument("--report", type=Path, default=Path("data/processed/lyrics_pilot_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit", type=int, help="Attempt at most this many uncached rows")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-ambiguous", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    songs, results = crawl_manifest_to_file(
        args.input, args.raw_output, args.report, limit=args.limit,
        max_attempts=args.max_attempts, retry_failed=args.retry_failed,
        retry_ambiguous=args.retry_ambiguous,
    )
    matched = sum(row["status"] == "success" for row in results)
    print(f"Collected {matched}/{len(results)} manifest songs with lyrics")
    if not songs:
        print(f"No lyrics available for Flow 1; inspect {args.report}")
        return
    run(args.raw_output, args.output)
    print(f"Flow 1 artifacts saved to {args.output}")


if __name__ == "__main__":
    main()
