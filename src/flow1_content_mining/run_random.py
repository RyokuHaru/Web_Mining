"""Crawl a random Genius sample and run the Flow 1 pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data_ingestion.genius_crawler import crawl_random_to_file
from src.flow1_content_mining.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl random Genius songs and create Flow 1 features"
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-searches", type=int, default=30)
    parser.add_argument("--raw-output", type=Path, default=Path("data/raw/songs.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    songs = crawl_random_to_file(
        args.count, args.raw_output, args.seed, args.max_searches
    )
    print(f"Collected {len(songs)} songs; starting Flow 1")
    run(args.raw_output, args.output)
    print(f"Flow 1 artifacts saved to {args.output}")


if __name__ == "__main__":
    main()
