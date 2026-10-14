"""Run Flow 2 sentiment analysis on collected YouTube comments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import settings
from src.flow2_social_mining.sentiment_analyzer import RobertaSentimentAnalyzer


def load_comments(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix.casefold() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("comments", []) if isinstance(payload, dict) else payload
    else:
        raise ValueError("Comments input must be .jsonl or .json")
    if not isinstance(rows, list):
        raise ValueError("Comments input must contain a list of comment objects")
    return rows


def run(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    comments = load_comments(input_path)
    analyzer = RobertaSentimentAnalyzer(settings.sentiment_model)
    scored = analyzer.analyze(comments)
    summary = analyzer.summarize(scored)

    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "youtube_comments_scored.jsonl"
    summary_path = output_dir / "social_valence.json"
    scored_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return scored_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze YouTube comment sentiment")
    parser.add_argument(
        "--input", type=Path, default=Path("data/raw/youtube_comments.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    scored_path, summary_path = run(args.input, args.output_dir)
    print(f"Saved scored comments to {scored_path}")
    print(f"Saved video sentiment to {summary_path}")


if __name__ == "__main__":
    main()
