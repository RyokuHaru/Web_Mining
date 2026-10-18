"""Score YouTube comments with a RoBERTa emotion classifier."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


class RobertaSentimentAnalyzer:
    POSITIVE = {
        "positive", "pos", "joy", "happy", "happiness", "enjoyment",
        "admiration", "amusement", "approval", "caring", "excitement",
        "gratitude", "love", "optimism", "pride", "relief",
    }
    NEGATIVE = {
        "negative", "neg", "sad", "sadness", "anger", "angry", "annoyance",
        "disappointment", "disapproval", "disgust", "fear", "grief", "nervousness",
        "remorse",
    }

    def __init__(
        self,
        model_name: str = "BaoNhan/xlm-roberta-base-UITVSMEC",
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        try:
            from transformers import AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError("Install transformers to analyze comment sentiment") from exc

        self.model_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=tokenizer,
            top_k=None,
        )
        raw_labels = getattr(self.classifier.model.config, "id2label", {}) or {}
        self.id2label = {}
        for key, value in raw_labels.items():
            label = str(value).casefold().replace(" ", "_")
            self.id2label[str(key).casefold()] = label
            self.id2label[f"label_{key}".casefold()] = label
        if "uitvsmec" in model_name.casefold():
            for index, label in enumerate(
                ("anger", "disgust", "enjoyment", "fear", "other", "sadness", "surprise")
            ):
                self.id2label[f"label_{index}"] = label
                self.id2label[str(index)] = label
        self.batch_size = batch_size
        self.max_length = 256

    def analyze(self, comments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        valid = [
            (index, str(row.get("text", "")).strip())
            for index, row in enumerate(comments)
            if str(row.get("text", "")).strip()
        ]
        if not valid:
            return []

        indices, texts = zip(*valid)
        predictions = self.classifier(
            list(texts),
            batch_size=self.batch_size,
            truncation=True,
            max_length=self.max_length,
        )
        results: list[dict[str, Any]] = []
        for index, prediction in zip(indices, predictions):
            scores = _score_map(prediction, self.id2label)
            label = max(scores, key=scores.get)
            results.append(
                {
                    **dict(comments[index]),
                    "sentiment": label,
                    "sentiment_confidence": scores[label],
                    "sentiment_scores": scores,
                    "valence": _valence(scores),
                    "sentiment_model": self.model_name,
                }
            )
        return results

    def summarize(
        self,
        comments: Sequence[Mapping[str, Any]],
        videos: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in comments:
            video_id = str(row.get("video_id") or "")
            grouped[video_id].append(row)

        video_by_id = {
            str(video.get("video_id")): video
            for video in videos
            if video.get("video_id") not in (None, "")
        }

        summaries = []
        for video_id in dict.fromkeys([*video_by_id, *grouped]):
            rows = grouped.get(video_id, [])
            counts = Counter(str(row.get("sentiment", "unknown")) for row in rows)
            vals = [
                float(row["valence"])
                for row in rows
                if row.get("valence") is not None
            ]
            video = video_by_id.get(video_id, {})
            summaries.append(
                {
                    "video_id": video_id,
                    "song_id": video.get("song_id")
                    or next((row.get("song_id") for row in rows if row.get("song_id")), None),
                    "title": video.get("title", ""),
                    "comment_count": len(rows),
                    "social_valence_score": sum(vals) / len(vals) if vals else None,
                    "sentiment_counts": dict(counts),
                    "sentiment_model": self.model_name,
                }
            )
        return summaries


def _score_map(prediction: Any, id2label: Mapping[str, str]) -> dict[str, float]:
    if isinstance(prediction, Mapping):
        prediction = [prediction]
    if prediction and isinstance(prediction[0], list):
        prediction = prediction[0]
    scores = {}
    for item in prediction:
        raw_label = str(item["label"]).casefold().replace(" ", "_")
        label = id2label.get(raw_label, raw_label)
        scores[label] = float(item["score"])
    if not scores:
        raise ValueError("The sentiment model returned no label scores")
    return scores


def _valence(scores: Mapping[str, float]) -> float:
    positive = sum(
        score
        for label, score in scores.items()
        if label.casefold() in RobertaSentimentAnalyzer.POSITIVE
    )
    negative = sum(
        score
        for label, score in scores.items()
        if label.casefold() in RobertaSentimentAnalyzer.NEGATIVE
    )
    return max(-1.0, min(1.0, positive - negative))
