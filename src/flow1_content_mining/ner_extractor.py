"""Named-entity extraction for lyrics and their structured metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


class NERExtractor:
    """Extract lyric entities and merge structured song metadata."""

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        genre_terms: Iterable[str] | None = None,
    ) -> None:
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError("Install spaCy with `pip install spacy`.") from exc

        try:
            self.nlp = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model {model_name!r} is missing. Install it with "
                f"`python -m spacy download {model_name}`."
            ) from exc

        terms = {term.strip() for term in (genre_terms or ()) if term.strip()}
        self.genre_terms = tuple(sorted(terms, key=len, reverse=True))

    def extract(
        self, lyrics: str, metadata: Mapping[str, Any] | None = None
    ) -> dict[str, list[str]]:
        metadata = metadata or {}
        doc = self.nlp(lyrics or "")

        person_candidates = _unique(
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ in {"PERSON", "ORG"} and ent.text.strip()
        )
        lyric_genres = [
            term
            for term in self.genre_terms
            if re.search(
                rf"(?<!\w){re.escape(term)}(?!\w)", lyrics or "", re.IGNORECASE
            )
        ]

        return {
            "artist": _as_list(_first(metadata, "artist", "artists")),
            "writers": _as_list(
                _first(metadata, "writers", "writer", "songwriters", "author")
            ),
            "genres": _unique(
                [*_as_list(_first(metadata, "genres", "genre")), *lyric_genres]
            ),
            "entity_candidates": person_candidates,
        }


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", []):
            return value
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, Iterable):
        values = list(value)
    else:
        values = [value]
    return _unique(str(item).strip() for item in values if str(item).strip())


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
