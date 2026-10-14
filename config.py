"""Environment-backed configuration for Music Flow."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    spotify_client_id: str | None = os.getenv("SPOTIFY_CLIENT_ID")
    spotify_client_secret: str | None = os.getenv("SPOTIFY_CLIENT_SECRET")
    genius_access_token: str | None = os.getenv("GENIUS_ACCESS_TOKEN")
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_database: str = os.getenv("MONGODB_DATABASE", "music_flow")
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    sbert_model: str = os.getenv(
        "SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    sentiment_model: str = os.getenv(
        "HF_SENTIMENT_MODEL", "BaoNhan/xlm-roberta-base-UITVSMEC"
    )


settings = Settings()
