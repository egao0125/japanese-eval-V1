"""Configuration via Pydantic BaseSettings — loads from .env and environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # API keys
    deepgram_api_key: str = Field(default="", description="Deepgram API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    google_application_credentials: str = Field(
        default="", description="Path to Google Cloud credentials JSON"
    )

    # Paths
    results_dir: Path = Field(default=Path("results"), description="Directory for evaluation results")
    corpus_base_dir: Path = Field(default=Path("corpora"), description="Base directory for corpus data")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Get application settings (singleton-like via module-level cache)."""
    return Settings()
