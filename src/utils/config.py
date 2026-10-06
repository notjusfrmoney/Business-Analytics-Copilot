from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def load_project_config() -> dict[str, str]:
    """Read environment/.env values, with Streamlit secrets as an LLM fallback."""
    return {
        "ENVIRONMENT": os.getenv("ENVIRONMENT", "development"),
        "DATABASE_PATH": os.getenv(
            "DATABASE_PATH",
            str(BASE_DIR / "data" / "processed" / "business_analytics.duckdb"),
        ),
        "OPENROUTER_API_KEY": _get_setting("OPENROUTER_API_KEY"),
        "OPENROUTER_MODEL": _get_setting("OPENROUTER_MODEL"),
        "OPENROUTER_BASE_URL": _get_setting(
            "OPENROUTER_BASE_URL",
            default="https://openrouter.ai/api/v1",
        ),
    }


def _get_setting(name: str, *, default: str = "") -> str:
    """Prefer process/.env values, then use Streamlit secrets when available."""
    environment_value = os.getenv(name)
    if environment_value:
        return environment_value

    try:
        secret_value = st.secrets[name]
    except (KeyError, StreamlitSecretNotFoundError):
        return default

    if secret_value is None or secret_value == "":
        return default
    if not isinstance(secret_value, str):
        raise ValueError(f"{name} must be configured as a string.")
    return secret_value


def get_database_path() -> Path:
    """Return the configured DuckDB file path as a Path object."""
    config = load_project_config()
    database_path = Path(config["DATABASE_PATH"])
    if not database_path.is_absolute():
        database_path = BASE_DIR / database_path
    return database_path


def get_openrouter_settings() -> tuple[str, str, str]:
    """Return OpenRouter settings, raising clear errors when required values are missing."""
    config = load_project_config()
    api_key = config["OPENROUTER_API_KEY"].strip()
    model = config["OPENROUTER_MODEL"].strip()
    base_url = config["OPENROUTER_BASE_URL"].strip().rstrip("/")
    missing = [
        setting
        for setting, value in (
            ("OPENROUTER_API_KEY", api_key),
            ("OPENROUTER_MODEL", model),
            ("OPENROUTER_BASE_URL", base_url),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"Missing required OpenRouter configuration: {', '.join(missing)}. "
            "Set these values in .env, the process environment, or Streamlit secrets."
        )
    if not base_url.startswith(("https://", "http://")):
        raise ValueError("OPENROUTER_BASE_URL must use an http:// or https:// URL.")
    return api_key, model, base_url
