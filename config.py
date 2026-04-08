import os
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

try:
    import streamlit as st
except Exception:
    st = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "drive_assistant.db"

def _get_setting(name: str, default: str = "") -> str:
    env_value = os.getenv(name)
    if env_value not in [None, ""]:
        return env_value
    if st is not None:
        try:
            secret_value = st.secrets.get(name, default)
            if isinstance(secret_value, Mapping):
                return json.dumps(dict(secret_value))
            if isinstance(secret_value, list):
                return json.dumps(secret_value)
            return str(secret_value)
        except Exception:
            return default
    return default


def _as_int(name: str, default: int) -> int:
    value = _get_setting(name, str(default)).strip()
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    drive_auth_mode: str = _get_setting("GOOGLE_DRIVE_AUTH_MODE", "service_account").lower()
    google_service_account_json: str = _get_setting("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    google_service_account_file: str = _get_setting(
        "GOOGLE_SERVICE_ACCOUNT_FILE",
        _get_setting("GOOGLE_CREDENTIALS_FILE", "credentials.json"),
    )
    google_oauth_client_file: str = _get_setting("GOOGLE_OAUTH_CLIENT_FILE", "credentials.json")
    google_oauth_token_file: str = _get_setting("GOOGLE_OAUTH_TOKEN_FILE", "token.json")
    google_drive_folder_id: str = _get_setting("GOOGLE_DRIVE_FOLDER_ID", _get_setting("DRIVE_FOLDER_ID", ""))
    google_shared_drive_id: str = _get_setting("GOOGLE_SHARED_DRIVE_ID", "")
    monitor_interval_seconds: int = _as_int("MONITOR_INTERVAL_SECONDS", 300)

    slack_webhook_url: str = _get_setting("SLACK_WEBHOOK_URL", "")
    smtp_host: str = _get_setting("SMTP_HOST", "")
    smtp_port: int = _as_int("SMTP_PORT", 587)
    smtp_username: str = _get_setting("SMTP_USERNAME", "")
    smtp_password: str = _get_setting("SMTP_PASSWORD", "")
    email_from: str = _get_setting("EMAIL_FROM", "")
    email_to: str = _get_setting("EMAIL_TO", "")

    ai_provider: str = _get_setting("AI_PROVIDER", "openai").lower()
    openai_api_key: str = _get_setting("OPENAI_API_KEY", "")
    openai_model: str = _get_setting("OPENAI_MODEL", "gpt-4o-mini")
    groq_api_key: str = _get_setting("GROQ_API_KEY", "")
    groq_model: str = _get_setting("GROQ_MODEL", "llama-3.1-8b-instant")
    gemini_api_key: str = _get_setting("GEMINI_API_KEY", "")
    gemini_model: str = _get_setting("GEMINI_MODEL", "gemini-1.5-flash")
    huggingface_api_key: str = _get_setting("HUGGINGFACE_API_KEY", "")
    huggingface_model: str = _get_setting("HUGGINGFACE_MODEL", "google/flan-t5-large")

    embedding_model: str = _get_setting("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    max_text_chars: int = _as_int("MAX_TEXT_CHARS", 20000)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
ensure_dirs()
