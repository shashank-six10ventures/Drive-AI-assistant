import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "drive_assistant.db"


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    drive_auth_mode: str = os.getenv("GOOGLE_DRIVE_AUTH_MODE", "service_account").lower()
    google_service_account_json: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    google_service_account_file: str = os.getenv(
        "GOOGLE_SERVICE_ACCOUNT_FILE",
        os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json"),
    )
    google_oauth_client_file: str = os.getenv("GOOGLE_OAUTH_CLIENT_FILE", "credentials.json")
    google_oauth_token_file: str = os.getenv("GOOGLE_OAUTH_TOKEN_FILE", "token.json")
    google_drive_folder_id: str = os.getenv("GOOGLE_DRIVE_FOLDER_ID", os.getenv("DRIVE_FOLDER_ID", ""))
    google_shared_drive_id: str = os.getenv("GOOGLE_SHARED_DRIVE_ID", "")
    monitor_interval_seconds: int = _as_int("MONITOR_INTERVAL_SECONDS", 300)

    slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = _as_int("SMTP_PORT", 587)
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    email_from: str = os.getenv("EMAIL_FROM", "")
    email_to: str = os.getenv("EMAIL_TO", "")

    ai_provider: str = os.getenv("AI_PROVIDER", "openai").lower()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")
    huggingface_model: str = os.getenv("HUGGINGFACE_MODEL", "google/flan-t5-large")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    max_text_chars: int = _as_int("MAX_TEXT_CHARS", 20000)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
ensure_dirs()
