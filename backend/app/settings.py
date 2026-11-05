from __future__ import annotations

import os
from pathlib import Path

# Load .env automatically in development
try:
    from dotenv import load_dotenv
    _backend_env = Path(__file__).resolve().parent.parent / ".env"
    _root_env = Path(__file__).resolve().parent.parent.parent / ".env"
    if _backend_env.exists():
        load_dotenv(_backend_env)
    if _root_env.exists():
        load_dotenv(_root_env)
except ImportError:
    pass  # python-dotenv is optional; env vars may be set externally


class Settings:
    session_secret = os.environ.get("SESSION_SECRET", "")
    environment = os.environ.get("ENVIRONMENT", "development")
    upload_directory = Path(os.environ.get("UPLOAD_DIRECTORY", "./storage/uploads"))
    max_upload_bytes = int(os.environ.get("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

    @property
    def database_url(self) -> str:
        url = os.environ.get("DATABASE_URL", "")
        if not url:
            raise RuntimeError(
                "DATABASE_URL must be set. Example: "
                "DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/cognicrew"
            )
        if url.startswith("postgresql+psycopg://"):
            try:
                import psycopg  # noqa: F401
            except Exception:
                url = url.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return url

    def validate_runtime_secrets(self) -> None:
        # database_url property already raises if unset
        _ = self.database_url
        if len(self.session_secret) < 32:
            raise RuntimeError("SESSION_SECRET must be at least 32 characters.")

    @property
    def ai_provider(self) -> str:
        return os.environ.get("AI_PROVIDER", "nvidia").lower().strip()

    @property
    def ai_model(self) -> str:
        return os.environ.get("AI_MODEL", "nvidia/ising-calibration-1.5-31b").strip()

    @property
    def ai_base_url(self) -> str:
        return os.environ.get("AI_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")

    @property
    def ai_timeout(self) -> float:
        return float(os.environ.get("AI_TIMEOUT", "120.0"))

    @property
    def ai_max_tokens(self) -> int:
        return int(os.environ.get("AI_MAX_TOKENS", "16384"))

    @property
    def ai_enable_thinking(self) -> bool:
        return os.environ.get("AI_ENABLE_THINKING", "true").strip().lower() in ("true", "1", "yes")

    @property
    def nvidia_api_key(self) -> str:
        return os.environ.get("NVIDIA_API_KEY", "").strip()

    @property
    def gemini_api_key(self) -> str:
        return os.environ.get("GEMINI_API_KEY", "").strip()

    @property
    def gemini_model(self) -> str:
        return os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()

    @property
    def gemini_base_url(self) -> str:
        return os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

    @property
    def exa_api_key(self) -> str:
        return os.environ.get("EXA_API_KEY", "").strip()


settings = Settings()
