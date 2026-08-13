"""Application settings — read from environment variables at import time.

Postgres-first by design: the same SQLAlchemy models run on SQLite (local dev,
zero infra) or Postgres (Supabase/Render) by setting ``DATABASE_URL``.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]          # backend/
REPO_ROOT = BASE_DIR.parent                             # repository root


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


class Settings:
    # Database
    DATABASE_URL: str = _env(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'faircv.db'}"
    )

    # Audit engine
    FAIRCV_DB_PATH: Path = Path(
        _env("FAIRCV_DB_PATH", str(REPO_ROOT / "FairCVdb.npy"))
    )
    AUDIT_N_BOOT: int = int(_env("AUDIT_N_BOOT", "2000"))
    AUDIT_SEED: int = int(_env("AUDIT_SEED", "42"))
    AUDIT_TEST_RATIO: float = float(_env("AUDIT_TEST_RATIO", "0.2"))
    TOP_N_DEMO: int = 1000          # demo test set is 4,800 → top-1000 (repo protocol)
    TOP_N_PCT: float = 0.10         # generic CSVs: default top-N = 10% of test set

    # Uploads / security
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    MAX_UPLOAD_MB: int = int(_env("MAX_UPLOAD_MB", "50"))
    ALLOWED_EXTENSIONS = {".csv"}

    # CORS
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in _env("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
        if o.strip()
    ]

    # AI Assistant (Groq) — key lives ONLY on the backend, never in the frontend
    GROQ_API_KEY: str = _env("GROQ_API_KEY", "")  # empty = assistant disabled
    GROQ_MODEL: str = _env("GROQ_MODEL", "llama-3.3-70b-versatile")


settings = Settings()
