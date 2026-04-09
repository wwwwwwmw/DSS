from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    secret_key: str
    database_url: str
    cars_csv_path: str
    model_path: str


def _normalize_database_url(url: str) -> str:
    s = str(url or "").strip()
    if s.startswith("postgres://"):
        # Render may provide postgres://...; normalize for SQLAlchemy 2.x.
        return "postgresql+psycopg://" + s[len("postgres://"):]
    if s.startswith("postgresql://") and not s.startswith("postgresql+psycopg://"):
        return "postgresql+psycopg://" + s[len("postgresql://"):]
    return s


def get_settings() -> Settings:
    # On Render, rely on dashboard environment variables instead of a committed local .env file.
    is_render = str(os.getenv("RENDER", "")).strip().lower() in {"1", "true", "yes", "on"}
    if not is_render:
        load_dotenv(override=False)

    secret_key = os.getenv("SECRET_KEY", "change-me")

    # SECURITY: never commit your real `.env` file.
    # Put secrets in `.env` locally and keep `.env.example` as a template.
    database_url_env = _normalize_database_url(os.getenv("DATABASE_URL", ""))
    if database_url_env and str(database_url_env).startswith("postgresql"):
        try:
            import psycopg  # noqa: F401
        except Exception:
            warnings.warn(
                "psycopg is unavailable for postgresql DATABASE_URL; using local default URL.",
                RuntimeWarning,
            )
            database_url_env = "postgresql+psycopg://postgres:postgres@localhost:5432/dss_car_advisor"

    if not database_url_env:
        warnings.warn(
            "DATABASE_URL is not set; using default PostgreSQL URL "
            "(postgresql+psycopg://postgres:postgres@localhost:5432/dss_car_advisor).",
            RuntimeWarning,
        )
    database_url = database_url_env or "postgresql+psycopg://postgres:postgres@localhost:5432/dss_car_advisor"

    cars_csv_path = os.getenv("CARS_CSV_PATH", "./cars.csv")
    model_path = os.getenv("MODEL_PATH", "./models/car_advisor_rf.pkl")

    return Settings(
        secret_key=secret_key,
        database_url=database_url,
        cars_csv_path=cars_csv_path,
        model_path=model_path,
    )
