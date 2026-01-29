from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    secret_key: str
    database_url: str
    cars_csv_path: str
    model_path: str


def get_settings() -> Settings:
    load_dotenv(override=False)

    secret_key = os.getenv("SECRET_KEY", "change-me")
    database_url = os.getenv("DATABASE_URL", "sqlite:///dss.sqlite3")
    cars_csv_path = os.getenv("CARS_CSV_PATH", "./cars.csv")
    model_path = os.getenv("MODEL_PATH", "./models/car_advisor_rf.pkl")

    return Settings(
        secret_key=secret_key,
        database_url=database_url,
        cars_csv_path=cars_csv_path,
        model_path=model_path,
    )
