from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def _is_postgresql_url(url: str) -> bool:
    return str(url).startswith("postgresql")


def migrate(database_url: str, *, dry_run: bool = False) -> None:
    if not _is_postgresql_url(database_url):
        raise ValueError("Unsupported DATABASE_URL scheme; expected postgresql+psycopg://...")

    statements = [
        "SELECT current_database();",
    ]

    if dry_run:
        print("DRY RUN - statements to execute:")
        for stmt in statements:
            print(stmt)
        return

    engine = create_engine(database_url, future=True)
    with engine.connect() as conn:
        for stmt in statements:
            conn.execute(text(stmt))

    print("PostgreSQL connection check succeeded. Use SQLAlchemy create_all or postgres_init.sql to initialize schema.")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DSS DB migration helper (PostgreSQL)")
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override DATABASE_URL (otherwise reads from .env/env vars).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without applying.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    load_dotenv(override=False)
    database_url = (
        args.database_url
        or os.getenv("DATABASE_URL")
        or "postgresql+psycopg://postgres:postgres@localhost:5432/dss_car_advisor"
    )

    migrate(database_url, dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
