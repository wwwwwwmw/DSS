from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


@dataclass(frozen=True)
class MigrationPlan:
    statements: list[str]


def _is_sqlite_url(url: str) -> bool:
    return str(url).startswith("sqlite")


def _is_mssql_url(url: str) -> bool:
    return str(url).startswith("mssql")


def _sqlite_path_from_url(url: str) -> str:
    # Supports sqlite:///relative.db and sqlite:////absolute/path.db
    m = re.match(r"^sqlite:(?P<slashes>/*)(?P<path>.*)$", url)
    if not m:
        raise ValueError("Not a sqlite URL")
    slashes = m.group("slashes")
    path = m.group("path")
    if not path:
        raise ValueError("SQLite URL has empty path")

    # sqlite:///dss.sqlite3 => relative path dss.sqlite3
    # sqlite:////C:/x/y.db => absolute path /C:/x/y.db (Windows) or /x/y.db (POSIX)
    if slashes == "///":
        return path
    if slashes.startswith("////"):
        # Drop one leading slash for Windows absolute paths if present
        return path.lstrip("/")
    return path


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _sql_list(plan: MigrationPlan) -> str:
    return "\n".join(plan.statements)


def plan_sqlite_migrations(database_url: str) -> MigrationPlan:
    # SQLite cannot ALTER COLUMN types; we rebuild `criteria_config` if needed.
    # created_at columns can remain as-is; SQLite is flexible with datetime storage.
    stmts: list[str] = []

    # Ensure foreign keys respected during rebuild.
    stmts.append("PRAGMA foreign_keys=OFF;")

    # Check whether criteria_config exists and whether default_weight looks like INTEGER.
    # We'll do the checks at runtime (script), but the actual rebuild SQL is static.

    # Rebuild table (id/key unique/index preserved)
    stmts += [
        "BEGIN;",
        "CREATE TABLE IF NOT EXISTS criteria_config_new (\n"
        "    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,\n"
        "    key VARCHAR(64) NOT NULL,\n"
        "    label VARCHAR(200) NOT NULL,\n"
        "    direction VARCHAR(16) NOT NULL,\n"
        "    default_weight REAL NOT NULL DEFAULT 0.0\n"
        ");",
        "INSERT INTO criteria_config_new (id, key, label, direction, default_weight)\n"
        "SELECT id, key, label, direction, CAST(default_weight AS REAL)\n"
        "FROM criteria_config;",
        "DROP TABLE criteria_config;",
        "ALTER TABLE criteria_config_new RENAME TO criteria_config;",
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_criteria_config_key ON criteria_config (key);",
        "COMMIT;",
        "PRAGMA foreign_keys=ON;",
    ]

    return MigrationPlan(statements=stmts)


def _mssql_drop_default_constraint_sql(table: str, column: str) -> str:
    # Generates dynamic SQL that drops a default constraint if present.
    return f"""
DECLARE @constraint_name sysname;
SELECT @constraint_name = dc.name
FROM sys.default_constraints dc
JOIN sys.columns c
  ON c.default_object_id = dc.object_id
JOIN sys.tables t
  ON t.object_id = c.object_id
JOIN sys.schemas s
  ON s.schema_id = t.schema_id
WHERE t.name = '{table}' AND c.name = '{column}' AND s.name = 'dbo';
IF @constraint_name IS NOT NULL
BEGIN
    EXEC('ALTER TABLE dbo.{table} DROP CONSTRAINT ' + @constraint_name);
END
""".strip()


def plan_mssql_migrations(
    *,
    migrate_default_weight: bool,
    migrate_history_created_at: bool,
    migrate_saved_created_at: bool,
    migrate_unicode_columns: list[tuple[str, str, str]],
) -> MigrationPlan:
    stmts: list[str] = []

    # 1) criteria_config.default_weight => FLOAT NOT NULL DEFAULT(0)
    if migrate_default_weight:
        stmts.append(_mssql_drop_default_constraint_sql("criteria_config", "default_weight"))
        stmts.append("ALTER TABLE dbo.criteria_config ALTER COLUMN default_weight FLOAT NOT NULL;")
        stmts.append(
            "ALTER TABLE dbo.criteria_config ADD CONSTRAINT DF_criteria_config_default_weight DEFAULT (0.0) FOR default_weight;"
        )

    # 2) created_at columns => DATETIMEOFFSET to match timezone-aware datetimes
    def add_created_at_migration(table: str) -> None:
        stmts.append(_mssql_drop_default_constraint_sql(table, "created_at"))
        stmts.append(
            f"IF COL_LENGTH('dbo.{table}', 'created_at_new') IS NULL ALTER TABLE dbo.{table} ADD created_at_new DATETIMEOFFSET(7) NULL;"
        )
        stmts.append(
            f"UPDATE dbo.{table} SET created_at_new = CASE WHEN created_at IS NULL THEN NULL ELSE TODATETIMEOFFSET(CAST(created_at AS datetime2), '+00:00') END WHERE created_at_new IS NULL;"
        )
        stmts.append(f"ALTER TABLE dbo.{table} ALTER COLUMN created_at_new DATETIMEOFFSET(7) NOT NULL;")
        stmts.append(f"ALTER TABLE dbo.{table} DROP COLUMN created_at;")
        stmts.append(f"EXEC sp_rename 'dbo.{table}.created_at_new', 'created_at', 'COLUMN';")
        stmts.append(
            f"ALTER TABLE dbo.{table} ADD CONSTRAINT DF_{table}_created_at DEFAULT (SYSDATETIMEOFFSET()) FOR created_at;"
        )

    if migrate_history_created_at:
        add_created_at_migration("recommendation_history")
    if migrate_saved_created_at:
        add_created_at_migration("saved_cars")

    # 3) Text columns that must support Vietnamese properly => NVARCHAR/NVARCHAR(MAX)
    for table, column, target_sql_type in migrate_unicode_columns:
        stmts.append(_mssql_drop_default_constraint_sql(table, column))
        stmts.append(f"ALTER TABLE dbo.{table} ALTER COLUMN {column} {target_sql_type} NOT NULL;")

    return MigrationPlan(statements=stmts)


def _sqlite_needs_criteria_rebuild(conn) -> bool:
    # criteria_config might not exist on fresh DB.
    row = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='criteria_config';")).fetchone()
    if not row:
        return False
    cols = conn.execute(text("PRAGMA table_info(criteria_config); ")).fetchall()
    # cols: (cid, name, type, notnull, dflt_value, pk)
    for _cid, name, coltype, _notnull, _dflt, _pk in cols:
        if name == "default_weight":
            t = (coltype or "").strip().upper()
            # Old schema used Integer; might show as INTEGER.
            return t in {"INTEGER", "INT"}
    # If missing, let app create a fresh DB.
    return False


def _mssql_column_type(conn, table: str, column: str) -> Optional[str]:
    r = conn.execute(
        text(
            "SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA='dbo' AND TABLE_NAME=:t AND COLUMN_NAME=:c"
        ),
        {"t": table, "c": column},
    ).fetchone()
    return None if not r else str(r[0]).lower()


def _mssql_needs_unicode(conn, table: str, column: str) -> bool:
    dt = _mssql_column_type(conn, table, column)
    if not dt:
        return False
    return dt in {"varchar", "char", "text"}


def apply_plan(database_url: str, plan: MigrationPlan, *, dry_run: bool) -> None:
    if dry_run:
        _print_header("DRY RUN - SQL to execute")
        print(_sql_list(plan))
        return

    engine = create_engine(database_url, future=True)
    with engine.begin() as conn:
        for stmt in plan.statements:
            # For multi-statement T-SQL blocks, SQLAlchemy needs `exec_driver_sql`.
            if _is_mssql_url(database_url) and ("DECLARE" in stmt or "EXEC(" in stmt):
                conn.exec_driver_sql(stmt)
            else:
                conn.execute(text(stmt))


def migrate(database_url: str, *, dry_run: bool = False) -> None:
    if _is_sqlite_url(database_url):
        _print_header("SQLite migration")
        engine = create_engine(database_url, future=True)
        with engine.connect() as conn:
            if not _sqlite_needs_criteria_rebuild(conn):
                print("No SQLite migrations needed.")
                return
        plan = plan_sqlite_migrations(database_url)
        apply_plan(database_url, plan, dry_run=dry_run)
        print("SQLite migration applied." if not dry_run else "SQLite dry-run complete.")
        return

    if _is_mssql_url(database_url):
        _print_header("SQL Server (mssql) migration")
        engine = create_engine(database_url, future=True)
        with engine.connect() as conn:
            dw = _mssql_column_type(conn, "criteria_config", "default_weight")
            ca1 = _mssql_column_type(conn, "recommendation_history", "created_at")
            ca2 = _mssql_column_type(conn, "saved_cars", "created_at")

            unicode_candidates: list[tuple[str, str, str, str]] = [
                ("recommendation_history", "summary", "NVARCHAR(400)", "recommendation_history.summary"),
                ("recommendation_history", "payload_json", "NVARCHAR(MAX)", "recommendation_history.payload_json"),
                ("saved_cars", "title", "NVARCHAR(200)", "saved_cars.title"),
                ("saved_cars", "car_json", "NVARCHAR(MAX)", "saved_cars.car_json"),
                ("criteria_config", "label", "NVARCHAR(200)", "criteria_config.label"),
            ]
            migrate_unicode_columns = [
                (t, c, ty)
                for (t, c, ty, _name) in unicode_candidates
                if _mssql_needs_unicode(conn, t, c)
            ]

        migrate_default_weight = bool(dw and dw != "float")
        migrate_history_created_at = bool(ca1 and ca1 != "datetimeoffset")
        migrate_saved_created_at = bool(ca2 and ca2 != "datetimeoffset")

        needs: list[str] = []
        if migrate_default_weight:
            needs.append("criteria_config.default_weight")
        if migrate_history_created_at:
            needs.append("recommendation_history.created_at")
        if migrate_saved_created_at:
            needs.append("saved_cars.created_at")
        for (_t, _c, _ty, name) in unicode_candidates:
            if any((t == _t and c == _c) for (t, c, _ty2) in migrate_unicode_columns):
                needs.append(name)

        if not needs:
            print("No SQL Server migrations needed.")
            return

        print("Will migrate:")
        for n in needs:
            print(f"- {n}")

        plan = plan_mssql_migrations(
            migrate_default_weight=migrate_default_weight,
            migrate_history_created_at=migrate_history_created_at,
            migrate_saved_created_at=migrate_saved_created_at,
            migrate_unicode_columns=migrate_unicode_columns,
        )
        apply_plan(database_url, plan, dry_run=dry_run)
        print("SQL Server migration applied." if not dry_run else "SQL Server dry-run complete.")
        return

    raise ValueError("Unsupported DATABASE_URL scheme; expected sqlite:///... or mssql+pyodbc://...")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DSS DB migration helper (SQLite + SQL Server)")
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override DATABASE_URL (otherwise reads from .env/env vars).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without applying.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    load_dotenv(override=False)
    database_url = args.database_url or os.getenv("DATABASE_URL") or "sqlite:///dss.sqlite3"

    migrate(database_url, dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
