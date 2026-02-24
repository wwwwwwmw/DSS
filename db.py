from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def create_session_factory(database_url: str):
    engine_kwargs = {"future": True}
    # SQL Server: keep pooled connections healthy over long-running app.
    if str(database_url).startswith("mssql"):
        engine_kwargs.update(
            {
                "pool_pre_ping": True,
                "pool_recycle": 1800,
            }
        )
    engine = create_engine(database_url, **engine_kwargs)
    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        future=True,
        expire_on_commit=False,
    )
    return engine, SessionLocal


@contextmanager
def session_scope(SessionLocal):
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
