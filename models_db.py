from __future__ import annotations

import datetime as dt

from flask_login import UserMixin
from sqlalchemy import DateTime, Float, Integer, String, Unicode, UnicodeText
from sqlalchemy.orm import Mapped, mapped_column

from db import Base


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base, UserMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="user")  # guest is unauthenticated


class RecommendationHistory(Base):
    __tablename__ = "recommendation_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    car_count: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str] = mapped_column(Unicode(400), nullable=False)
    payload_json: Mapped[str] = mapped_column(UnicodeText, nullable=False)


class SavedCar(Base):
    __tablename__ = "saved_cars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    title: Mapped[str] = mapped_column(Unicode(200), nullable=False, default="")
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")  # manual | stock
    car_json: Mapped[str] = mapped_column(UnicodeText, nullable=False)


class CriteriaConfig(Base):
    __tablename__ = "criteria_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    label: Mapped[str] = mapped_column(Unicode(200), nullable=False)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)  # 'benefit' | 'cost'
    default_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value_json: Mapped[str] = mapped_column(UnicodeText, nullable=False)
