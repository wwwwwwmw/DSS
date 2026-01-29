from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import re
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from config import get_settings
from db import Base, create_session_factory, session_scope
from ml import CRITERIA, ahp_score, choose_option, load_models, parse_mpg, predict, serialize_payload
from models_db import CriteriaConfig, RecommendationHistory, User


def create_app() -> Flask:
    settings = get_settings()

    app = Flask(__name__)
    app.secret_key = settings.secret_key

    engine, SessionLocal = create_session_factory(settings.database_url)
    Base.metadata.create_all(bind=engine)

    login_manager = LoginManager()
    login_manager.login_view = "login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        with session_scope(SessionLocal) as s:
            return s.get(User, int(user_id))

    def ensure_default_admin():
        with session_scope(SessionLocal) as s:
            admin = s.query(User).filter(User.role == "admin").first()
            if admin:
                return
            email = "admin@example.com"
            pw = "admin123"
            s.add(
                User(
                    email=email,
                    password_hash=generate_password_hash(pw),
                    role="admin",
                )
            )
        app.logger.warning("Created default admin: %s / %s", email, pw)

    ensure_default_admin()

    def ensure_default_criteria():
        with session_scope(SessionLocal) as s:
            exists = s.query(CriteriaConfig).first()
            if exists:
                return
            for c in CRITERIA:
                s.add(
                    CriteriaConfig(
                        key=c["key"],
                        label=c["label"],
                        direction=c["direction"],
                        default_weight=int(c.get("default", 5)),
                    )
                )

    ensure_default_criteria()

    def load_criteria() -> List[Dict[str, Any]]:
        with session_scope(SessionLocal) as s:
            items = s.query(CriteriaConfig).order_by(CriteriaConfig.id.asc()).all()
            return [
                {
                    "key": it.key,
                    "label": it.label,
                    "direction": it.direction,
                    "default": int(it.default_weight),
                }
                for it in items
            ]

    def get_models():
        return load_models(settings.model_path)

    def parse_cars_from_form(prefix: str = "car") -> List[Dict[str, Any]]:
        cars: List[Dict[str, Any]] = []

        # Support dynamic indices: car0_*, car1_*, car7_* ...
        pat = re.compile(rf"^{re.escape(prefix)}(\d+)_")
        indices = set()
        for k in request.form.keys():
            m = pat.match(k)
            if m:
                indices.add(int(m.group(1)))

        def g(i: int, field: str):
            return request.form.get(f"{prefix}{i}_{field}", "").strip()

        for i in sorted(indices):
            any_field = any(
                g(i, k)
                for k in [
                    "price",
                    "mileage",
                    "year",
                    "accidents_or_damage",
                    "one_owner",
                    "driver_rating",
                    "seller_rating",
                    "mpg",
                    "price_drop",
                ]
            )
            if not any_field:
                continue

            cars.append(
                {
                    "price": g(i, "price"),
                    "mileage": g(i, "mileage"),
                    "year": g(i, "year"),
                    "accidents_or_damage": g(i, "accidents_or_damage"),
                    "one_owner": g(i, "one_owner"),
                    "driver_rating": g(i, "driver_rating"),
                    "seller_rating": g(i, "seller_rating"),
                    "mpg": g(i, "mpg"),
                    "price_drop": g(i, "price_drop"),
                }
            )

        return cars

    def validate_cars(cars: List[Dict[str, Any]], min_cars: int) -> List[str]:
        errors: List[str] = []
        if len(cars) < min_cars:
            errors.append(f"Cần nhập tối thiểu {min_cars} xe (ít nhất 1 trường có giá trị).")
            return errors

        def f(v: Any):
            try:
                if v is None or str(v).strip() == "":
                    return None
                return float(v)
            except Exception:
                return None

        def is01(v: Any) -> bool:
            s = str(v).strip()
            return s == "0" or s == "1"

        for idx, car in enumerate(cars, start=1):
            missing = [k for k in ["price", "mileage", "year", "accidents_or_damage", "one_owner"] if str(car.get(k, "")).strip() == ""]
            if missing:
                errors.append(f"Xe #{idx}: thiếu trường bắt buộc: {', '.join(missing)}")

            price = f(car.get("price"))
            if car.get("price", "").strip() != "" and (price is None or price < 0):
                errors.append(f"Xe #{idx}: price phải là số >= 0")

            mileage = f(car.get("mileage"))
            if car.get("mileage", "").strip() != "" and (mileage is None or mileage < 0):
                errors.append(f"Xe #{idx}: mileage phải là số >= 0")

            year = f(car.get("year"))
            if car.get("year", "").strip() != "" and (year is None or year < 1980 or year > 2035):
                errors.append(f"Xe #{idx}: year không hợp lệ (1980-2035)")

            aod = car.get("accidents_or_damage", "").strip()
            if aod and not is01(aod):
                errors.append(f"Xe #{idx}: accidents_or_damage chỉ nhận 0 hoặc 1")

            oo = car.get("one_owner", "").strip()
            if oo and not is01(oo):
                errors.append(f"Xe #{idx}: one_owner chỉ nhận 0 hoặc 1")

            mpg_raw = car.get("mpg", "").strip()
            if mpg_raw and parse_mpg(mpg_raw) is None:
                errors.append(f"Xe #{idx}: mpg sai định dạng (vd: 30 hoặc 39-38)")

            dr = car.get("driver_rating", "").strip()
            if dr:
                n = f(dr)
                if n is None or n < 0 or n > 5:
                    errors.append(f"Xe #{idx}: driver_rating phải trong khoảng 0-5")

            sr = car.get("seller_rating", "").strip()
            if sr:
                n = f(sr)
                if n is None or n < 0 or n > 5:
                    errors.append(f"Xe #{idx}: seller_rating phải trong khoảng 0-5")

            pd = car.get("price_drop", "").strip()
            if pd:
                n = f(pd)
                if n is None or n < 0:
                    errors.append(f"Xe #{idx}: price_drop phải là số >= 0")

        return errors

    def parse_weights_from_form(criteria: List[Dict[str, Any]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for c in criteria:
            key = c["key"]
            raw = request.form.get(f"w_{key}", str(c.get("default", 5)))
            try:
                v = float(raw)
            except Exception:
                v = float(c.get("default", 5))
            out[key] = max(1.0, min(9.0, v))
        return out

    @app.get("/")
    def home():
        criteria = load_criteria()
        return render_template("index.html", criteria=criteria, results=None, top_recommendations=None)

    @app.post("/recommend")
    def recommend():
        criteria = load_criteria()
        cars = parse_cars_from_form()
        errors = validate_cars(cars, min_cars=3)
        if errors:
            for e in errors[:6]:
                flash(e, "danger")
            return redirect(url_for("home"))

        weights = parse_weights_from_form(criteria)
        scores = ahp_score(cars, weights)

        models = get_models()
        if not models:
            flash("Chưa có model. Admin hãy retrain hoặc chạy train.py trước.", "danger")
            accident_probs = [0.5 for _ in cars]
            maint_monthly = [300.0 for _ in cars]
        else:
            accident_probs, maint_monthly = predict(models, cars)

        results: List[Dict[str, Any]] = []
        for idx, (s, ap, mm) in enumerate(zip(scores, accident_probs, maint_monthly), start=1):
            option, badge, message = choose_option(s, ap, mm)
            results.append(
                {
                    "idx": idx,
                    "ahp_score": float(s),
                    "accident_risk_pct": float(ap) * 100.0,
                    "maintenance_monthly": int(round(mm)),
                    "maintenance_annual": int(round(mm * 12)),
                    "option": option,
                    "badge": badge,
                    "message": message,
                }
            )

        # Sort by option priority then by score
        priority = {"green": 0, "yellow": 1, "red": 2}
        results.sort(key=lambda r: (priority.get(r["badge"], 9), -r["ahp_score"]))

        top_recommendations = [r for r in results if r["badge"] == "green"]

        if current_user.is_authenticated:
            payload = serialize_payload(weights, cars, results)
            summary = f"Top: xe #{top_recommendations[0]['idx']}" if top_recommendations else "Không có xe xanh"
            with session_scope(SessionLocal) as s:
                s.add(
                    RecommendationHistory(
                        user_id=int(current_user.get_id()),
                        created_at=dt.datetime.utcnow(),
                        car_count=len(cars),
                        summary=summary,
                        payload_json=payload,
                    )
                )

        return render_template(
            "index.html",
            criteria=criteria,
            results=results,
            top_recommendations=top_recommendations,
        )

    @app.route("/compare", methods=["GET", "POST"])
    def compare():
        if request.method == "GET":
            return render_template("compare.html", cars=None, rows=None)

        cars = parse_cars_from_form()
        errors = validate_cars(cars, min_cars=2)
        if errors:
            for e in errors[:6]:
                flash(e, "danger")
            return redirect(url_for("compare"))

        criteria = load_criteria()

        # For each criterion, determine best cell.
        def to_float(v, key: str):
            try:
                if key == "mpg":
                    return parse_mpg(v)
                return float(v)
            except Exception:
                return None

        def best_index(vals: List[Any], direction: str, key: str):
            xs = [(i, to_float(v, key)) for i, v in enumerate(vals)]
            xs = [(i, v) for i, v in xs if v is not None]
            if not xs:
                return None
            return min(xs, key=lambda t: t[1])[0] if direction == "cost" else max(xs, key=lambda t: t[1])[0]

        rows = []
        for c in criteria:
            key = c["key"]
            direction = c["direction"]
            label = c["label"].split(":", 1)[0]
            vals = [car.get(key) for car in cars]
            b = best_index(vals, direction, key)
            cells = []
            for i, v in enumerate(vals):
                cells.append({"value": v if v != "" else "—", "best": (b == i)})
            rows.append({"label": label, "cells": cells})

        return render_template("compare.html", cars=cars, rows=rows)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            return render_template("auth_login.html")

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        with session_scope(SessionLocal) as s:
            user = s.query(User).filter(User.email == email).first()
            if not user or not check_password_hash(user.password_hash, password):
                flash("Sai email hoặc mật khẩu.", "danger")
                return redirect(url_for("login"))

            login_user(user)
            flash("Đăng nhập thành công.", "success")
            return redirect(url_for("home"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "GET":
            return render_template("auth_register.html")

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if len(password) < 6:
            flash("Mật khẩu tối thiểu 6 ký tự.", "danger")
            return redirect(url_for("register"))

        with session_scope(SessionLocal) as s:
            exists = s.query(User).filter(User.email == email).first()
            if exists:
                flash("Email đã tồn tại.", "danger")
                return redirect(url_for("register"))

            s.add(User(email=email, password_hash=generate_password_hash(password), role="user"))

        flash("Đăng ký thành công. Hãy đăng nhập.", "success")
        return redirect(url_for("login"))

    @app.get("/logout")
    def logout():
        if current_user.is_authenticated:
            logout_user()
        flash("Đã đăng xuất.", "success")
        return redirect(url_for("home"))

    @app.get("/history")
    @login_required
    def history():
        with session_scope(SessionLocal) as s:
            items = (
                s.query(RecommendationHistory)
                .filter(RecommendationHistory.user_id == int(current_user.get_id()))
                .order_by(RecommendationHistory.created_at.desc())
                .limit(50)
                .all()
            )
        return render_template("history.html", items=items)

    def require_admin():
        if not current_user.is_authenticated or current_user.role != "admin":
            flash("Chỉ admin mới truy cập được.", "danger")
            return False
        return True

    @app.get("/admin")
    @login_required
    def admin():
        if not require_admin():
            return redirect(url_for("home"))
        with session_scope(SessionLocal) as s:
            users = s.query(User).order_by(User.id.asc()).all()
            criteria = s.query(CriteriaConfig).order_by(CriteriaConfig.id.asc()).all()
        return render_template("admin.html", users=users, criteria=criteria)

    @app.post("/admin/criteria")
    @login_required
    def admin_update_criteria():
        if not require_admin():
            return redirect(url_for("home"))

        with session_scope(SessionLocal) as s:
            items = s.query(CriteriaConfig).all()
            for it in items:
                raw = request.form.get(f"w_{it.key}")
                if raw is None:
                    continue
                try:
                    v = int(float(raw))
                except Exception:
                    continue
                it.default_weight = max(1, min(9, v))

        flash("Đã cập nhật trọng số mặc định.", "success")
        return redirect(url_for("admin"))

    @app.post("/admin/retrain")
    @login_required
    def admin_retrain():
        if not require_admin():
            return redirect(url_for("home"))

        csv_path = Path(settings.cars_csv_path)
        uploaded = request.files.get("csv_file")
        if uploaded and uploaded.filename:
            csv_path = Path("./data") / "cars_uploaded.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            uploaded.save(str(csv_path))

        # Run training as a subprocess for simplicity.
        try:
            subprocess.check_call([
                sys.executable,
                "train.py",
                "--cars",
                str(csv_path),
                "--out",
                settings.model_path,
            ])
            flash("Retrain thành công.", "success")
        except Exception as e:
            flash(f"Retrain thất bại: {e}", "danger")

        return redirect(url_for("admin"))

    @app.post("/admin/make-admin/<int:user_id>")
    @login_required
    def admin_make_admin(user_id: int):
        if not require_admin():
            return redirect(url_for("home"))

        with session_scope(SessionLocal) as s:
            u = s.get(User, user_id)
            if not u:
                flash("User không tồn tại.", "danger")
                return redirect(url_for("admin"))
            u.role = "admin"

        flash("Đã set admin.", "success")
        return redirect(url_for("admin"))

    return app


app = create_app()


if __name__ == "__main__":
    # For production on Windows: waitress-serve --call app:create_app
    app.run(debug=True, port=5002)
