from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from config import get_settings
from db import Base, create_session_factory, session_scope
from ml import (
    CRITERIA,
    ahp_score,
    ahp_score_dataframe,
    choose_option,
    load_models,
    normalize_weights,
    parse_mpg,
    parse_mpg_series,
    predict,
    serialize_payload,
)
from models_db import CriteriaConfig, RecommendationHistory, SavedCar, User


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
            user = s.get(User, int(user_id))
            if user is None:
                return None
            # Detach instance so template access won't require an active session.
            s.expunge(user)
            return user

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

    def get_risk_level(risk_pct: float) -> Dict[str, str]:
        """Chuyển đổi tỷ lệ rủi ro (%) thành nhãn cấp độ.
        
        Returns dict với 'label' và 'badge_class'.
        Cấp độ:
        - Rất thấp: 0-20%
        - Thấp: 20-40%
        - Trung bình: 40-60%
        - Cao: 60-80%
        - Rất cao: 80-100%
        """
        if risk_pct < 20:
            return {"label": "Rất thấp", "badge_class": "success"}
        elif risk_pct < 40:
            return {"label": "Thấp", "badge_class": "info"}
        elif risk_pct < 60:
            return {"label": "Trung bình", "badge_class": "warning"}
        elif risk_pct < 80:
            return {"label": "Cao", "badge_class": "danger"}
        else:
            return {"label": "Rất cao", "badge_class": "danger-dark"}

    def get_maintenance_level(annual_cost: float) -> Dict[str, str]:
        """Chuyển đổi chi phí bảo dưỡng hàng năm thành nhãn cấp độ.
        
        Returns dict với 'label' và 'badge_class'.
        Cấp độ (dựa trên chi phí hàng năm):
        - Rất thấp: < 3 triệu/năm
        - Thấp: 3-6 triệu/năm
        - Trung bình: 6-9 triệu/năm
        - Cao: 9-12 triệu/năm
        - Rất cao: >= 12 triệu/năm
        """
        if annual_cost < 3000:
            return {"label": "Rất thấp", "badge_class": "success"}
        elif annual_cost < 6000:
            return {"label": "Thấp", "badge_class": "info"}
        elif annual_cost < 9000:
            return {"label": "Trung bình", "badge_class": "warning"}
        elif annual_cost < 12000:
            return {"label": "Cao", "badge_class": "danger"}
        else:
            return {"label": "Rất cao", "badge_class": "danger-dark"}

    def get_models():
        return load_models(settings.model_path)

    def _safe_json_loads(s: str) -> Any:
        try:
            return json.loads(s)
        except Exception:
            return None

    def sanitize_for_json(obj: Any) -> Any:
        """Convert NaN/Infinity to None so JSON is standards-compliant."""

        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, (int, str, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {str(k): sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize_for_json(v) for v in obj]
        # Fallback for numpy types / other objects
        try:
            if hasattr(obj, "item"):
                return sanitize_for_json(obj.item())
        except Exception:
            pass
        return str(obj)

    def save_history(*, action: str, cars: List[Dict[str, Any]], payload: Dict[str, Any], summary: str):
        if not current_user.is_authenticated:
            return
        # Store action inside payload (schema can evolve without DB migrations).
        payload2 = dict(payload)
        payload2["action"] = action
        payload_json = json.dumps(payload2, ensure_ascii=False)
        with session_scope(SessionLocal) as s:
            s.add(
                RecommendationHistory(
                    user_id=int(current_user.get_id()),
                    created_at=dt.datetime.utcnow(),
                    car_count=len(cars),
                    summary=summary,
                    payload_json=payload_json,
                )
            )

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

    def sstrip(v: Any) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        return "" if s.lower() in {"nan", "none"} else s

    def normalize_01(v: Any, *, default: Optional[int] = None) -> Any:
        """Normalize various boolean-ish inputs to 0/1.

        If default is None, empty/unknown values are kept as-is.
        If default is 0 or 1, empty/unknown values fall back to that.
        """

        raw = sstrip(v)
        if raw == "":
            return default if default is not None else v

        try:
            n = float(raw)
            if abs(n - 0.0) < 1e-9:
                return 0
            if abs(n - 1.0) < 1e-9:
                return 1
        except Exception:
            pass

        s = str(raw).strip().lower()
        if s in {"true", "t", "yes", "y"}:
            return 1
        if s in {"false", "f", "no", "n"}:
            return 0
        return default if default is not None else v

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
            if v is None:
                return False
            try:
                n = float(v)
                if abs(n - 0.0) < 1e-9 or abs(n - 1.0) < 1e-9:
                    return True
            except Exception:
                pass

            s = str(v).strip().lower()
            if s in {"0", "1", "0.0", "1.0"}:
                return True
            if s in {"true", "false", "t", "f", "yes", "no", "y", "n"}:
                return True
            return False

        for idx, car in enumerate(cars, start=1):
            missing = [k for k in ["price", "mileage", "year", "accidents_or_damage", "one_owner"] if sstrip(car.get(k, "")) == ""]
            if missing:
                errors.append(f"Xe #{idx}: thiếu trường bắt buộc: {', '.join(missing)}")

            price = f(car.get("price"))
            if sstrip(car.get("price", "")) != "" and (price is None or price < 0):
                errors.append(f"Xe #{idx}: price phải là số >= 0")

            mileage = f(car.get("mileage"))
            if sstrip(car.get("mileage", "")) != "" and (mileage is None or mileage < 0):
                errors.append(f"Xe #{idx}: mileage phải là số >= 0")

            year = f(car.get("year"))
            if sstrip(car.get("year", "")) != "" and (year is None or year < 1980 or year > 2035):
                errors.append(f"Xe #{idx}: year không hợp lệ (1980-2035)")

            aod = sstrip(car.get("accidents_or_damage", ""))
            if aod and not is01(aod):
                errors.append(f"Xe #{idx}: accidents_or_damage chỉ nhận 0 hoặc 1")

            oo = sstrip(car.get("one_owner", ""))
            if oo and not is01(oo):
                errors.append(f"Xe #{idx}: one_owner chỉ nhận 0 hoặc 1")

            mpg_raw = sstrip(car.get("mpg", ""))
            if mpg_raw and parse_mpg(mpg_raw) is None:
                errors.append(f"Xe #{idx}: mpg sai định dạng (vd: 30 hoặc 39-38)")

            dr = sstrip(car.get("driver_rating", ""))
            if dr:
                n = f(dr)
                if n is None or n < 0 or n > 5:
                    errors.append(f"Xe #{idx}: driver_rating phải trong khoảng 0-5")

            sr = sstrip(car.get("seller_rating", ""))
            if sr:
                n = f(sr)
                if n is None or n < 0 or n > 5:
                    errors.append(f"Xe #{idx}: seller_rating phải trong khoảng 0-5")

            pd = sstrip(car.get("price_drop", ""))
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

    def ahp_score_single_against_df(car: Dict[str, Any], weights: Dict[str, float], df) -> float:
        """Compute AHP score for a single car using min/max bounds from a reference dataframe."""

        import pandas as pd

        ws = normalize_weights(weights)
        score = 0.0

        for c in CRITERIA:
            key = c["key"]
            direction = c["direction"]
            w = float(ws.get(key, 0.0))
            if w <= 0:
                continue
            if key not in df.columns:
                continue

            if key == "mpg":
                col = parse_mpg_series(df[key])
                x = parse_mpg(car.get(key))
            else:
                col = pd.to_numeric(df[key], errors="coerce")
                try:
                    raw = sstrip(car.get(key, ""))
                    x = None if raw == "" else float(raw)
                except Exception:
                    x = None

            xs = col.dropna()
            if xs.empty or x is None or (isinstance(x, float) and not math.isfinite(x)):
                scaled = 0.0
            else:
                lo = float(xs.min())
                hi = float(xs.max())
                if abs(hi - lo) < 1e-12:
                    scaled = 0.5
                else:
                    t = (float(x) - lo) / (hi - lo)
                    if t < 0:
                        t = 0.0
                    elif t > 1:
                        t = 1.0
                    scaled = (1.0 - t) if direction == "cost" else t

            score += w * float(scaled)

        return float(score)

    @app.get("/")
    def home():
        criteria = load_criteria()
        return render_template("index.html", criteria=criteria, results=None, top_recommendations=None)

    @app.route("/evaluate", methods=["GET", "POST"])
    def evaluate():
        """Đánh giá xe ngoài (tối thiểu 1 xe) và có thể so sánh với xe trong kho. Khách vãng lai có thể truy cập."""

        criteria = load_criteria()
        if request.method == "GET":
            return render_template(
                "evaluate.html",
                criteria=criteria,
                results=None,
                cars=None,
                weights=None,
                compare_stock=False,
                stock_info=None,
                stock_results=None,
                top_n=10,
            )

        cars = parse_cars_from_form()
        errors = validate_cars(cars, min_cars=1)
        if errors:
            for e in errors[:6]:
                flash(e, "danger")
            return redirect(url_for("evaluate"))

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
            risk_pct = float(ap) * 100.0
            annual_cost = int(round(mm * 12))
            risk_level = get_risk_level(risk_pct)
            maint_level = get_maintenance_level(annual_cost)
            results.append(
                {
                    "idx": idx,
                    "ahp_score": float(s),
                    "accident_risk_pct": risk_pct,
                    "risk_level_label": risk_level["label"],
                    "risk_level_badge": risk_level["badge_class"],
                    "maintenance_monthly": int(round(mm)),
                    "maintenance_annual": annual_cost,
                    "maintenance_level_label": maint_level["label"],
                    "maintenance_level_badge": maint_level["badge_class"],
                    "option": option,
                    "badge": badge,
                    "message": message,
                }
            )

        compare_stock = (request.form.get("compare_stock") or "").strip() in {"1", "true", "on", "yes"}
        stock_info = None
        stock_results = None
        top_n = 10

        if compare_stock:
            top_n_raw = request.form.get("top_n", "10")
            try:
                top_n = int(float(top_n_raw))
            except Exception:
                top_n = 10
            top_n = max(1, min(50, top_n))

            import pandas as pd

            usecols = [
                "manufacturer",
                "model",
                "year",
                "mileage",
                "mpg",
                "accidents_or_damage",
                "one_owner",
                "personal_use_only",
                "seller_rating",
                "driver_rating",
                "driver_reviews_num",
                "price_drop",
                "price",
            ]

            try:
                df = pd.read_csv(settings.cars_csv_path, usecols=usecols, low_memory=False)
            except Exception as e:
                flash(f"Không đọc được cars.csv để so sánh: {e}", "danger")
                df = None

            if df is not None and not df.empty:
                df_score = df.copy()
                stock_scores = ahp_score_dataframe(df_score, weights)
                df_score["_ahp_score"] = stock_scores

                # Percentile for each outside car
                percents = []
                for car in cars:
                    s_ref = ahp_score_single_against_df(car, weights, df_score)
                    pct = float((stock_scores <= s_ref).mean()) * 100.0
                    percents.append({"ahp_ref": float(s_ref), "percentile": pct})

                top = df_score.nlargest(top_n, "_ahp_score")
                top_cars: List[Dict[str, Any]] = []
                for _, row in top.iterrows():
                    top_cars.append(
                        {
                            "title": f"{row.get('manufacturer', '')} {row.get('model', '')}".strip(),
                            "year": row.get("year", ""),
                            "mileage": row.get("mileage", ""),
                            "price": row.get("price", ""),
                            "ahp_score": float(row.get("_ahp_score", 0.0)),
                            "mpg": row.get("mpg", ""),
                            "accidents_or_damage": row.get("accidents_or_damage", ""),
                            "one_owner": row.get("one_owner", ""),
                            "personal_use_only": row.get("personal_use_only", ""),
                            "seller_rating": row.get("seller_rating", ""),
                            "driver_rating": row.get("driver_rating", ""),
                            "driver_reviews_num": row.get("driver_reviews_num", ""),
                            "price_drop": row.get("price_drop", ""),
                        }
                    )

                if models:
                    ap2, mm2 = predict(models, top_cars)
                else:
                    ap2 = [0.5 for _ in top_cars]
                    mm2 = [300.0 for _ in top_cars]

                stock_results = []
                for car, ap, mm in zip(top_cars, ap2, mm2):
                    risk_pct = float(ap) * 100.0
                    annual_cost = int(round(mm * 12))
                    risk_level = get_risk_level(risk_pct)
                    maint_level = get_maintenance_level(annual_cost)
                    stock_results.append(
                        {
                            **car,
                            "accident_risk_pct": risk_pct,
                            "risk_level_label": risk_level["label"],
                            "risk_level_badge": risk_level["badge_class"],
                            "maintenance_monthly": int(round(mm)),
                            "maintenance_annual": annual_cost,
                            "maintenance_level_label": maint_level["label"],
                            "maintenance_level_badge": maint_level["badge_class"],
                        }
                    )

                stock_info = {
                    "top_n": top_n,
                    "percents": percents,
                }

        # History
        payload = {
            "weights": weights,
            "cars": cars,
            "results": results,
            "compare_stock": compare_stock,
            "stock_info": stock_info,
        }
        summary = "Đánh giá: Đã tính cho 1 xe" if len(cars) == 1 else f"Đánh giá: {len(cars)} xe"
        if compare_stock:
            summary += f" • so với kho Top {top_n}"
        save_history(action="evaluate", cars=cars, payload=payload, summary=summary)

        return render_template(
            "evaluate.html",
            criteria=criteria,
            results=results,
            cars=cars,
            weights=weights,
            compare_stock=compare_stock,
            stock_info=stock_info,
            stock_results=stock_results,
            top_n=top_n,
        )

    @app.post("/recommend")
    def recommend():
        criteria = load_criteria()
        cars = parse_cars_from_form()
        errors = validate_cars(cars, min_cars=2)
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
            risk_pct = float(ap) * 100.0
            annual_cost = int(round(mm * 12))
            risk_level = get_risk_level(risk_pct)
            maint_level = get_maintenance_level(annual_cost)
            results.append(
                {
                    "idx": idx,
                    "ahp_score": float(s),
                    "accident_risk_pct": risk_pct,
                    "risk_level_label": risk_level["label"],
                    "risk_level_badge": risk_level["badge_class"],
                    "maintenance_monthly": int(round(mm)),
                    "maintenance_annual": annual_cost,
                    "maintenance_level_label": maint_level["label"],
                    "maintenance_level_badge": maint_level["badge_class"],
                    "option": option,
                    "badge": badge,
                    "message": message,
                }
            )

        # Sort by option priority then by score
        priority = {"green": 0, "yellow": 1, "red": 2}
        results.sort(key=lambda r: (priority.get(r["badge"], 9), -r["ahp_score"]))

        top_recommendations = [r for r in results if r["badge"] == "green"]

        payload = _safe_json_loads(serialize_payload(weights, cars, results)) or {"weights": weights, "cars": cars, "results": results}
        summary = f"Tư vấn: Top xe #{top_recommendations[0]['idx']}" if top_recommendations else "Tư vấn: Không có xe xanh"
        save_history(action="recommend", cars=cars, payload=payload, summary=summary)

        return render_template(
            "index.html",
            criteria=criteria,
            results=results,
            top_recommendations=top_recommendations,
            cars=cars,
            weights=weights,
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

        # Auto history
        save_history(
            action="compare",
            cars=cars,
            payload={"cars": cars, "criteria": criteria, "rows": rows},
            summary=f"So sánh: {len(cars)} xe",
        )

        return render_template("compare.html", cars=cars, rows=rows)

    @app.route("/stock", methods=["GET", "POST"])
    def stock():
        criteria = load_criteria()
        if request.method == "GET":
            return render_template("stock.html", criteria=criteria, results=None, top_n=10)

        # POST
        top_n_raw = request.form.get("top_n", "10")
        try:
            top_n = int(float(top_n_raw))
        except Exception:
            top_n = 10
        top_n = max(1, min(50, top_n))

        weights = parse_weights_from_form(criteria)

        import pandas as pd

        usecols = [
            "manufacturer",
            "model",
            "year",
            "mileage",
            "mpg",
            "accidents_or_damage",
            "one_owner",
            "personal_use_only",
            "seller_rating",
            "driver_rating",
            "driver_reviews_num",
            "price_drop",
            "price",
        ]

        try:
            df = pd.read_csv(settings.cars_csv_path, usecols=usecols, low_memory=False)
        except Exception as e:
            flash(f"Không đọc được cars.csv: {e}", "danger")
            return redirect(url_for("stock"))

        # Compute AHP score across entire dataset
        df_score = df.copy()
        scores = ahp_score_dataframe(df_score, weights)
        df_score["_ahp_score"] = scores

        top = df_score.nlargest(top_n, "_ahp_score")
        cars = []
        for _, row in top.iterrows():
            cars.append(
                {
                    "title": f"{row.get('manufacturer', '')} {row.get('model', '')}".strip(),
                    "year": row.get("year", ""),
                    "mileage": row.get("mileage", ""),
                    "price": row.get("price", ""),
                    "ahp_score": float(row.get("_ahp_score", 0.0)),
                    "mpg": row.get("mpg", ""),
                    "accidents_or_damage": row.get("accidents_or_damage", ""),
                    "one_owner": row.get("one_owner", ""),
                    "personal_use_only": row.get("personal_use_only", ""),
                    "seller_rating": row.get("seller_rating", ""),
                    "driver_rating": row.get("driver_rating", ""),
                    "driver_reviews_num": row.get("driver_reviews_num", ""),
                    "price_drop": row.get("price_drop", ""),
                }
            )

        models = get_models()
        if models:
            accident_probs, maint_monthly = predict(models, cars)
        else:
            accident_probs = [0.5 for _ in cars]
            maint_monthly = [300.0 for _ in cars]

        results = []
        for car, ap, mm in zip(cars, accident_probs, maint_monthly):
            risk_pct = float(ap) * 100.0
            annual_cost = int(round(mm * 12))
            risk_level = get_risk_level(risk_pct)
            maint_level = get_maintenance_level(annual_cost)
            results.append(
                {
                    **car,
                    "accident_risk_pct": risk_pct,
                    "risk_level_label": risk_level["label"],
                    "risk_level_badge": risk_level["badge_class"],
                    "maintenance_monthly": int(round(mm)),
                    "maintenance_annual": annual_cost,
                    "maintenance_level_label": maint_level["label"],
                    "maintenance_level_badge": maint_level["badge_class"],
                }
            )

        save_history(
            action="stock",
            cars=cars,
            payload={"weights": weights, "top_n": top_n, "cars": cars, "results": results},
            summary=f"Xe kho: Top {top_n}",
        )

        return render_template("stock.html", criteria=criteria, results=results, top_n=top_n)

    @app.get("/my-cars")
    @login_required
    def my_cars():
        with session_scope(SessionLocal) as s:
            items = (
                s.query(SavedCar)
                .filter(SavedCar.user_id == int(current_user.get_id()))
                .order_by(SavedCar.created_at.desc())
                .limit(200)
                .all()
            )
        # Detach/serialize for templates
        cars_out = []
        for it in items:
            cars_out.append(
                {
                    "id": it.id,
                    "created_at": it.created_at,
                    "title": it.title,
                    "source": it.source,
                    "car": _safe_json_loads(it.car_json) or {},
                }
            )
        return render_template("my_cars.html", items=cars_out)

    @app.get("/my-cars/<int:item_id>")
    @login_required
    def my_car_detail(item_id: int):
        with session_scope(SessionLocal) as s:
            it = s.get(SavedCar, item_id)
            if not it or it.user_id != int(current_user.get_id()):
                flash("Không tìm thấy xe.", "danger")
                return redirect(url_for("my_cars"))
            car = sanitize_for_json(_safe_json_loads(it.car_json) or {})

        return render_template(
            "my_car_detail.html",
            item={
                "id": it.id,
                "created_at": it.created_at,
                "title": it.title,
                "source": it.source,
            },
            car=car,
        )

    @app.route("/my-cars/<int:item_id>/edit", methods=["GET", "POST"])
    @login_required
    def my_car_edit(item_id: int):
        with session_scope(SessionLocal) as s:
            it = s.get(SavedCar, item_id)
            if not it or it.user_id != int(current_user.get_id()):
                flash("Không tìm thấy xe.", "danger")
                return redirect(url_for("my_cars"))

            car = sanitize_for_json(_safe_json_loads(it.car_json) or {})

            if request.method == "GET":
                return render_template(
                    "my_car_edit.html",
                    item={
                        "id": it.id,
                        "created_at": it.created_at,
                        "title": it.title,
                        "source": it.source,
                    },
                    car=car,
                )

            # POST
            new_title = request.form.get("title", "").strip()
            cars_edit = parse_cars_from_form()
            if not cars_edit:
                flash("Không có dữ liệu xe để cập nhật.", "danger")
                return redirect(url_for("my_car_edit", item_id=item_id))
            car_new = cars_edit[0]

            # Normalize + sanitize
            if "accidents_or_damage" in car_new:
                car_new["accidents_or_damage"] = normalize_01(car_new.get("accidents_or_damage"), default=None)
            if "one_owner" in car_new:
                car_new["one_owner"] = normalize_01(car_new.get("one_owner"), default=None)
            car_new.update(sanitize_for_json(car_new))

            errors = validate_cars([car_new], min_cars=1)
            if errors:
                for e in errors[:6]:
                    flash(e, "danger")
                return redirect(url_for("my_car_edit", item_id=item_id))

            if new_title:
                it.title = new_title
            it.car_json = json.dumps(sanitize_for_json(car_new), ensure_ascii=False, allow_nan=False)

        flash("Đã cập nhật xe.", "success")
        return redirect(url_for("my_car_detail", item_id=item_id))

    @app.get("/api/my-cars")
    @login_required
    def api_my_cars():
        with session_scope(SessionLocal) as s:
            items = (
                s.query(SavedCar)
                .filter(SavedCar.user_id == int(current_user.get_id()))
                .order_by(SavedCar.created_at.desc())
                .limit(200)
                .all()
            )

        out = []
        for it in items:
            out.append(
                {
                    "id": it.id,
                    "created_at": it.created_at.isoformat() if it.created_at else "",
                    "title": it.title,
                    "source": it.source,
                }
            )
        return jsonify({"items": out})

    @app.get("/api/my-cars/<int:item_id>")
    @login_required
    def api_my_car(item_id: int):
        with session_scope(SessionLocal) as s:
            it = s.get(SavedCar, item_id)
            if not it or it.user_id != int(current_user.get_id()):
                return jsonify({"error": "not_found"}), 404
            car = sanitize_for_json(_safe_json_loads(it.car_json) or {})

        return jsonify(
            {
                "id": it.id,
                "created_at": it.created_at.isoformat() if it.created_at else "",
                "title": it.title,
                "source": it.source,
                "car": car,
            }
        )

    @app.post("/my-cars/save")
    @login_required
    def save_my_cars():
        # Accept either a JSON list (cars_json) or a single car payload (car_json)
        cars_json = request.form.get("cars_json", "").strip()
        car_json = request.form.get("car_json", "").strip()
        title_override = request.form.get("title", "").strip()
        source = request.form.get("source", "manual").strip() or "manual"
        return_to = request.form.get("return_to", "").strip() or (request.referrer or "")

        cars_to_save: List[Dict[str, Any]] = []
        if cars_json:
            obj = _safe_json_loads(cars_json)
            if isinstance(obj, list):
                cars_to_save = [c for c in obj if isinstance(c, dict)]
        elif car_json:
            obj = _safe_json_loads(car_json)
            if isinstance(obj, dict):
                cars_to_save = [obj]
        else:
            # fallback: parse from dynamic form
            cars_to_save = parse_cars_from_form()

        if not cars_to_save:
            flash("Không có xe để lưu.", "danger")
            return redirect(return_to or url_for("home"))

        # Normalize boolean-ish fields from dataset/JSON saves.
        if source == "stock":
            for car in cars_to_save:
                if not isinstance(car, dict):
                    continue
                car["accidents_or_damage"] = normalize_01(car.get("accidents_or_damage"), default=0)
                car["one_owner"] = normalize_01(car.get("one_owner"), default=0)
                if "personal_use_only" in car:
                    car["personal_use_only"] = normalize_01(car.get("personal_use_only"), default=1)
                # Ensure standards-compliant JSON for later API import.
                car.update(sanitize_for_json(car))
        else:
            for car in cars_to_save:
                if not isinstance(car, dict):
                    continue
                if "accidents_or_damage" in car:
                    car["accidents_or_damage"] = normalize_01(car.get("accidents_or_damage"), default=None)
                if "one_owner" in car:
                    car["one_owner"] = normalize_01(car.get("one_owner"), default=None)
                car.update(sanitize_for_json(car))

        # Stock cars may contain missing fields; allow saving and let user adjust after import.
        if source != "stock":
            errors = validate_cars(cars_to_save, min_cars=1)
            if errors:
                for e in errors[:6]:
                    flash(e, "danger")
                return redirect(return_to or url_for("home"))

        with session_scope(SessionLocal) as s:
            for car in cars_to_save:
                title = str(car.get("title") or "").strip()
                if title_override and len(cars_to_save) == 1:
                    title = title_override
                if not title:
                    mk = str(car.get("manufacturer") or "").strip()
                    md = str(car.get("model") or "").strip()
                    if mk or md:
                        title = f"{mk} {md}".strip()
                if not title:
                    year = str(car.get("year") or "").strip()
                    price = str(car.get("price") or "").strip()
                    title = f"Xe {year} • {price}".strip(" •")

                s.add(
                    SavedCar(
                        user_id=int(current_user.get_id()),
                        created_at=dt.datetime.utcnow(),
                        title=title,
                        source=source,
                        car_json=json.dumps(sanitize_for_json(car), ensure_ascii=False, allow_nan=False),
                    )
                )

        flash(f"Đã lưu {len(cars_to_save)} xe vào 'Xe của tôi'.", "success")
        return redirect(url_for("my_cars"))

    @app.post("/my-cars/delete/<int:item_id>")
    @login_required
    def delete_my_car(item_id: int):
        with session_scope(SessionLocal) as s:
            it = s.get(SavedCar, item_id)
            if not it or it.user_id != int(current_user.get_id()):
                flash("Không tìm thấy xe.", "danger")
                return redirect(url_for("my_cars"))
            s.delete(it)

        flash("Đã xóa xe khỏi 'Xe của tôi'.", "success")
        return redirect(url_for("my_cars"))

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

    @app.get("/history/<int:item_id>")
    @login_required
    def history_detail(item_id: int):
        with session_scope(SessionLocal) as s:
            it = s.get(RecommendationHistory, item_id)
            if not it or it.user_id != int(current_user.get_id()):
                flash("Không tìm thấy bản ghi lịch sử.", "danger")
                return redirect(url_for("history"))
            payload = _safe_json_loads(it.payload_json) or {}

        return render_template(
            "history_detail.html",
            item={
                "id": it.id,
                "created_at": it.created_at,
                "car_count": it.car_count,
                "summary": it.summary,
            },
            payload=payload,
        )

    @app.post("/history/<int:item_id>/delete")
    @login_required
    def history_delete(item_id: int):
        with session_scope(SessionLocal) as s:
            it = s.get(RecommendationHistory, item_id)
            if not it or it.user_id != int(current_user.get_id()):
                flash("Không tìm thấy bản ghi lịch sử.", "danger")
                return redirect(url_for("history"))
            s.delete(it)

        flash("Đã xóa bản ghi lịch sử.", "success")
        return redirect(url_for("history"))

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
