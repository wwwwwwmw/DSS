from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import re
import math
import csv
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash

from config import get_settings
from db import Base, create_session_factory, session_scope
from ml import (
    CRITERIA,
    ahp_score,
    ahp_score_dataframe,
    choose_option,
    compute_ahp_weights,
    evaluate_market_position,
    generate_explanation,
    load_market_stats,
    load_models,
    normalize_weights,
    parse_mpg,
    parse_mpg_series,
    parse_pairwise_matrix_text,
    predict,
    serialize_payload,
)
from models_db import CriteriaConfig, RecommendationHistory, SavedCar, User


def create_app() -> Flask:
    settings = get_settings()

    app = Flask(__name__)
    app.secret_key = settings.secret_key

    # Upload limit (admin retrain CSV)
    max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))  # default 5MB
    app.config["MAX_CONTENT_LENGTH"] = max_upload_bytes

    @app.errorhandler(RequestEntityTooLarge)
    def _handle_413(_e):
        flash(f"File upload quá lớn. Giới hạn {max_upload_bytes // (1024 * 1024)}MB.", "danger")
        return redirect(request.referrer or url_for("admin"))

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
            # Refresh before expunge to avoid DetachedInstanceError surprises
            # (e.g., if any lazy-loaded attrs/relationships are accessed later).
            try:
                s.refresh(user)
            except Exception:
                pass
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
                # Always sync labels/directions from source to fix encoding
                src = {c["key"]: c for c in CRITERIA}
                for it in s.query(CriteriaConfig).all():
                    if it.key in src:
                        it.label = src[it.key]["label"]
                        it.direction = src[it.key]["direction"]
                return
            # Seed defaults as a normalized float weight vector.
            seed_raw = {c["key"]: float(c.get("default", 5)) for c in CRITERIA}
            seed = normalize_weights(seed_raw)
            for c in CRITERIA:
                s.add(
                    CriteriaConfig(
                        key=c["key"],
                        label=c["label"],
                        direction=c["direction"],
                        default_weight=float(seed.get(c["key"], 0.0)),
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
                    "default": float(it.default_weight),
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

    def get_maintenance_level(monthly_usd: float) -> Dict[str, str]:
        """Label maintenance cost levels (USD/month).

        Thresholds are tuned for USD/month to match model output.
        - Rất thấp: < $100/mo
        - Thấp: $100-$200/mo
        - Trung bình: $200-$300/mo
        - Cao: $300-$450/mo
        - Rất cao: >= $450/mo
        """
        try:
            x = float(monthly_usd)
        except Exception:
            x = 1e9

        if x < 100:
            return {"label": "Rất thấp", "badge_class": "success"}
        if x < 200:
            return {"label": "Thấp", "badge_class": "info"}
        if x < 300:
            return {"label": "Trung bình", "badge_class": "warning"}
        if x < 450:
            return {"label": "Cao", "badge_class": "danger"}
        return {"label": "Rất cao", "badge_class": "danger-dark"}

    def get_models():
        return load_models(settings.model_path)

    # ------------------------------------------------------------------
    # Market stats cache (loaded once at startup, refreshed on retrain)
    # ------------------------------------------------------------------
    _market_stats_cache: Dict[str, Any] = {}
    _market_stats_lock = threading.Lock()

    def get_market_stats() -> Optional[Dict[str, Dict[str, float]]]:
        if "stats" not in _market_stats_cache:
            with _market_stats_lock:
                if "stats" not in _market_stats_cache:
                    _market_stats_cache["stats"] = load_market_stats(settings.cars_csv_path)
        return _market_stats_cache.get("stats")

    def refresh_market_stats() -> None:
        with _market_stats_lock:
            _market_stats_cache["stats"] = load_market_stats(settings.cars_csv_path)

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

    def _repair_mojibake_text(s: Any) -> Any:
        """Best-effort fix for UTF-8 text that was mis-decoded as latin-1/cp1252."""
        if not isinstance(s, str) or not s:
            return s

        def _normalize_known_vi_glitches(txt: str) -> str:
            # Common Vietnamese fragments that become unreadable after codepage mismatch.
            replacements = {
                "Tu v?n": "Tư vấn",
                "Kh�ng c�": "Không có",
                "Kh�ng": "Không",
                "v?i": "với",
                "th?": "thị",
                "tru?ng": "trường",
                "xe xanh": "xe xanh",
                "��nh gi�": "Đánh giá",
                "Ðánh giá": "Đánh giá",
                "R?t th?p": "Rất thấp",
                "R?t cao": "Rất cao",
                "Th?p": "Thấp",
                "Trung b?nh": "Trung bình",
                "C?n c?n nh?c": "Cần cân nhắc",
                "c?n c?n nh?c": "cần cân nhắc",
                "NÊN CẦN NH?C": "NÊN CÂN NHẮC",
                "Kh?ng n?n mua": "Không nên mua",
                "R?i ro": "Rủi ro",
                "b?o d??ng": "bảo dưỡng",
                "nh?c": "nhắc",
            }
            out = txt
            for bad, good in replacements.items():
                out = out.replace(bad, good)
            return out

        # Fast path: only attempt repair when suspicious mojibake markers appear.
        suspicious_tokens = ("Ã", "Â", "Ä", "Å", "Æ", "Ð", "Ø", "Þ", "â€", "â€“", "â€”", "�", "?")
        if not any(tok in s for tok in suspicious_tokens):
            return s

        candidates = [s]
        for src_enc in ("latin-1", "cp1252"):
            for dst_enc in ("utf-8", "cp1258"):
                try:
                    candidates.append(s.encode(src_enc).decode(dst_enc))
                except Exception:
                    pass

        def score(txt: str) -> int:
            # Higher is better: reward Vietnamese letters, penalize replacement chars / stray '?'.
            vi_chars = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵĂÂĐÊÔƠƯÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ")
            good = sum(1 for ch in txt if ch in vi_chars)
            bad = txt.count("�") * 4 + txt.count("?")
            mojibake = sum(txt.count(tok) for tok in ("Ã", "Â", "Ä", "Å", "Æ", "Ð", "â€"))
            return good - bad - mojibake

        best = max(candidates, key=score)
        out = best if score(best) >= score(s) else s
        return _normalize_known_vi_glitches(out)

    def _repair_mojibake_obj(obj: Any) -> Any:
        if isinstance(obj, str):
            return _repair_mojibake_text(obj)
        if isinstance(obj, dict):
            return {k: _repair_mojibake_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_repair_mojibake_obj(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_repair_mojibake_obj(v) for v in obj)
        return obj

    def save_history(*, action: str, cars: List[Dict[str, Any]], payload: Dict[str, Any], summary: str):
        if not current_user.is_authenticated:
            return
        # Store action inside payload (schema can evolve without DB migrations).
        payload2 = dict(payload)
        payload2["action"] = action
        payload2 = _repair_mojibake_obj(payload2)
        payload_json = json.dumps(payload2, ensure_ascii=False)
        safe_summary = _repair_mojibake_text(summary)
        with session_scope(SessionLocal) as s:
            s.add(
                RecommendationHistory(
                    user_id=int(current_user.get_id()),
                    created_at=dt.datetime.now(dt.timezone.utc),
                    car_count=len(cars),
                    summary=safe_summary,
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

            price_drop_val = sstrip(car.get("price_drop", ""))
            if price_drop_val:
                n = f(price_drop_val)
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
            # Accept any non-negative weights; scoring will normalize.
            if not math.isfinite(v):
                v = float(c.get("default", 0.0))
            out[key] = max(0.0, min(1e6, float(v)))
        return out

    def parse_pairwise_matrix_from_form(n: int) -> Optional[List[List[float]]]:
        """Parse pairwise matrix from hidden JSON field and enforce reciprocal shape."""

        raw = request.form.get("pairwise_matrix_json", "").strip()
        if not raw:
            return None

        try:
            obj = json.loads(raw)
        except Exception as e:
            raise ValueError(f"Không đọc được ma trận so sánh cặp: {e}") from e

        if not isinstance(obj, list) or len(obj) != n:
            raise ValueError(f"Ma trận so sánh cặp phải có đúng {n} hàng.")

        out: List[List[float]] = [[1.0 for _ in range(n)] for _ in range(n)]

        for i, row in enumerate(obj):
            if not isinstance(row, list) or len(row) != n:
                raise ValueError(f"Hàng {i + 1} của ma trận phải có đúng {n} cột.")
            for j in range(i + 1, n):
                try:
                    v = float(row[j])
                except Exception as e:
                    raise ValueError(f"Giá trị ô [{i + 1}, {j + 1}] không hợp lệ.") from e

                if not math.isfinite(v) or v <= 0:
                    raise ValueError(f"Giá trị ô [{i + 1}, {j + 1}] phải là số > 0.")

                out[i][j] = v
                out[j][i] = 1.0 / v

        return out

    def build_option_score_matrix(
        *,
        criteria: List[Dict[str, Any]],
        weights: Dict[str, float],
        details_by_car_idx: Dict[int, Dict[str, float]],
        badge_by_car_idx: Dict[int, str],
    ) -> Dict[str, Any]:
        """Build AHP alternative matrices for 3 recommendation options.

        Returns:
            - summary rows/totals (0-100 scale) for each criterion and option
            - 9 pairwise 3x3 matrices (1 per criterion) with a column-sum row
        """

        normalized_weights = normalize_weights(weights)
        groups: Dict[str, List[Dict[str, float]]] = {"green": [], "yellow": [], "red": []}

        for idx, detail in details_by_car_idx.items():
            badge = badge_by_car_idx.get(int(idx))
            if badge in groups and isinstance(detail, dict):
                groups[badge].append(detail)

        option_meta = [
            ("green", "Phương án 1: Nên mua ngay"),
            ("yellow", "Phương án 2: Nên cân nhắc"),
            ("red", "Phương án 3: Không nên mua"),
        ]

        def avg_scaled(key: str, badge: str) -> Optional[float]:
            rows = groups.get(badge) or []
            if not rows:
                return None

            w = float(normalized_weights.get(key, 0.0))
            if w <= 1e-12:
                return 0.0

            vals: List[float] = []
            for d in rows:
                try:
                    contrib = float(d.get(key, 0.0))
                except Exception:
                    contrib = 0.0
                scaled = contrib / w
                if scaled < 0:
                    scaled = 0.0
                elif scaled > 1:
                    scaled = 1.0
                vals.append(float(scaled))

            return float(sum(vals) / len(vals)) if vals else None

        rows_out: List[Dict[str, Any]] = []
        criterion_pairwise_tables: List[Dict[str, Any]] = []

        def clamp_ahp_ratio(v: float) -> float:
            if v < (1.0 / 9.0):
                return 1.0 / 9.0
            if v > 9.0:
                return 9.0
            return float(v)

        for c in criteria:
            key = c["key"]
            raw_label = str(c.get("label") or key)
            if ":" in raw_label:
                # Prefer the human-friendly Vietnamese part after the key prefix.
                label = raw_label.split(":", 1)[1].strip()
            else:
                label = raw_label.strip()
            row = {
                "key": key,
                "label": label,
                "weight_pct": float(normalized_weights.get(key, 0.0)) * 100.0,
                "green": None,
                "yellow": None,
                "red": None,
            }

            for b in ("green", "yellow", "red"):
                val = avg_scaled(key, b)
                row[b] = (float(val) * 100.0) if val is not None else None

            rows_out.append(row)

            # Build pairwise comparison matrix (Alternatives 3x3) for this criterion.
            scores_for_ratio: List[float] = []
            for badge, _lbl in option_meta:
                v = avg_scaled(key, badge)
                # If a group is absent, use neutral baseline so matrix remains defined.
                scores_for_ratio.append(0.5 if v is None else float(v))

            m = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
            for i in range(3):
                for j in range(i + 1, 3):
                    den = float(scores_for_ratio[j])
                    num = float(scores_for_ratio[i])
                    ratio = 9.0 if den <= 1e-12 else (num / den)
                    ratio = clamp_ahp_ratio(ratio)
                    m[i][j] = float(ratio)
                    m[j][i] = float(1.0 / ratio)

            col_sums = [
                float(m[0][j] + m[1][j] + m[2][j])
                for j in range(3)
            ]

            criterion_pairwise_tables.append(
                {
                    "criterion_key": key,
                    "criterion_label": label,
                    "weight_pct": float(normalized_weights.get(key, 0.0)) * 100.0,
                    "option_labels": [lbl for _badge, lbl in option_meta],
                    "matrix": m,
                    "col_sums": col_sums,
                }
            )

        totals: Dict[str, Optional[float]] = {"green": None, "yellow": None, "red": None}
        for b in ("green", "yellow", "red"):
            if not groups.get(b):
                continue
            s = 0.0
            for c in criteria:
                key = c["key"]
                val = avg_scaled(key, b)
                if val is None:
                    continue
                s += float(normalized_weights.get(key, 0.0)) * float(val)
            totals[b] = s * 100.0

        return {
            "rows": rows_out,
            "counts": {
                "green": len(groups["green"]),
                "yellow": len(groups["yellow"]),
                "red": len(groups["red"]),
            },
            "totals": totals,
            "criterion_pairwise_tables": criterion_pairwise_tables,
        }

    def build_car_option_matrix(
        *,
        criteria: List[Dict[str, Any]],
        weights: Dict[str, float],
        details_by_car_idx: Dict[int, Dict[str, float]],
    ) -> Dict[str, Any]:
        """Build per-criterion pairwise matrices where alternatives are user-entered cars.

        For evaluate mode, options are dynamic: Xe #1, Xe #2, ..., Xe #N.
        """

        normalized_weights = normalize_weights(weights)
        car_indices = sorted(int(i) for i in details_by_car_idx.keys())
        option_labels = [f"Xe #{i}" for i in car_indices]

        def clamp_ahp_ratio(v: float) -> float:
            if v < (1.0 / 9.0):
                return 1.0 / 9.0
            if v > 9.0:
                return 9.0
            return float(v)

        def scaled_value(detail: Dict[str, float], key: str, w: float) -> float:
            if w <= 1e-12:
                return 0.0
            try:
                contrib = float(detail.get(key, 0.0))
            except Exception:
                contrib = 0.0
            v = contrib / w
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return float(v)

        criterion_pairwise_tables: List[Dict[str, Any]] = []
        m = len(option_labels)

        for c in criteria:
            key = c["key"]
            raw_label = str(c.get("label") or key)
            if ":" in raw_label:
                label = raw_label.split(":", 1)[1].strip()
            else:
                label = raw_label.strip()

            w = float(normalized_weights.get(key, 0.0))
            scores_for_ratio: List[float] = []
            for idx in car_indices:
                detail = details_by_car_idx.get(idx) or {}
                scores_for_ratio.append(scaled_value(detail, key, w))

            matrix = [[1.0 for _ in range(m)] for _ in range(m)]
            for i in range(m):
                for j in range(i + 1, m):
                    num = float(scores_for_ratio[i])
                    den = float(scores_for_ratio[j])

                    if num <= 1e-12 and den <= 1e-12:
                        ratio = 1.0
                    else:
                        ratio = 9.0 if den <= 1e-12 else (num / den)

                    ratio = clamp_ahp_ratio(ratio)
                    matrix[i][j] = float(ratio)
                    matrix[j][i] = float(1.0 / ratio)

            col_sums = [
                float(sum(matrix[i][j] for i in range(m)))
                for j in range(m)
            ]

            criterion_pairwise_tables.append(
                {
                    "criterion_key": key,
                    "criterion_label": label,
                    "weight_pct": w * 100.0,
                    "option_labels": option_labels,
                    "matrix": matrix,
                    "col_sums": col_sums,
                }
            )

        return {
            "car_count": m,
            "option_labels": option_labels,
            "criterion_pairwise_tables": criterion_pairwise_tables,
        }

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
        return render_template(
            "index.html",
            criteria=criteria,
            results=None,
            top_recommendations=None,
            high_risk_results=None,
            cars=None,
            weights=None,
            chart_data=None,
            pairwise_matrix=None,
            option_matrix=None,
        )

    @app.route("/evaluate", methods=["GET", "POST"])
    def evaluate():
        """Đánh giá thị trường: so sánh xe của người dùng với kho CSV (market stats)."""

        criteria = load_criteria()
        if request.method == "GET":
            return render_template(
                "evaluate.html",
                criteria=criteria,
                results=None,
                cars=None,
                weights=None,
                chart_data=None,
                pairwise_matrix=None,
                option_matrix=None,
            )

        pairwise_matrix = None
        weights = parse_weights_from_form(criteria)
        try:
            pairwise_matrix = parse_pairwise_matrix_from_form(len(criteria))
            if pairwise_matrix:
                ahp_res = compute_ahp_weights(pairwise_matrix)
                weights = {
                    criteria[i]["key"]: float(ahp_res.weights[i])
                    for i in range(len(criteria))
                }
        except Exception as e:
            flash(f"Ma trận AHP không hợp lệ: {e}", "danger")

        cars = parse_cars_from_form()
        errors = validate_cars(cars, min_cars=3)
        if errors:
            for e in errors[:6]:
                flash(e, "danger")
            return render_template(
                "evaluate.html",
                criteria=criteria,
                results=None,
                cars=cars,
                weights=weights,
                chart_data=None,
                pairwise_matrix=pairwise_matrix,
                option_matrix=None,
            )

        scores, ahp_details = ahp_score(cars, weights)

        models = get_models()
        if not models:
            flash("Chưa có model. Admin hãy retrain hoặc chạy train.py trước.", "danger")
            accident_probs = [0.5 for _ in cars]
            maint_monthly = [300.0 for _ in cars]
        else:
            accident_probs, maint_monthly = predict(models, cars)

        market_stats = get_market_stats()

        results: List[Dict[str, Any]] = []
        for idx, (car, s, ap, mm) in enumerate(
            zip(cars, scores, accident_probs, maint_monthly), start=1
        ):
            risk_pct = float(ap) * 100.0
            annual_cost = int(round(mm * 12))
            risk_level = get_risk_level(risk_pct)
            maint_level = get_maintenance_level(float(mm))

            explanation = None
            percentile = None
            if market_stats:
                mkt_pos = evaluate_market_position(car, market_stats, weights)
                percentile = mkt_pos["overall_percentile"]
                explanation = generate_explanation(
                    car, market_stats, float(ap), float(mm), mkt_pos, float(s),
                )

            _option_eval, badge, message = choose_option(float(s), float(ap), float(mm), percentile=percentile)
            option = f"Xe #{idx}"

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
                    "explanation": explanation,
                }
            )

        details_by_car_idx: Dict[int, Dict[str, float]] = {
            i + 1: ahp_details[i] for i in range(len(ahp_details))
        }
        option_matrix = build_car_option_matrix(
            criteria=criteria,
            weights=weights,
            details_by_car_idx=details_by_car_idx,
        )

        # History
        payload = {
            "weights": weights,
            "cars": cars,
            "results": [
                {k: v for k, v in r.items() if k != "explanation"}
                for r in results
            ],
        }
        summary = "Đánh giá: Đã tính cho 1 xe" if len(cars) == 1 else f"Đánh giá: {len(cars)} xe"
        if market_stats:
            summary += " • so với thị trường"
        save_history(action="evaluate", cars=cars, payload=payload, summary=summary)

        # --- Build chart data for frontend ---
        normalized_weights = normalize_weights(weights)
        chart_data = sanitize_for_json({
            "criteria_keys": [c["key"] for c in criteria],
            "criteria_labels": [c["label"].split(":")[0].strip() for c in criteria],
            "criteria_directions": [c["direction"] for c in criteria],
            "weights": normalized_weights,
            "cars": cars,
            "ahp_details": ahp_details,
            "scores": scores,
        })

        return render_template(
            "evaluate.html",
            criteria=criteria,
            results=results,
            cars=cars,
            weights=weights,
            chart_data=chart_data,
            pairwise_matrix=pairwise_matrix,
            option_matrix=option_matrix,
        )

    @app.post("/recommend")
    def recommend():
        criteria = load_criteria()

        pairwise_matrix = None
        weights = parse_weights_from_form(criteria)
        try:
            pairwise_matrix = parse_pairwise_matrix_from_form(len(criteria))
            if pairwise_matrix:
                ahp_res = compute_ahp_weights(pairwise_matrix)
                weights = {
                    criteria[i]["key"]: float(ahp_res.weights[i])
                    for i in range(len(criteria))
                }
        except Exception as e:
            flash(f"Ma trận AHP không hợp lệ: {e}", "danger")

        cars = parse_cars_from_form()
        errors = validate_cars(cars, min_cars=1)
        if errors:
            for e in errors[:6]:
                flash(e, "danger")
            return render_template(
                "index.html",
                criteria=criteria,
                results=None,
                high_risk_results=None,
                top_recommendations=None,
                cars=cars,
                weights=weights,
                chart_data=None,
                pairwise_matrix=pairwise_matrix,
                option_matrix=None,
            )

        # Predict risk first, then filter high-risk cars out BEFORE AHP ranking.
        models = get_models()
        if not models:
            flash("Chưa có model. Admin hãy retrain hoặc chạy train.py trước.", "danger")
            accident_probs = [0.5 for _ in cars]
            maint_monthly = [300.0 for _ in cars]
        else:
            accident_probs, maint_monthly = predict(models, cars)

        high_risk_idx = []
        safe_idx = []
        for i, ap in enumerate(accident_probs):
            if float(ap) > 0.60:
                high_risk_idx.append(i)
            else:
                safe_idx.append(i)

        high_risk_results: List[Dict[str, Any]] = []
        for i in high_risk_idx:
            ap = float(accident_probs[i])
            mm = float(maint_monthly[i])
            # AHP score is not computed for high-risk cars (excluded from ranking)
            option, badge, message = choose_option(0.0, ap, mm)
            risk_pct = ap * 100.0
            risk_level = get_risk_level(risk_pct)
            maint_level = get_maintenance_level(mm)
            high_risk_results.append(
                {
                    "idx": i + 1,
                    "ahp_score": 0.0,
                    "accident_risk_pct": risk_pct,
                    "risk_level_label": risk_level["label"],
                    "risk_level_badge": risk_level["badge_class"],
                    "maintenance_monthly": int(round(mm)),
                    "maintenance_level_label": maint_level["label"],
                    "maintenance_level_badge": maint_level["badge_class"],
                    "option": option,
                    "badge": badge,
                    "message": message,
                }
            )

        safe_cars = [cars[i] for i in safe_idx]
        safe_acc = [accident_probs[i] for i in safe_idx]
        safe_maint = [maint_monthly[i] for i in safe_idx]
        safe_scores, safe_ahp_details = ahp_score(safe_cars, weights) if safe_cars else ([], [])

        results: List[Dict[str, Any]] = []
        for local_i, (s, ap, mm) in enumerate(zip(safe_scores, safe_acc, safe_maint)):
            original_idx = safe_idx[local_i] + 1
            option, badge, message = choose_option(float(s), float(ap), float(mm))
            risk_pct = float(ap) * 100.0
            risk_level = get_risk_level(risk_pct)
            maint_level = get_maintenance_level(float(mm))
            results.append(
                {
                    "idx": original_idx,
                    "ahp_score": float(s),
                    "accident_risk_pct": risk_pct,
                    "risk_level_label": risk_level["label"],
                    "risk_level_badge": risk_level["badge_class"],
                    "maintenance_monthly": int(round(float(mm))),
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

        # --- Option matrix (per-criterion scoring for the 3 recommendation options) ---
        _all_scores, all_ahp_details = ahp_score(cars, weights) if cars else ([], [])
        details_by_car_idx: Dict[int, Dict[str, float]] = {
            i + 1: all_ahp_details[i] for i in range(len(all_ahp_details))
        }
        badge_by_car_idx: Dict[int, str] = {}
        for r in results:
            badge_by_car_idx[int(r["idx"])] = str(r.get("badge") or "")
        for r in high_risk_results:
            badge_by_car_idx[int(r["idx"])] = "red"

        option_matrix = build_option_score_matrix(
            criteria=criteria,
            weights=weights,
            details_by_car_idx=details_by_car_idx,
            badge_by_car_idx=badge_by_car_idx,
        )

        payload = _safe_json_loads(serialize_payload(weights, cars, results)) or {"weights": weights, "cars": cars, "results": results}
        summary = f"Tư vấn: Top xe #{top_recommendations[0]['idx']}" if top_recommendations else "Tư vấn: Không có xe xanh"
        save_history(action="recommend", cars=cars, payload=payload, summary=summary)

        # --- Build chart data for frontend ---
        normalized_weights = normalize_weights(weights)
        chart_data = sanitize_for_json({
            "criteria_keys": [c["key"] for c in criteria],
            "criteria_labels": [c["label"].split(":")[0].strip() for c in criteria],
            "criteria_directions": [c["direction"] for c in criteria],
            "weights": normalized_weights,
            "cars": safe_cars,
            "ahp_details": safe_ahp_details,
            "scores": safe_scores,
        })

        return render_template(
            "index.html",
            criteria=criteria,
            results=results,
            high_risk_results=high_risk_results,
            top_recommendations=top_recommendations,
            cars=cars,
            weights=weights,
            chart_data=chart_data,
            pairwise_matrix=pairwise_matrix,
            option_matrix=option_matrix,
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
            "seller_rating",
            "driver_rating",
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
                    "seller_rating": row.get("seller_rating", ""),
                    "driver_rating": row.get("driver_rating", ""),
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
            maint_level = get_maintenance_level(float(mm))
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
                    "title": _repair_mojibake_text(it.title),
                    "source": it.source,
                    "car": _repair_mojibake_obj(_safe_json_loads(it.car_json) or {}),
                }
            )
        return render_template("my_cars.html", items=cars_out)

    @app.get("/my-cars/new")
    @login_required
    def my_car_new():
        return render_template("my_car_new.html")

    @app.get("/my-cars/<int:item_id>")
    @login_required
    def my_car_detail(item_id: int):
        with session_scope(SessionLocal) as s:
            it = s.get(SavedCar, item_id)
            if not it or it.user_id != int(current_user.get_id()):
                flash("Không tìm thấy xe.", "danger")
                return redirect(url_for("my_cars"))
            car = sanitize_for_json(_safe_json_loads(it.car_json) or {})
            car = _repair_mojibake_obj(car)

        return render_template(
            "my_car_detail.html",
            item={
                "id": it.id,
                "created_at": it.created_at,
                    "title": _repair_mojibake_text(it.title),
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
                        created_at=dt.datetime.now(dt.timezone.utc),
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
            rows = (
                s.query(RecommendationHistory)
                .filter(RecommendationHistory.user_id == int(current_user.get_id()))
                .order_by(RecommendationHistory.created_at.desc())
                .limit(50)
                .all()
            )
        items = [
            {
                "id": it.id,
                "created_at": it.created_at,
                "car_count": it.car_count,
                "summary": _repair_mojibake_text(it.summary),
            }
            for it in rows
        ]
        return render_template("history.html", items=items)

    @app.get("/history/<int:item_id>")
    @login_required
    def history_detail(item_id: int):
        with session_scope(SessionLocal) as s:
            it = s.get(RecommendationHistory, item_id)
            if not it or it.user_id != int(current_user.get_id()):
                flash("Không tìm thấy bản ghi lịch sử.", "danger")
                return redirect(url_for("history"))
            payload = _repair_mojibake_obj(_safe_json_loads(it.payload_json) or {})

        return render_template(
            "history_detail.html",
            item={
                "id": it.id,
                "created_at": it.created_at,
                "car_count": it.car_count,
                "summary": _repair_mojibake_text(it.summary),
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

    def _count_csv_rows(csv_file: str) -> Optional[int]:
        try:
            p = Path(csv_file)
            if not p.exists() or not p.is_file():
                return None
            with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                row_count = sum(1 for _ in csv.reader(f))
            return max(0, row_count - 1)
        except Exception:
            return None

    def _build_model_metrics() -> Dict[str, Any]:
        models = get_models()
        if not models:
            return {
                "available": False,
                "trained_at": None,
                "train_rows": None,
                "accident_accuracy": None,
                "maintenance_mae": None,
                "feature_importance": [],
                "cars_csv": None,
                "note": None,
            }

        meta = dict(models.meta or {})
        feature_names = (
            meta.get("feature_cols")
            or meta.get("numeric_features")
            or meta.get("features")
            or []
        )

        fi_pairs: List[Dict[str, Any]] = []
        raw_fi = meta.get("feature_importance")
        if isinstance(raw_fi, list):
            for it in raw_fi:
                if not isinstance(it, dict):
                    continue
                name = str(it.get("name") or "").strip()
                val = it.get("value")
                try:
                    fval = float(val)
                except Exception:
                    continue
                if name:
                    fi_pairs.append({"name": name, "value": fval})

        if not fi_pairs:
            try:
                importances = list(getattr(models.accident_clf, "feature_importances_", []) or [])
                for i, val in enumerate(importances):
                    name = feature_names[i] if i < len(feature_names) else f"feature_{i + 1}"
                    fi_pairs.append({"name": str(name), "value": float(val)})
            except Exception:
                fi_pairs = []

        fi_pairs.sort(key=lambda x: x["value"], reverse=True)

        train_rows = meta.get("train_rows")
        if train_rows is not None:
            try:
                train_rows = int(train_rows)
            except Exception:
                train_rows = None

        cars_csv = meta.get("cars_csv") or settings.cars_csv_path
        if train_rows is None and cars_csv:
            train_rows = _count_csv_rows(str(cars_csv))

        def _f(v: Any) -> Optional[float]:
            try:
                fv = float(v)
                return fv if math.isfinite(fv) else None
            except Exception:
                return None

        return {
            "available": True,
            "trained_at": meta.get("trained_at"),
            "train_rows": train_rows,
            "accident_accuracy": _f(meta.get("accident_accuracy")),
            "maintenance_mae": _f(meta.get("maintenance_mae")),
            "feature_importance": fi_pairs[:12],
            "cars_csv": str(cars_csv) if cars_csv else None,
            "log_file": meta.get("log_file"),
            "note": meta.get("note"),
        }

    def _build_admin_file_reports() -> Dict[str, Any]:
        logs_dir = Path("./logs")
        data_dir = Path("./data")

        log_files = []
        if logs_dir.exists():
            for p in sorted(logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True):
                st = p.stat()
                log_files.append(
                    {
                        "name": p.name,
                        "path": str(p),
                        "size_kb": round(st.st_size / 1024.0, 1),
                        "modified_at": dt.datetime.fromtimestamp(st.st_mtime),
                    }
                )

        latest_log_text = ""
        if log_files:
            latest = Path(log_files[0]["path"])
            try:
                lines = latest.read_text(encoding="utf-8", errors="ignore").splitlines()
                latest_log_text = "\n".join(lines[-80:])
            except Exception:
                latest_log_text = "Không đọc được nội dung file log gần nhất."

        csv_files = []
        if data_dir.exists():
            for p in sorted(data_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
                st = p.stat()
                csv_files.append(
                    {
                        "name": p.name,
                        "path": str(p),
                        "size_kb": round(st.st_size / 1024.0, 1),
                        "rows": _count_csv_rows(str(p)),
                        "modified_at": dt.datetime.fromtimestamp(st.st_mtime),
                    }
                )

        default_csv = Path(settings.cars_csv_path)
        default_csv_info = None
        if default_csv.exists() and default_csv.is_file():
            st = default_csv.stat()
            default_csv_info = {
                "name": default_csv.name,
                "path": str(default_csv),
                "size_kb": round(st.st_size / 1024.0, 1),
                "rows": _count_csv_rows(str(default_csv)),
                "modified_at": dt.datetime.fromtimestamp(st.st_mtime),
            }

        return {
            "log_files": log_files,
            "latest_log_text": latest_log_text,
            "csv_files": csv_files,
            "default_csv": default_csv_info,
        }

    def _build_admin_usage_stats() -> Dict[str, Any]:
        with session_scope(SessionLocal) as s:
            user_count = int(s.query(User).count())
            history_count = int(s.query(RecommendationHistory).count())
            saved_car_count = int(s.query(SavedCar).count())
            histories = (
                s.query(RecommendationHistory)
                .order_by(RecommendationHistory.created_at.desc())
                .limit(500)
                .all()
            )

        action_counts = {"recommend": 0, "evaluate": 0, "compare": 0, "stock": 0, "other": 0}
        badge_counts = {"green": 0, "yellow": 0, "red": 0}

        daily_map: Dict[str, int] = {}
        for it in histories:
            dkey = (it.created_at or dt.datetime.now()).date().isoformat()
            daily_map[dkey] = int(daily_map.get(dkey, 0)) + 1

            payload = _safe_json_loads(it.payload_json) or {}
            action = str(payload.get("action") or "other")
            if action not in action_counts:
                action = "other"
            action_counts[action] = int(action_counts[action]) + 1

            results = payload.get("results")
            if isinstance(results, list):
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    badge = str(r.get("badge") or "").strip()
                    if badge in badge_counts:
                        badge_counts[badge] = int(badge_counts[badge]) + 1

        today = dt.date.today()
        daily_labels: List[str] = []
        daily_values: List[int] = []
        for i in range(13, -1, -1):
            d = today - dt.timedelta(days=i)
            key = d.isoformat()
            daily_labels.append(d.strftime("%d/%m"))
            daily_values.append(int(daily_map.get(key, 0)))

        total_recommendation_badges = int(sum(badge_counts.values()))
        green_rate = (badge_counts["green"] / total_recommendation_badges * 100.0) if total_recommendation_badges else 0.0
        red_rate = (badge_counts["red"] / total_recommendation_badges * 100.0) if total_recommendation_badges else 0.0

        return {
            "user_count": user_count,
            "history_count": history_count,
            "saved_car_count": saved_car_count,
            "action_counts": action_counts,
            "badge_counts": badge_counts,
            "daily_labels": daily_labels,
            "daily_values": daily_values,
            "green_rate": green_rate,
            "red_rate": red_rate,
        }

    @app.get("/admin")
    @login_required
    def admin():
        if not require_admin():
            return redirect(url_for("home"))
        with session_scope(SessionLocal) as s:
            users = s.query(User).order_by(User.id.asc()).all()
            criteria = s.query(CriteriaConfig).order_by(CriteriaConfig.id.asc()).all()
        model_metrics = _build_model_metrics()
        usage_stats = _build_admin_usage_stats()
        file_reports = _build_admin_file_reports()
        return render_template(
            "admin.html",
            users=users,
            criteria=criteria,
            model_metrics=model_metrics,
            usage_stats=usage_stats,
            file_reports=file_reports,
        )

    @app.post("/admin/criteria")
    @login_required
    def admin_update_criteria():
        if not require_admin():
            return redirect(url_for("home"))

        matrix_text = request.form.get("pairwise_matrix", "")

        with session_scope(SessionLocal) as s:
            items = s.query(CriteriaConfig).order_by(CriteriaConfig.id.asc()).all()
            keys = [it.key for it in items]
            n = len(keys)

            try:
                mat = parse_pairwise_matrix_from_form(n)
                if mat is None:
                    mat = parse_pairwise_matrix_text(matrix_text, n)
                res = compute_ahp_weights(mat)
            except Exception as e:
                flash(f"Ma trận AHP không hợp lệ: {e}", "danger")
                return redirect(url_for("admin"))

            if not res.is_valid:
                flash(f"Ma trận AHP không nhất quán (CR={res.cr:.3f} >= 0.100). Vui lòng nhập lại.", "danger")
                return redirect(url_for("admin"))

            for it, w in zip(items, res.weights):
                it.default_weight = float(w)

        flash("Đã cập nhật trọng số mặc định (AHP).", "success")
        return redirect(url_for("admin"))

    @app.post("/admin/retrain")
    @login_required
    def admin_retrain():
        if not require_admin():
            return redirect(url_for("home"))

        return_tab = (request.form.get("return_tab") or "").strip()
        safe_tab = return_tab if return_tab in {"tab-ahp", "tab-model", "tab-usage", "tab-logs"} else None

        csv_path = Path(settings.cars_csv_path)
        uploaded = request.files.get("csv_file")
        if uploaded and uploaded.filename:
            filename = secure_filename(uploaded.filename)
            if not filename.lower().endswith(".csv"):
                flash("Chỉ cho phép upload file .csv", "danger")
                return redirect(url_for("admin", tab=safe_tab) if safe_tab else url_for("admin"))

            # Enforce per-file size limit (in addition to MAX_CONTENT_LENGTH)
            max_bytes = int(app.config.get("MAX_CONTENT_LENGTH") or (5 * 1024 * 1024))
            try:
                uploaded.stream.seek(0, os.SEEK_END)
                size = int(uploaded.stream.tell())
                uploaded.stream.seek(0)
            except Exception:
                size = int(request.content_length or 0)

            if size and size > max_bytes:
                flash(f"File quá lớn ({size // 1024}KB). Giới hạn {max_bytes // (1024 * 1024)}MB.", "danger")
                return redirect(url_for("admin", tab=safe_tab) if safe_tab else url_for("admin"))

            # Save with a timestamped name to avoid overwriting.
            ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            csv_path = Path("./data") / f"cars_uploaded_{ts}.csv"
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
            ], timeout=600)
            refresh_market_stats()
            flash("Retrain thành công.", "success")
        except Exception as e:
            flash(f"Retrain thất bại: {e}", "danger")

        return redirect(url_for("admin", tab=safe_tab) if safe_tab else url_for("admin"))

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

    @app.get("/admin/users/new")
    @login_required
    def admin_user_new():
        if not require_admin():
            return redirect(url_for("home"))
        return render_template("admin_user_form.html", mode="new", user_item=None)

    @app.post("/admin/users/new")
    @login_required
    def admin_user_create():
        if not require_admin():
            return redirect(url_for("home"))

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "user").strip().lower()

        if not email:
            flash("Email không được để trống.", "danger")
            return redirect(url_for("admin_user_new"))
        if len(password) < 6:
            flash("Mật khẩu tối thiểu 6 ký tự.", "danger")
            return redirect(url_for("admin_user_new"))
        if role not in {"user", "admin"}:
            role = "user"

        with session_scope(SessionLocal) as s:
            exists = s.query(User).filter(User.email == email).first()
            if exists:
                flash("Email đã tồn tại.", "danger")
                return redirect(url_for("admin_user_new"))
            s.add(User(email=email, password_hash=generate_password_hash(password), role=role))

        flash("Đã thêm user mới.", "success")
        return redirect(url_for("admin", tab="tab-usage"))

    @app.get("/admin/users/<int:user_id>")
    @login_required
    def admin_user_detail(user_id: int):
        if not require_admin():
            return redirect(url_for("home"))

        with session_scope(SessionLocal) as s:
            u = s.get(User, user_id)
            if not u:
                flash("User không tồn tại.", "danger")
                return redirect(url_for("admin", tab="tab-usage"))

            history_count = (
                s.query(RecommendationHistory)
                .filter(RecommendationHistory.user_id == u.id)
                .count()
            )
            saved_count = s.query(SavedCar).filter(SavedCar.user_id == u.id).count()

            recent_history = (
                s.query(RecommendationHistory)
                .filter(RecommendationHistory.user_id == u.id)
                .order_by(RecommendationHistory.created_at.desc())
                .limit(10)
                .all()
            )

            user_out = {"id": u.id, "email": u.email, "role": u.role}
            hist_out = [
                {
                    "id": h.id,
                    "created_at": h.created_at,
                    "summary": _repair_mojibake_text(h.summary),
                    "car_count": h.car_count,
                }
                for h in recent_history
            ]

        return render_template(
            "admin_user_detail.html",
            user_item=user_out,
            history_count=int(history_count),
            saved_count=int(saved_count),
            recent_history=hist_out,
        )

    @app.get("/admin/users/<int:user_id>/edit")
    @login_required
    def admin_user_edit(user_id: int):
        if not require_admin():
            return redirect(url_for("home"))

        with session_scope(SessionLocal) as s:
            u = s.get(User, user_id)
            if not u:
                flash("User không tồn tại.", "danger")
                return redirect(url_for("admin", tab="tab-usage"))
            user_out = {"id": u.id, "email": u.email, "role": u.role}

        return render_template("admin_user_form.html", mode="edit", user_item=user_out)

    @app.post("/admin/users/<int:user_id>/edit")
    @login_required
    def admin_user_update(user_id: int):
        if not require_admin():
            return redirect(url_for("home"))

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "user").strip().lower()
        if role not in {"user", "admin"}:
            role = "user"

        if not email:
            flash("Email không được để trống.", "danger")
            return redirect(url_for("admin_user_edit", user_id=user_id))
        if password and len(password) < 6:
            flash("Nếu đổi mật khẩu, cần tối thiểu 6 ký tự.", "danger")
            return redirect(url_for("admin_user_edit", user_id=user_id))

        with session_scope(SessionLocal) as s:
            u = s.get(User, user_id)
            if not u:
                flash("User không tồn tại.", "danger")
                return redirect(url_for("admin", tab="tab-usage"))

            exists = s.query(User).filter(User.email == email, User.id != user_id).first()
            if exists:
                flash("Email đã được user khác sử dụng.", "danger")
                return redirect(url_for("admin_user_edit", user_id=user_id))

            u.email = email
            u.role = role
            if password:
                u.password_hash = generate_password_hash(password)

        flash("Đã cập nhật user.", "success")
        return redirect(url_for("admin_user_detail", user_id=user_id))

    @app.post("/admin/users/<int:user_id>/delete")
    @login_required
    def admin_user_delete(user_id: int):
        if not require_admin():
            return redirect(url_for("home"))

        current_id = int(current_user.get_id()) if current_user.is_authenticated else -1
        if user_id == current_id:
            flash("Không thể tự xóa chính mình.", "danger")
            return redirect(url_for("admin", tab="tab-usage"))

        with session_scope(SessionLocal) as s:
            u = s.get(User, user_id)
            if not u:
                flash("User không tồn tại.", "danger")
                return redirect(url_for("admin", tab="tab-usage"))

            s.query(RecommendationHistory).filter(RecommendationHistory.user_id == user_id).delete()
            s.query(SavedCar).filter(SavedCar.user_id == user_id).delete()
            s.delete(u)

        flash("Đã xóa user cùng dữ liệu liên quan.", "success")
        return redirect(url_for("admin", tab="tab-usage"))

    return app


app = create_app()


if __name__ == "__main__":
    # Render (and most PaaS) requires binding to 0.0.0.0 and the injected PORT.
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5002"))
    debug = str(os.getenv("FLASK_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}

    if debug:
        app.run(debug=True, host=host, port=port)
    else:
        try:
            from waitress import serve

            serve(app, host=host, port=port)
        except Exception:
            app.run(debug=False, host=host, port=port)
