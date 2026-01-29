from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


CRITERIA = [
    {"key": "price", "label": "price: Giá bán xe", "direction": "cost", "default": 7},
    {"key": "mileage", "label": "mileage: Số dặm đã chạy", "direction": "cost", "default": 6},
    {"key": "year", "label": "year: Năm sản xuất", "direction": "benefit", "default": 6},
    {"key": "accidents_or_damage", "label": "accidents_or_damage: Tai nạn/hư hại", "direction": "cost", "default": 8},
    {"key": "one_owner", "label": "one_owner: Một chủ", "direction": "benefit", "default": 5},
    {"key": "driver_rating", "label": "driver_rating: Đánh giá người lái", "direction": "benefit", "default": 4},
    {"key": "seller_rating", "label": "seller_rating: Uy tín người bán", "direction": "benefit", "default": 5},
    {"key": "mpg", "label": "mpg: Hiệu suất nhiên liệu", "direction": "benefit", "default": 4},
    {"key": "price_drop", "label": "price_drop: Mức giảm giá", "direction": "benefit", "default": 3},
]


@dataclass
class LoadedModels:
    preprocessor: Any
    accident_clf: Any
    maint_reg: Any
    meta: Dict[str, Any]


def parse_mpg(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)(?:\s*[-–]\s*(\d+(?:\.\d+)?))?$", s)
    if not m:
        try:
            return float(s)
        except Exception:
            return None
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) else None
    return (a + b) / 2.0 if b is not None else a


def load_models(model_path: str) -> Optional[LoadedModels]:
    path = Path(model_path)
    if not path.exists():
        return None
    with path.open("rb") as f:
        pkg = pickle.load(f)
    return LoadedModels(
        preprocessor=pkg["preprocessor"],
        accident_clf=pkg["accident_clf"],
        maint_reg=pkg["maint_reg"],
        meta=pkg.get("meta", {}),
    )


def normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    weights = {k: max(0.0, float(v)) for k, v in raw.items()}
    s = sum(weights.values())
    if s <= 0:
        n = len(weights) or 1
        return {k: 1.0 / n for k in weights}
    return {k: v / s for k, v in weights.items()}


def _minmax(values: List[Optional[float]], direction: str) -> List[float]:
    xs = [v for v in values if v is not None and not math.isnan(float(v))]
    if not xs:
        return [0.0 for _ in values]
    lo, hi = float(min(xs)), float(max(xs))
    if abs(hi - lo) < 1e-12:
        scaled = [0.5 if (v is not None) else 0.0 for v in values]
        return scaled

    out: List[float] = []
    for v in values:
        if v is None or math.isnan(float(v)):
            out.append(0.0)
            continue
        t = (float(v) - lo) / (hi - lo)
        out.append(1.0 - t if direction == "cost" else t)
    return out


def ahp_score(cars: List[Dict[str, Any]], weights: Dict[str, float]) -> List[float]:
    # Here "AHP" is implemented as normalized weights + min-max scoring per criterion.
    ws = normalize_weights(weights)

    per_key_scaled: Dict[str, List[float]] = {}
    for c in CRITERIA:
        key = c["key"]
        direction = c["direction"]
        values: List[Optional[float]] = []
        for car in cars:
            v = car.get(key)
            if key == "mpg":
                values.append(parse_mpg(v))
            else:
                try:
                    values.append(None if v is None or v == "" else float(v))
                except Exception:
                    values.append(None)
        per_key_scaled[key] = _minmax(values, direction)

    scores: List[float] = []
    for i in range(len(cars)):
        s = 0.0
        for c in CRITERIA:
            key = c["key"]
            s += ws.get(key, 0.0) * per_key_scaled[key][i]
        scores.append(float(s))
    return scores


def predict(models: LoadedModels, cars: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    # Build a minimal feature set. Categorical features are not required for the UI input.
    # Training will create the same columns; missing columns will be handled by the preprocessor.
    rows: List[Dict[str, Any]] = []
    for car in cars:
        rows.append(
            {
                "year": _to_float(car.get("year")),
                "mileage": _to_float(car.get("mileage")),
                "mpg": parse_mpg(car.get("mpg")),
                "one_owner": _to_float(car.get("one_owner")),
                "personal_use_only": 1.0,
                "seller_rating": _to_float(car.get("seller_rating")),
                "driver_rating": _to_float(car.get("driver_rating")),
                "driver_reviews_num": 0.0,
                "price_drop": _to_float(car.get("price_drop")),
                "price": _to_float(car.get("price")),
            }
        )

    import pandas as pd  # lazy import

    X = pd.DataFrame(rows)
    Xp = models.preprocessor.transform(X)

    # Accident risk probability
    if hasattr(models.accident_clf, "predict_proba"):
        proba = models.accident_clf.predict_proba(Xp)[:, 1]
    else:
        proba = models.accident_clf.predict(Xp)
    maint = models.maint_reg.predict(Xp)
    return [float(x) for x in proba], [float(x) for x in maint]


def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def choose_option(ahp: float, accident_risk: float, maint_monthly: float) -> Tuple[str, str, str]:
    # Simple decision rules. You can tune thresholds in admin later.
    # accident_risk in [0,1]
    accident_pct = accident_risk * 100.0

    if accident_pct <= 25.0 and maint_monthly <= 220.0 and ahp >= 0.60:
        return (
            "Phương án 1: NÊN MUA NGAY",
            "green",
            "Ưu tiên cao. Liên hệ người bán sớm.",
        )
    if accident_pct <= 55.0 and maint_monthly <= 350.0:
        return (
            "Phương án 2: CẦN CÂN NHẮC",
            "yellow",
            "So sánh thêm và mang xe kiểm tra tại gara.",
        )
    return (
        "Phương án 3: RỦI RO CAO",
        "red",
        "Rủi ro kỹ thuật cao hoặc chi phí quá lớn.",
    )


def serialize_payload(weights: Dict[str, float], cars: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> str:
    return json.dumps({"weights": weights, "cars": cars, "results": results}, ensure_ascii=False)
