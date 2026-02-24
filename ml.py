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


@dataclass(frozen=True)
class AHPResult:
    weights: List[float]
    cr: float
    ci: float
    lambda_max: float
    is_valid: bool


_SAATY_RI: Dict[int, float] = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}


def compute_ahp_weights(matrix: Any) -> AHPResult:
    """Compute AHP weights using Saaty's method.

    Steps:
    1) Normalize each column of the pairwise comparison matrix.
    2) Take the average of each row => priority (weights) vector.
    3) Compute Consistency Ratio (CR). If CR >= 0.1, consider invalid.

    Returns:
        AHPResult(weights, cr, ci, lambda_max, is_valid)
    """

    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be a square n x n array")

    n = int(a.shape[0])
    if n == 0:
        raise ValueError("matrix must be non-empty")

    if not np.isfinite(a).all():
        raise ValueError("matrix contains NaN/Inf")
    if (a <= 0).any():
        raise ValueError("matrix entries must be > 0")

    col_sum = a.sum(axis=0)
    if (col_sum <= 0).any():
        raise ValueError("matrix has a zero-sum column")

    norm = a / col_sum
    w = norm.mean(axis=1)
    w_sum = float(w.sum())
    if not math.isfinite(w_sum) or w_sum <= 0:
        raise ValueError("failed to compute a valid weight vector")
    w = w / w_sum

    # Consistency
    aw = a.dot(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(w > 0, aw / w, np.nan)
    lambda_max = float(np.nanmean(ratios))

    if n < 3:
        ci = 0.0
    else:
        ci = float((lambda_max - n) / (n - 1))

    ri = float(_SAATY_RI.get(n, _SAATY_RI[10]))
    cr = 0.0 if ri <= 0 else float(ci / ri)
    is_valid = bool(cr < 0.1)

    return AHPResult(
        weights=[float(x) for x in w.tolist()],
        cr=cr,
        ci=ci,
        lambda_max=lambda_max,
        is_valid=is_valid,
    )


def parse_pairwise_matrix_text(text: str, n: int) -> np.ndarray:
    """Parse an n x n pairwise comparison matrix from a textarea.

    Accepts separators: whitespace and/or commas.
    Supports fractions like "1/3".
    """

    if n <= 0:
        raise ValueError("n must be > 0")
    if text is None:
        raise ValueError("matrix text is empty")

    lines = [ln.strip() for ln in str(text).strip().splitlines() if ln.strip()]
    if len(lines) != n:
        raise ValueError(f"expected {n} rows, got {len(lines)}")

    def parse_token(tok: str) -> float:
        t = tok.strip()
        if not t:
            raise ValueError("empty token")
        if "/" in t:
            a, b = t.split("/", 1)
            return float(a) / float(b)
        return float(t)

    rows: List[List[float]] = []
    for ln in lines:
        parts = [p for p in re.split(r"[\s,;]+", ln) if p]
        if len(parts) != n:
            raise ValueError(f"each row must have {n} values")
        rows.append([parse_token(p) for p in parts])

    return np.asarray(rows, dtype=float)


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


def parse_mpg_series(s):
    import pandas as pd  # lazy import

    if s is None:
        return s
    ss = s.astype(str).str.strip()
    ss = ss.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Extract either a single number or a range a-b
    m = ss.str.extract(r"^(?P<a>\d+(?:\.\d+)?)(?:\s*[-–]\s*(?P<b>\d+(?:\.\d+)?))?$")
    a = pd.to_numeric(m["a"], errors="coerce")
    b = pd.to_numeric(m["b"], errors="coerce")
    out = a.where(b.isna(), (a + b) / 2.0)
    return out


def ahp_score_dataframe(df, weights: Dict[str, float]):
    import pandas as pd  # lazy import

    ws = normalize_weights(weights)
    score = pd.Series(0.0, index=df.index)

    for c in CRITERIA:
        key = c["key"]
        direction = c["direction"]
        w = float(ws.get(key, 0.0))
        if w <= 0:
            continue

        col = df.get(key)
        if col is None:
            continue

        if key == "mpg":
            x = parse_mpg_series(col)
        else:
            x = pd.to_numeric(col, errors="coerce")

        xs = x.dropna()
        if xs.empty:
            scaled = pd.Series(0.0, index=df.index)
        else:
            lo = float(xs.min())
            hi = float(xs.max())
            if abs(hi - lo) < 1e-12:
                scaled = pd.Series(0.0, index=df.index)
                scaled.loc[x.notna()] = 0.5
            else:
                t = (x - lo) / (hi - lo)
                scaled = (1.0 - t) if direction == "cost" else t
                scaled = scaled.fillna(0.0)

        score = score + (w * scaled)

    return score


def predict(models: LoadedModels, cars: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    # Build feature rows aligned to training metadata.
    # We always create all raw columns expected by the preprocessor, and fill missing
    # values with None so the imputer can handle them.
    meta_features = models.meta.get("feature_cols") or models.meta.get("features")
    feature_cols = list(meta_features) if isinstance(meta_features, (list, tuple)) and meta_features else [
        "year",
        "mileage",
        "mpg",
        "one_owner",
        "seller_rating",
        "driver_rating",
        "price_drop",
        "price",
    ]

    rows: List[Dict[str, Any]] = []
    for car in cars:
        row = {
            "year": _to_float(car.get("year")),
            "mileage": _to_float(car.get("mileage")),
            "mpg": parse_mpg(car.get("mpg")),
            "one_owner": _to_float(car.get("one_owner")),
            "seller_rating": _to_float(car.get("seller_rating")),
            "driver_rating": _to_float(car.get("driver_rating")),
            "price_drop": _to_float(car.get("price_drop")),
            "price": _to_float(car.get("price")),
        }

        # Ensure all expected columns exist.
        for col in feature_cols:
            if col not in row:
                row[col] = None

        rows.append({k: row.get(k) for k in feature_cols})

    import pandas as pd  # lazy import

    X = pd.DataFrame(rows, columns=feature_cols)
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
    """Decision logic with risk-first rule and more realistic AHP thresholds.

    Rules:
    - If accident risk > 60% => red (reject/warn)
    - Otherwise, use a softer scoring based on AHP + risk + maintenance.
    """

    accident_risk = float(accident_risk) if math.isfinite(float(accident_risk)) else 1.0
    maint_monthly = float(maint_monthly) if math.isfinite(float(maint_monthly)) else 1e9
    ahp = float(ahp) if math.isfinite(float(ahp)) else 0.0

    accident_pct = accident_risk * 100.0
    if accident_risk > 0.60:
        return (
            "Phương án 3: RỦI RO TAI NẠN CAO",
            "red",
            f"Rủi ro tai nạn {accident_pct:.0f}% > 60%. Nên loại hoặc kiểm tra lịch sử tai nạn rất kỹ.",
        )

    # Soft composite score (bounded) to avoid brittle hard thresholds.
    # AHP in [0,1], risk in [0,1], maint roughly [25..900] USD/month.
    maint_pen = min(1.0, max(0.0, maint_monthly / 500.0))
    composite = (0.62 * ahp) + (0.28 * (1.0 - accident_risk)) + (0.10 * (1.0 - maint_pen))

    # Practical recommendation bands.
    if ahp >= 0.45 and composite >= 0.58:
        return (
            "Phương án 1: NÊN MUA (ƯU TIÊN)",
            "green",
            "Điểm AHP tốt và rủi ro/chi phí hợp lý. Ưu tiên thương lượng và kiểm tra nhanh.",
        )
    if composite >= 0.48:
        return (
            "Phương án 2: CẦN CÂN NHẮC",
            "yellow",
            "Khá ổn nhưng nên so sánh thêm và kiểm định tại gara trước khi chốt.",
        )
    return (
        "Phương án 3: RỦI RO/CHI PHÍ CAO",
        "red",
        "Điểm tổng hợp thấp do rủi ro hoặc chi phí bảo dưỡng cao. Nên tìm lựa chọn khác.",
    )


def serialize_payload(weights: Dict[str, float], cars: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> str:
    return json.dumps({"weights": weights, "cars": cars, "results": results}, ensure_ascii=False)
