from __future__ import annotations

import argparse
import datetime as dt
import gzip
import logging
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight


def parse_mpg_series(s: pd.Series) -> pd.Series:
    def _parse(v):
        if pd.isna(v):
            return np.nan
        txt = str(v).strip()
        if not txt:
            return np.nan
        if "-" in txt:
            parts = txt.split("-")
            try:
                a = float(parts[0])
                b = float(parts[1])
                return (a + b) / 2.0
            except Exception:
                return np.nan
        try:
            return float(txt)
        except Exception:
            return np.nan

    return s.apply(_parse)


def make_synthetic_maintenance(df: pd.DataFrame, year_now: int = 2026) -> pd.Series:
    """Synthetic maintenance cost target (USD/month) for demo purposes ONLY.

    Formula (requested):
    - Base by age: $8/month per year of age
    - Base by mileage: $0.3/month per 1000 miles
    - Heavy penalty if accidents: +$40/month
    - Add reasonable noise
    """

    rng = np.random.default_rng(42)

    year = pd.to_numeric(df.get("year"), errors="coerce").fillna(year_now)
    mileage = pd.to_numeric(df.get("mileage"), errors="coerce").fillna(0)
    accidents = pd.to_numeric(df.get("accidents_or_damage"), errors="coerce").fillna(0).clip(0, 1)

    age = (year_now - year).clip(lower=0)
    mileage_k = (mileage / 1000.0).clip(lower=0)

    base = (age * 8.0) + (mileage_k * 0.3)
    base = base + (accidents * 40.0)

    noise = rng.normal(0.0, 10.0, size=len(df))
    out = (base + noise).clip(lower=25.0, upper=900.0)
    return out


def _fmt_seconds(sec: float) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _print_progress(label: str, done: int, total: int, started: float) -> None:
    pct = 0.0 if total <= 0 else (done / total) * 100.0
    width = 24
    filled = int(round(width * (pct / 100.0)))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = _fmt_seconds(time.perf_counter() - started)
    msg = f"[{bar}] {pct:6.2f}% | {label} | elapsed {elapsed}"
    print(msg, flush=True)


def train(
    cars_csv: str,
    model_path: str,
    sample_rows: int | None = None,
    sample_frac: float | None = None,
    fast: bool = False,
    n_estimators_clf: int = 420,
    n_estimators_reg: int = 520,
    max_depth_clf: int = 24,
    max_depth_reg: int = 24,
    min_samples_leaf_clf: int = 2,
    min_samples_leaf_reg: int = 2,
    max_samples: float = 0.9,
    step: int = 20,
):
    started_all = time.perf_counter()

    logs_dir = Path("./logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    now_utc = dt.datetime.now(dt.timezone.utc)
    log_file = logs_dir / f"train_{now_utc.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
    )
    logger = logging.getLogger("train")

    try:
        logger.info("Reading CSV: %s", cars_csv)

        # Raw input columns (must be preserved in meta to keep predict() aligned).
        numeric_features = [
            "year",
            "mileage",
            "mpg",
            "one_owner",
            "seller_rating",
            "driver_rating",
            "price_drop",
            "price",
        ]
        categorical_features: list[str] = []
        feature_cols = list(numeric_features)
        usecols = feature_cols + ["accidents_or_damage"]

        dtype_map = {
            "year": "float64",
            "mileage": "float64",
            "mpg": "string",
            "one_owner": "float64",
            "seller_rating": "float64",
            "driver_rating": "float64",
            "price_drop": "float64",
            "price": "float64",
            "accidents_or_damage": "float64",
        }

        # If user requests sample_rows, don't read whole file.
        read_nrows = sample_rows if (sample_rows is not None and sample_rows > 0) else None
        df = pd.read_csv(
            cars_csv,
            usecols=usecols,
            low_memory=False,
            nrows=read_nrows,
            dtype=dtype_map,
        )
        logger.info("Loaded rows: %d", len(df))

        if read_nrows is None and sample_rows is not None and sample_rows > 0 and len(df) > sample_rows:
            df = df.sample(n=sample_rows, random_state=42)
            logger.info("Sampled rows (n=%d)", len(df))
        elif read_nrows is None and sample_frac is not None and 0 < sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
            logger.info("Sampled rows (frac=%.3f -> n=%d)", sample_frac, len(df))

        if "accidents_or_damage" not in df.columns:
            raise ValueError("cars.csv must contain 'accidents_or_damage' column")

        # Cleaning
        df["mpg"] = parse_mpg_series(df["mpg"])

        # Targets
        y_accident = df["accidents_or_damage"].fillna(0).astype(int).clip(0, 1)
        y_maint = make_synthetic_maintenance(df)

        X = df[feature_cols].copy()

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        X_train, X_test, yA_train, yA_test, yM_train, yM_test = train_test_split(
            X,
            y_accident,
            y_maint,
            test_size=0.2,
            random_state=42,
            stratify=y_accident if y_accident.nunique() > 1 else None,
        )

        X_train_np = preprocessor.fit_transform(X_train)
        X_test_np = preprocessor.transform(X_test)

        if fast:
            n_estimators_clf = min(n_estimators_clf, 70)
            n_estimators_reg = min(n_estimators_reg, 90)
            max_depth_clf = min(max_depth_clf, 10)
            max_depth_reg = min(max_depth_reg, 10)
            min_samples_leaf_clf = max(min_samples_leaf_clf, 20)
            min_samples_leaf_reg = max(min_samples_leaf_reg, 24)
            max_samples = min(max_samples, 0.25)
            step = min(step, 20)
            logger.info(
                "FAST mode enabled: clf=%d reg=%d depth=(%d,%d) leaf=(%d,%d) max_samples=%.2f step=%d",
                n_estimators_clf,
                n_estimators_reg,
                max_depth_clf,
                max_depth_reg,
                min_samples_leaf_clf,
                min_samples_leaf_reg,
                max_samples,
                step,
            )

        # Stable class weights for warm_start
        class_weight = None
        if yA_train.nunique() > 1:
            classes = np.array(sorted(yA_train.unique()))
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=yA_train.to_numpy())
            class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        logger.info("Training RandomForestClassifier...")
        clf_started = time.perf_counter()
        accident_clf = RandomForestClassifier(
            n_estimators=0,
            warm_start=True,
            random_state=42,
            class_weight=class_weight,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=max(1, int(min_samples_leaf_clf)),
            max_depth=max(2, int(max_depth_clf)),
            max_leaf_nodes=2048,
            bootstrap=True,
            max_samples=min(1.0, max(0.05, float(max_samples))),
        )

        done = 0
        while done < n_estimators_clf:
            done = min(n_estimators_clf, done + step)
            accident_clf.set_params(n_estimators=done)
            accident_clf.fit(X_train_np, yA_train)
            _print_progress("accident_clf", done, n_estimators_clf, clf_started)

        yA_pred = accident_clf.predict(X_test_np)
        accident_accuracy = float(accuracy_score(yA_test, yA_pred))
        logger.info("=== Accident classifier report ===\n%s", classification_report(yA_test, yA_pred, digits=3))
        logger.info("Accident accuracy: %.4f", accident_accuracy)

        logger.info("Training RandomForestRegressor...")
        reg_started = time.perf_counter()
        maint_reg = RandomForestRegressor(
            n_estimators=0,
            warm_start=True,
            random_state=42,
            n_jobs=-1,
            max_features=0.7,
            min_samples_leaf=max(1, int(min_samples_leaf_reg)),
            max_depth=max(2, int(max_depth_reg)),
            max_leaf_nodes=2048,
            bootstrap=True,
            max_samples=min(1.0, max(0.05, float(max_samples))),
        )

        done = 0
        while done < n_estimators_reg:
            done = min(n_estimators_reg, done + step)
            maint_reg.set_params(n_estimators=done)
            maint_reg.fit(X_train_np, yM_train)
            _print_progress("maint_reg", done, n_estimators_reg, reg_started)

        yM_pred = maint_reg.predict(X_test_np)
        maintenance_mae = float(mean_absolute_error(yM_test, yM_pred))
        logger.info("Maintenance MAE: %.4f", maintenance_mae)

        fi_raw = getattr(accident_clf, "feature_importances_", None)
        if fi_raw is None:
            fi_raw = []

        feature_importance = []
        for i, val in enumerate(fi_raw):
            feature_importance.append(
                {
                    "name": feature_cols[i] if i < len(feature_cols) else f"feature_{i + 1}",
                    "value": float(val),
                }
            )

        pkg = {
            "preprocessor": preprocessor,
            "accident_clf": accident_clf,
            "maint_reg": maint_reg,
            "meta": {
                "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "cars_csv": str(cars_csv),
                "note": "Maintenance target is SYNTHETIC for demo.",
                # Backward-compatible key + explicit raw feature order.
                "features": feature_cols,
                "feature_cols": feature_cols,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                # Useful for debugging transformed feature order.
                "feature_names_out": (
                    preprocessor.get_feature_names_out().tolist()
                    if hasattr(preprocessor, "get_feature_names_out")
                    else None
                ),
                "log_file": str(log_file),
                "train_rows": int(len(df)),
                "accident_accuracy": accident_accuracy,
                "maintenance_mae": maintenance_mae,
                "feature_importance": feature_importance,
            },
        }

        out = Path(model_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out, "wb", compresslevel=3) as f:
            pickle.dump(pkg, f, protocol=pickle.HIGHEST_PROTOCOL)

        out_size_mb = os.path.getsize(out) / (1024 * 1024)

        logger.info("Saved model package to: %s", out)
        logger.info("Model file size: %.2f MB", out_size_mb)
        logger.info("Total elapsed: %s", _fmt_seconds(time.perf_counter() - started_all))
        logger.info("Log file: %s", log_file)

    except Exception as e:
        logger.error("Training failed: %s", e)
        logger.error(traceback.format_exc())
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", default="./cars.csv")
    ap.add_argument("--out", default="./models/car_advisor_rf.pkl")
    ap.add_argument("--sample-rows", type=int, default=None, help="Subsample N rows for faster training")
    ap.add_argument("--sample-frac", type=float, default=None, help="Subsample fraction (0-1) for faster training")
    ap.add_argument("--fast", action="store_true", help="Use smaller models for quick demo")
    ap.add_argument("--n-estimators-clf", type=int, default=420)
    ap.add_argument("--n-estimators-reg", type=int, default=520)
    ap.add_argument("--max-depth-clf", type=int, default=24)
    ap.add_argument("--max-depth-reg", type=int, default=24)
    ap.add_argument("--min-samples-leaf-clf", type=int, default=2)
    ap.add_argument("--min-samples-leaf-reg", type=int, default=2)
    ap.add_argument("--max-samples", type=float, default=0.9, help="Per-tree row sampling ratio (0-1]")
    ap.add_argument("--step", type=int, default=20, help="Progress step for warm_start training")
    args = ap.parse_args()

    train(
        args.cars,
        args.out,
        sample_rows=args.sample_rows,
        sample_frac=args.sample_frac,
        fast=args.fast,
        n_estimators_clf=args.n_estimators_clf,
        n_estimators_reg=args.n_estimators_reg,
        max_depth_clf=args.max_depth_clf,
        max_depth_reg=args.max_depth_reg,
        min_samples_leaf_clf=args.min_samples_leaf_clf,
        min_samples_leaf_reg=args.min_samples_leaf_reg,
        max_samples=args.max_samples,
        step=args.step,
    )


if __name__ == "__main__":
    main()
