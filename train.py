from __future__ import annotations

import argparse
import datetime as dt
import logging
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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
    # Synthetic target for demo purposes ONLY.
    age = (year_now - df["year"].fillna(year_now)).clip(lower=0)
    mileage_k = (df["mileage"].fillna(0) / 1000.0).clip(lower=0)
    accidents = df["accidents_or_damage"].fillna(0).clip(lower=0, upper=1)
    one_owner = df["one_owner"].fillna(0).clip(lower=0, upper=1)

    base = 70.0 + age * 6.0 + mileage_k * 2.5 + accidents * 55.0 + (1 - one_owner) * 8.0
    noise = np.random.default_rng(42).normal(0, 12.0, size=len(df))
    return (base + noise).clip(lower=30.0)


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
    n_estimators_clf: int = 200,
    n_estimators_reg: int = 260,
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

        # For speed on large files: use only columns needed by UI (numeric-focused).
        feature_cols = [
            "year",
            "mileage",
            "mpg",
            "one_owner",
            "personal_use_only",
            "seller_rating",
            "driver_rating",
            "driver_reviews_num",
            "price_drop",
            "price",
        ]
        usecols = feature_cols + ["accidents_or_damage"]

        dtype_map = {
            "year": "float64",
            "mileage": "float64",
            "one_owner": "float64",
            "personal_use_only": "float64",
            "seller_rating": "float64",
            "driver_rating": "float64",
            "driver_reviews_num": "float64",
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

        # Preprocess numeric only
        preprocessor = SimpleImputer(strategy="median")

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
            n_estimators_clf = min(n_estimators_clf, 120)
            n_estimators_reg = min(n_estimators_reg, 160)
            step = min(step, 20)
            logger.info("FAST mode enabled: clf=%d reg=%d step=%d", n_estimators_clf, n_estimators_reg, step)

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
            min_samples_leaf=2,
        )

        done = 0
        while done < n_estimators_clf:
            done = min(n_estimators_clf, done + step)
            accident_clf.set_params(n_estimators=done)
            accident_clf.fit(X_train_np, yA_train)
            _print_progress("accident_clf", done, n_estimators_clf, clf_started)

        yA_pred = accident_clf.predict(X_test_np)
        logger.info("=== Accident classifier report ===\n%s", classification_report(yA_test, yA_pred, digits=3))

        logger.info("Training RandomForestRegressor...")
        reg_started = time.perf_counter()
        maint_reg = RandomForestRegressor(
            n_estimators=0,
            warm_start=True,
            random_state=42,
            n_jobs=-1,
            max_features=0.7,
            min_samples_leaf=2,
        )

        done = 0
        while done < n_estimators_reg:
            done = min(n_estimators_reg, done + step)
            maint_reg.set_params(n_estimators=done)
            maint_reg.fit(X_train_np, yM_train)
            _print_progress("maint_reg", done, n_estimators_reg, reg_started)

        pkg = {
            "preprocessor": preprocessor,
            "accident_clf": accident_clf,
            "maint_reg": maint_reg,
            "meta": {
                "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "cars_csv": str(cars_csv),
                "note": "Maintenance target is SYNTHETIC for demo.",
                "features": feature_cols,
                "log_file": str(log_file),
            },
        }

        out = Path(model_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump(pkg, f)

        logger.info("Saved model package to: %s", out)
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
    ap.add_argument("--n-estimators-clf", type=int, default=200)
    ap.add_argument("--n-estimators-reg", type=int, default=260)
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
        step=args.step,
    )


if __name__ == "__main__":
    main()
