"""Shared modelling frame, metrics and threshold utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, median_absolute_error, precision_score, recall_score, roc_auc_score)

from omnisupport.config import PROCESSED_DIR, SEED
from omnisupport.data.cleaning import clean_support_records
from omnisupport.data.features import FEATURE_COLUMNS, RAW_INPUT_FIELDS, build_features
from omnisupport.data.leakage import assert_features_valid
from omnisupport.data.loader import load_primary
from omnisupport.data.splits import temporal_split

CLEAN_PATH = PROCESSED_DIR / "clean.parquet"
FRAME_PATH = PROCESSED_DIR / "modelling_frame.parquet"
# Columns kept only for slicing/explanation (NEVER passed to a model as features).
META_COLUMNS = ["ticket_id", "customer_id", "timestamp", "escalated", "resolution_time_hours", "issue_category", "priority",
                "customer_age_band", "preferred_language", "accessibility_support_flag", "customer_sentiment",
                "csat_score", "final_agent_disposition", "refund_amount", "refund_requested"]


def load_clean() -> pd.DataFrame:
    if CLEAN_PATH.exists():
        return pd.read_parquet(CLEAN_PATH)
    clean, _ = clean_support_records(load_primary())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(CLEAN_PATH, index=False)
    return clean


def modelling_frame(rebuild: bool = False) -> pd.DataFrame:
    """Cleaned rows + engineered T0 features (+ meta columns for slicing). Cached on disk."""
    if FRAME_PATH.exists() and not rebuild:
        return pd.read_parquet(FRAME_PATH)
    clean = load_clean()
    feats = build_features(clean[RAW_INPUT_FIELDS])
    assert_features_valid(list(feats.columns))
    frame = pd.concat([clean[META_COLUMNS].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FRAME_PATH, index=False)
    return frame


def split_frame(frame: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    frame = modelling_frame() if frame is None else frame
    parts = temporal_split(frame)
    return {k: v.reset_index(drop=True) for k, v in parts.items()}


def cv_subsample(train: pd.DataFrame, n: int = 120_000, seed: int = SEED) -> pd.DataFrame:
    """Random subsample of TRAIN rows for model comparison/tuning (keeps runtime sane; test untouched)."""
    return train.sample(min(n, len(train)), random_state=seed).reset_index(drop=True)


# ----------------------------------------------------------------- classification
def class_metrics(y, p, thr: float = 0.5) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, yhat, zero_division=0)), "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)), "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "threshold": float(thr), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "base_rate": float(y.mean()),
        "flag_rate": float(yhat.mean()),
    }


def threshold_table(y, p, fn_cost: float = 3.0, fp_cost: float = 1.0, grid: np.ndarray | None = None) -> pd.DataFrame:
    y = np.asarray(y); p = np.asarray(p)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    rows = []
    for t in grid:
        yh = p >= t
        tp = int(((yh) & (y == 1)).sum()); fp = int(((yh) & (y == 0)).sum()); fn = int(((~yh) & (y == 1)).sum()); tn = int(((~yh) & (y == 0)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        rows.append({"threshold": t, "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
                     "flag_rate": float(yh.mean()), "fp": fp, "fn": fn, "expected_cost_per_ticket": (fn * fn_cost + fp * fp_cost) / len(y)})
    return pd.DataFrame(rows)


def pick_thresholds(tab: pd.DataFrame, min_recall: float = 0.60) -> dict:
    """Three defensible operating points; the cost-optimal one is the default."""
    cost = tab.loc[tab["expected_cost_per_ticket"].idxmin()]
    f1 = tab.loc[tab["f1"].idxmax()]
    feas = tab[tab["recall"] >= min_recall]
    rec = feas.loc[feas["precision"].idxmax()] if len(feas) else cost
    return {"cost_optimal": float(cost["threshold"]), "f1_optimal": float(f1["threshold"]), f"max_precision_at_recall>={min_recall}": float(rec["threshold"])}


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y); p = np.asarray(p)
    q = pd.qcut(p, bins, duplicates="drop")
    d = pd.DataFrame({"y": y, "p": p, "bin": q}).groupby("bin", observed=True).agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    d["abs_gap"] = (d["mean_pred"] - d["observed"]).abs()
    return d


def expected_calibration_error(tab: pd.DataFrame) -> float:
    return float((tab["abs_gap"] * tab["n"]).sum() / tab["n"].sum())


# ----------------------------------------------------------------- regression
def reg_metrics(y, yhat) -> dict:
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    return {"mae": float(mean_absolute_error(y, yhat)), "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
            "median_ae": float(median_absolute_error(y, yhat)), "r2": float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "mape_pct": float(np.mean(np.abs(y - yhat) / np.maximum(y, 1e-9)) * 100)}


# ----------------------------------------------------------------- intervals
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 300, seed: int = SEED, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    if len(values) == 0:
        return (np.nan, np.nan)
    boots = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS]
