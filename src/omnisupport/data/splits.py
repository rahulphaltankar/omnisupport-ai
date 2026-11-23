"""Leakage-aware splitting.

Operationally the model is trained on the past and scores future tickets, so the primary split is
*temporal*: train Jan-Aug 2025, validation Sep-Oct (model/threshold selection), test Nov-Dec (touched once,
for the final report). Customers re-contact, so customer overlap across periods is measured and reported
(it mirrors production, where returning customers are normal) and CV *within train* is grouped by customer.
"""
from __future__ import annotations

import pandas as pd

TRAIN_END = pd.Timestamp("2025-09-01")
VAL_END = pd.Timestamp("2025-11-01")


def temporal_split(df: pd.DataFrame, time_col: str = "timestamp") -> dict[str, pd.DataFrame]:
    ts = pd.to_datetime(df[time_col])
    parts = {
        "train": df[ts < TRAIN_END],
        "val": df[(ts >= TRAIN_END) & (ts < VAL_END)],
        "test": df[ts >= VAL_END],
    }
    assert sum(len(p) for p in parts.values()) == len(df), "split must be a partition"
    return parts


def split_diagnostics(parts: dict[str, pd.DataFrame], target: str | None = None) -> pd.DataFrame:
    rows = []
    train_customers = set(parts["train"]["customer_id"])
    train_tickets = set(parts["train"]["ticket_id"])
    for name, p in parts.items():
        r = {"split": name, "rows": len(p), "start": pd.to_datetime(p["timestamp"]).min(), "end": pd.to_datetime(p["timestamp"]).max(),
             "customers": p["customer_id"].nunique(),
             "customer_overlap_with_train": float(p["customer_id"].isin(train_customers).mean()) if name != "train" else 1.0,
             "ticket_id_overlap_with_train": int(p["ticket_id"].isin(train_tickets).sum()) if name != "train" else len(p)}
        if target:
            r[f"{target}_mean"] = float(p[target].mean())
        rows.append(r)
    return pd.DataFrame(rows)
