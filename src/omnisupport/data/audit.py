"""Data-quality audit functions (pure, testable, no plotting)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_AUDIT_COLS = ["support_channel", "customer_region", "customer_age_band", "preferred_language", "product_category",
                          "issue_category", "priority", "customer_sentiment", "resolution_status_after_7d", "final_agent_disposition"]
NUMERIC_AUDIT_COLS = ["customer_tenure_months", "order_value", "delivery_delay_days", "previous_ticket_count",
                      "resolution_time_hours", "refund_amount", "csat_score"]


def schema_and_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        if c == "_shard":
            continue
        s = df[c]
        is_text = pd.api.types.is_string_dtype(s) or s.dtype == object
        blank = int((s.astype("string").str.strip() == "").sum()) if is_text else 0
        rows.append({"column": c, "dtype": str(s.dtype), "null_count": int(s.isna().sum()), "blank_string_count": blank,
                     "missing_or_blank_pct": round(100 * (int(s.isna().sum()) + blank) / len(s), 3), "n_unique": int(s.nunique(dropna=True))})
    return pd.DataFrame(rows)


def category_inconsistencies(df: pd.DataFrame, cols=CATEGORICAL_AUDIT_COLS) -> pd.DataFrame:
    """Values whose normalised form collides with a more frequent spelling (casing/whitespace/separator variants)."""
    rows = []
    for c in cols:
        vc = df[c].astype("string").value_counts(dropna=False)
        norm: dict[str, list] = {}
        for v, n in vc.items():
            key = "" if pd.isna(v) else str(v).strip().lower().replace(" ", "_").replace("-", "_")
            norm.setdefault(key, []).append((v, int(n)))
        for key, variants in norm.items():
            if len(variants) > 1:
                variants = sorted(variants, key=lambda x: -x[1])
                for v, n in variants[1:]:
                    rows.append({"column": c, "variant": v, "count": n, "canonical": variants[0][0], "canonical_count": variants[0][1]})
        blanks = int((df[c].astype("string").str.strip() == "").sum())
        if blanks:
            rows.append({"column": c, "variant": "<blank>", "count": blanks, "canonical": "Unknown (explicit)", "canonical_count": np.nan})
    return pd.DataFrame(rows)


def outlier_table(df: pd.DataFrame, cols=NUMERIC_AUDIT_COLS) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna().astype(float)
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        hi = q3 + 3 * iqr
        rows.append({"column": c, "min": s.min(), "p50": s.median(), "p99": s.quantile(.99), "p999": s.quantile(.999), "max": s.max(),
                     "iqr_fence_3x_high": hi, "n_above_fence": int((s > hi).sum()), "pct_above_fence": round(100 * (s > hi).mean(), 3),
                     "max_over_p99": round(s.max() / max(s.quantile(.99), 1e-9), 2)})
    return pd.DataFrame(rows)


def duplicate_summary(df: pd.DataFrame) -> dict:
    cols = [c for c in df.columns if c != "_shard"]
    exact = df.duplicated(subset=cols, keep=False)
    return {
        "rows": int(len(df)),
        "exact_duplicate_rows_extra_copies": int(df.duplicated(subset=cols).sum()),
        "rows_involved_in_exact_duplication": int(exact.sum()),
        "duplicate_ticket_id_extra": int(df["ticket_id"].duplicated().sum()),
        "duplicates_cross_shard_fraction": float((df.loc[df.duplicated(subset=cols, keep=False)].groupby("ticket_id")["_shard"].nunique() > 1).mean())
        if "_shard" in df and exact.any() else None,
        "unique_tickets": int(df["ticket_id"].nunique()),
    }


def imbalance_table(df: pd.DataFrame, cols=("escalated", "refund_requested", "attachment_available", "accessibility_support_flag", "resolution_status_after_7d", "priority", "issue_category")) -> pd.DataFrame:
    rows = []
    for c in cols:
        vc = df[c].value_counts(normalize=True)
        for k, v in vc.items():
            rows.append({"column": c, "value": k, "share": round(float(v), 5), "count": int((df[c] == k).sum())})
    return pd.DataFrame(rows)


def text_noise_summary(df: pd.DataFrame) -> dict:
    d = df["issue_description"].astype("string")
    filled = d.fillna("")
    letters = filled.str.count(r"[A-Za-z]").clip(lower=1)
    upper = (filled.str.count(r"[A-Z]") / letters) > 0.5
    return {
        "blank_or_missing_description": int((filled.str.strip() == "").sum()),
        "mostly_uppercase_description": int(upper.sum()),
        "with_noise_footer_(!!!,...,pls,thx,ASAP,#help,mobile)": int(filled.str.contains(r"(?i)(?:!!!|\.\.\.$|pls$|thx$|asap$|#help$|sent from mobile$)").sum()),
        "unique_titles": int(df["issue_title"].nunique()),
    }


def single_field_leak_scan(df: pd.DataFrame, target: str = "escalated", fields=None, sample: int = 200_000, seed: int = 42) -> pd.DataFrame:
    """Rank each candidate field by its stand-alone ability to predict the target (AUC / target purity).

    A very high single-field score for a field that is not knowable at prediction time is the signature of leakage.
    """
    from sklearn.metrics import roc_auc_score

    d = df.sample(min(sample, len(df)), random_state=seed)
    fields = fields or [c for c in d.columns if c not in {target, "_shard", "ticket_id", "customer_id", "order_id", "timestamp", "issue_description", "image_id"}]
    y = d[target].astype(int)
    rows = []
    for c in fields:
        s = d[c]
        if pd.api.types.is_numeric_dtype(s):
            x = s.fillna(s.median())
            auc = roc_auc_score(y, x)
            auc = max(auc, 1 - auc)
        else:
            rate = y.groupby(s.astype("string").fillna("NA")).transform("mean")
            auc = roc_auc_score(y, rate)
        rows.append({"field": c, "single_field_auc": round(float(auc), 4)})
    return pd.DataFrame(rows).sort_values("single_field_auc", ascending=False).reset_index(drop=True)
