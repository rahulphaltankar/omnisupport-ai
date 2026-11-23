"""Prediction-time-valid feature engineering.

``build_features`` is a *stateless* function of one row's T0-available fields, so exactly the same
code runs for training, held-out evaluation and live API requests. Everything that is learnt from data
(imputation values, clipping bounds, encoders, scalers) lives in the sklearn ``ColumnTransformer``
returned by ``make_preprocessor`` and is fitted on training rows only.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from omnisupport.data.leakage import assert_raw_columns_allowed

# --------------------------------------------------------------- feature spec ---
NUMERIC_FEATURES = [
    "customer_tenure_months", "order_value", "delivery_delay_days", "previous_ticket_count", "attachment_available",
    "hour", "dayofweek", "month", "is_weekend", "is_night",
    "desc_len", "desc_words", "desc_upper_ratio", "desc_exclaim", "desc_missing", "desc_days_mentioned",
    "desc_urgent_phrase", "desc_mobile_footer",
    "log_order_value", "delay_x_prev_tickets", "value_x_delay", "tenure_bucket_new", "repeat_contact",
]
CATEGORICAL_FEATURES = ["support_channel", "customer_region", "product_category", "issue_title"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# raw source field(s) of every engineered feature (used by the leakage guard)
FEATURE_SOURCES: dict[str, list[str]] = {
    "customer_tenure_months": ["customer_tenure_months"], "order_value": ["order_value"],
    "delivery_delay_days": ["delivery_delay_days"], "previous_ticket_count": ["previous_ticket_count"],
    "attachment_available": ["attachment_available"],
    "hour": ["timestamp"], "dayofweek": ["timestamp"], "month": ["timestamp"], "is_weekend": ["timestamp"], "is_night": ["timestamp"],
    "desc_len": ["issue_description"], "desc_words": ["issue_description"], "desc_upper_ratio": ["issue_description"],
    "desc_exclaim": ["issue_description"], "desc_missing": ["issue_description"], "desc_days_mentioned": ["issue_description"],
    "desc_urgent_phrase": ["issue_description"], "desc_mobile_footer": ["issue_description"],
    "log_order_value": ["order_value"], "delay_x_prev_tickets": ["delivery_delay_days", "previous_ticket_count"],
    "value_x_delay": ["order_value", "delivery_delay_days"],
    "tenure_bucket_new": ["customer_tenure_months"], "repeat_contact": ["previous_ticket_count"],
    "support_channel": ["support_channel"], "customer_region": ["customer_region"], "product_category": ["product_category"],
    "issue_title": ["issue_title"],
}
SENSITIVE_ADDONS = ["customer_age_band", "preferred_language", "accessibility_support_flag"]  # ablation only

_DAYS = r"(\d+)\s+days"
_URGENT = r"(?i)\b(?:asap|resolved soon|urgent|immediately|essential)\b"
_MOBILE = r"(?i)sent from mobile"
RAW_INPUT_FIELDS = ["timestamp", "support_channel", "customer_region", "customer_tenure_months", "product_category",
                    "order_value", "delivery_delay_days", "previous_ticket_count", "issue_title", "issue_description",
                    "attachment_available"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless T0 feature construction. Raises ``LeakageError`` if forbidden raw fields are supplied."""
    assert_raw_columns_allowed(df.columns)
    d = df
    ts = pd.to_datetime(d["timestamp"])
    desc = d["issue_description"].astype("string")
    filled = desc.fillna("")
    out = pd.DataFrame(index=d.index)
    out["customer_tenure_months"] = d["customer_tenure_months"].astype(float)
    out["order_value"] = d["order_value"].astype(float)
    out["delivery_delay_days"] = d["delivery_delay_days"].astype(float)
    out["previous_ticket_count"] = d["previous_ticket_count"].astype(float)
    out["attachment_available"] = d["attachment_available"].astype(float)
    out["hour"] = ts.dt.hour.astype(float)
    out["dayofweek"] = ts.dt.dayofweek.astype(float)
    out["month"] = ts.dt.month.astype(float)
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)
    out["is_night"] = ((ts.dt.hour < 6) | (ts.dt.hour >= 22)).astype(float)
    out["desc_len"] = filled.str.len().astype(float)
    out["desc_words"] = filled.str.split().str.len().astype(float)
    letters = filled.str.count(r"[A-Za-z]").clip(lower=1)
    out["desc_upper_ratio"] = (filled.str.count(r"[A-Z]") / letters).astype(float)
    out["desc_exclaim"] = filled.str.count("!").astype(float)
    out["desc_missing"] = desc.isna().astype(float)
    days = filled.str.extract(_DAYS)[0].astype(float)
    out["desc_days_mentioned"] = days
    out["desc_urgent_phrase"] = filled.str.contains(_URGENT).astype(float)
    out["desc_mobile_footer"] = filled.str.contains(_MOBILE).astype(float)
    out["log_order_value"] = np.log1p(out["order_value"])
    out["delay_x_prev_tickets"] = out["delivery_delay_days"] * out["previous_ticket_count"]
    out["value_x_delay"] = out["log_order_value"] * out["delivery_delay_days"]
    out["tenure_bucket_new"] = (out["customer_tenure_months"] < 6).astype(float)
    out["repeat_contact"] = (out["previous_ticket_count"] >= 3).astype(float)
    for c in CATEGORICAL_FEATURES:
        out[c] = d[c].astype("string").fillna("missing").astype(object)
    return out[FEATURE_COLUMNS]


class TrainQuantileClipper(BaseEstimator, TransformerMixin):
    """Learns [q_low, q_high] from TRAIN data only and winsorises extreme values (rare outliers)."""

    def __init__(self, lower: float = 0.0, upper: float = 0.999):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, names=None):
        return np.asarray(names)


def make_preprocessor(scale: bool = False, clip_cols: tuple[str, ...] = ("order_value", "delivery_delay_days", "delay_x_prev_tickets", "value_x_delay"),
                      numeric: list[str] | None = None, categorical: list[str] | None = None,
                      extra_numeric: tuple[str, ...] = (), extra_categorical: tuple[str, ...] = ()) -> ColumnTransformer:
    """Fit-on-train preprocessing: imputation, outlier clipping, optional scaling, one-hot encoding.

    ``numeric``/``categorical`` select feature subsets (ablations); ``extra_*`` exist ONLY for the documented
    leakage/sensitive-attribute experiments - production code never passes them.
    """
    num_cols = list(NUMERIC_FEATURES if numeric is None else numeric) + list(extra_numeric)
    cat_cols = list(CATEGORICAL_FEATURES if categorical is None else categorical) + list(extra_categorical)
    clip = [c for c in num_cols if c in clip_cols]
    rest = [c for c in num_cols if c not in clip]
    clip_steps = [("impute", SimpleImputer(strategy="median", add_indicator=False)), ("clip", TrainQuantileClipper(0.0, 0.999))]
    rest_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        clip_steps.append(("scale", StandardScaler()))
        rest_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num_clip", Pipeline(clip_steps), clip),
            ("num", Pipeline(rest_steps), rest),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), cat_cols),
        ],
        remainder="drop", verbose_feature_names_out=False,
    )
