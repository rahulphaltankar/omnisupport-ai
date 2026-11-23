"""Deterministic, row-wise cleaning with an auditable before/after report.

Only *rule-based* (non-fitted) cleaning lives here. Anything that learns from data
(imputation values, clipping bounds, encoders, scalers) is fitted inside sklearn
Pipelines on training data only - see ``features.py`` / ``ml/``.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

CANONICAL_REGIONS = {
    "london": "London", "south east": "South East", "north west": "North West", "west midlands": "West Midlands",
    "yorkshire": "Yorkshire", "south west": "South West", "scotland": "Scotland", "east midlands": "East Midlands",
    "wales": "Wales", "northern ireland": "Northern Ireland",
}
SNAKE_COLS = ["support_channel", "product_category", "issue_category", "priority", "customer_sentiment",
              "resolution_status_after_7d", "final_agent_disposition"]
TEXT_COLS = ["issue_title", "issue_description"]


def _snake(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)


def clean_support_records(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean_df, report). ``report`` lists each rule with rows affected/removed."""
    rows: list[dict] = []
    n0 = len(df)
    out = df.copy()

    def log(rule: str, affected: int, removed: int = 0, note: str = "") -> None:
        rows.append({"rule": rule, "rows_affected": int(affected), "rows_removed": int(removed),
                     "rows_after": int(len(out) if removed else len(out)), "note": note})

    # 1. whitespace trimming on every string column
    str_cols = [c for c in out.columns if pd.api.types.is_string_dtype(out[c]) or out[c].dtype == object]
    changed = 0
    for c in str_cols:
        s = out[c].astype("string")
        trimmed = s.str.strip()
        changed += int((s != trimmed).fillna(False).sum())
        out[c] = trimmed
    log("strip_whitespace", changed, note="leading/trailing whitespace on string columns")

    # 2. category normalisation (casing / separators)
    changed = 0
    for c in SNAKE_COLS:
        if c in out:
            new = _snake(out[c])
            changed += int((out[c].astype("string") != new).fillna(False).sum())
            out[c] = new
    log("normalise_category_case", changed, note="e.g. 'Web Chat'->'web_chat', 'Electronics'->'electronics'")

    # 3. region: canonical title-case; blank -> explicit 'Unknown' (missingness is informative, not dropped)
    reg = out["customer_region"].astype("string").str.strip()
    blank = reg.isna() | (reg == "")
    canon = reg.str.lower().map(CANONICAL_REGIONS)
    fixed_case = int((canon.notna() & (canon != reg)).sum())
    out["customer_region"] = canon.fillna("Unknown").where(~blank, "Unknown")
    log("normalise_region", fixed_case + int(blank.sum()), note=f"{int(blank.sum())} blank -> 'Unknown'; {fixed_case} casing fixes")

    # 4. free text: blank -> missing, keep explicit flag
    for c in TEXT_COLS:
        s = out[c].astype("string")
        out[c] = s.where(s.str.strip() != "", pd.NA)
    out["description_missing"] = out["issue_description"].isna().astype("int8")
    log("blank_text_to_missing", int(out["description_missing"].sum()), note="blank issue_description set to NA with flag")

    # 5. timestamps
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    bad_ts = int(ts.isna().sum())
    out["timestamp"] = ts
    log("parse_timestamp", bad_ts, note="unparseable timestamps become NaT and are dropped below")

    # 6. invalid values (impossible, not merely extreme)
    invalid = (
        ts.isna()
        | (out["order_value"] <= 0)
        | (out["delivery_delay_days"] < 0)
        | (out["customer_tenure_months"] < 0)
        | (out["previous_ticket_count"] < 0)
        | (out["resolution_time_hours"] <= 0)
        | (out["csat_score"].notna() & ~out["csat_score"].between(1, 5))
    )
    n_inv = int(invalid.sum())
    out = out.loc[~invalid]
    log("drop_impossible_values", n_inv, n_inv, note="non-positive value/time, negative counts, csat outside 1-5, bad timestamp")

    # 7. exact duplicates (source-system duplication)
    before = len(out)
    dup_mask = out.drop(columns=[c for c in ("_shard",) if c in out]).duplicated(keep="first")
    out = out.loc[~dup_mask]
    log("drop_exact_duplicates", int(dup_mask.sum()), int(dup_mask.sum()), note="all-column duplicates; first kept")

    # 8. residual ticket_id duplicates with conflicting content (should be 0 after step 7)
    tid_dup = out["ticket_id"].duplicated(keep="first")
    out = out.loc[~tid_dup]
    log("drop_conflicting_ticket_id", int(tid_dup.sum()), int(tid_dup.sum()), note="same ticket_id, different content")

    out = out.reset_index(drop=True)
    report = pd.DataFrame(rows)
    report.loc[len(report)] = {"rule": "TOTAL", "rows_affected": n0 - len(out), "rows_removed": n0 - len(out),
                               "rows_after": len(out), "note": f"{n0:,} -> {len(out):,}"}
    # rows_after column is only meaningful at the end; recompute cumulative counts for removal rules
    return out, report


_AMOUNT = re.compile(r"GBP\s*([0-9]+(?:\.[0-9]+)?)")


def parse_amount(text: str | None) -> float:
    if not text:
        return np.nan
    m = _AMOUNT.search(text)
    return float(m.group(1)) if m else np.nan
