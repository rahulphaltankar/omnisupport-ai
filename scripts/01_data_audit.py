"""Epic DATA-01/02: audit all ten primary shards, clean reproducibly, write the leakage register and figures.

Run:  python scripts/01_data_audit.py
Outputs: artifacts/data_audit/*, reports/leakage_register.md, data/processed/clean.parquet (git-ignored cache)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from omnisupport.config import EXPECTED_ROWS, PROCESSED_DIR, REPORTS, SEED, artifact_dir  # noqa: E402
from omnisupport.data import audit as A  # noqa: E402
from omnisupport.data.cleaning import clean_support_records  # noqa: E402
from omnisupport.data.leakage import register_table  # noqa: E402
from omnisupport.data.loader import discover_shards, load_primary, shard_row_counts, validate_counts  # noqa: E402
from omnisupport.utils import plotting as P  # noqa: E402
from omnisupport.utils.runlog import RunRecord, write_json  # noqa: E402


def main() -> None:
    out = artifact_dir("data_audit")
    with RunRecord(out, "data_audit", config={"expected_rows": EXPECTED_ROWS}, split_method="n/a (audit of full primary data)") as run:
        paths = discover_shards()
        counts = shard_row_counts(paths)
        total = validate_counts(counts)  # raises unless exactly 10 shards / 1,000,000 rows
        pd.DataFrame({"shard": list(counts), "rows": list(counts.values())}).to_csv(out / "shard_counts.csv", index=False)
        print(f"[OK] {len(paths)} primary shards, {total:,} rows")

        raw = load_primary()
        assert len(raw) == EXPECTED_ROWS

        schema = A.schema_and_missingness(raw); schema.to_csv(out / "schema_missingness_raw.csv", index=False)
        cats = A.category_inconsistencies(raw); cats.to_csv(out / "category_inconsistencies_raw.csv", index=False)
        outl = A.outlier_table(raw); outl.to_csv(out / "outliers_raw.csv", index=False)
        dups = A.duplicate_summary(raw)
        imb = A.imbalance_table(raw); imb.to_csv(out / "imbalance_raw.csv", index=False)
        noise = A.text_noise_summary(raw)

        clean, report = clean_support_records(raw)
        report.to_csv(out / "cleaning_report.csv", index=False)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(PROCESSED_DIR / "clean.parquet", index=False)

        # ---------------------------------------------------------------- leakage evidence
        leak_scan = A.single_field_leak_scan(clean, "escalated")
        leak_scan.to_csv(out / "single_field_leak_scan_escalated.csv", index=False)
        reg = register_table(); reg.to_csv(out / "leakage_register.csv", index=False)
        crosstab = pd.crosstab(clean["escalated"], clean["final_agent_disposition"])
        crosstab.to_csv(out / "escalated_vs_final_disposition.csv")

        summary = {
            "primary_shards": len(paths), "primary_rows": total, "rows_after_cleaning": int(len(clean)),
            "rows_removed": int(total - len(clean)), "duplicates": dups, "text_noise": noise,
            "date_range": [str(clean["timestamp"].min()), str(clean["timestamp"].max())],
            "target_balance": {"escalated_rate": round(float(clean["escalated"].mean()), 4),
                               "resolution_time_hours_mean": round(float(clean["resolution_time_hours"].mean()), 3),
                               "resolution_time_hours_p99": round(float(clean["resolution_time_hours"].quantile(.99)), 3)},
            "supporting_assets_not_counted": ["dataset_preview.csv", "transformer_finetune_10000.parquet", "agent_store/*.csv", "images/"],
        }
        write_json(out / "audit_summary.json", summary)
        run.add(**{k: summary[k] for k in ("primary_rows", "rows_after_cleaning")})

        # ---------------------------------------------------------------- figures
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        m = schema.sort_values("missing_or_blank_pct", ascending=False).head(6)
        ax[0].barh(m["column"][::-1], m["missing_or_blank_pct"][::-1]); ax[0].set_title("Missing/blank % (raw, top 6)")
        ax[0].set_xlabel("% of 1,000,000 rows")
        c = cats.groupby("column")["count"].sum().sort_values()
        ax[1].barh(c.index, c.values); ax[1].set_title("Rows with inconsistent category spelling"); ax[1].set_xlabel("rows")
        rr = report[report["rule"] != "TOTAL"].sort_values("rows_affected")
        rr = rr[rr["rows_affected"] > 0]
        ax[2].barh(rr["rule"], rr["rows_affected"], color=[P.PALETTE[4] if r > 0 else P.PALETTE[0] for r in rr["rows_removed"]])
        ax[2].set_title(f"Rows touched per cleaning rule (red = removed); {total:,} -> {len(clean):,}"); ax[2].set_xscale("log")
        P.save(fig, out / "fig_data_quality_overview.png")

        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for a, col in zip(ax, ["order_value", "delivery_delay_days", "resolution_time_hours"]):
            a.hist(np.log10(clean[col].clip(lower=0.1)), bins=60); a.set_title(f"{col} (log10 scale)")
            p999 = clean[col].quantile(.999); a.axvline(np.log10(max(p999, .1)), color="crimson", ls="--", label=f"p99.9={p999:,.1f}")
            a.axvline(np.log10(max(clean[col].max(), .1)), color="black", ls=":", label=f"max={clean[col].max():,.0f}"); a.legend()
        P.save(fig, out / "fig_outliers_log_scale.png")

        fig, ax = plt.subplots(figsize=(7, 3.8))
        top = leak_scan.head(12).iloc[::-1]
        colors = [P.PALETTE[4] if f in {"final_agent_disposition", "customer_sentiment", "csat_score", "priority", "resolution_status_after_7d", "resolution_time_hours", "refund_amount"} else P.PALETTE[0] for f in top["field"]]
        ax.barh(top["field"], top["single_field_auc"], color=colors); ax.axvline(0.5, color="grey", ls=":")
        ax.set_title("Single-field AUC for 'escalated' (red = excluded as post-outcome/triage)")
        P.save(fig, out / "fig_leakage_single_field_auc.png")

        print(report.to_string(index=False))
        print(leak_scan.head(8).to_string(index=False))

    # human-readable register
    lines = ["# Field-level leakage register", "",
             "Prediction moment **T0 = ticket creation, before human triage**. Generated by `scripts/01_data_audit.py` from "
             "`src/omnisupport/data/leakage.py` and enforced in code by `assert_raw_columns_allowed` / `assert_features_valid`.", "",
             "| Field | Role | Available at T0 | Rationale |", "|---|---|---|---|"]
    for _, r in reg.iterrows():
        lines.append(f"| `{r['field']}` | {r['role']} | {'yes' if r['available_at_prediction_time_T0'] else 'no'} | {r['rationale']} |")
    lines += ["", "## Empirical confirmation", "",
              "`final_agent_disposition == 'specialist_escalation'` coincides exactly with `escalated == 1` "
              "(`artifacts/data_audit/escalated_vs_final_disposition.csv`). Single-field scores are in "
              "`artifacts/data_audit/single_field_leak_scan_escalated.csv`; the leakage-uplift experiment is in "
              "`artifacts/ml_classification/leakage_demonstration.csv`."]
    (REPORTS / "leakage_register.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] data audit complete")


if __name__ == "__main__":
    main()
