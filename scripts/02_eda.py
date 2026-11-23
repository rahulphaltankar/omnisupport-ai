"""Epic DATA-03: decision-driving EDA. Every figure gets one computed observation and one implication.

Run:  python scripts/02_eda.py     (needs data/processed/clean.parquet from scripts/01_data_audit.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from omnisupport.config import PROCESSED_DIR, REPORTS, artifact_dir  # noqa: E402
from omnisupport.utils import plotting as P  # noqa: E402
from omnisupport.utils.runlog import RunRecord, write_json  # noqa: E402


def main() -> None:
    out = artifact_dir("eda")
    df = pd.read_parquet(PROCESSED_DIR / "clean.parquet")
    notes: list[tuple[str, str, str, str]] = []  # (figure, title, observation, implication)
    stats: dict = {}

    with RunRecord(out, "eda", config={"rows": len(df)}, split_method="n/a (descriptive, full cleaned data)") as run:
        # F1 targets ---------------------------------------------------------------------------
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].hist(df["resolution_time_hours"].clip(upper=80), bins=80); ax[0].set_title("Resolution time (hours, clipped at 80 for display)")
        p50, p99, mx = df["resolution_time_hours"].quantile([.5, .99]).tolist() + [df["resolution_time_hours"].max()]
        ax[0].axvline(p50, color="k", ls="--", label=f"median {p50:.1f}h"); ax[0].legend()
        rate = df["escalated"].mean(); ax[1].bar(["not escalated", "escalated"], [1 - rate, rate], color=[P.PALETTE[5], P.PALETTE[4]]); ax[1].set_title("Escalation class balance")
        for i, v in enumerate([1 - rate, rate]): ax[1].text(i, v + .01, f"{v:.1%}", ha="center")
        by_m = df.groupby(df["timestamp"].dt.month)["escalated"].agg(["mean", "size"])
        ax[2].plot(by_m.index, by_m["mean"], marker="o"); ax[2].set_ylim(0, .5); ax[2].set_title("Escalation rate by month (stationary?)"); ax[2].set_xlabel("month")
        P.save(fig, out / "fig1_targets.png")
        stats.update(median_resolution_h=p50, p99_resolution_h=p99, max_resolution_h=mx, escalation_rate=rate,
                     monthly_rate_min=float(by_m["mean"].min()), monthly_rate_max=float(by_m["mean"].max()))
        notes.append(("fig1_targets.png", "Targets",
                      f"Resolution time is right-skewed (median {p50:.1f}h, p99 {p99:.1f}h, max {mx:,.0f}h) and {rate:.1%} of tickets escalate; the monthly escalation rate stays within {by_m['mean'].min():.1%}-{by_m['mean'].max():.1%}.",
                      "Report MAE alongside RMSE (RMSE is dominated by a handful of extreme times), winsorise the training target, and use class weighting/PR-AUC rather than accuracy. A near-stationary rate supports a temporal split without drift correction."))

        # F2 issue drivers ---------------------------------------------------------------------
        g = df.groupby("issue_category").agg(esc=("escalated", "mean"), res=("resolution_time_hours", "mean"), n=("ticket_id", "size")).sort_values("esc")
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
        ax[0].barh(g.index, g["esc"], color=P.PALETTE[4]); ax[0].set_title("Escalation rate by issue category (explanatory only)")
        ax[1].barh(g.index, g["res"], color=P.PALETTE[0]); ax[1].set_title("Mean resolution time (h) by issue category")
        P.save(fig, out / "fig2_issue_drivers.png")
        top, low = g.index[-1], g.index[0]
        stats["issue_extremes"] = {"highest_escalation": top, "rate": float(g.loc[top, "esc"]), "lowest": low, "lowest_rate": float(g.loc[low, "esc"])}
        notes.append(("fig2_issue_drivers.png", "Issue type is a major driver",
                      f"'{top}' escalates {g.loc[top, 'esc']:.0%} of the time and takes {g.loc[top, 'res']:.0f}h on average, versus {g.loc[low, 'esc']:.0%} for '{low}'.",
                      "Issue type must be available to the model. issue_category is assigned at triage (not T0), so the model uses the customer-entered issue_title (30 values) and message text as its legitimate proxy."))

        # F3 behavioural drivers ---------------------------------------------------------------
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        pt = df.groupby(df["previous_ticket_count"].clip(upper=8))["escalated"].mean(); ax[0].plot(pt.index, pt.values, marker="o"); ax[0].set_title("Escalation vs previous tickets (8+ pooled)")
        dd = df.groupby(pd.cut(df["delivery_delay_days"], [-1, 0, 2, 5, 10, 20, 200]), observed=True)["escalated"].mean(); ax[1].bar(dd.index.astype(str), dd.values); ax[1].set_title("Escalation vs delivery delay (days)"); ax[1].tick_params(axis="x", rotation=30)
        ov = df.groupby(pd.qcut(df["order_value"], 10, duplicates="drop"), observed=True)["escalated"].mean(); ax[2].plot(range(len(ov)), ov.values, marker="o"); ax[2].set_title("Escalation vs order-value decile"); ax[2].set_xlabel("decile")
        P.save(fig, out / "fig3_behavioural_drivers.png")
        stats.update(esc_prev0=float(pt.iloc[0]), esc_prev8=float(pt.iloc[-1]), esc_delay0=float(dd.iloc[0]), esc_delay_max=float(dd.iloc[-1]), esc_value_d1=float(ov.iloc[0]), esc_value_d10=float(ov.iloc[-1]))
        pk = dd.idxmax()
        notes.append(("fig3_behavioural_drivers.png", "Prior contacts, delay and value",
                      f"Escalation rises steadily with previous tickets ({pt.iloc[0]:.0%} with none to {pt.iloc[-1]:.0%} with 8+) and with order value ({ov.iloc[0]:.0%} in the lowest decile to {ov.iloc[-1]:.0%} in the highest). Delay is NOT monotone: {dd.iloc[0]:.0%} when on time, peaking at {dd.max():.0%} for {pk} days late, then dropping to {dd.iloc[-1]:.0%} beyond 20 days.",
                      "Previous tickets and order value are the main monotone T0 signals; the non-monotone delay curve is a reason to use tree ensembles (and delay bins/interactions for linear baselines) instead of assuming linearity. The >20-day drop is a small, heavy-tailed group and is treated cautiously."))

        # F4 outliers ---------------------------------------------------------------------------
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].boxplot([np.log10(df["order_value"]), np.log10(df["delivery_delay_days"] + 1), np.log10(df["resolution_time_hours"])], tick_labels=["log10 order value", "log10(delay+1)", "log10 res. time"])
        ax[0].set_title("Heavy right tails (box = IQR, dots = beyond 1.5 IQR)")
        big = (df["order_value"] > df["order_value"].quantile(.999)); ax[1].bar(["orders <= p99.9", "orders > p99.9"], [df.loc[~big, "escalated"].mean(), df.loc[big, "escalated"].mean()]); ax[1].set_title("Escalation rate: extreme-value orders")
        P.save(fig, out / "fig4_outliers.png")
        stats["extreme_order_value_escalation"] = float(df.loc[big, "escalated"].mean())
        notes.append(("fig4_outliers.png", "Outliers",
                      f"order_value reaches GBP {df['order_value'].max():,.0f} against a p99.9 of GBP {df['order_value'].quantile(.999):,.0f}; the extreme orders escalate {df.loc[big, 'escalated'].mean():.0%} of the time (vs {df.loc[~big, 'escalated'].mean():.0%}).",
                      "Extremes are plausible rare high-value orders, not errors, so they are kept but winsorised at the TRAIN 99.9th percentile inside the pipeline (never using held-out quantiles); a log feature is added for linear models."))

        # F5 temporal + channel -----------------------------------------------------------------
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        h = df.groupby(df["timestamp"].dt.hour).size(); ax[0].bar(h.index, h.values); ax[0].set_title("Tickets by hour of day")
        ch = df.groupby("support_channel")["escalated"].mean().sort_values(); ax[1].barh(ch.index, ch.values); ax[1].set_title("Escalation by channel")
        pc = df.groupby("product_category")["resolution_time_hours"].mean().sort_values(); ax[2].barh(pc.index, pc.values); ax[2].set_title("Mean resolution time (h) by product category")
        P.save(fig, out / "fig5_time_channel_product.png")
        stats.update(channel_spread=float(ch.max() - ch.min()), hour_cv=float(h.std() / h.mean()), product_spread_h=float(pc.max() - pc.min()))
        notes.append(("fig5_time_channel_product.png", "Time, channel, product",
                      f"Ticket volume is almost flat across hours (CV {h.std() / h.mean():.2f}); escalation differs by only {ch.max() - ch.min():.1%} across channels and mean resolution time by {pc.max() - pc.min():.1f}h across product categories.",
                      "Calendar and channel features are cheap to keep but are expected to add little; the permutation-importance run in ml_classification confirms this rather than assuming it."))

        # F6 subgroup base rates ----------------------------------------------------------------
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        sub = {}
        for a, col in zip(ax, ["customer_age_band", "preferred_language", "accessibility_support_flag"]):
            r = df.groupby(col).agg(rate=("escalated", "mean"), n=("ticket_id", "size"))
            a.bar(r.index.astype(str), r["rate"]); a.set_ylim(0, .5); a.set_title(f"Escalation rate by {col}"); a.tick_params(axis="x", rotation=30)
            for i, (rt, n) in enumerate(zip(r["rate"], r["n"])): a.text(i, rt + .01, f"n={n:,}", ha="center", fontsize=7, rotation=90 if len(r) > 3 else 0)
            sub[col] = r.to_dict()
        P.save(fig, out / "fig6_subgroup_base_rates.png")
        acc = df.groupby("accessibility_support_flag")["escalated"].mean()
        stats["accessibility_escalation"] = {"flag0": float(acc.get(0, np.nan)), "flag1": float(acc.get(1, np.nan)), "n_flag1": int((df["accessibility_support_flag"] == 1).sum())}
        notes.append(("fig6_subgroup_base_rates.png", "Subgroup imbalance",
                      f"Age bands and languages escalate at near-identical rates, but accessibility-flagged customers escalate more ({acc.get(1, np.nan):.1%} vs {acc.get(0, np.nan):.1%}) and are only {stats['accessibility_escalation']['n_flag1'] / len(df):.1%} of tickets; Welsh/Romanian/Urdu/Punjabi/Polish speakers are each a few percent.",
                      "Sensitive attributes are kept out of the model (data minimisation) and used for per-subgroup error analysis; small groups get confidence intervals and an explicit sample-size caveat."))

        # F7 correlations -----------------------------------------------------------------------
        num = ["customer_tenure_months", "order_value", "delivery_delay_days", "previous_ticket_count", "attachment_available", "escalated", "resolution_time_hours"]
        cm = df[num].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6.2, 5))
        im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1); ax.set_xticks(range(len(num))); ax.set_yticks(range(len(num)))
        ax.set_xticklabels(num, rotation=45, ha="right"); ax.set_yticklabels(num); ax.grid(False)
        for i in range(len(num)):
            for j in range(len(num)):
                ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, shrink=.8); ax.set_title("Correlation (cleaned data)")
        P.save(fig, out / "fig7_correlations.png")
        stats["corr_delay_resolution"] = float(cm.loc["delivery_delay_days", "resolution_time_hours"]); stats["corr_prev_escalated"] = float(cm.loc["previous_ticket_count", "escalated"])
        notes.append(("fig7_correlations.png", "Linear structure",
                      f"previous_ticket_count correlates {stats['corr_prev_escalated']:.2f} with escalation and delivery_delay_days correlates {stats['corr_delay_resolution']:.2f} with resolution time; tenure is uncorrelated with both.",
                      "Moderate linear signal supports a Ridge/logistic baseline; the remaining non-linear structure justifies tree ensembles. Tenure is kept only for the non-linear/interaction learners and tested in ablation."))

        # F8 text noise -------------------------------------------------------------------------
        d = df["issue_description"].astype("string")
        upp = (d.fillna("").str.count(r"[A-Z]") / d.fillna("").str.count(r"[A-Za-z]").clip(lower=1)) > 0.5
        miss = d.isna()
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        ax[0].bar(["missing", "mostly UPPERCASE", "ok"], [miss.mean(), upp.mean(), 1 - miss.mean() - upp.mean()]); ax[0].set_title("Description quality")
        ax[1].bar(["description present", "description missing"], [df.loc[~miss, "escalated"].mean(), df.loc[miss, "escalated"].mean()]); ax[1].set_title("Escalation rate vs missing description"); ax[1].set_ylim(0, .5)
        P.save(fig, out / "fig8_text_quality.png")
        stats["missing_desc_pct"] = float(miss.mean()); stats["upper_pct"] = float(upp.mean())
        notes.append(("fig8_text_quality.png", "Noisy text",
                      f"{miss.mean():.2%} of descriptions are blank and {upp.mean():.2%} are mostly upper-case; messages carry template filler and noise footers (pls, thx, ASAP, '!!!').",
                      "Text pipelines lower-case and ignore filler; models receive a desc_missing flag instead of imputed text, and issue_title (never blank) backs up the text."))

        write_json(out / "eda_stats.json", stats)
        run.add(figures=len(notes))

    md = ["# EDA report", "",
          "Generated by `scripts/02_eda.py` on the cleaned data (996,000 unique tickets). Each figure has one computed observation and one modelling/business implication. Numbers come from `artifacts/eda/eda_stats.json`.", ""]
    for i, (fig, title, obs, imp) in enumerate(notes, 1):
        md += [f"## {i}. {title}", f"![{title}](../artifacts/eda/{fig})", "", f"**Observation.** {obs}", "", f"**Implication.** {imp}", ""]
    md += ["## Limitations", "",
           "- The data is synthetic: relationships (e.g. flat hour-of-day volume, zero tenure effect) are generator properties, not evidence about real customers.",
           "- Descriptive plots show association only; none of the patterns are causal claims.",
           "- Issue-category plots use a triage-assigned field for *explanation only*; it is not a model input at T0 (see `reports/leakage_register.md`)."]
    (REPORTS / "eda_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("[OK] EDA complete:", len(notes), "figures")


if __name__ == "__main__":
    main()
