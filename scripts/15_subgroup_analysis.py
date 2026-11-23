"""Subgroup performance & fairness analysis on the frozen TEST predictions (classification) and regression test predictions.

Sensitive attributes (age band, language, accessibility flag) were NOT model inputs (leakage register: evaluation_only_sensitive);
they are used here only to audit outcomes. Reports gaps with confidence intervals and flags small groups instead of over-claiming.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from omnisupport.config import PROCESSED_DIR, REPORTS, artifact_dir
from omnisupport.ml.common import wilson_ci
from omnisupport.utils.runlog import write_json

OUT = artifact_dir("subgroups")
thr = json.loads((artifact_dir("ml_classification") / "threshold_selection.json").read_text())["chosen"]
df = pd.read_parquet(PROCESSED_DIR / "pred_cls_test.parquet")
df["accessibility_support_flag"] = df["accessibility_support_flag"].astype(str)
df["y"] = df["escalated"].astype(int)
df["flag"] = (df["p"] >= thr).astype(int)
overall_recall = df.loc[df.y == 1, "flag"].mean(); overall_fpr = df.loc[df.y == 0, "flag"].mean(); overall_flag = df.flag.mean()
MIN_N = 1000
rows = []
for attr in ["customer_age_band", "preferred_language", "accessibility_support_flag", "customer_region", "support_channel", "product_category"]:
    for val, g in df.groupby(attr, observed=True):
        pos, neg = g[g.y == 1], g[g.y == 0]
        tp, fn, fp, tn = int(pos.flag.sum()), int((1 - pos.flag).sum()), int(neg.flag.sum()), int((1 - neg.flag).sum())
        lo, hi = wilson_ci(tp, len(pos))
        rows.append(dict(attribute=attr, value=val, n=len(g), base_rate=g.y.mean(), mean_pred=g.p.mean(), calib_gap=g.p.mean() - g.y.mean(), flag_rate=g.flag.mean(),
                         recall=tp / max(len(pos), 1), recall_lo=lo, recall_hi=hi, fpr=fp / max(len(neg), 1), precision=tp / max(tp + fp, 1), small_group=len(g) < MIN_N))
t = pd.DataFrame(rows)
t["recall_gap_vs_overall"] = t.recall - overall_recall
t["fpr_gap_vs_overall"] = t.fpr - overall_fpr
t["flag_rate_ratio_vs_overall"] = t.flag_rate / overall_flag
t["recall_gap_significant"] = (t.recall_hi < overall_recall) | (t.recall_lo > overall_recall)
t.round(4).to_csv(OUT / "subgroup_classification_test.csv", index=False)

# regression subgroups
reg = None
pr = PROCESSED_DIR / "pred_reg_test.parquet"
if pr.exists():
    r = pd.read_parquet(PROCESSED_DIR / "pred_reg_test.parquet")
    tgt = [c for c in r.columns if c not in ("ticket_id", "customer_age_band", "preferred_language", "accessibility_support_flag", "customer_region", "support_channel", "product_category", "issue_title", "pred")][0]
    r["accessibility_support_flag"] = r["accessibility_support_flag"].astype(str); r["ae"] = (r.pred - r[tgt]).abs(); r["e"] = r.pred - r[tgt]
    rr = []
    for attr in ["customer_age_band", "preferred_language", "accessibility_support_flag"]:
        for val, g in r.groupby(attr, observed=True):
            rr.append(dict(attribute=attr, value=val, n=len(g), mae=g.ae.mean(), bias=g.e.mean(), mean_actual=g[tgt].mean()))
    reg = pd.DataFrame(rr); reg["mae_ratio_vs_overall"] = reg.mae / r.ae.mean(); reg.round(4).to_csv(OUT / "subgroup_regression_test.csv", index=False)

sens = t[t.attribute.isin(["customer_age_band", "preferred_language", "accessibility_support_flag"])]
worst = sens.reindex(sens.recall_gap_vs_overall.abs().sort_values(ascending=False).index[:1])
summary = {"threshold": thr, "overall": {"recall": overall_recall, "fpr": overall_fpr, "flag_rate": overall_flag, "n": len(df)},
           "max_abs_recall_gap_sensitive": float(sens.recall_gap_vs_overall.abs().max()), "max_abs_fpr_gap_sensitive": float(sens.fpr_gap_vs_overall.abs().max()),
           "significant_recall_gaps": t[t.recall_gap_significant & ~t.small_group][["attribute", "value", "recall_gap_vs_overall"]].round(4).to_dict("records")}
write_json(OUT / "summary.json", summary)


def md(d: pd.DataFrame, cols, fmt=3):
    d = d[cols].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(fmt)
    return d.to_markdown(index=False) if hasattr(d, "to_markdown") else d.to_string(index=False)


try:
    import tabulate  # noqa: F401
except ImportError:
    md = lambda d, cols, fmt=3: "```\n" + d[cols].round(fmt).to_string(index=False) + "\n```"  # noqa: E731

lines = ["# Subgroup analysis (frozen test set)", "", f"Classification: escalation model at the validation-chosen threshold **{thr}** on **{len(df):,}** held-out test tickets "
         f"(overall recall {overall_recall:.3f}, FPR {overall_fpr:.3f}, flag rate {overall_flag:.3f}).", "",
         "Age band, language and accessibility flag were **not** model inputs (leakage/sensitivity register); they are used only to audit outcomes. "
         f"Groups with n < {MIN_N} are flagged and not interpreted. Recall CIs are Wilson 95%.", ""]
for attr in t.attribute.unique():
    lines += [f"## {attr}", "", md(t[t.attribute == attr].sort_values("value"), ["value", "n", "base_rate", "mean_pred", "calib_gap", "flag_rate", "recall", "recall_lo", "recall_hi", "fpr", "precision"]), ""]
lines += ["## Findings", ""]
sg = summary["significant_recall_gaps"]
lines.append(f"- Largest absolute recall gap on a sensitive/support attribute: **{summary['max_abs_recall_gap_sensitive']:.3f}**; largest FPR gap: **{summary['max_abs_fpr_gap_sensitive']:.3f}**.")
lines.append(f"- Groups (n >= {MIN_N}) whose recall CI excludes the overall recall: " + (", ".join(f"`{x['attribute']}={x['value']}` ({x['recall_gap_vs_overall']:+.3f})" for x in sg) if sg else "none."))
big_flag = t.sort_values("flag_rate_ratio_vs_overall").iloc[[0, -1]]
lines.append("- Flag-rate spread (relative to overall): " + "; ".join(f"`{r.attribute}={r.value}` x{r.flag_rate_ratio_vs_overall:.2f}" for r in big_flag.itertuples()) + ". Differences in flag rate largely track differences in base rate (see `base_rate` column), which is expected for a risk score; recall/FPR parity is the better fairness lens.")
if reg is not None:
    lines += ["", "## Regression (resolution time) by sensitive attribute", "", md(reg, ["attribute", "value", "n", "mae", "bias", "mean_actual", "mae_ratio_vs_overall"]), ""]
lines += ["", "## Interpretation and limits", "",
          "- The data are synthetic; a gap here is a property of the generator and model, not evidence about real customers. The procedure is what transfers.",
          "- Equal recall across groups was **not** enforced; the model is advisory and a human decides. If a significant gap were operationally material we would (1) check base-rate differences, (2) evaluate group-specific thresholds on validation only, (3) re-audit on a fresh period.",
          "- Accessibility-flagged customers should never receive a *worse* service because of a score: use the score only to raise priority for review (a false positive costs a few minutes; a false negative costs a delayed escalation).",
          "- Intersectional groups were not analysed (small n); this is a known limitation (see risk register)."]
(REPORTS / "subgroup_report.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:12])); print(json.dumps(summary, indent=1, default=float)[:900])
