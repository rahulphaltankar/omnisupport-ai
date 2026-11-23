"""Epic UNSUP-01: ticket segmentation (unit = ticket at T0) with validation and cautious personas.

Clustering uses ONLY T0-valid features; outcomes (escalation, resolution time) are used afterwards to *describe* segments.
Run:  python scripts/05_segmentation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans, MiniBatchKMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from omnisupport.config import REPORTS, SEED, artifact_dir  # noqa: E402
from omnisupport.ml import common as C  # noqa: E402
from omnisupport.utils import plotting as P  # noqa: E402
from omnisupport.utils.runlog import RunRecord, write_json  # noqa: E402

NUM = ["log_order_value", "log_delay", "previous_ticket_count", "customer_tenure_months", "attachment_available"]
FAMILY_RULES = [("delivery", r"parcel|order still|delivery|late|delayed|lost"), ("damage", r"damage|broken|arrived"), ("wrong_item", r"not what i ordered|wrong"),
                ("refund", r"refund"), ("return", r"send this back|return"), ("account_payment_other", r".*")]


def family(title: pd.Series) -> pd.Series:
    t = title.astype(str).str.lower(); out = pd.Series("account_payment_other", index=t.index)
    for name, pat in reversed(FAMILY_RULES[:-1]):
        out = out.where(~t.str.contains(pat, regex=True), name)
    return out


def design(df: pd.DataFrame, scaler: StandardScaler | None = None, families: list[str] | None = None):
    X = pd.DataFrame({"log_order_value": np.log1p(df["order_value"]), "log_delay": np.log1p(df["delivery_delay_days"]), "previous_ticket_count": df["previous_ticket_count"].clip(upper=10),
                      "customer_tenure_months": df["customer_tenure_months"], "attachment_available": df["attachment_available"]})
    fam = family(df["issue_title"]); families = families or sorted(fam.unique())
    oh = pd.DataFrame({f"fam_{f}": (fam == f).astype(float) * 0.7 for f in families}, index=df.index)  # 0.7 weight: categorical shouldn't dominate
    scaler = scaler or StandardScaler().fit(X[NUM])
    return np.hstack([scaler.transform(X[NUM]), oh.values]), scaler, families, fam


def main() -> None:
    out = artifact_dir("segmentation")
    frame = C.modelling_frame(); parts = C.split_frame(frame)
    tr, va = parts["train"], parts["val"]
    with RunRecord(out, "segmentation", config={"unit": "ticket", "features": NUM + ["issue_family (weight 0.7)"], "k_range": "2-9", "algorithm": "MiniBatchKMeans n_init=5", "fit_sample": 150000},
                   split_method="fit on train sample; stability and profile consistency checked on validation period", model_version="segmentation-v1",
                   limitations=["Synthetic data: segments describe generator structure, not real behaviour.", "Personas are descriptive associations; no causal claim."]) as run:
        fit = tr.sample(150_000, random_state=SEED)
        Xf, scaler, fams, fam_fit = design(fit)
        sil_idx = np.random.default_rng(SEED).choice(len(Xf), 20_000, replace=False)
        rows = []
        for k in range(2, 10):
            km = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=SEED, batch_size=4096).fit(Xf)
            lab = km.labels_
            # stability: refit on two disjoint halves, label the same evaluation set, compare partitions
            half = np.random.default_rng(k).permutation(len(Xf)); a, b = half[:len(half) // 2], half[len(half) // 2:]
            ka = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=1, batch_size=4096).fit(Xf[a]); kb = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=2, batch_size=4096).fit(Xf[b])
            ari = adjusted_rand_score(ka.predict(Xf[sil_idx]), kb.predict(Xf[sil_idx]))
            rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Xf[sil_idx], lab[sil_idx]), "davies_bouldin": davies_bouldin_score(Xf[sil_idx], lab[sil_idx]),
                         "stability_ari_halves": ari, "min_cluster_share": float(np.bincount(lab).min() / len(lab))})
            print(rows[-1])
        sel = pd.DataFrame(rows); sel.to_csv(out / "k_selection.csv", index=False)
        ok = sel[(sel["stability_ari_halves"] >= 0.7) & (sel["min_cluster_share"] >= 0.03)]
        k_best = int((ok if len(ok) else sel).sort_values(["silhouette"], ascending=False).iloc[0]["k"])
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].plot(sel["k"], sel["inertia"], marker="o"); ax[0].set_title("Inertia (elbow)"); ax[1].plot(sel["k"], sel["silhouette"], marker="o", label="silhouette"); ax[1].plot(sel["k"], sel["stability_ari_halves"], marker="s", label="stability ARI"); ax[1].axvline(k_best, color="k", ls="--"); ax[1].legend(); ax[1].set_title("Silhouette & stability")
        ax[2].plot(sel["k"], sel["davies_bouldin"], marker="o"); ax[2].set_title("Davies-Bouldin (lower=better)")
        for a in ax: a.set_xlabel("k")
        P.save(fig, out / "fig_cluster_selection.png")

        km = KMeans(n_clusters=k_best, n_init=10, random_state=SEED).fit(Xf)
        fit = fit.assign(segment=km.labels_, family=fam_fit.values)
        Xv, _, _, fam_v = design(va, scaler, fams); va = va.assign(segment=km.predict(Xv), family=fam_v.values)

        def profile(d):
            g = d.groupby("segment").agg(share=("ticket_id", lambda s: len(s) / len(d)), order_value_median=("order_value", "median"), delay_days_mean=("delivery_delay_days", "mean"),
                                        previous_tickets_mean=("previous_ticket_count", "mean"), tenure_months_mean=("customer_tenure_months", "mean"), attachment_rate=("attachment_available", "mean"),
                                        escalation_rate=("escalated", "mean"), resolution_h_mean=("resolution_time_hours", "mean"),
                                        top_issue_family=("family", lambda s: s.value_counts().index[0]), top_family_share=("family", lambda s: s.value_counts(normalize=True).iloc[0]))
            return g
        pf, pv = profile(fit), profile(va); pf.round(3).to_csv(out / "segment_profile_train.csv"); pv.round(3).to_csv(out / "segment_profile_validation.csv")
        consistency = pd.DataFrame({"share_train": pf["share"], "share_val": pv["share"], "esc_train": pf["escalation_rate"], "esc_val": pv["escalation_rate"], "res_h_train": pf["resolution_h_mean"], "res_h_val": pv["resolution_h_mean"]}).round(3)
        consistency.to_csv(out / "segment_consistency_train_vs_validation.csv")
        pca = PCA(2, random_state=SEED).fit(Xf); Z = pca.transform(Xf[sil_idx][:6000])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4)); sc = ax[0].scatter(Z[:, 0], Z[:, 1], c=km.labels_[sil_idx][:6000], s=4, cmap="tab10"); ax[0].set_title(f"Segments in PCA space (k={k_best}; PCA explains {pca.explained_variance_ratio_.sum():.0%})")
        ax[1].bar(pf.index.astype(str), pf["escalation_rate"], label="train"); ax[1].plot(pv.index.astype(str), pv["escalation_rate"], "ro", label="validation"); ax[1].set_title("Escalation rate by segment (descriptive only)"); ax[1].legend()
        P.save(fig, out / "fig_segments.png")

        # cautious persona text, generated from the profile
        gm = fit["escalated"].mean(); gr = fit["resolution_time_hours"].mean(); lines = []
        for s, r in pf.iterrows():
            traits = []
            if r["previous_tickets_mean"] > fit["previous_ticket_count"].mean() + 0.5: traits.append("frequent previous contacts")
            elif r["previous_tickets_mean"] < fit["previous_ticket_count"].mean() - 0.5: traits.append("few previous contacts")
            if r["delay_days_mean"] > fit["delivery_delay_days"].mean() * 1.5: traits.append("long delivery delays")
            if r["order_value_median"] > fit["order_value"].median() * 1.25: traits.append("higher-value orders")
            if r["order_value_median"] < fit["order_value"].median() * 0.8: traits.append("lower-value orders")
            if r["attachment_rate"] > 0.5: traits.append("usually attach a file")
            traits = traits or ["close to the overall average on the numeric features"]
            lines.append(f"- **Segment {s}** ({r['share']:.0%} of tickets; mostly '{r['top_issue_family']}' tickets, {r['top_family_share']:.0%} of the segment): tickets that tend to show {', '.join(traits)}. "
                         f"In training {r['escalation_rate']:.0%} escalate (overall {gm:.0%}) and mean resolution time is {r['resolution_h_mean']:.1f}h (overall {gr:.1f}h); validation shows {pv.loc[s, 'escalation_rate']:.0%} and {pv.loc[s, 'resolution_h_mean']:.1f}h.")
        silk = float(sel.set_index("k").loc[k_best, "silhouette"]); ari = float(sel.set_index("k").loc[k_best, "stability_ari_halves"])
        md = ["# Ticket segmentation and cautious personas", "",
              f"*Unit:* a ticket at creation (T0). *Features:* log order value, log delay, previous tickets (capped), tenure, attachment, issue family. *Method:* MiniBatchKMeans scan k=2..9, final KMeans k={k_best}.",
              f"*Selection:* the highest-silhouette k among those with half-split stability ARI >= 0.7 and no cluster below 3% share. Chosen k={k_best}: silhouette {silk:.3f}, stability ARI {ari:.3f}. See `artifacts/segmentation/k_selection.csv` and `fig_cluster_selection.png`.", "",
              "![k selection](../artifacts/segmentation/fig_cluster_selection.png)", "![segments](../artifacts/segmentation/fig_segments.png)", "", "## Personas (descriptive, not causal)", "", *lines, "",
              "## How far to trust these", "",
              f"- Silhouette {silk:.2f} is {'modest' if silk < 0.35 else 'reasonable'}: segments overlap and boundaries are soft; they are **useful groupings, not natural species**.",
              "- Outcome columns were not used to form segments; train-vs-validation consistency is in `segment_consistency_train_vs_validation.csv`.",
              "- Escalation/resolution differences are **associations**: a segment does not *cause* its outcome and these must not drive individual customer decisions.",
              "- The data is synthetic; real-world segments would need re-validation, and sensitive attributes (age, language, accessibility) were deliberately excluded from clustering."]
        (REPORTS / "segmentation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        write_json(out / "selection_summary.json", {"k": k_best, "silhouette": silk, "stability_ari": ari, "consistency": consistency.to_dict()})
        run.add(k=k_best)
    print("[OK] segmentation complete k =", k_best)


if __name__ == "__main__":
    main()
