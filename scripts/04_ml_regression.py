"""Epic ML-01/03: resolution-time regression at T0 (MAE primary, RMSE reported).

Run:  python scripts/04_ml_regression.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from omnisupport.config import MODEL_VERSION, MODELS, SEED, artifact_dir  # noqa: E402
from omnisupport.data.features import FEATURE_COLUMNS, make_preprocessor  # noqa: E402
from omnisupport.data.leakage import assert_features_valid  # noqa: E402
from omnisupport.ml import common as C  # noqa: E402
from omnisupport.ml.models import MedianRegressor, WinsorisedRegressor  # noqa: E402
from omnisupport.utils import plotting as P  # noqa: E402
from omnisupport.utils.runlog import RunRecord, write_json  # noqa: E402

TARGET = "resolution_time_hours"
SLA_HOURS = 36.0


def hgbr(**kw):
    return HistGradientBoostingRegressor(random_state=SEED, **kw)


def pipe(model, scale=False):
    return Pipeline([("pre", make_preprocessor(scale=scale)), ("m", WinsorisedRegressor(model))])


def main() -> None:
    out = artifact_dir("ml_regression")
    frame = C.modelling_frame(); parts = C.split_frame(frame)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    assert_features_valid(FEATURE_COLUMNS)

    cfg = {"target": TARGET, "features": FEATURE_COLUMNS, "winsorise_train_target_at": 0.999, "sla_hours": SLA_HOURS,
           "cv": "GroupKFold(5) by customer_id on 120k train subsample", "tuning": "RandomizedSearchCV n_iter=14 GroupKFold(3) neg_MAE"}
    with RunRecord(out, "ml_regression", config=cfg, split_method="temporal: train 2025-01..08, val 09..10, test 11..12; CV grouped by customer",
                   model_version=f"{MODEL_VERSION}/resolution-hgb", limitations=["MAE is the headline: RMSE is dominated by <0.1% extreme resolution times.", "Synthetic data."]) as run:
        sub = C.cv_subsample(tr, 120_000); groups = sub["customer_id"]
        cands = {
            "baseline_median": Pipeline([("pre", make_preprocessor()), ("m", MedianRegressor("median"))]),
            "baseline_mean": Pipeline([("pre", make_preprocessor()), ("m", MedianRegressor("mean"))]),
            "ridge": pipe(Ridge(alpha=3.0), scale=True),
            "random_forest": pipe(RandomForestRegressor(n_estimators=120, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=SEED)),
            "hist_gradient_boosting": pipe(hgbr(max_iter=250, learning_rate=0.08, early_stopping=False)),
        }
        rows = []
        for name, mp in cands.items():
            t0 = time.perf_counter()
            r = cross_validate(mp, sub[FEATURE_COLUMNS], sub[TARGET], groups=groups, cv=GroupKFold(5), n_jobs=1,
                               scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"})
            rows.append({"model": name, "cv_mae_mean": -r["test_mae"].mean(), "cv_mae_std": r["test_mae"].std(), "cv_rmse_mean": -r["test_rmse"].mean(), "cv_rmse_std": r["test_rmse"].std(), "seconds": time.perf_counter() - t0})
            print(rows[-1])
        cv_tab = pd.DataFrame(rows).sort_values("cv_mae_mean"); cv_tab.to_csv(out / "cv_model_comparison.csv", index=False)

        space = {"m__base__learning_rate": [0.03, 0.05, 0.08, 0.12], "m__base__max_leaf_nodes": [15, 31, 63, 127], "m__base__min_samples_leaf": [20, 50, 100, 200],
                 "m__base__l2_regularization": [0.0, 0.5, 2.0, 5.0], "m__base__max_iter": [150, 250, 400], "m__base__loss": ["squared_error", "absolute_error"]}
        search = RandomizedSearchCV(pipe(hgbr(early_stopping=False)), space, n_iter=14, scoring="neg_mean_absolute_error", cv=GroupKFold(3), random_state=SEED, n_jobs=1, refit=False)
        search.fit(sub[FEATURE_COLUMNS], sub[TARGET], groups=groups)
        pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]].sort_values("rank_test_score").to_csv(out / "tuning_results.csv", index=False)
        best = {k.replace("m__base__", ""): v for k, v in search.best_params_.items()}
        write_json(out / "best_params.json", {"best_params": best, "cv_mae_tuned": float(-search.best_score_), "search_space": space})
        print("best", best, round(-search.best_score_, 3))

        final = pipe(hgbr(early_stopping=False, **best)).fit(tr[FEATURE_COLUMNS], tr[TARGET])
        base_med = cands["baseline_median"].fit(tr[FEATURE_COLUMNS], tr[TARGET])
        ridge = cands["ridge"].fit(tr[FEATURE_COLUMNS], tr[TARGET])
        pv = final.predict(va[FEATURE_COLUMNS])
        val_tab = pd.DataFrame({"hgb_tuned": C.reg_metrics(va[TARGET], pv), "ridge": C.reg_metrics(va[TARGET], ridge.predict(va[FEATURE_COLUMNS])),
                                "baseline_median": C.reg_metrics(va[TARGET], base_med.predict(va[FEATURE_COLUMNS]))}).T
        val_tab.to_csv(out / "validation_model_comparison.csv")
        print(val_tab.round(3))

        # prediction intervals (quantile HGB) trained on the same train rows
        q_lo = Pipeline([("pre", make_preprocessor()), ("m", hgbr(loss="quantile", quantile=0.05, max_iter=200, learning_rate=0.08, early_stopping=False))]).fit(tr[FEATURE_COLUMNS], tr[TARGET].clip(upper=tr[TARGET].quantile(.999)))
        q_hi = Pipeline([("pre", make_preprocessor()), ("m", hgbr(loss="quantile", quantile=0.95, max_iter=200, learning_rate=0.08, early_stopping=False))]).fit(tr[FEATURE_COLUMNS], tr[TARGET].clip(upper=tr[TARGET].quantile(.999)))
        lo_v, hi_v = q_lo.predict(va[FEATURE_COLUMNS]), q_hi.predict(va[FEATURE_COLUMNS])
        cover_v = float(((va[TARGET] >= lo_v) & (va[TARGET] <= hi_v)).mean())

        # SLA-risk use case: flag tickets predicted above a cut-off; choose cut-off on validation (F1 for >36h)
        y_sla_v = (va[TARGET] > SLA_HOURS).astype(int).values
        cuts = np.arange(20, 50, 1.0); res = []
        for c in cuts:
            f = pv >= c; tp = int((f & (y_sla_v == 1)).sum()); fp = int((f & (y_sla_v == 0)).sum()); fn = int((~f & (y_sla_v == 1)).sum())
            pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); res.append({"cutoff_h": c, "precision": pr, "recall": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-9), "flag_rate": float(f.mean())})
        sla_tab = pd.DataFrame(res); sla_tab.to_csv(out / "sla_risk_cutoff_validation.csv", index=False)
        sla_cut = float(sla_tab.loc[sla_tab["f1"].idxmax(), "cutoff_h"])

        # residual / error-slice analysis on VALIDATION
        v = va.assign(pred=pv, err=pv - va[TARGET], abs_err=np.abs(pv - va[TARGET]))
        v["delay_bucket"] = pd.cut(v["delivery_delay_days"], [-1, 0, 2, 5, 10, 20, 500], labels=["0", "1-2", "3-5", "6-10", "11-20", ">20"])
        v["prev_bucket"] = pd.cut(v["previous_ticket_count"], [-1, 0, 1, 2, 4, 20], labels=["0", "1", "2", "3-4", "5+"])
        v["true_bucket"] = pd.cut(v[TARGET], [0, 12, 24, 36, 60, 1000], labels=["<12h", "12-24h", "24-36h", "36-60h", ">60h"])
        sl = []
        for col in ["issue_title", "delay_bucket", "prev_bucket", "true_bucket", "support_channel", "product_category"]:
            g = v.groupby(col, observed=True).agg(n=("abs_err", "size"), mae=("abs_err", "mean"), bias=("err", "mean"), mean_true=(TARGET, "mean")).reset_index().rename(columns={col: "value"}); g.insert(0, "slice", col); sl.append(g)
        sl = pd.concat(sl); sl.to_csv(out / "error_slices_validation.csv", index=False)
        big = sl[sl["n"] >= 2000]
        patterns = [f"Largest MAE: `{r['slice']}`={r['value']} (MAE {r['mae']:.1f}h, bias {r['bias']:+.1f}h, n={int(r['n']):,})" for _, r in big.sort_values("mae", ascending=False).head(3).iterrows()]
        patterns += [f"Systematic bias: `{r['slice']}`={r['value']} (mean error {r['bias']:+.1f}h, n={int(r['n']):,})" for _, r in big.sort_values("bias", key=lambda x: x.abs(), ascending=False).head(2).iterrows()]
        write_json(out / "error_patterns.json", {"patterns": patterns, "sla_cutoff_h": sla_cut, "interval_coverage_validation_90pct_nominal": cover_v}); print("\n".join(patterns))

        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        ax[0].scatter(va[TARGET].clip(upper=120)[:6000], pv[:6000], s=3, alpha=.3); ax[0].plot([0, 120], [0, 120], "k:"); ax[0].set_xlabel("actual (h, clipped 120)"); ax[0].set_ylabel("predicted"); ax[0].set_title("Predicted vs actual (validation sample)")
        ax[1].hist(v["err"].clip(-30, 30), bins=80); ax[1].axvline(0, color="k"); ax[1].set_title("Residuals pred-actual (h, clipped +/-30)")
        tb = v.groupby("true_bucket", observed=True)["err"].mean(); ax[2].bar(tb.index.astype(str), tb.values); ax[2].axhline(0, color="k"); ax[2].set_title("Mean error by actual duration (regression to the mean)")
        P.save(fig, out / "fig_residuals.png")

        # ------------------------------------------------------------ frozen single TEST evaluation
        pt = final.predict(te[FEATURE_COLUMNS]); lo_t, hi_t = q_lo.predict(te[FEATURE_COLUMNS]), q_hi.predict(te[FEATURE_COLUMNS])
        test = {"hgb_tuned (selected)": C.reg_metrics(te[TARGET], pt), "ridge": C.reg_metrics(te[TARGET], ridge.predict(te[FEATURE_COLUMNS])), "baseline_median": C.reg_metrics(te[TARGET], base_med.predict(te[FEATURE_COLUMNS]))}
        pd.DataFrame(test).T.to_csv(out / "heldout_test_results.csv")
        trimmed = te[TARGET] <= te[TARGET].quantile(.999)
        sel = test["hgb_tuned (selected)"]; basem = test["baseline_median"]
        y_sla_t = (te[TARGET] > SLA_HOURS).astype(int).values; f = pt >= sla_cut
        tp = int((f & (y_sla_t == 1)).sum()); fp = int((f & (y_sla_t == 0)).sum()); fn = int((~f & (y_sla_t == 1)).sum())
        extra = {"mae_reduction_vs_median_pct": 100 * (1 - sel["mae"] / basem["mae"]), "rmse_reduction_vs_median_pct": 100 * (1 - sel["rmse"] / basem["rmse"]),
                 "mae_excluding_top_0.1pct_targets": float(np.abs(pt[trimmed.values] - te[TARGET][trimmed]).mean()),
                 "interval_90_coverage_test": float(((te[TARGET] >= lo_t) & (te[TARGET] <= hi_t)).mean()), "mean_interval_width_h": float((hi_t - lo_t).mean()),
                 "sla_36h_precision": tp / max(tp + fp, 1), "sla_36h_recall": tp / max(tp + fn, 1), "sla_cutoff_h": sla_cut, "test_evaluations_performed": 1,
                 "pre_registered_success_criteria": {"mae_reduction>=30pct": 100 * (1 - sel["mae"] / basem["mae"]) >= 30}}
        write_json(out / "heldout_test_selected_model.json", {"metrics": sel, **extra})
        print("TEST", {k: round(v, 3) for k, v in sel.items()}, {k: v for k, v in extra.items() if not isinstance(v, dict)})

        for nm, part, pr in (("val", va, pv), ("test", te, pt)):
            part[["ticket_id", "customer_age_band", "preferred_language", "accessibility_support_flag", "customer_region", "support_channel", "product_category", "issue_title", TARGET]].assign(pred=pr).to_parquet(C.PROCESSED_DIR / f"pred_reg_{nm}.parquet", index=False)
        serving = MODELS / "serving"; serving.mkdir(parents=True, exist_ok=True)
        joblib.dump({"point": final, "q05": q_lo, "q95": q_hi}, serving / "resolution_time_hgb.joblib", compress=3)
        write_json(serving / "resolution_time_metadata.json", {"model_version": f"{MODEL_VERSION}/resolution-hgb", "task": "predict resolution_time_hours at ticket creation (T0)", "features": FEATURE_COLUMNS,
                                                              "hyperparameters": best, "sklearn_version": sklearn.__version__, "seed": SEED, "validation_metrics": C.reg_metrics(va[TARGET], pv), "test_metrics": sel,
                                                              "interval": "5th-95th percentile quantile models (nominal 90%)", "sla_hours": SLA_HOURS, "sla_cutoff_h": sla_cut})
        run.add(test_mae=sel["mae"], test_rmse=sel["rmse"])
    print("[OK] regression complete")


if __name__ == "__main__":
    main()
