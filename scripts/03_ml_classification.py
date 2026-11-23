"""Epic ML-02/03: escalation classification at T0.

Flow: baselines -> grouped CV comparison -> randomised tuning (CV) -> validation-period threshold selection ->
frozen single test-period evaluation -> error analysis, importance, ablations, leakage demonstration -> serving model.

Run:  python scripts/03_ml_classification.py
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
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from omnisupport.config import MODEL_VERSION, MODELS, SEED, artifact_dir  # noqa: E402
from omnisupport.data.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, make_preprocessor  # noqa: E402
from omnisupport.data.leakage import assert_features_valid  # noqa: E402
from omnisupport.ml import common as C  # noqa: E402
from omnisupport.utils import plotting as P  # noqa: E402
from omnisupport.utils.runlog import RunRecord, write_json  # noqa: E402

FN_COST, FP_COST = 3.0, 1.0
TARGET = "escalated"


def hgb(**kw):
    return HistGradientBoostingClassifier(random_state=SEED, **kw)


def pipe(model, scale=False, **pp):
    return Pipeline([("pre", make_preprocessor(scale=scale, **pp)), ("m", model)])


def fit_eval(model_pipe, tr, va, cols=FEATURE_COLUMNS):
    model_pipe.fit(tr[cols], tr[TARGET])
    p = model_pipe.predict_proba(va[cols])[:, 1]
    return p


def main() -> None:
    out = artifact_dir("ml_classification")
    frame = C.modelling_frame()
    parts = C.split_frame(frame)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    assert_features_valid(FEATURE_COLUMNS)  # enforce the leakage register on the final feature list
    print({k: len(v) for k, v in parts.items()})

    cfg = {"target": TARGET, "features": FEATURE_COLUMNS, "fn_cost": FN_COST, "fp_cost": FP_COST, "cv": "GroupKFold(5) by customer_id on a 120k train subsample",
           "tuning": "RandomizedSearchCV n_iter=14, GroupKFold(3), scoring=average_precision"}
    with RunRecord(out, "ml_classification", config=cfg, split_method="temporal: train 2025-01..08, val 09..10, test 11..12; CV grouped by customer",
                   model_version=f"{MODEL_VERSION}/escalation-hgb",
                   limitations=["Synthetic data; temporal split but 96% of later customers appear in train.",
                                "CV/tuning on a 120k train subsample for runtime; the final model is refit on all train rows."]) as run:
        # ------------------------------------------------------------ 1. CV comparison
        sub = C.cv_subsample(tr, 120_000)
        groups = sub["customer_id"]
        gkf = GroupKFold(n_splits=5)
        candidates = {
            "dummy_prior": pipe(DummyClassifier(strategy="prior")),
            "logistic_regression": pipe(LogisticRegression(max_iter=400, class_weight="balanced", C=1.0), scale=True),
            "random_forest": pipe(RandomForestClassifier(n_estimators=120, min_samples_leaf=25, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
            "hist_gradient_boosting": pipe(hgb(max_iter=200, learning_rate=0.1, early_stopping=False)),
        }
        rows = []
        for name, mp in candidates.items():
            t0 = time.perf_counter()
            r = cross_validate(mp, sub[FEATURE_COLUMNS], sub[TARGET], groups=groups, cv=gkf, scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"}, n_jobs=1)
            rows.append({"model": name, "cv_pr_auc_mean": r["test_pr_auc"].mean(), "cv_pr_auc_std": r["test_pr_auc"].std(), "cv_roc_auc_mean": r["test_roc_auc"].mean(),
                         "cv_roc_auc_std": r["test_roc_auc"].std(), "fit_time_s": r["fit_time"].mean(), "seconds_total": time.perf_counter() - t0})
            print(rows[-1])
        cv_tab = pd.DataFrame(rows).sort_values("cv_pr_auc_mean", ascending=False)
        cv_tab.to_csv(out / "cv_model_comparison.csv", index=False)

        # ------------------------------------------------------------ 2. tuning
        space = {"m__learning_rate": [0.03, 0.05, 0.08, 0.12, 0.2], "m__max_leaf_nodes": [15, 31, 63, 127], "m__min_samples_leaf": [20, 50, 100, 200],
                 "m__l2_regularization": [0.0, 0.1, 1.0, 5.0], "m__max_iter": [150, 250, 400], "m__max_depth": [None, 6, 10]}
        search = RandomizedSearchCV(pipe(hgb(early_stopping=False)), space, n_iter=14, scoring="average_precision", cv=GroupKFold(n_splits=3), random_state=SEED, n_jobs=1, refit=False)
        t0 = time.perf_counter()
        search.fit(sub[FEATURE_COLUMNS], sub[TARGET], groups=groups)
        tune = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]].sort_values("rank_test_score")
        tune.to_csv(out / "tuning_results.csv", index=False)
        best_params = {k.replace("m__", ""): v for k, v in search.best_params_.items()}
        default_cv = float(cv_tab.set_index("model").loc["hist_gradient_boosting", "cv_pr_auc_mean"])
        print("best", best_params, round(search.best_score_, 4), "default", round(default_cv, 4), f"{time.perf_counter() - t0:.0f}s")
        write_json(out / "best_params.json", {"best_params": best_params, "cv_pr_auc_tuned": float(search.best_score_), "cv_pr_auc_default_same_data": default_cv,
                                              "search_space": space})

        # ------------------------------------------------------------ 3. final fit on full train, validation analysis
        final = pipe(hgb(early_stopping=False, **best_params))
        final.fit(tr[FEATURE_COLUMNS], tr[TARGET])
        p_val = final.predict_proba(va[FEATURE_COLUMNS])[:, 1]
        base_models = {n: m for n, m in candidates.items() if n != "hist_gradient_boosting"}
        val_scores = {"hgb_tuned": C.class_metrics(va[TARGET], p_val, 0.5)}
        fitted_baselines = {}
        for n, m in base_models.items():
            m.fit(tr[FEATURE_COLUMNS], tr[TARGET])
            fitted_baselines[n] = m
            val_scores[n] = C.class_metrics(va[TARGET], m.predict_proba(va[FEATURE_COLUMNS])[:, 1], 0.5)
        pd.DataFrame(val_scores).T.to_csv(out / "validation_model_comparison.csv")

        tt = C.threshold_table(va[TARGET], p_val, FN_COST, FP_COST)
        tt.to_csv(out / "threshold_table_validation.csv", index=False)
        thr = C.pick_thresholds(tt, 0.60)
        chosen = thr["cost_optimal"]
        cal = C.calibration_table(va[TARGET], p_val); cal.to_csv(out / "calibration_validation.csv", index=False)
        ece = C.expected_calibration_error(cal)
        write_json(out / "threshold_selection.json", {"thresholds": thr, "chosen": chosen, "rule": f"minimise expected cost with FN={FN_COST} FP={FP_COST} on validation", "ece_validation": ece,
                                                      "metrics_at_chosen_validation": C.class_metrics(va[TARGET], p_val, chosen)})
        print("thresholds", thr, "ECE", round(ece, 4))

        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        ax[0].plot(tt["threshold"], tt["precision"], label="precision"); ax[0].plot(tt["threshold"], tt["recall"], label="recall"); ax[0].plot(tt["threshold"], tt["f1"], label="F1")
        ax[0].axvline(chosen, color="k", ls="--", label=f"chosen {chosen:.2f}"); ax[0].set_xlabel("threshold"); ax[0].legend(); ax[0].set_title("Validation: precision/recall vs threshold")
        ax[1].plot(tt["threshold"], tt["expected_cost_per_ticket"]); ax[1].axvline(chosen, color="k", ls="--"); ax[1].set_title(f"Expected cost/ticket (FN={FN_COST:g}, FP={FP_COST:g})"); ax[1].set_xlabel("threshold")
        ax[2].plot(cal["mean_pred"], cal["observed"], marker="o"); ax[2].plot([0, 1], [0, 1], "k:"); ax[2].set_title(f"Calibration (validation, ECE={ece:.3f})"); ax[2].set_xlabel("mean predicted"); ax[2].set_ylabel("observed rate")
        P.save(fig, out / "fig_threshold_and_calibration.png")

        # ------------------------------------------------------------ 4. error analysis on VALIDATION (test stays untouched)
        v = va.assign(p=p_val); v["pred"] = (v["p"] >= chosen).astype(int); v["fn"] = ((v["pred"] == 0) & (v[TARGET] == 1)).astype(int); v["fp"] = ((v["pred"] == 1) & (v[TARGET] == 0)).astype(int)
        v["prev_bucket"] = pd.cut(v["previous_ticket_count"], [-1, 0, 1, 2, 4, 20], labels=["0", "1", "2", "3-4", "5+"])
        v["delay_bucket"] = pd.cut(v["delivery_delay_days"], [-1, 0, 2, 5, 10, 20, 500], labels=["0", "1-2", "3-5", "6-10", "11-20", ">20"])
        v["value_bucket"] = pd.qcut(v["order_value"], 5, labels=["q1", "q2", "q3", "q4", "q5"])
        err_rows = []
        for col in ["issue_title", "prev_bucket", "delay_bucket", "value_bucket", "support_channel", "product_category", "desc_missing"]:
            g = v.groupby(col, observed=True).agg(n=("p", "size"), pos_rate=(TARGET, "mean"), fn_rate_of_all=("fn", "mean"), fp_rate_of_all=("fp", "mean"))
            pos = v[v[TARGET] == 1].groupby(col, observed=True)["fn"].mean().rename("miss_rate_among_escalations")
            neg = v[v[TARGET] == 0].groupby(col, observed=True)["fp"].mean().rename("false_alarm_rate_among_non_escalations")
            g = g.join(pos).join(neg).reset_index().rename(columns={col: "value"}); g.insert(0, "slice", col)
            err_rows.append(g)
        err = pd.concat(err_rows); err.to_csv(out / "error_analysis_validation.csv", index=False)
        big = err[err["n"] >= 2000]
        worst_fn = big.sort_values("miss_rate_among_escalations", ascending=False).head(3)
        worst_fp = big.sort_values("false_alarm_rate_among_non_escalations", ascending=False).head(3)
        patterns = [f"Missed escalations concentrate where `{r['slice']}`={r['value']} (miss rate {r['miss_rate_among_escalations']:.0%} of true escalations, n={int(r['n']):,}, base rate {r['pos_rate']:.0%})" for _, r in worst_fn.iterrows()]
        patterns += [f"False alarms concentrate where `{r['slice']}`={r['value']} ({r['false_alarm_rate_among_non_escalations']:.0%} of non-escalations flagged, n={int(r['n']):,})" for _, r in worst_fp.iterrows()]
        write_json(out / "error_patterns.json", {"threshold": chosen, "patterns": patterns})
        print("\n".join(patterns))

        # ------------------------------------------------------------ 5. importance (permutation, validation sample)
        vs = va.sample(40_000, random_state=SEED)
        pi = permutation_importance(final, vs[FEATURE_COLUMNS], vs[TARGET], scoring="average_precision", n_repeats=3, random_state=SEED, n_jobs=1)
        imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "pr_auc_drop": pi.importances_mean, "std": pi.importances_std}).sort_values("pr_auc_drop", ascending=False)
        imp.to_csv(out / "permutation_importance_validation.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 5)); top = imp.head(14).iloc[::-1]
        ax.barh(top["feature"], top["pr_auc_drop"], xerr=top["std"]); ax.set_title("Permutation importance (PR-AUC drop, validation)")
        P.save(fig, out / "fig_permutation_importance.png")

        # ------------------------------------------------------------ 6. ablations + leakage demonstration (200k train rows, validation scoring)
        sub2 = C.cv_subsample(tr, 200_000, seed=7)
        hp = dict(max_iter=200, learning_rate=0.1, early_stopping=False)
        groups_defs = {
            "all_T0_features": dict(numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES),
            "minus_text_features": dict(numeric=[c for c in NUMERIC_FEATURES if not c.startswith("desc_")], categorical=CATEGORICAL_FEATURES),
            "minus_calendar": dict(numeric=[c for c in NUMERIC_FEATURES if c not in {"hour", "dayofweek", "month", "is_weekend", "is_night"}], categorical=CATEGORICAL_FEATURES),
            "minus_interactions": dict(numeric=[c for c in NUMERIC_FEATURES if c not in {"delay_x_prev_tickets", "value_x_delay", "log_order_value"}], categorical=CATEGORICAL_FEATURES),
            "minus_tenure": dict(numeric=[c for c in NUMERIC_FEATURES if c not in {"customer_tenure_months", "tenure_bucket_new"}], categorical=CATEGORICAL_FEATURES),
            "minus_issue_title": dict(numeric=NUMERIC_FEATURES, categorical=[c for c in CATEGORICAL_FEATURES if c != "issue_title"]),
        }
        abl = []
        for name, kw in groups_defs.items():
            mp = Pipeline([("pre", make_preprocessor(**kw)), ("m", hgb(**hp))]).fit(sub2[kw["numeric"] + kw["categorical"]], sub2[TARGET])
            pv = mp.predict_proba(va[kw["numeric"] + kw["categorical"]])[:, 1]
            abl.append({"variant": name, "val_pr_auc": average_precision_score(va[TARGET], pv), "val_roc_auc": roc_auc_score(va[TARGET], pv)})
        pd.DataFrame(abl).to_csv(out / "ablation_feature_groups.csv", index=False)

        sens = {"+age_band,+language,+accessibility (sensitive)": dict(extra_numeric=("accessibility_support_flag",), extra_categorical=("customer_age_band", "preferred_language"))}
        leak = {
            "+issue_category,+priority (T1 triage labels - not T0)": dict(extra_categorical=("issue_category", "priority")),
            "+customer_sentiment (post-contact review)": dict(extra_categorical=("customer_sentiment",)),
            "+csat_score (post-closure)": dict(extra_numeric=("csat_score",)),
            "+resolution_time_hours (post-outcome)": dict(extra_numeric=("resolution_time_hours",)),
            "+final_agent_disposition (is the target)": dict(extra_categorical=("final_agent_disposition",)),
        }
        base = abl[0]
        def run_extra(defs):
            res = []
            for name, kw in defs.items():
                en, ec = kw.get("extra_numeric", ()), kw.get("extra_categorical", ())
                mp = Pipeline([("pre", make_preprocessor(extra_numeric=en, extra_categorical=ec)), ("m", hgb(**hp))]).fit(sub2[FEATURE_COLUMNS + list(en) + list(ec)], sub2[TARGET])
                pv = mp.predict_proba(va[FEATURE_COLUMNS + list(en) + list(ec)])[:, 1]
                res.append({"variant": name, "val_pr_auc": average_precision_score(va[TARGET], pv), "val_roc_auc": roc_auc_score(va[TARGET], pv),
                            "delta_pr_auc_vs_T0": average_precision_score(va[TARGET], pv) - base["val_pr_auc"], "delta_roc_auc_vs_T0": roc_auc_score(va[TARGET], pv) - base["val_roc_auc"]})
            return pd.DataFrame(res)
        sens_tab = run_extra(sens); sens_tab.to_csv(out / "ablation_sensitive.csv", index=False)
        leak_tab = run_extra(leak)
        leak_tab = pd.concat([pd.DataFrame([{"variant": "T0-valid baseline (this project)", "val_pr_auc": base["val_pr_auc"], "val_roc_auc": base["val_roc_auc"], "delta_pr_auc_vs_T0": 0.0, "delta_roc_auc_vs_T0": 0.0}]), leak_tab])
        leak_tab.to_csv(out / "leakage_demonstration.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.barh(leak_tab["variant"][::-1], leak_tab["val_roc_auc"][::-1], color=[P.PALETTE[4] if "T0-valid" not in x else P.PALETTE[0] for x in leak_tab["variant"][::-1]])
        ax.set_xlim(0.5, 1.02); ax.set_title("Validation ROC-AUC if post-outcome/triage fields leak in (red) vs honest T0 model (blue)"); ax.title.set_fontsize(9)
        P.save(fig, out / "fig_leakage_demonstration.png")
        print(leak_tab.round(4).to_string(index=False)); print(sens_tab.round(4).to_string(index=False))

        # ------------------------------------------------------------ 7. FROZEN single test evaluation
        p_test = final.predict_proba(te[FEATURE_COLUMNS])[:, 1]
        test_scores = {"hgb_tuned (selected)": C.class_metrics(te[TARGET], p_test, chosen)}
        for n, m in fitted_baselines.items():
            thr_n = 0.5 if n != "dummy_prior" else chosen
            test_scores[n] = C.class_metrics(te[TARGET], m.predict_proba(te[FEATURE_COLUMNS])[:, 1], thr_n)
        test_tab = pd.DataFrame(test_scores).T
        test_tab.to_csv(out / "heldout_test_results.csv")
        m_test = C.class_metrics(te[TARGET], p_test, chosen)
        rng = np.random.default_rng(SEED); idx = np.arange(len(te)); bs = []
        for _ in range(200):
            b = rng.choice(idx, 50_000, replace=True); bs.append((average_precision_score(te[TARGET].values[b], p_test[b]), roc_auc_score(te[TARGET].values[b], p_test[b])))
        bs = np.array(bs)
        m_test["pr_auc_ci95_bootstrap50k"] = [float(np.quantile(bs[:, 0], .025)), float(np.quantile(bs[:, 0], .975))]
        m_test["roc_auc_ci95_bootstrap50k"] = [float(np.quantile(bs[:, 1], .025)), float(np.quantile(bs[:, 1], .975))]
        thresholds_pre_registered = {"pr_auc>=0.50": m_test["pr_auc"] >= 0.50, "roc_auc>=0.70": m_test["roc_auc"] >= 0.70, "recall>=0.60": m_test["recall"] >= 0.60}
        write_json(out / "heldout_test_selected_model.json", {"metrics": m_test, "pre_registered_success_criteria": thresholds_pre_registered, "test_period": [str(te["timestamp"].min()), str(te["timestamp"].max())],
                                                              "test_evaluations_performed": 1, "note": "Model, features, hyper-parameters and threshold were frozen on train/validation before this call."})
        print("TEST", {k: round(v, 4) if isinstance(v, float) else v for k, v in m_test.items() if k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "brier")}, thresholds_pre_registered)

        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        RocCurveDisplay.from_predictions(te[TARGET], p_test, ax=ax[0], name="HGB tuned"); ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_title("ROC (test)")
        PrecisionRecallDisplay.from_predictions(te[TARGET], p_test, ax=ax[1], name="HGB tuned"); ax[1].axhline(te[TARGET].mean(), color="k", ls=":", label="no-skill"); ax[1].legend(); ax[1].set_title("Precision-recall (test)")
        cm = np.array([[m_test["tn"], m_test["fp"]], [m_test["fn"], m_test["tp"]]])
        ax[2].imshow(cm, cmap="Blues"); ax[2].set_xticks([0, 1]); ax[2].set_yticks([0, 1]); ax[2].set_xticklabels(["pred 0", "pred 1"]); ax[2].set_yticklabels(["true 0", "true 1"]); ax[2].grid(False)
        for i in range(2):
            for j in range(2):
                ax[2].text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax[2].set_title(f"Confusion matrix (test, threshold {chosen:.2f})")
        P.save(fig, out / "fig_test_roc_pr_confusion.png")

        # persist predictions (git-ignored cache) for subgroup analysis; serving model + metadata (committed)
        for nm, part, pr in (("val", va, p_val), ("test", te, p_test)):
            part[["ticket_id", "customer_age_band", "preferred_language", "accessibility_support_flag", "customer_region", "support_channel", "product_category", "issue_title", TARGET]].assign(p=pr).to_parquet(C.PROCESSED_DIR / f"pred_cls_{nm}.parquet", index=False)
        serving = MODELS / "serving"; serving.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, serving / "escalation_hgb.joblib", compress=3)
        write_json(serving / "escalation_metadata.json", {
            "model_version": f"{MODEL_VERSION}/escalation-hgb", "task": "predict escalated at ticket creation (T0)", "features": FEATURE_COLUMNS, "decision_threshold": chosen,
            "threshold_rule": f"min expected cost FN={FN_COST} FP={FP_COST} on validation", "trained_on": "train period 2025-01..2025-08 (662,878 rows)", "hyperparameters": best_params,
            "sklearn_version": sklearn.__version__, "seed": SEED, "validation_metrics": C.class_metrics(va[TARGET], p_val, chosen), "test_metrics": m_test,
            "intended_use": "advisory routing support; human decides. Not for automated customer-facing decisions."})
        run.add(chosen_threshold=chosen, test_pr_auc=m_test["pr_auc"], test_roc_auc=m_test["roc_auc"])
    print("[OK] classification complete")


if __name__ == "__main__":
    main()
