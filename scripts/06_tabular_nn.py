"""Epic DL-01: tabular neural network for escalation (same T0 features/splits as the classical models).

Five optimiser/regularisation settings, learning curves, best-validation checkpoint, one frozen test evaluation,
reproducibility check and an explicit classical-vs-NN conclusion.
Run:  python scripts/06_tabular_nn.py
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
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from omnisupport.config import MODEL_VERSION, MODELS, SEED, artifact_dir  # noqa: E402
from omnisupport.data.features import FEATURE_COLUMNS, make_preprocessor  # noqa: E402
from omnisupport.data.leakage import assert_features_valid  # noqa: E402
from omnisupport.ml import common as C  # noqa: E402
from omnisupport.utils import plotting as P  # noqa: E402
from omnisupport.utils.runlog import RunRecord, read_json, set_seed, write_json  # noqa: E402

TARGET = "escalated"
EPOCHS = 25


class TabularMLP(nn.Module):
    def __init__(self, d_in: int, hidden=(128, 64), dropout: float = 0.0):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


CONFIGS = {
    "A_adam_noreg": dict(hidden=(128, 64), dropout=0.0, opt="adam", lr=1e-3, wd=0.0, sched=None),
    "B_adam_dropout_wd": dict(hidden=(128, 64), dropout=0.3, opt="adam", lr=1e-3, wd=1e-4, sched=None),
    "C_sgd_momentum": dict(hidden=(128, 64), dropout=0.3, opt="sgd", lr=0.05, wd=1e-4, sched=None),
    "D_adamw_plateau": dict(hidden=(128, 64), dropout=0.3, opt="adamw", lr=3e-3, wd=1e-2, sched="plateau"),
    "E_big_noreg_overfit_probe": dict(hidden=(512, 512, 256), dropout=0.0, opt="adam", lr=2e-3, wd=0.0, sched=None),
}


def make_loader(X, y, bs, shuffle, seed):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=bs, shuffle=shuffle, generator=g if shuffle else None)


@torch.no_grad()
def predict_proba(model, X, bs=8192):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + bs]))).numpy())
    return np.concatenate(out)


def train(cfg: dict, Xtr, ytr, Xva, yva, seed=SEED, epochs=EPOCHS, bs=2048, tag="", verbose=True):
    set_seed(seed); torch.manual_seed(seed)
    model = TabularMLP(Xtr.shape[1], cfg["hidden"], cfg["dropout"])
    opt = {"adam": lambda: torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]),
           "adamw": lambda: torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]),
           "sgd": lambda: torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["wd"])}[cfg["opt"]]()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2) if cfg["sched"] == "plateau" else None
    loss_fn = nn.BCEWithLogitsLoss(); loader = make_loader(Xtr, ytr, bs, True, seed)
    best_loss, best_state, best_ep, hist = 1e9, None, 0, []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0
        for xb, yb in loader:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step(); tl += loss.item() * len(xb)
        pv = predict_proba(model, Xva)
        vloss = float(-np.mean(yva * np.log(np.clip(pv, 1e-7, 1)) + (1 - yva) * np.log(np.clip(1 - pv, 1e-7, 1))))
        ptr = predict_proba(model, Xtr[:100_000]); trl_eval = float(-np.mean(ytr[:100_000] * np.log(np.clip(ptr, 1e-7, 1)) + (1 - ytr[:100_000]) * np.log(np.clip(1 - ptr, 1e-7, 1))))
        if sched: sched.step(vloss)
        hist.append({"epoch": ep, "train_loss_running": tl / len(Xtr), "train_loss_eval": trl_eval, "val_loss": vloss, "val_pr_auc": average_precision_score(yva, pv), "val_roc_auc": roc_auc_score(yva, pv), "lr": opt.param_groups[0]["lr"]})
        if vloss < best_loss:
            best_loss, best_ep, best_state = vloss, ep, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"[{tag}] ep{ep:02d} train {trl_eval:.4f} val {vloss:.4f} pr-auc {hist[-1]['val_pr_auc']:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), best_ep, time.perf_counter() - t0


def main() -> None:
    out = artifact_dir("tabular_nn")
    frame = C.modelling_frame(); parts = C.split_frame(frame); tr, va, te = parts["train"], parts["val"], parts["test"]
    assert_features_valid(FEATURE_COLUMNS)
    with RunRecord(out, "tabular_nn", config={"configs": {k: {**v, "hidden": list(v["hidden"])} for k, v in CONFIGS.items()}, "epochs": EPOCHS, "batch_size": 2048, "loss": "BCEWithLogits", "features": FEATURE_COLUMNS},
                   split_method="temporal train/val/test identical to classical models; preprocessor fitted on train only", model_version=f"{MODEL_VERSION}/tabular-mlp",
                   limitations=["CPU-only; 25 epochs.", "Single seed per configuration (plus one reproducibility rerun)."]) as run:
        pre = make_preprocessor(scale=True).fit(tr[FEATURE_COLUMNS])  # fit on TRAIN only
        Xtr, Xva, Xte = (pre.transform(d[FEATURE_COLUMNS]).astype(np.float32) for d in (tr, va, te))
        ytr, yva, yte = (d[TARGET].values.astype(np.float32) for d in (tr, va, te))
        print("input dim", Xtr.shape[1])
        rows, hists, states = [], {}, {}
        for name, cfg in CONFIGS.items():
            model, h, best_ep, secs = train(cfg, Xtr, ytr, Xva, yva, tag=name)
            h.to_csv(out / f"history_{name}.csv", index=False); hists[name] = h; states[name] = model
            bv = h.loc[h["epoch"] == best_ep].iloc[0]; last = h.iloc[-1]
            rows.append({"config": name, "optimizer": cfg["opt"], "hidden": str(cfg["hidden"]), "dropout": cfg["dropout"], "weight_decay": cfg["wd"], "lr": cfg["lr"], "best_epoch": best_ep,
                         "best_val_loss": bv["val_loss"], "best_val_pr_auc": bv["val_pr_auc"], "best_val_roc_auc": bv["val_roc_auc"], "final_train_loss": last["train_loss_eval"], "final_val_loss": last["val_loss"],
                         "overfit_gap_final(val-train)": last["val_loss"] - last["train_loss_eval"], "params": sum(p.numel() for p in model.parameters()), "seconds": secs})
        comp = pd.DataFrame(rows); comp.to_csv(out / "config_comparison.csv", index=False); print(comp.round(4).to_string(index=False))

        fig, ax = plt.subplots(1, len(CONFIGS), figsize=(4 * len(CONFIGS), 3.4), sharey=True)
        for a, (n, h) in zip(ax, hists.items()):
            a.plot(h["epoch"], h["train_loss_eval"], label="train"); a.plot(h["epoch"], h["val_loss"], label="validation"); a.axvline(int(comp.set_index("config").loc[n, "best_epoch"]), color="k", ls=":", label="best epoch")
            a.set_title(n, fontsize=8); a.set_xlabel("epoch")
        ax[0].set_ylabel("log-loss"); ax[0].legend(fontsize=7); P.save(fig, out / "fig_learning_curves.png")

        best_name = comp.sort_values("best_val_pr_auc", ascending=False).iloc[0]["config"]; best_model = states[best_name]
        # reproducibility: retrain best config with the same seed for a few epochs and compare the loss curve
        _, h1, _, _ = train(CONFIGS[best_name], Xtr, ytr, Xva, yva, epochs=3, tag="repro-1", verbose=False)
        _, h2, _, _ = train(CONFIGS[best_name], Xtr, ytr, Xva, yva, epochs=3, tag="repro-2", verbose=False)
        repro = float(np.abs(h1["val_loss"].values - h2["val_loss"].values).max())

        # threshold on validation (same cost rule as classical), frozen test evaluation once
        pv = predict_proba(best_model, Xva); tt = C.threshold_table(yva, pv, 3.0, 1.0); thr = C.pick_thresholds(tt)["cost_optimal"]
        pt = predict_proba(best_model, Xte); m_nn = C.class_metrics(yte, pt, thr)
        classical = read_json(Path(out).parent / "ml_classification" / "heldout_test_selected_model.json")["metrics"] if (Path(out).parent / "ml_classification" / "heldout_test_selected_model.json").exists() else None
        write_json(out / "heldout_test_selected_model.json", {"selected_config": best_name, "threshold_from_validation": thr, "metrics": m_nn, "classical_selected_test_metrics": classical, "reproducibility_max_abs_val_loss_diff_3_epochs": repro, "test_evaluations_performed": 1})
        mdir = MODELS / "tabular_nn"; mdir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_model.state_dict(), "d_in": Xtr.shape[1], "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CONFIGS[best_name].items()}}, mdir / "best.pt")
        joblib.dump(pre, mdir / "preprocessor.joblib", compress=3)
        write_json(mdir / "metadata.json", {"model_version": f"{MODEL_VERSION}/tabular-mlp", "config_name": best_name, "seed": SEED, "feature_version": "T0-features-v1 (" + str(len(FEATURE_COLUMNS)) + " columns)", "input_dim": int(Xtr.shape[1]),
                                            "best_epoch": int(comp.set_index("config").loc[best_name, "best_epoch"]), "val_pr_auc": float(comp.set_index("config").loc[best_name, "best_val_pr_auc"]), "test_metrics": m_nn, "torch": torch.__version__})
        # ---- conclusion (computed)
        e = comp.set_index("config"); a, b, big = e.loc["A_adam_noreg"], e.loc["B_adam_dropout_wd"], e.loc["E_big_noreg_overfit_probe"]
        concl = [f"- Best configuration: **{best_name}** (best epoch {int(e.loc[best_name, 'best_epoch'])}, validation PR-AUC {e.loc[best_name, 'best_val_pr_auc']:.4f}).",
                 f"- Overfitting evidence: the unregularised large network E reaches a train loss of {big['final_train_loss']:.4f} but validation loss {big['final_val_loss']:.4f} (gap {big['overfit_gap_final(val-train)']:+.4f}; best epoch {int(big['best_epoch'])} of {EPOCHS}). The small unregularised net A has gap {a['overfit_gap_final(val-train)']:+.4f}; dropout + weight decay (B) gives {b['overfit_gap_final(val-train)']:+.4f}.",
                 f"- Optimiser comparison (validation PR-AUC): Adam {b['best_val_pr_auc']:.4f}, SGD+momentum {e.loc['C_sgd_momentum', 'best_val_pr_auc']:.4f}, AdamW+plateau {e.loc['D_adamw_plateau', 'best_val_pr_auc']:.4f}.",
                 f"- Held-out test (threshold {thr:.2f} from validation): NN PR-AUC {m_nn['pr_auc']:.4f}, ROC-AUC {m_nn['roc_auc']:.4f}" + (f" vs classical selected model PR-AUC {classical['pr_auc']:.4f}, ROC-AUC {classical['roc_auc']:.4f}." if classical else "."),
                 f"- Reproducibility: re-running 3 epochs with the same seed changes validation loss by at most {repro:.2e}."]
        (out / "conclusion.md").write_text("# Tabular NN conclusion (generated)\n\n" + "\n".join(concl) + "\n", encoding="utf-8")
        print("\n".join(concl)); run.add(best=best_name, test_pr_auc=m_nn["pr_auc"])
    print("[OK] tabular NN complete")


if __name__ == "__main__":
    main()
