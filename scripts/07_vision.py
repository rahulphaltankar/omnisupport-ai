"""Epic CV-01/02: basic CNN vs ResNet18 transfer learning on the 480 synthetic return images.

Same stratified 70/15/15 split, augmentation, optimiser family, metrics. Extras: data-size curve and corruption robustness.
Run:  python scripts/07_vision.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from omnisupport.config import MODEL_VERSION, MODELS, SEED, artifact_dir  # noqa: E402
from omnisupport.utils import plotting as P  # noqa: E402
from omnisupport.utils.runlog import RunRecord, sha256_file, write_json  # noqa: E402
from omnisupport.vision import core as V  # noqa: E402


def make_cnn():
    return V.Normalised(V.BasicCNN(), imagenet_norm=False)


def make_transfer():
    net = V.build_resnet18(pretrained=True)
    return V.Normalised(net, imagenet_norm=True, resize=128)


def fit_cnn(Xtr, ytr, Xva, yva, seed, epochs=40, tag="cnn", log_every=5):
    torch.manual_seed(seed)
    m = make_cnn()
    r = V.train_model(m, Xtr, ytr, Xva, yva, epochs=epochs, lr=3e-3, wd=1e-4, seed=seed, tag=tag, log_every=log_every)
    return m, r


def fit_transfer(Xtr, ytr, Xva, yva, seed, epochs=12, tag="resnet18", log_every=2):
    """Stage-wise fine-tuning: ImageNet backbone with layer4 + new head trainable (lower LR for pretrained layers)."""
    torch.manual_seed(seed)
    m = make_transfer(); net = m.net
    for p in net.parameters():
        p.requires_grad = False
    for p in list(net.layer4.parameters()) + list(net.fc.parameters()):
        p.requires_grad = True
    groups = [{"params": net.layer4.parameters(), "lr": 5e-4}, {"params": net.fc.parameters(), "lr": 3e-3}]
    r = V.train_model(m, Xtr, ytr, Xva, yva, epochs=epochs, lr=3e-3, wd=1e-4, seed=seed, param_groups=groups, tag=tag, log_every=log_every)
    return m, r


def main() -> None:
    out = artifact_dir("vision")
    df = V.load_labels(); sp = V.stratified_split(df)
    pd.concat([s.assign(split=n) for n, s in sp.items()])[["image_id", "label", "split"]].to_csv(out / "split_assignment.csv", index=False)
    pd.DataFrame({n: s["label"].value_counts() for n, s in sp.items()}).fillna(0).astype(int).to_csv(out / "split_class_counts.csv")
    X = {n: V.load_tensor(s["relative_path"]) for n, s in sp.items()}; y = {n: torch.tensor(s["class_idx"].values) for n, s in sp.items()}
    print({n: tuple(v.shape) for n, v in X.items()})
    cfg = {"classes": V.CLASSES, "split": "stratified 70/15/15 (336/72/72), seed 42", "augmentation": "affine rot+-12deg/shift 8%/scale 0.9-1.1, hflip, brightness/contrast +-20%, gaussian noise 0.01 (train only)",
           "cnn": "BasicCNN 4 blocks, AdamW lr 3e-3 OneCycle, 40 epochs", "transfer": "ResNet18 ImageNet, layer4+head trainable, AdamW (5e-4 / 3e-3), 12 epochs, input resized to 128", "seeds_for_variance": [42, 43, 44]}
    with RunRecord(out, "vision", config=cfg, split_method="stratified 70/15/15 by class, fixed seed; identical for both models", model_version=f"{MODEL_VERSION}/vision",
                   limitations=["480 synthetic, near-templated images: results do NOT transfer to real photographs.", "Test set has only 72 images (18/class): one error = 1.4 points of accuracy; use seed spread, not single numbers."]) as run:
        results, models = {}, {}
        for name, fit in (("basic_cnn", fit_cnn), ("resnet18_transfer", fit_transfer)):
            seed_rows = []
            for seed in (42, 43, 44):
                m, r = fit(X["train"], y["train"], X["val"], y["val"], seed, tag=f"{name}-s{seed}")
                ev = V.evaluate(m, X["test"], y["test"].numpy())
                seed_rows.append({"seed": seed, "test_accuracy": ev["accuracy"], "test_macro_f1": ev["macro_f1"], "val_macro_f1": r["best_val_macro_f1"], "train_seconds": r["train_seconds"], "ms_per_image": ev["ms_per_image"]})
                if seed == 42:
                    results[name] = {"ev": ev, "hist": r["history"], "train_seconds": r["train_seconds"]}; models[name] = m
                    r["history"].to_csv(out / f"history_{name}.csv", index=False)
            pd.DataFrame(seed_rows).to_csv(out / f"seed_runs_{name}.csv", index=False)
            results[name]["seeds"] = pd.DataFrame(seed_rows)
            print(name, pd.DataFrame(seed_rows).round(3).to_string(index=False))

        # ---- summary + per-class metrics + confusion matrices
        summ = []
        for n, r in results.items():
            s = r["seeds"]
            net = models[n]
            summ.append({"model": n, "params_total": V.count_params(net), "params_trainable": V.count_params(net, True), "test_acc_seed42": r["ev"]["accuracy"], "test_macro_f1_seed42": r["ev"]["macro_f1"],
                         "test_macro_f1_mean_3seeds": s["test_macro_f1"].mean(), "test_macro_f1_std_3seeds": s["test_macro_f1"].std(), "train_seconds_mean": s["train_seconds"].mean(), "ms_per_image": r["ev"]["ms_per_image"]})
        pd.DataFrame(summ).to_csv(out / "model_comparison.csv", index=False)
        pc = [{"model": n, "class": c, **{k: v for k, v in r["ev"]["per_class"][c].items()}} for n, r in results.items() for c in V.CLASSES]
        pd.DataFrame(pc).to_csv(out / "per_class_metrics.csv", index=False)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a, (n, r) in zip(ax, results.items()):
            cm = np.array(r["ev"]["confusion"]); a.imshow(cm, cmap="Blues"); a.grid(False)
            a.set_xticks(range(4)); a.set_yticks(range(4)); a.set_xticklabels(V.CLASSES, rotation=30, ha="right"); a.set_yticklabels(V.CLASSES)
            for i in range(4):
                for j in range(4):
                    a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
            a.set_title(f"{n}: test confusion (acc {r['ev']['accuracy']:.3f})")
        P.save(fig, out / "fig_confusion_matrices.png")

        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for a, (n, r) in zip(ax, results.items()):
            h = r["hist"]; a.plot(h["epoch"], h["train_loss"], label="train loss"); a.plot(h["epoch"], h["val_loss"], label="val loss"); a2 = a.twinx(); a2.plot(h["epoch"], h["val_macro_f1"], "g--", label="val macro-F1"); a2.set_ylim(0, 1.02); a2.grid(False)
            a.set_title(f"{n} learning curves"); a.set_xlabel("epoch"); a.legend(loc="center right")
        P.save(fig, out / "fig_learning_curves.png")

        # ---- data-size curve (does pretraining help when labels are scarce?)
        rows = []
        for frac in (0.1, 0.25, 0.5, 1.0):
            n = max(int(len(X["train"]) * frac), 8)
            idx = torch.tensor(sp["train"].groupby("class_idx", group_keys=False).apply(lambda d: d.sample(frac=frac, random_state=SEED) if frac < 1 else d).index.values)
            for name, fit in (("basic_cnn", fit_cnn), ("resnet18_transfer", fit_transfer)):
                m, _ = fit(X["train"][idx], y["train"][idx], X["val"], y["val"], SEED, tag=f"{name}@{frac}", log_every=999)
                ev = V.evaluate(m, X["test"], y["test"].numpy()); rows.append({"train_fraction": frac, "n_train": len(idx), "model": name, "test_macro_f1": ev["macro_f1"], "test_accuracy": ev["accuracy"]})
        curve = pd.DataFrame(rows); curve.to_csv(out / "data_size_curve.csv", index=False); print(curve.round(3).to_string(index=False))

        # ---- robustness: corruptions the model never saw
        rob = []
        corruptions = [("none", 0), ("noise", 0.05), ("noise", 0.15), ("blur", 1), ("blur", 2), ("rotate", 20), ("rotate", 40), ("darken", 0.5), ("occlude", 0.35)]
        for kind, lvl in corruptions:
            Xc = X["test"] if kind == "none" else V.perturb(X["test"], kind, lvl)
            for n, m in models.items():
                ev = V.evaluate(m, Xc, y["test"].numpy()); rob.append({"corruption": f"{kind}({lvl})", "model": n, "accuracy": ev["accuracy"], "macro_f1": ev["macro_f1"]})
        rob = pd.DataFrame(rob); rob.to_csv(out / "robustness.csv", index=False)
        piv = rob.pivot(index="corruption", columns="model", values="macro_f1").loc[[f"{k}({l})" for k, l in corruptions]]
        fig, ax = plt.subplots(figsize=(8, 3.6)); piv.plot.bar(ax=ax); ax.set_ylabel("test macro-F1"); ax.set_title("Robustness to unseen corruptions (same trained models)"); ax.set_ylim(0, 1.05); plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        P.save(fig, out / "fig_robustness.png")

        # ---- error gallery: test errors (clean) or, if none, the lowest-confidence/worst-corruption errors
        fig_rows = []
        for n, m in models.items():
            for kind, lvl in [("none", 0), ("rotate", 40), ("noise", 0.15), ("occlude", 0.35)]:
                Xc = X["test"] if kind == "none" else V.perturb(X["test"], kind, lvl); ev = V.evaluate(m, Xc, y["test"].numpy())
                for i in np.where(ev["pred"] != y["test"].numpy())[0]:
                    fig_rows.append((n, f"{kind}({lvl})", i, int(y["test"][i]), int(ev["pred"][i]), float(ev["prob"][i].max())))
        err = pd.DataFrame(fig_rows, columns=["model", "condition", "test_idx", "true", "pred", "confidence"])
        err["true"] = err["true"].map(dict(enumerate(V.CLASSES))); err["pred"] = err["pred"].map(dict(enumerate(V.CLASSES)))
        err["image_id"] = err["test_idx"].map(sp["test"]["image_id"]); err.to_csv(out / "error_analysis.csv", index=False)
        pairs = err.groupby(["model", "condition", "true", "pred"]).size().reset_index(name="count").sort_values("count", ascending=False); pairs.to_csv(out / "error_confusion_pairs.csv", index=False)
        show = err[err["model"] == "basic_cnn"].head(8) if len(err) else err
        shown = err.drop_duplicates(["model", "image_id"]).head(8)
        if len(shown):
            fig, ax = plt.subplots(2, 4, figsize=(11, 6))
            for a, (_, r) in zip(ax.ravel(), shown.iterrows()):
                kind, lvl = r["condition"].rstrip(")").split("("); Xc = X["test"] if kind == "none" else V.perturb(X["test"], kind, float(lvl))
                a.imshow(Xc[int(r["test_idx"])].permute(1, 2, 0).numpy()); a.axis("off"); a.set_title(f"{r['model'][:6]} {r['condition']}\ntrue {r['true']} / pred {r['pred']} ({r['confidence']:.2f})", fontsize=7)
            for a in ax.ravel()[len(shown):]: a.axis("off")
            P.save(fig, out / "fig_error_gallery.png")

        # ---- save: small CNN checkpoint (committed), ResNet weights (git-ignored) + metadata
        vm = MODELS / "vision"; vm.mkdir(parents=True, exist_ok=True)
        torch.save(models["basic_cnn"].state_dict(), vm / "basic_cnn.pt"); torch.save(models["resnet18_transfer"].state_dict(), vm / "resnet18_transfer.pt")
        write_json(vm / "metadata.json", {"classes": V.CLASSES, "model_version": f"{MODEL_VERSION}/vision", "seed": SEED, "basic_cnn_sha256_12": sha256_file(vm / "basic_cnn.pt"),
                                          "resnet18_sha256_12": sha256_file(vm / "resnet18_transfer.pt"), "summary": summ, "input": "RGB 96x96 in [0,1]", "torch": torch.__version__})
        run.add(summary=summ)
    print("[OK] vision complete")


if __name__ == "__main__":
    main()
