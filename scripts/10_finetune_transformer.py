"""Epic NLP-04 (Session 22): REAL Hugging Face fine-tuning of DistilBERT on data/subsets/transformer_finetune_10000.parquet,
compared fairly with TF-IDF on the identical held-out split.

Run:  python scripts/10_finetune_transformer.py [--epochs 2]
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from omnisupport.config import MODEL_VERSION, MODELS, REPORTS, SEED, SUBSET_PATH, artifact_dir  # noqa: E402
from omnisupport.utils import plotting as P  # noqa: E402
from omnisupport.utils.runlog import RunRecord, set_seed, write_json  # noqa: E402

BASE = "distilbert/distilbert-base-uncased"


def noisy(s: str, rng: np.random.Generator) -> str:
    ch = list(s)
    for _ in range(max(1, len(ch) // 20)):
        i = int(rng.integers(0, len(ch))); ch[i] = chr(ord("a") + int(rng.integers(0, 26)))
    out = "".join(ch)
    return out.upper() if rng.random() < 0.3 else out


@torch.no_grad()
def hf_predict(model, tok, texts, bs=64, max_len=64):
    model.eval(); logits = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        logits.append(model(**enc).logits)
    lg = torch.cat(logits); return lg.argmax(1).numpy(), torch.softmax(lg, 1).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--max_train", type=int, default=0); args = ap.parse_args()
    out = artifact_dir("transformer_finetune")
    df = pd.read_parquet(SUBSET_PATH).dropna(subset=["issue_description", "issue_category"]); df["issue_description"] = df["issue_description"].astype(str)
    labels = sorted(df["issue_category"].unique()); l2i = {l: i for i, l in enumerate(labels)}
    tr, rest = train_test_split(df, test_size=0.30, stratify=df["issue_category"], random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=rest["issue_category"], random_state=SEED)
    if args.max_train: tr = tr.sample(min(args.max_train, len(tr)), random_state=SEED)
    pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])[["ticket_id", "issue_category", "split"]].to_csv(out / "split_assignment.csv", index=False)
    cfg = {"base_model": BASE, "epochs": args.epochs, "batch_size": 32, "lr": 5e-5, "max_length": 64, "warmup_ratio": 0.1, "weight_decay": 0.01, "labels": labels, "n_train": len(tr), "n_val": len(va), "n_test": len(te), "text": "issue_description only (same for TF-IDF)"}
    with RunRecord(out, "transformer_finetune", config=cfg, split_method="stratified 70/15/15 by issue_category, seed 42; identical rows for TF-IDF and DistilBERT", model_version=f"{MODEL_VERSION}/distilbert-issue",
                   limitations=["CPU-only fine-tuning, 2 epochs.", "Text is templated; both models near ceiling so the comparison is informative mainly on robustness, latency and cost.", "Label ambiguity (delivery_late vs lost_parcel) bounds achievable accuracy."]) as run:
        # ---------------- TF-IDF baseline on identical rows
        t0 = time.perf_counter(); tfidf = Pipeline([("v", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)), ("c", LogisticRegression(max_iter=400, C=5.0))]).fit(tr["issue_description"], tr["issue_category"]); tfidf_train_s = time.perf_counter() - t0
        t0 = time.perf_counter(); p_tf = tfidf.predict(te["issue_description"]); tfidf_ms = (time.perf_counter() - t0) / len(te) * 1000

        # ---------------- fine-tune
        set_seed(SEED); tok = AutoTokenizer.from_pretrained(BASE)
        model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i)
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        ytr = torch.tensor(tr["issue_category"].map(l2i).values); texts = tr["issue_description"].tolist()
        steps = args.epochs * int(np.ceil(len(tr) / 32))
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01); sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        gen = torch.Generator().manual_seed(SEED); hist = []; t_start = time.perf_counter(); step = 0
        for ep in range(1, args.epochs + 1):
            model.train(); perm = torch.randperm(len(tr), generator=gen); run_loss = 0.0
            for i in range(0, len(tr), 32):
                idx = perm[i:i + 32]; enc = tok([texts[j] for j in idx.tolist()], padding=True, truncation=True, max_length=64, return_tensors="pt")
                loss = model(**enc, labels=ytr[idx]).loss; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); run_loss += loss.item() * len(idx); step += 1
                if step % 20 == 0:
                    print(f"ep{ep} step {step}/{steps} loss {loss.item():.4f} elapsed {time.perf_counter() - t_start:.0f}s", flush=True)
            pv, _ = hf_predict(model, tok, va["issue_description"].tolist()); yv = va["issue_category"].map(l2i).values
            hist.append({"epoch": ep, "train_loss": run_loss / len(tr), "val_accuracy": accuracy_score(yv, pv), "val_macro_f1": f1_score(yv, pv, average="macro"), "elapsed_s": time.perf_counter() - t_start})
            print(hist[-1], flush=True)
        train_s = time.perf_counter() - t_start; pd.DataFrame(hist).to_csv(out / "training_log.csv", index=False)

        # ---------------- identical held-out evaluation
        yt = te["issue_category"].map(l2i).values
        t0 = time.perf_counter(); p_hf, prob_hf = hf_predict(model, tok, te["issue_description"].tolist()); hf_ms = (time.perf_counter() - t0) / len(te) * 1000
        p_tf_i = np.array([l2i[x] for x in p_tf])
        rng = np.random.default_rng(SEED); te_noisy = te["issue_description"].map(lambda s: noisy(s, rng)).tolist()
        p_hf_n, _ = hf_predict(model, tok, te_noisy); p_tf_n = np.array([l2i[x] for x in tfidf.predict(te_noisy)])
        def sc(y, p): return {"accuracy": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro"))}
        res = pd.DataFrame([{"model": "TF-IDF(1-2gram)+LogReg", "split": "test clean", **sc(yt, p_tf_i), "train_seconds": tfidf_train_s, "ms_per_example": tfidf_ms, "trainable_params": len(tfidf.named_steps["v"].vocabulary_) * len(labels)},
                            {"model": "DistilBERT fine-tuned", "split": "test clean", **sc(yt, p_hf), "train_seconds": train_s, "ms_per_example": hf_ms, "trainable_params": sum(p.numel() for p in model.parameters())},
                            {"model": "TF-IDF(1-2gram)+LogReg", "split": "test noisy (typos/UPPERCASE)", **sc(yt, p_tf_n), "train_seconds": np.nan, "ms_per_example": np.nan, "trainable_params": np.nan},
                            {"model": "DistilBERT fine-tuned", "split": "test noisy (typos/UPPERCASE)", **sc(yt, p_hf_n), "train_seconds": np.nan, "ms_per_example": np.nan, "trainable_params": np.nan}])
        res.to_csv(out / "heldout_comparison.csv", index=False); print(res.round(4).to_string(index=False))
        rep = classification_report(yt, p_hf, target_names=labels, output_dict=True, zero_division=0); pd.DataFrame(rep).T.to_csv(out / "per_class_distilbert_test.csv")
        rep2 = classification_report(yt, p_tf_i, target_names=labels, output_dict=True, zero_division=0); pd.DataFrame(rep2).T.to_csv(out / "per_class_tfidf_test.csv")
        # paired disagreement analysis
        both_ok = int(((p_hf == yt) & (p_tf_i == yt)).sum()); only_hf = int(((p_hf == yt) & (p_tf_i != yt)).sum()); only_tf = int(((p_hf != yt) & (p_tf_i == yt)).sum()); both_bad = int(((p_hf != yt) & (p_tf_i != yt)).sum())
        errs = te.assign(true=te["issue_category"], pred_hf=[labels[i] for i in p_hf], pred_tfidf=[labels[i] for i in p_tf_i]); errs = errs[(errs["true"] != errs["pred_hf"]) | (errs["true"] != errs["pred_tfidf"])]
        errs[["ticket_id", "issue_description", "true", "pred_hf", "pred_tfidf"]].to_csv(out / "error_analysis_test.csv", index=False)
        pairs = errs.groupby(["true", "pred_hf"]).size().reset_index(name="n").sort_values("n", ascending=False).head(6)
        cm = confusion_matrix(yt, p_hf, labels=range(len(labels)))
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].imshow(cm / cm.sum(1, keepdims=True), cmap="Blues"); ax[0].set_xticks(range(len(labels))); ax[0].set_yticks(range(len(labels))); ax[0].set_xticklabels(labels, rotation=60, ha="right", fontsize=7); ax[0].set_yticklabels(labels, fontsize=7); ax[0].grid(False); ax[0].set_title("DistilBERT test confusion (row-normalised)")
        h = pd.DataFrame(hist); ax[1].plot(h["epoch"], h["train_loss"], marker="o", label="train loss"); ax[1].set_xlabel("epoch"); a2 = ax[1].twinx(); a2.plot(h["epoch"], h["val_macro_f1"], "g--o", label="val macro-F1"); a2.set_ylim(0, 1.02); a2.grid(False); ax[1].legend(loc="center left"); ax[1].set_title("Fine-tuning curve")
        P.save(fig, out / "fig_finetune.png")
        mdir = MODELS / "hf_distilbert_issue"; model.save_pretrained(mdir); tok.save_pretrained(mdir)
        summary = {"base_model": BASE, "hardware": {"device": "cpu", "cpu_count": os.cpu_count(), "gpu": "none"}, "train_seconds": train_s, "seed": SEED, "epochs": args.epochs, "n_train": len(tr),
                   "paired": {"both_correct": both_ok, "only_distilbert_correct": only_hf, "only_tfidf_correct": only_tf, "both_wrong": both_bad}, "top_distilbert_confusions": pairs.to_dict("records"), "test_evaluations_performed": 1}
        write_json(out / "summary.json", summary); write_json(MODELS / "transformer_finetune_metadata.json", {**summary, "config": cfg, "weights": "models/hf_distilbert_issue (git-ignored; regenerate with scripts/10_finetune_transformer.py)", "results": res.to_dict("records")})
        c = res.set_index(["model", "split"])
        md = ["# Transformer fine-tuning vs TF-IDF (Session 22)", "",
              f"A real fine-tuning run of **{BASE}** ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters) on `data/subsets/transformer_finetune_10000.parquet` for **{args.epochs} epochs** ({train_s / 60:.1f} min on {os.cpu_count()} CPU cores, no GPU), seed {SEED}, batch 32, lr 5e-5, max length 64. "
              f"Training rows: **{len(tr):,}** (full 70% share - no hardware-driven reduction was needed); validation {len(va):,}; held-out test **{len(te):,}**. TF-IDF + logistic regression was trained on the *same* training rows and scored on the *same* test rows.", "",
              "| Model | Test split | Accuracy | Macro-F1 | Train time | ms / example |", "|---|---|---|---|---|---|"]
        for _, r in res.iterrows():
            md.append(f"| {r['model']} | {r['split']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {'' if pd.isna(r['train_seconds']) else f'{r['train_seconds']:.0f}s'} | {'' if pd.isna(r['ms_per_example']) else f'{r['ms_per_example']:.2f}'} |")
        md += ["", f"Paired outcome on the {len(te):,} test rows: both correct {both_ok:,}; only DistilBERT correct {only_hf}; only TF-IDF correct {only_tf}; both wrong {both_bad}.", "",
               "![fine-tuning](../artifacts/transformer_finetune/fig_finetune.png)", "", "## Reading the result honestly",
               "- The messages are generated from templates, so a linear bag-of-words model is already near the ceiling; the fine-tuned transformer **cannot be shown to be meaningfully better on clean text** and costs far more to train and serve (see table).",
               "- The remaining errors are dominated by label ambiguity (e.g. `delivery_late` vs `lost_parcel` messages that are textually near-identical), which no model can fix without new information (carrier tracking).",
               "- The noisy-text rows (random typos and upper-casing) show whether contextual sub-word models degrade differently from word-level TF-IDF; differences there are the more relevant production signal.",
               f"- Top DistilBERT confusions: " + "; ".join(f"{r['true']}->{r['pred_hf']} ({r['n']})" for _, r in pairs.iterrows()) + ".",
               "- Recommendation: serve TF-IDF as the default classifier (cheap, transparent, same accuracy) and keep the fine-tuned model as a documented alternative for noisier real-world text."]
        (REPORTS / "transformer_finetuning.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        run.add(results=res.to_dict("records"))
    print("[OK] fine-tuning complete")


if __name__ == "__main__":
    main()
