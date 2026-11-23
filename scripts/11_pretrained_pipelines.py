"""Epic NLP-03: two pretrained Hugging Face pipelines on support text (no training).

1. sentiment-analysis  (distilbert-base-uncased-finetuned-sst-2-english)
2. zero-shot-classification (typeform/distilbert-base-uncased-mnli) over the 10 issue categories
Run:  python scripts/11_pretrained_pipelines.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from transformers import pipeline  # noqa: E402

from omnisupport.config import REPORTS, SEED, artifact_dir  # noqa: E402
from omnisupport.ml import common as C  # noqa: E402
from omnisupport.utils.runlog import RunRecord, write_json  # noqa: E402

SENT = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
ZS = "typeform/distilbert-base-uncased-mnli"
LABEL_TEXT = {"delivery_late": "late delivery", "damaged_item": "damaged item", "refund_status": "refund status", "wrong_item": "wrong item received", "return_request": "return request",
              "payment_issue": "payment problem", "account_access": "account login problem", "product_question": "product question", "warranty": "warranty claim", "lost_parcel": "lost parcel"}


def main() -> None:
    out = artifact_dir("pretrained_pipelines")
    df = C.load_clean(); df = df[df["issue_description"].notna()]
    sample = df.sample(frac=1, random_state=SEED).groupby("issue_category").head(10).reset_index(drop=True)  # 100 rows, balanced
    with RunRecord(out, "pretrained_pipelines", config={"sentiment_model": SENT, "zero_shot_model": ZS, "n": len(sample)}, split_method="n/a (inference only on 100 balanced rows)", model_version=f"{SENT} | {ZS}",
                   limitations=["SST-2 sentiment is trained on movie reviews; support text is out-of-domain.", "Zero-shot NLI needs one forward pass per label (10x cost)."]) as run:
        sa = pipeline("sentiment-analysis", model=SENT, device=-1)
        t0 = time.perf_counter(); s = sa(sample["issue_description"].tolist(), batch_size=16, truncation=True); sa_ms = (time.perf_counter() - t0) / len(sample) * 1000
        sample["sentiment_label"] = [r["label"] for r in s]; sample["sentiment_score"] = [r["score"] for r in s]
        # post-contact review sentiment exists in the data: used ONLY to check domain transfer, never as a feature
        agree = pd.crosstab(sample["customer_sentiment"], sample["sentiment_label"]); agree.to_csv(out / "sentiment_vs_review_sentiment_crosstab.csv")
        zs = pipeline("zero-shot-classification", model=ZS, device=-1)
        labels = list(LABEL_TEXT.values()); inv = {v: k for k, v in LABEL_TEXT.items()}
        t0 = time.perf_counter(); z = zs(sample["issue_description"].tolist(), candidate_labels=labels, hypothesis_template="This customer message is about a {}.", batch_size=8); zs_ms = (time.perf_counter() - t0) / len(sample) * 1000
        sample["zeroshot_pred"] = [inv[r["labels"][0]] for r in z]; sample["zeroshot_conf"] = [r["scores"][0] for r in z]
        acc = accuracy_score(sample["issue_category"], sample["zeroshot_pred"]); f1 = f1_score(sample["issue_category"], sample["zeroshot_pred"], average="macro")
        sample[["ticket_id", "issue_description", "issue_category", "customer_sentiment", "sentiment_label", "sentiment_score", "zeroshot_pred", "zeroshot_conf"]].to_csv(out / "pipeline_outputs_100.csv", index=False)
        summ = {"sentiment_ms_per_message": sa_ms, "zero_shot_ms_per_message": zs_ms, "zero_shot_accuracy_vs_gold": float(acc), "zero_shot_macro_f1_vs_gold": float(f1),
                "sentiment_negative_share": float((sample["sentiment_label"] == "NEGATIVE").mean()), "n": int(len(sample))}
        write_json(out / "summary.json", summ); run.add(**summ)
        ex = sample.groupby("issue_category").head(1).head(4)
        md = ["# Pretrained transformer pipelines (no fine-tuning)", "", f"100 balanced real tickets (10 per category). Outputs: `artifacts/pretrained_pipelines/pipeline_outputs_100.csv`.", "",
              "## 1. Sentiment analysis (`" + SENT + "`)", f"- {summ['sentiment_ms_per_message']:.1f} ms/message on CPU. {summ['sentiment_negative_share']:.0%} of messages are labelled NEGATIVE.",
              "- Cross-tab against the dataset's post-contact review sentiment (**evaluation only; that field is a leakage field and is never a feature**):", "", agree.to_markdown() if hasattr(agree, "to_markdown") else agree.to_string(), "",
              "- Limitation: trained on movie reviews, so polite-but-unhappy support wording is routinely mis-scored; treat as a weak signal.", "",
              "## 2. Zero-shot classification (`" + ZS + "`)", f"- Candidate labels are the 10 issue categories phrased in plain words; {summ['zero_shot_ms_per_message']:.0f} ms/message (one NLI pass per label).",
              f"- Against the gold `issue_category`: **accuracy {acc:.2f}, macro-F1 {f1:.2f}** without any training - useful for cold-start, but well below the supervised TF-IDF/DistilBERT models (macro-F1 ~0.9+), at ~{zs_ms / max(sa_ms, 1e-9):.0f}x the sentiment cost.", "",
              "| Message | Gold | Zero-shot (confidence) | Sentiment |", "|---|---|---|---|", *[f"| {r['issue_description'][:80]}... | {r['issue_category']} | {r['zeroshot_pred']} ({r['zeroshot_conf']:.2f}) | {r['sentiment_label']} ({r['sentiment_score']:.2f}) |" for _, r in ex.iterrows()], ""]
        (REPORTS / "pretrained_pipelines.md").write_text("\n".join(md), encoding="utf-8")
        print(summ)
    print("[OK] pipelines complete")


if __name__ == "__main__":
    main()
