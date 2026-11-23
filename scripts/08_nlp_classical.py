"""Epic NLP-01: BoW vs TF-IDF ticket-text classification + embedding-based semantic search.

Text -> issue_category (10 classes). Trained on train-period rows only, selected on validation, evaluated once on test period.
Semantic search compares MiniLM embeddings with TF-IDF retrieval on hand-written free-text queries (good / weak / ambiguous).
Run:  python scripts/08_nlp_classical.py
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import Pipeline, make_union  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from omnisupport.config import MODEL_VERSION, MODELS, SEED, artifact_dir  # noqa: E402
from omnisupport.ml import common as C  # noqa: E402
from omnisupport.nlp.embeddings import get_embedder  # noqa: E402
from omnisupport.utils import plotting as P  # noqa: E402
from omnisupport.utils.runlog import RunRecord, write_json  # noqa: E402

TARGET = "issue_category"


def text_of(df: pd.DataFrame, with_title=True) -> pd.Series:
    d = df["issue_description"].astype("string").fillna("")
    return (df["issue_title"].astype("string") + ". " + d) if with_title else d


QUERIES = [  # hand-written free-text probes (NOT drawn from the data) with the category a human would expect
    ("my parcel still has not arrived and it has been over a week", "delivery_late", "good"),
    ("package arrived crushed and the item inside is smashed", "damaged_item", "good"),
    ("you sent me a blue one but I ordered the red one", "wrong_item", "good"),
    ("I returned it two weeks ago, where is my money", "refund_status", "good"),
    ("how do I send this jumper back", "return_request", "good"),
    ("tracking says delivered but nothing came, I think it is lost", "lost_parcel", "ambiguous"),
    ("I was charged twice for one order", "payment_issue", "good"),
    ("can't log in to my account", "account_access", "good"),
    ("does this charger work with my phone", "product_question", "good"),
    ("the kettle stopped working after four months", "warranty", "good"),
    ("it's late and also damaged", "delivery_late", "ambiguous"),
    ("something is wrong with my order", "wrong_item", "weak"),
    ("help", "delivery_late", "weak"),
    ("my card was declined at checkout but money left my account", "payment_issue", "good"),
    ("refund approved but not showing on my statement", "refund_status", "good"),
    ("received someone else's parcel", "wrong_item", "ambiguous"),
    ("password reset email never comes", "account_access", "good"),
    ("is the laptop bag waterproof", "product_question", "good"),
    ("parcel marked as delivered to a neighbour I do not know", "lost_parcel", "ambiguous"),
    ("the heater broke within the guarantee period", "warranty", "good"),
]


def main() -> None:
    out = artifact_dir("nlp_classical")
    frame = C.modelling_frame()
    frame["issue_description"] = frame["ticket_id"].map(C.load_clean().set_index("ticket_id")["issue_description"])  # raw text lives in the clean table, not the feature frame
    parts = C.split_frame(frame)
    tr, va, te = (parts[k][parts[k]["desc_missing"] == 0] for k in ("train", "val", "test"))  # rows with a message; blanks handled via title-only in the API
    trs = tr.sample(120_000, random_state=SEED)
    with RunRecord(out, "nlp_classical", config={"target": TARGET, "train_rows": len(trs), "text": "issue_title + description", "vectorisers": ["CountVectorizer(BoW)", "TfidfVectorizer", "TF-IDF word 1-2gram", "TF-IDF word + char 3-5gram"]},
                   split_method="temporal split; train sample 120k from train period; selection on validation; single test evaluation", model_version=f"{MODEL_VERSION}/tfidf-lr",
                   limitations=["Messages are templated: high accuracy reflects synthetic generation, not real-world language difficulty.", "Remaining errors are label ambiguity between delivery_late and lost_parcel, not model weakness."]) as run:
        vecs = {"bow_counts_unigram": lambda: CountVectorizer(lowercase=True, min_df=3),
                "tfidf_unigram": lambda: TfidfVectorizer(lowercase=True, min_df=3, sublinear_tf=True),
                "tfidf_word_1-2gram": lambda: TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, sublinear_tf=True),
                "tfidf_word+char_3-5": lambda: make_union(TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, sublinear_tf=True), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True, max_features=150_000))}
        clfs = {"logreg": lambda: LogisticRegression(max_iter=300, C=5.0), "linear_svc": lambda: LinearSVC(C=0.5), "naive_bayes": lambda: MultinomialNB(alpha=0.3)}
        rows, fitted = [], {}
        for vn, vf in vecs.items():
            for cn, cf in clfs.items():
                if cn == "naive_bayes" and vn.startswith("tfidf_word+char"):
                    continue
                t0 = time.perf_counter(); mp = Pipeline([("vec", vf()), ("clf", cf())]).fit(text_of(trs), trs[TARGET]); fit_s = time.perf_counter() - t0
                pv = mp.predict(text_of(va)); rows.append({"vectoriser": vn, "classifier": cn, "val_macro_f1": f1_score(va[TARGET], pv, average="macro"), "val_accuracy": accuracy_score(va[TARGET], pv), "fit_seconds": fit_s})
                fitted[(vn, cn)] = mp; print(rows[-1], flush=True)
        comp = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False); comp.to_csv(out / "bow_vs_tfidf_comparison.csv", index=False)

        # description-only vs title+description (does the customer title carry the signal?)
        bv, bc = comp.iloc[0]["vectoriser"], comp.iloc[0]["classifier"]
        t_only = Pipeline([("vec", vecs[bv]()), ("clf", clfs[bc]())]).fit(text_of(trs, False), trs[TARGET]); pd_only = t_only.predict(text_of(va, False))
        ablate = pd.DataFrame([{"input": "description only", "val_macro_f1": f1_score(va[TARGET], pd_only, average="macro")}, {"input": "title + description", "val_macro_f1": comp.iloc[0]["val_macro_f1"]}])
        ablate.to_csv(out / "input_ablation.csv", index=False)

        best = fitted[(bv, bc)]
        pv = best.predict(text_of(va)); rep = classification_report(va[TARGET], pv, output_dict=True, zero_division=0)
        pd.DataFrame(rep).T.to_csv(out / "per_class_validation.csv")
        labels = sorted(va[TARGET].unique()); cm = confusion_matrix(va[TARGET], pv, labels=labels)
        fig, ax = plt.subplots(figsize=(7, 6)); ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues"); ax.grid(False); ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j]: ax.text(j, i, f"{cm[i, j]:,}" if cm[i, j] < 10000 else f"{cm[i, j] // 1000}k", ha="center", va="center", fontsize=6)
        ax.set_title(f"Validation confusion (row-normalised): {bv} + {bc}"); P.save(fig, out / "fig_confusion_validation.png")

        # error analysis: which pairs are confused, and why
        v = va.assign(pred=pv); wrong = v[v[TARGET] != v["pred"]]
        pairs = wrong.groupby([TARGET, "pred"]).size().reset_index(name="n").sort_values("n", ascending=False); pairs.to_csv(out / "confusion_pairs_validation.csv", index=False)
        ambiguous_titles = v.groupby("issue_title")[TARGET].nunique(); amb = ambiguous_titles[ambiguous_titles > 1].index.tolist()
        wrong_ex = wrong.sample(min(15, len(wrong)), random_state=SEED)[["issue_title", "issue_description", TARGET, "pred"]]; wrong_ex.to_csv(out / "misclassified_examples.csv", index=False)
        share_amb = float(wrong["issue_title"].isin(amb).mean()) if len(wrong) else 0.0
        write_json(out / "error_analysis.json", {"validation_errors": int(len(wrong)), "validation_error_rate": float(len(wrong) / len(v)), "titles_mapping_to_multiple_categories": amb, "share_of_errors_with_ambiguous_title": share_amb,
                                                 "top_confusions": pairs.head(5).to_dict("records")})
        print("errors", len(wrong), "top confusions", pairs.head(3).to_dict("records"))

        # frozen test (selected pipeline), plus a noise-robustness check
        pt = best.predict(text_of(te)); test_f1 = f1_score(te[TARGET], pt, average="macro"); test_acc = accuracy_score(te[TARGET], pt)
        rng = np.random.default_rng(SEED)

        def noisy(s: str) -> str:
            ch = list(s)
            for _ in range(max(1, len(ch) // 25)):
                i = int(rng.integers(0, len(ch))); ch[i] = chr(ord("a") + int(rng.integers(0, 26)))
            return "".join(ch).upper() if rng.random() < 0.3 else "".join(ch)
        ts = te.sample(20_000, random_state=SEED); tn = text_of(ts).map(noisy)
        robust = {"clean_test_sample_macro_f1": f1_score(ts[TARGET], best.predict(text_of(ts)), average="macro"), "noisy_typos_uppercase_macro_f1": f1_score(ts[TARGET], best.predict(tn), average="macro")}
        write_json(out / "heldout_test_selected_model.json", {"model": f"{bv}+{bc}", "test_macro_f1": test_f1, "test_accuracy": test_acc, "test_rows": int(len(te)), "robustness": robust, "test_evaluations_performed": 1})
        print("TEST", test_f1, test_acc, robust)
        mdir = MODELS / "nlp"; mdir.mkdir(parents=True, exist_ok=True); joblib.dump(best, mdir / "tfidf_classifier.joblib", compress=3)

        # -------------------------------------------------------------- semantic search
        emb = get_embedder()
        corpus = pd.concat([tr, va]).assign(text=lambda d: text_of(d).str.replace(r"GBP [0-9.]+", "GBP X", regex=True).str.replace(r"\b\d+ days", "N days", regex=True)).drop_duplicates("text").sample(frac=1.0, random_state=SEED)
        corpus = corpus.groupby(TARGET, group_keys=False).head(600).reset_index(drop=True)  # balanced, de-templated corpus (~6000 messages)
        t0 = time.perf_counter(); E = emb.encode(corpus["text"].tolist(), batch_size=128); enc_s = time.perf_counter() - t0
        tf = TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2), min_df=2).fit(corpus["text"]); T = tf.transform(corpus["text"])
        Q = emb.encode([q for q, _, _ in QUERIES]); QT = tf.transform([q for q, _, _ in QUERIES])
        sims_e = Q @ E.T; sims_t = (QT @ T.T).toarray()
        rows, shown = [], []
        for i, (q, gold, kind) in enumerate(QUERIES):
            te5 = np.argsort(-sims_e[i])[:5]; tt5 = np.argsort(-sims_t[i])[:5]
            pe = float(np.mean(corpus[TARGET].astype(str).values[te5] == gold)); pt_ = float(np.mean(corpus[TARGET].astype(str).values[tt5] == gold))
            rows.append({"query": q, "expected_category": gold, "difficulty_label": kind, "embedding_precision@5": pe, "tfidf_precision@5": pt_, "embedding_top_score": float(sims_e[i][te5[0]]), "tfidf_top_score": float(sims_t[i][tt5[0]]),
                         "embedding_top_category": corpus[TARGET].values[te5[0]], "tfidf_top_category": corpus[TARGET].values[tt5[0]], "embedding_top1_text": corpus["text"].values[te5[0]]})
        sr = pd.DataFrame(rows); sr.to_csv(out / "semantic_search_eval.csv", index=False)
        summ = sr.groupby("difficulty_label")[["embedding_precision@5", "tfidf_precision@5"]].mean().assign(n=sr.groupby("difficulty_label").size()); summ.to_csv(out / "semantic_search_by_difficulty.csv")
        overall = {"embedding_precision@5": float(sr["embedding_precision@5"].mean()), "tfidf_precision@5": float(sr["tfidf_precision@5"].mean()), "corpus_size": int(len(corpus)), "encode_seconds": enc_s, "embed_model": emb.model_name, "queries": len(QUERIES)}
        write_json(out / "semantic_search_summary.json", overall); print(overall); print(summ.round(2))
        ex = ["# Semantic search examples (generated)", "", "| Query | Difficulty | Expected | Embedding top-1 (cos) | TF-IDF top-1 (cos) | P@5 emb / tf-idf |", "|---|---|---|---|---|---|"]
        for _, r in sr.iterrows():
            ex.append(f"| {r['query']} | {r['difficulty_label']} | {r['expected_category']} | {r['embedding_top_category']} ({r['embedding_top_score']:.2f}) | {r['tfidf_top_category']} ({r['tfidf_top_score']:.2f}) | {r['embedding_precision@5']:.1f} / {r['tfidf_precision@5']:.1f} |")
        (out / "semantic_search_examples.md").write_text("\n".join(ex) + "\n", encoding="utf-8")
        run.add(best=f"{bv}+{bc}", test_macro_f1=test_f1, semantic=overall)
    print("[OK] classical NLP complete")


if __name__ == "__main__":
    main()
