"""RAG: chunking comparison, abstention calibration (own dev probes), evaluation on public RAG01-15 + held-out own probes."""
from __future__ import annotations
import json, re, argparse
import numpy as np, pandas as pd
from omnisupport.config import EVAL_DIR, REPORTS, INDEX_VERSION, MODELS
from omnisupport.rag.pipeline import PolicyRAG, STRATEGIES, INDEX_DIR
from omnisupport.utils.runlog import write_json, set_seed
from omnisupport.config import artifact_dir

set_seed(42); OUT = artifact_dir("rag"); INDEX_DIR.mkdir(parents=True, exist_ok=True)
pub = json.loads((EVAL_DIR / "rag_questions.json").read_text(encoding="utf-8"))
own = json.loads((EVAL_DIR / "own_rag_probes.json").read_text(encoding="utf-8"))
def hints(h): return [x.strip() for x in re.split(r"[/,]", h)]
def hit(res, hs, k): return any(any(h in r["doc_id"] or r["doc_id"] in h for h in hs) for r in res[:k])
rows = []
sets = {"public_RAG01-15": [(q["id"], q["question"], hints(q["expected_source_hint"])) for q in pub],
        "own_answerable": [(q["id"], q["q"], hints(q["hint"])) for q in own["answerable"]]}
rags = {}
for be in ("tfidf", "dense"):
    for st in STRATEGIES:
        r = PolicyRAG(st, be, threshold=0.0); rags[(be, st)] = r
        for sname, qs in sets.items():
            h1 = h3 = 0
            for _, q, hs in qs:
                res = r.retrieve(q, 3); h1 += hit(res, hs, 1); h3 += hit(res, hs, 3)
            rows.append(dict(backend=r.backend_name, chunking=st, set=sname, n=len(qs), hit_at_1=h1 / len(qs), hit_at_3=h3 / len(qs), n_chunks=len(r.chunks), build_s=round(r.build_seconds, 2)))
cmp = pd.DataFrame(rows); cmp.to_csv(OUT / "chunking_comparison.csv", index=False); print(cmp.to_string())
# choose chunking by OWN answerable set only (public set is for reporting, not selection)
sel = cmp[(cmp.set == "own_answerable") & (cmp.backend == "dense")].sort_values(["hit_at_1", "hit_at_3"], ascending=False, kind="stable").iloc[0]
best = sel.chunking; print("selected chunking:", best)
# abstention calibration on own dev probes (even index = dev, odd index = held-out test)
r = rags[("dense", best)]
if r.backend_name != "dense": raise SystemExit("dense backend unavailable")
def top(q): return r.retrieve(q, 1)[0]["score"]
A = [(q["id"], q["q"], top(q["q"])) for q in own["answerable"]]; U = [(q["id"], q["q"], top(q["q"])) for q in own["unanswerable"]]
devA, devU = A[::2], U[::2]; testA, testU = A[1::2], U[1::2]
cands = np.linspace(0.1, 0.8, 71)
def f(th, a, u): tp = sum(s >= th for *_, s in a); tn = sum(s < th for *_, s in u); return (tp / len(a) + 2 * tn / len(u)) / 3   # safety-weighted: wrongly answering an unsupported question costs 2x a wrong abstention
th = float(max(cands, key=lambda t: (f(t, devA, devU), -abs(t - 0.4))))
print("threshold", th, "dev bal-acc", f(th, devA, devU), "test bal-acc", f(th, testA, testU))
sc = pd.DataFrame([dict(kind="answerable", id=i, top_score=s) for i, _, s in A] + [dict(kind="unanswerable", id=i, top_score=s) for i, _, s in U]); sc.to_csv(OUT / "abstention_scores.csv", index=False)
# metadata (also calibrate the TF-IDF fallback backend so CI/offline deployments abstain sensibly)
rt = rags[("tfidf", best)]
def top_t(q): return rt.retrieve(q, 1)[0]["score"]
At = [(i, q, top_t(q)) for i, q, _ in A]; Ut = [(i, q, top_t(q)) for i, q, _ in U]
th_t = float(max(cands, key=lambda t: (f(t, At[::2], Ut[::2]), -abs(t - 0.2)))) if True else 0.2
print("tfidf threshold", th_t, "dev", f(th_t, At[::2], Ut[::2]), "test", f(th_t, At[1::2], Ut[1::2]))
rag = PolicyRAG(best, "dense", threshold=th); meta = rag.metadata()
meta["thresholds"] = {"dense": {"abstain_below": round(th, 3), "calibrated_on": "own dev probes (6 answerable / 6 unanswerable), objective = (recall_answerable + 2*specificity_unanswerable)/3", "heldout_weighted_score": round(f(th, testA, testU), 3)},
                      "tfidf": {"abstain_below": round(th_t, 3), "calibrated_on": "same own dev probes", "heldout_weighted_score": round(f(th_t, At[1::2], Ut[1::2]), 3)}}
write_json(INDEX_DIR / "index_metadata.json", meta)
# full evaluation
ev = []
def run(q_id, q, kind, hs=None):
    a = rag.answer(q); ok_src = bool(hs) and any(hit([{"doc_id": c["doc_id"]} for c in a.citations], hs, 3) for _ in [0]) if hs else None
    ev.append(dict(id=q_id, kind=kind, question=q, abstained=a.abstained, conflict=a.conflict, human_review=a.needs_human_review, reason=a.reason, top_score=round(a.top_score, 3), cited=";".join(c["doc_id"] for c in a.citations), source_hit=ok_src, answer=a.answer[:300], latency_ms=round(a.latency_ms, 1)))
for q in pub: run(q["id"], q["question"], "public", hints(q["expected_source_hint"]))
for q in own["answerable"]: run(q["id"], q["q"], "own_answerable", hints(q["hint"]))
for q in own["unanswerable"]: run(q["id"], q["q"], "own_unanswerable")
for q in own["adversarial"]: run(q["id"], q["q"], "own_adversarial")
E = pd.DataFrame(ev); E.to_csv(OUT / "rag_evaluation.csv", index=False)
g = E.groupby("kind").agg(n=("id", "count"), abstain_rate=("abstained", "mean"), source_hit=("source_hit", "mean"), conflict_rate=("conflict", "mean"), review_rate=("human_review", "mean"), avg_latency_ms=("latency_ms", "mean")).round(3)
print(g.to_string()); print(E[E.kind == "public"][["id", "abstained", "conflict", "cited", "source_hit", "top_score"]].to_string())
print(E[E.kind == "own_adversarial"][["id", "abstained", "reason", "answer"]].to_string())
# fallback (TF-IDF) end-to-end behaviour at its calibrated threshold - reported so the degradation is visible, not hidden
fb = PolicyRAG(best, "tfidf", threshold=th_t); fbr = {"pub_abstain_rate": float(np.mean([fb.answer(q["question"]).abstained for q in pub])), "pub_source_hit_when_answered": None,
      "unanswerable_abstain_rate": float(np.mean([fb.answer(q["q"]).abstained for q in own["unanswerable"]])), "threshold": th_t}
ans_ = [(fb.answer(q["question"]), hints(q["expected_source_hint"])) for q in pub]; ans_ = [(a, h) for a, h in ans_ if not a.abstained]
fbr["pub_source_hit_when_answered"] = float(np.mean([hit([{"doc_id": c["doc_id"]} for c in a.citations], h, 3) for a, h in ans_])) if ans_ else None; print("tfidf fallback:", fbr)
write_json(OUT / "summary.json", {"tfidf_fallback": fbr, "selected_chunking": best, "threshold": th, "dev_balanced_acc": f(th, devA, devU), "heldout_balanced_acc": f(th, testA, testU), "by_kind": json.loads(g.reset_index().to_json(orient="records"))})
