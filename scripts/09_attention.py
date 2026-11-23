"""Epic NLP-02 (Session 20): query/key/value, self-attention and positional information on ONE real support message,
contrasted with TF-IDF.

Run:  python scripts/09_attention.py        -> artifacts/attention/*, reports/attention_explained.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from omnisupport.config import REPORTS, SEED, artifact_dir  # noqa: E402
from omnisupport.ml import common as C  # noqa: E402
from omnisupport.nlp.attention import cosine, self_attention, sinusoidal_positions  # noqa: E402
from omnisupport.nlp.embeddings import get_embedder  # noqa: E402
from omnisupport.utils import plotting as P  # noqa: E402
from omnisupport.utils.runlog import RunRecord, write_json  # noqa: E402

MODEL = "distilbert/distilbert-base-uncased"
FOCUS = "I returned the item but no refund is visible."               # first sentence of the real ticket below
REORDER = "No refund is visible but I returned the item."             # same words, different order  (constructed contrast)
NEGATE = "I returned the item and the refund is visible."              # one clause flipped            (constructed contrast)


def word_span(tokens: list[str], word: str) -> list[int]:
    """Indices of the (sub)word pieces that spell `word` (WordPiece splits e.g. 'refund' into 'ref' + '##und')."""
    words, cur = [], []
    for i, t in enumerate(tokens):
        if t.startswith("##") and cur:
            cur.append(i)
        else:
            if cur: words.append(cur)
            cur = [i]
    words.append(cur)
    for idx in words:
        if "".join(tokens[i].replace("##", "") for i in idx) == word:
            return idx
    raise ValueError(f"{word!r} not found in {tokens}")


def pick_real_message() -> tuple[str, str]:
    df = C.load_clean()
    hit = df[df["issue_description"].astype("string").str.contains("I returned the item but no refund is visible", na=False)].iloc[0]
    return str(hit["ticket_id"]), str(hit["issue_description"])


def fmt(a: np.ndarray, nd: int = 2) -> list:
    return np.round(a, nd).tolist()


def md_matrix(rows, cols, M, nd=2):
    out = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for r, vals in zip(rows, M):
        out.append(f"| **{r}** | " + " | ".join(f"{v:.{nd}f}" for v in vals) + " |")
    return "\n".join(out)


def main() -> None:
    out = artifact_dir("attention")
    tid, full_text = pick_real_message()
    with RunRecord(out, "attention", config={"model": MODEL, "focus_sentence": FOCUS, "source_ticket": tid}, split_method="n/a (explanatory demonstration)", model_version=MODEL,
                   limitations=["Attention weights are not explanations of model decisions; they show information routing in one head/layer.", "DistilBERT uses learned absolute position embeddings, not the sinusoidal ones shown for the toy example."]) as run:
        # ------------------------------------------------------------------ 1. toy worked example (hand-checkable)
        toks = ["not", "receive", "my", "refund"]
        E = np.array([[1.0, 0.0, 1.0, 0.0],      # not
                      [0.0, 1.0, 0.0, 1.0],      # receive
                      [1.0, 1.0, 0.0, 0.0],      # my
                      [0.0, 1.0, 1.0, 0.5]])      # refund
        PE = sinusoidal_positions(4, 4); X = E + PE
        Wq = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float); Wk = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=float); Wv = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
        toy = self_attention(X, Wq, Wk, Wv)
        toy_nopos = self_attention(E, Wq, Wk, Wv)
        # the SAME word at two positions: without PE the two 'not' rows are identical, with PE they differ
        Etwo = np.vstack([E[0], E[1], E[0]]); Xtwo = Etwo + sinusoidal_positions(3, 4); q_two = (Xtwo @ Wq); q_two_nopos = Etwo @ Wq
        assert np.allclose(toy["weights"].sum(1), 1.0)
        write_json(out / "toy_example.json", {"tokens": toks, "E": fmt(E), "PE": fmt(PE, 3), "X": fmt(X, 3), "Wq": fmt(Wq), "Wk": fmt(Wk), "Wv": fmt(Wv), "Q": fmt(toy["Q"], 3), "K": fmt(toy["K"], 3), "V": fmt(toy["V"], 3),
                                              "raw_scores": fmt(toy["raw_scores"], 3), "scaled_scores": fmt(toy["scaled_scores"], 3), "weights": fmt(toy["weights"], 3), "output": fmt(toy["output"], 3)})

        # ------------------------------------------------------------------ 2. real DistilBERT: recompute Q,K,V from its weights and verify
        tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModel.from_pretrained(MODEL, attn_implementation="eager").eval()
        enc = tok(FOCUS, return_tensors="pt"); tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
        with torch.no_grad():
            o = model(**enc, output_attentions=True, output_hidden_states=True)
        L0 = model.transformer.layer[0].attention; x0 = o.hidden_states[0][0]                                    # input to layer 0 = word + position embeddings (+LayerNorm)
        H, dk = model.config.n_heads, model.config.dim // model.config.n_heads
        q = (x0 @ L0.q_lin.weight.T + L0.q_lin.bias).detach().numpy(); k = (x0 @ L0.k_lin.weight.T + L0.k_lin.bias).detach().numpy(); v = (x0 @ L0.v_lin.weight.T + L0.v_lin.bias).detach().numpy()
        head = 0; sl = slice(head * dk, (head + 1) * dk)
        manual = self_attention(x0.detach().numpy(), (L0.q_lin.weight.T.detach().numpy())[:, sl], (L0.k_lin.weight.T.detach().numpy())[:, sl], (L0.v_lin.weight.T.detach().numpy())[:, sl],
                                bq=L0.q_lin.bias.detach().numpy()[sl], bk=L0.k_lin.bias.detach().numpy()[sl], bv=L0.v_lin.bias.detach().numpy()[sl])
        official = o.attentions[0][0, head].numpy(); max_diff = float(np.abs(manual["weights"] - official).max())
        assert max_diff < 1e-4, f"manual attention does not reproduce the model ({max_diff})"
        n = len(tokens)
        # full multi-head layer-0 output check
        ctx = []
        for h in range(H):
            s_ = slice(h * dk, (h + 1) * dk); ah = torch.softmax(torch.from_numpy(q[:, s_] @ k[:, s_].T / np.sqrt(dk)), -1).numpy(); ctx.append(ah @ v[:, s_])
        manual_out = (torch.from_numpy(np.concatenate(ctx, 1)).float() @ L0.out_lin.weight.T + L0.out_lin.bias).detach().numpy()
        with torch.no_grad():
            module_out = L0(o.hidden_states[0])[0][0].numpy()
        layer_diff = float(np.abs(manual_out - module_out).max()); assert layer_diff < 1e-3
        pd.DataFrame({"token": tokens, "q_first4": [fmt(r[:4], 3) for r in q[:, sl]], "k_first4": [fmt(r[:4], 3) for r in k[:, sl]], "v_first4": [fmt(r[:4], 3) for r in v[:, sl]]}).to_csv(out / "real_qkv_head0_layer0.csv", index=False)

        # ------------------------------------------------------------------ 3. heatmaps (layer 0 head 0, layer 0 mean, last layer mean) + toy
        att = [a[0].numpy() for a in o.attentions]                         # layers x [heads, n, n]
        fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
        for a, (title, M) in zip(ax[:3], [("Layer 1, head 1 (exact recompute from Q/K)", official), ("Layer 1, mean of 12 heads", att[0].mean(0)), ("Layer 6 (last), mean of 12 heads", att[-1].mean(0))]):
            a.imshow(M, cmap="viridis"); a.set_xticks(range(n)); a.set_yticks(range(n)); a.set_xticklabels(tokens, rotation=60, ha="right"); a.set_yticklabels(tokens); a.set_title(title, fontsize=9); a.grid(False); a.set_xlabel("key (attended-to)"); a.set_ylabel("query (attending)")
        ax[3].imshow(toy["weights"], cmap="viridis"); ax[3].set_xticks(range(4)); ax[3].set_yticks(range(4)); ax[3].set_xticklabels(toks, rotation=45); ax[3].set_yticklabels(toks); ax[3].grid(False); ax[3].set_title("Toy example weights (softmax rows sum to 1)", fontsize=9)
        for i in range(4):
            for j in range(4): ax[3].text(j, i, f"{toy['weights'][i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        P.save(fig, out / "fig_attention_heatmaps.png")

        fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
        pe_big = sinusoidal_positions(24, 32); ax[0].imshow(pe_big.T, cmap="RdBu", aspect="auto"); ax[0].set_title("Sinusoidal positional encoding\n(rows=dimensions, columns=positions)", fontsize=9); ax[0].set_xlabel("position"); ax[0].grid(False)
        pos_w = model.embeddings.position_embeddings.weight.detach().numpy()[:n]; sim = np.array([[cosine(pos_w[i], pos_w[j]) for j in range(n)] for i in range(n)])
        ax[1].imshow(sim, cmap="magma"); ax[1].set_title("DistilBERT learned position embeddings:\ncosine similarity between positions", fontsize=9); ax[1].grid(False)
        ax[2].bar(["without PE", "with PE"], [float(np.abs(q_two_nopos[0] - q_two_nopos[2]).sum()), float(np.abs(q_two[0] - q_two[2]).sum())]); ax[2].set_title("Toy: |Q('not' @ pos0) - Q('not' @ pos2)|\n(0 => position-blind)", fontsize=9)
        P.save(fig, out / "fig_positional_information.png")

        # ------------------------------------------------------------------ 4. contrast with TF-IDF (same sentence pairs)
        corpus = C.load_clean()["issue_description"].dropna().astype(str).sample(20000, random_state=SEED).tolist()
        tf = TfidfVectorizer(lowercase=True).fit(corpus + [FOCUS, REORDER, NEGATE])
        Vf, Vr, Vn = (tf.transform([s]).toarray()[0] for s in (FOCUS, REORDER, NEGATE))
        nz = Vf.nonzero()[0]; tfidf_terms = sorted(((tf.get_feature_names_out()[i], float(Vf[i])) for i in nz), key=lambda x: -x[1])
        emb = get_embedder(); Es = emb.encode([FOCUS, REORDER, NEGATE])
        def sent_vec(text):
            e = tok(text, return_tensors="pt")
            with torch.no_grad(): h = model(**e).last_hidden_state[0]
            return h.numpy(), tok.convert_ids_to_tokens(e["input_ids"][0])
        hf, tf_toks = sent_vec(FOCUS); hr, tr_toks = sent_vec(REORDER)
        sp = lambda t, w: word_span(t, w)
        refund_f = hf[sp(tf_toks, "refund")].mean(0); refund_r = hr[sp(tr_toks, "refund")].mean(0); item_f = hf[sp(tf_toks, "item")].mean(0); item_r = hr[sp(tr_toks, "item")].mean(0)
        with torch.no_grad(): orr = model(**tok(REORDER, return_tensors="pt"), output_attentions=True)
        atts_r = np.stack([a[0].numpy().mean(0) for a in orr.attentions]).mean(0); atts_f = np.stack([a.mean(0) for a in att]).mean(0)
        pair_f = float(atts_f[np.ix_(sp(tf_toks, "refund"), sp(tf_toks, "no"))].mean()); pair_r = float(atts_r[np.ix_(sp(tr_toks, "refund"), sp(tr_toks, "no"))].mean())
        tfidf_dim = len(tf.get_feature_names_out())
        contrast = {"tfidf_vocab_dim": tfidf_dim, "tfidf_nonzero_in_focus_sentence": int(len(nz)), "tfidf_cosine_focus_vs_reorder": cosine(Vf, Vr), "tfidf_cosine_focus_vs_negation": cosine(Vf, Vn),
                    "minilm_sentence_cosine_focus_vs_reorder": cosine(Es[0], Es[1]), "minilm_sentence_cosine_focus_vs_negation": cosine(Es[0], Es[2]),
                    "distilbert_token_cosine_refund_focus_vs_reorder": cosine(refund_f, refund_r), "distilbert_token_cosine_item_focus_vs_reorder": cosine(item_f, item_r),
                    "attention_refund_to_no_focus": pair_f, "attention_refund_to_no_reorder": pair_r, "contextual_vector_dim": int(hf.shape[1]), "manual_vs_model_attention_max_abs_diff_head0": max_diff, "manual_vs_model_layer0_output_max_abs_diff": layer_diff}
        write_json(out / "tfidf_contrast.json", contrast)
        run.add(**contrast)

        # ------------------------------------------------------------------ 5. write the explained report
        qkv_rows = pd.read_csv(out / "real_qkv_head0_layer0.csv")
        top_att = sorted(((tokens[i], tokens[j], float(official[i, j])) for i in range(n) for j in range(n) if i != j), key=lambda x: -x[2])[:5]
        r = ["# Attention explained on one real support message (Session 20)", "",
             f"**Real message** (ticket `{tid}`): *\"{full_text}\"*", "",
             f"To keep the matrices readable the walkthrough uses its first sentence: **\"{FOCUS}\"**. All numbers below are produced by `scripts/09_attention.py` and stored in `artifacts/attention/`.", "",
             "## 1. The idea in one paragraph",
             "Each token is turned into three vectors by three learned matrices: a **query** (*what am I looking for?*), a **key** (*what do I offer to be found by?*) and a **value** (*what information do I hand over if I am attended to?*). "
             "The score between token *i* (query) and token *j* (key) is the dot product q·k, divided by √d_k so the softmax does not saturate. A softmax over *j* turns the scores of token *i* into weights that sum to 1, and the new representation of token *i* is the **weighted sum of all value vectors**. "
             "That is **self-attention**: every token rebuilds itself from all the other tokens in the same message, so *\"refund\"* can pull in information from *\"no\"* and *\"returned\"*. "
             "Attention itself has no notion of order — swapping two tokens just swaps two rows — so **positional information** is added to the token embeddings first.", "",
             "## 2. Worked numerical example (4 tokens, d_model = 4, d_k = 2) — checkable by hand",
             "Tokens taken from the message: `not receive my refund`. Embeddings E, the positional encoding PE (sinusoidal: PE[pos,2i]=sin(pos/10000^(2i/d)), PE[pos,2i+1]=cos(…)) and the input X = E + PE:", "",
             "**E (embeddings)**", md_matrix(toks, ["d0", "d1", "d2", "d3"], E), "", "**PE (position 0..3)**", md_matrix(toks, ["d0", "d1", "d2", "d3"], PE, 3), "", "**X = E + PE**", md_matrix(toks, ["d0", "d1", "d2", "d3"], X, 3), "",
             "With small hand-chosen projection matrices W_Q, W_K, W_V (4×2, in `toy_example.json`): **Q = XW_Q**, **K = XW_K**, **V = XW_V**:", "",
             "**Q**", md_matrix(toks, ["q0", "q1"], toy["Q"], 3), "", "**K**", md_matrix(toks, ["k0", "k1"], toy["K"], 3), "", "**V**", md_matrix(toks, ["v0", "v1"], toy["V"], 3), "",
             f"Scores QKᵀ (raw) then divided by √d_k = √2 = {np.sqrt(2):.3f}:", "", md_matrix(toks, toks, toy["scaled_scores"], 3), "",
             "Row-wise softmax = **attention weights** (each row sums to 1):", "", md_matrix(toks, toks, toy["weights"], 3), "",
             "Output = weights · V (the new, context-mixed representation of each token):", "", md_matrix(toks, ["o0", "o1"], toy["output"], 3), "",
             f"*Reading it:* the row for **refund** says it takes {toy['weights'][3, 0]:.0%} of its new representation from `not`, {toy['weights'][3, 1]:.0%} from `receive`, {toy['weights'][3, 2]:.0%} from `my` and {toy['weights'][3, 3]:.0%} from itself — "
             "so \"refund\" now carries a trace of the negation, which a bag-of-words vector cannot express.", "",
             "![attention heatmaps](../artifacts/attention/fig_attention_heatmaps.png)", "",
             "## 3. The same maths on a real pretrained model (DistilBERT, 6 layers × 12 heads, d_k = 64)",
             f"Tokens: `{' '.join(tokens)}`. I re-computed layer 1 / head 1 **from the model's own weights**: Q = x·W_Qᵀ + b_Q, K = x·W_Kᵀ + b_K, softmax(QKᵀ/√64). The result matches the weights the library returns to within **{max_diff:.1e}** (max absolute difference), and the full 12-head layer output including the output projection matches to **{layer_diff:.1e}**. "
             "So the diagram is not an illustration — it is the computation. First 4 of 64 dimensions of q, k, v for head 1 (`real_qkv_head0_layer0.csv`):", "",
             "| token | q[:4] | k[:4] | v[:4] |", "|---|---|---|---|", *[f"| {t} | {a} | {b} | {c} |" for t, a, b, c in zip(qkv_rows['token'], qkv_rows['q_first4'], qkv_rows['k_first4'], qkv_rows['v_first4'])], "",
             f"Strongest off-diagonal weights in that head: " + ", ".join(f"`{a}`→`{b}` {w:.2f}" for a, b, w in top_att) + ". Different heads and layers specialise differently; attention weights show *where information flows*, they are not a faithful explanation of the final prediction.", "",
             "## 4. Positional information",
             "Self-attention is permutation-equivariant: without positions, `not` at position 0 and `not` at position 2 would receive identical queries and keys. Position is injected by adding a position vector to each token embedding. "
             f"In the toy example the L1 distance between the queries of the *same* word at positions 0 and 2 is **{float(np.abs(q_two_nopos[0] - q_two_nopos[2]).sum()):.2f} without** and **{float(np.abs(q_two[0] - q_two[2]).sum()):.2f} with** positional encoding. "
             "The original Transformer uses fixed sinusoids; DistilBERT learns an absolute position embedding table instead (right-hand panels below show both).", "",
             "![positions](../artifacts/attention/fig_positional_information.png)", "",
             "## 5. Contrast with TF-IDF (same sentences)",
             f"TF-IDF represents the message as a **sparse {tfidf_dim:,}-dimensional vector with {len(nz)} non-zero entries** (one weight per distinct word: " + ", ".join(f"{w} {s:.2f}" for w, s in tfidf_terms[:6]) + ", …). Word order and context do not exist in that vector. Two constructed contrast sentences make the difference measurable:", "",
             "| Pair | TF-IDF cosine | MiniLM sentence cosine |", "|---|---|---|",
             f"| *{FOCUS}* vs *{REORDER}* (same words, different order) | **{contrast['tfidf_cosine_focus_vs_reorder']:.3f}** | {contrast['minilm_sentence_cosine_focus_vs_reorder']:.3f} |",
             f"| *{FOCUS}* vs *{NEGATE}* (meaning flipped, most words shared) | {contrast['tfidf_cosine_focus_vs_negation']:.3f} | {contrast['minilm_sentence_cosine_focus_vs_negation']:.3f} |", "",
             f"- TF-IDF cannot tell the reordered sentence from the original (cosine {contrast['tfidf_cosine_focus_vs_reorder']:.3f}, identical vectors) because it is a bag of words.",
             f"- In DistilBERT the *contextual* vector of the token `refund` is **not the same** in the two sentences (cosine {contrast['distilbert_token_cosine_refund_focus_vs_reorder']:.3f}; `item`: {contrast['distilbert_token_cosine_item_focus_vs_reorder']:.3f}) and the MiniLM sentence vectors differ (cosine {contrast['minilm_sentence_cosine_focus_vs_reorder']:.3f} vs TF-IDF's exact 1.000). The difference is small, but it exists *only* because each token is rebuilt from different neighbours at different positions - a bag of words has no mechanism to produce it.",
             f"- A caution against over-reading heatmaps: the average attention from `refund` to `no` is {contrast['attention_refund_to_no_focus']:.3f} in the original and {contrast['attention_refund_to_no_reorder']:.3f} when reordered - practically unchanged. Attention weights show where information *can* flow; they do not by themselves explain a prediction.",
             f"- **Honest limits.** (1) Both representations under-react to negation: the meaning-flipped sentence has cosine {contrast['tfidf_cosine_focus_vs_negation']:.3f} in TF-IDF and **{contrast['minilm_sentence_cosine_focus_vs_negation']:.3f}** in MiniLM (higher than the harmless reordering, {contrast['minilm_sentence_cosine_focus_vs_reorder']:.3f}), so semantic similarity is not entailment and a negation-sensitive task would need fine-tuning. (2) Attention costs O(n^2) in message length and needs a pretrained model, whereas TF-IDF is a cheap, transparent baseline that performs as well on this templated classification task (see `reports/transformer_finetuning.md`).",
             "- Dense contextual vectors are 768-dimensional and mostly non-zero; TF-IDF vectors are sparse, interpretable word weights. They answer different questions: *which words matter* (TF-IDF) vs *what does each word mean here* (attention).", ""]
        (REPORTS / "attention_explained.md").write_text("\n".join(r), encoding="utf-8")
    print("[OK] attention demo complete", {k: round(v, 4) for k, v in contrast.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
