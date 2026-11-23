# Policy RAG: design, chunking comparison and evaluation

Code: `src/omnisupport/rag/pipeline.py` · Evaluation: `scripts/13_rag_eval.py` · Evidence: `artifacts/rag/` · Index metadata: `models/rag/index_metadata.json` (`policy-index-v1.0.0`)

## Design

* **Corpus**: the 40 supplied policy files in `knowledge_base/` (each ~450 characters). The boilerplate "Operational notes" block is parsed for **scope/version** and removed from the retrievable text so that every chunk is not dominated by identical boilerplate; each answer cites `doc_id` + **policy version** (e.g. `[refund_limits v1.09]`).
* **Retrieval**: `all-MiniLM-L6-v2` embeddings (plain `transformers`, mean pooling, L2-normalised) in a FAISS `IndexFlatIP`; one best chunk per document, top-3. A TF-IDF backend is the automatic fallback when torch/FAISS are unavailable (used in CI) and is also calibrated.
* **Answering**: *extractive* and cited. We return the question-relevant sentences from the top documents, not free text, so the answer cannot contain a claim the policy does not contain. No LLM is needed for the grounded answer (an LLM could be layered on top with the same citations).
* **Abstention**: if the best retrieval score is below a threshold calibrated on our own probes, the response is `No answer: ... insufficient policy evidence`, `abstained=true`, `needs_human_review=true`, with no citations.
* **Conflicts**: if two relevant *return* policies state different day windows (21-day marketplace vs 30-day standard) or the user asks about conflicting policies, the answer is prefixed with the conflict-resolution rule from `rag_conflict` (prefer the more specific policy for the channel; otherwise flag for human review) and `needs_human_review=true`.
* **Safety**: requests for another customer's data or credentials return a refusal that cites the privacy policy; instruction-override text in the *question* is ignored (and flagged); instruction-like sentences in *retrieved* text are dropped from quotes (`_best_sentences`).

## Chunking comparison (hit = expected policy among the top-k retrieved documents)

| backend | chunking | chunks | public RAG01–15 hit@1 | hit@3 | own answerable (12) hit@1 | hit@3 |
|---|---|---:|---:|---:|---:|---:|
| TF-IDF | document | 40 | 0.73 | 0.93 | 0.58 | 0.83 |
| TF-IDF | section | 76 | 0.67 | 0.93 | 0.58 | 0.83 |
| TF-IDF | fixed 220/60 | 74 | 0.73 | 0.87 | 0.67 | 0.83 |
| MiniLM+FAISS | document | 40 | 0.80 | 0.87 | 0.75 | 1.00 |
| MiniLM+FAISS | **section (selected)** | 76 | **0.87** | **0.93** | **0.83** | **1.00** |
| MiniLM+FAISS | fixed 220/60 | 74 | 0.87 | 0.93 | 0.83 | 1.00 |

Selection used **only our own 12 paraphrased probes** (`evaluation/own_rag_probes.json`), not the public questions. `section` and `fixed` tie on retrieval; `section` wins the tie because
sentence-aligned chunks never cut a rule in half and give cleaner quotes. Whole-document chunks are slightly worse for dense retrieval because a 450-character document mixes the rule with
unrelated "see also" sentences. With only 15 + 12 questions a difference of one question is 7–8 points, so the ranking between `section`/`fixed` is **not** statistically meaningful; the dense-over-TF-IDF gain (≈ +0.2 hit@1) is the robust finding.

## Evaluation

| set | n | result |
|---|---:|---|
| Public RAG01–15 | 15 | expected source cited for **14/15**; conflict flagged on RAG04 (marketplace 21 vs 30 days) and RAG07 (explicit conflict question); no abstentions |
| Own answerable paraphrases | 12 | expected source cited 10/12 |
| Own unanswerable / out-of-scope | 12 | abstained on **11/12**; the miss (OU12 "warranty for cars sold in Germany") is topically close to the warranty policies (score 0.44) |
| Own adversarial | 3 | injection ignored + flagged; other-customer-data request refused with privacy citation; credential request refused |

Failures we kept in the report rather than tuning away: **RAG13** (password reset failing + unrecognised orders) returned `account_access` and `delivery_lost`, missing `account_takeover` — a dense retriever
anchored on "password reset" over "orders I don't recognise"; a hybrid (BM25+dense) or query rewriting would likely fix it. **OA07** ("customer thinks their account was hacked") also missed `account_takeover` at the cited rank.
Both are *safety-relevant* misses, so the product answer for account-security wording should always add a human-review flag (future work: lexical keyword boost for security terms).

### Abstention threshold

Scores for answerable vs unanswerable probes are well separated on average (answerable mean 0.59, unanswerable mean 0.17; `artifacts/rag/abstention_scores.csv`) but overlap in the tail.
The objective is safety-weighted — `(recall on answerable + 2 × specificity on unanswerable) / 3` — because answering an unsupported policy question is worse than asking a human. Threshold **0.34** (dense) scores **1.00** on the dev half (6 + 6 probes) and **0.89** on the held-out half (6 + 6).
These are tiny samples, so treat the threshold as a sensible default, not a tuned guarantee; re-calibrate whenever the corpus or embedding model changes (one script run).

### Degraded mode (no torch / FAISS, e.g. CI)

The service falls back to TF-IDF retrieval (`/rag/ask` reports `retrieval_backend`). Because TF-IDF cannot match paraphrases, its calibrated threshold (0.20, with an out-of-vocabulary penalty so unseen words like "teleporting" or "mars" lower the score) makes it **abstain on 60 % of the public questions**,
cite the expected source on **100 % of those it does answer**, and abstain on **100 % of our unanswerable probes** (`artifacts/rag/summary.json → tfidf_fallback`). The fallback is therefore precise but low-recall, which is the safe direction for a degraded mode.

## Limits

* The answer is extractive; it will not synthesise across policies beyond quoting the top two or three. This is deliberate for safety.
* Probe sets are small and written by the same author who built the retriever (optimistic by construction); the public RAG01–15 set is the more independent measure.
* Only the knowledge-base version `1.x` snapshot exists; there is no time-based conflict resolution (effective dates are not in the source documents).
