# Classical NLP: BoW vs TF-IDF and semantic search

Script `scripts/08_nlp_classical.py` · Evidence `artifacts/nlp_classical/` · Task: ticket text → `issue_category` (10 classes). Train sample 120k rows from the training period, selection on validation (164,894 rows), one test evaluation.

## BoW vs TF-IDF (`bow_vs_tfidf_comparison.csv`)

Four vectorisers (count BoW, TF-IDF unigram, TF-IDF 1–2-gram, TF-IDF word + char 3–5-gram) × three classifiers (logistic regression, linear SVC, naive Bayes): **every combination reaches validation macro-F1 1.000**, with zero errors on validation, and the selected model scores **1.000 on the test period**. Using the description only, without the title, is also 1.000 (`input_ablation.csv`). On text with random typos and upper-casing the selected model keeps macro-F1 **0.995**.

**Honest reading:** the messages are generated from templates, so the comparison cannot rank BoW against TF-IDF on accuracy; the only separating differences are cost (fit time 7–9 s for unigram BoW/NB or LR vs 76–136 s for word+char features) and robustness to noise. We therefore use the cheapest strong option (TF-IDF/BoW unigram + logistic regression). There are no misclassified examples to analyse (`misclassified_examples.csv` is empty); the transformer comparison (`reports/transformer_finetuning.md`) carries the same caveat. Real tickets would need a re-run of this comparison.

## Semantic search (MiniLM embeddings vs TF-IDF retrieval)

6,000-ticket corpus, 20 hand-written free-text queries (not from the data) labelled good / ambiguous / weak; metric = precision@5 of the retrieved tickets' category versus the expected one (`semantic_search_eval.csv`, `semantic_search_examples.md`).

| query type | n | embedding p@5 | TF-IDF p@5 |
|---|---:|---:|---:|
| good | 14 | **0.80** | 0.64 |
| ambiguous | 4 | 0.25 | **0.50** |
| weak ("help") | 2 | 0.20 | 0.00 |
| all | 20 | **0.63** | 0.55 |

Embeddings win on natural paraphrases ("how do I send this jumper back"); they lose on ambiguous queries where several categories are legitimately close, and neither is meaningful on vague queries. 20 queries is small (one query ≈ 5 points), so the overall gap is suggestive only. Encoding 6,000 tickets took 68 s on CPU.
