# Viva notes: questions to expect and the evidence-backed answers

Short answers first, then where to look. Numbers are copied from `artifacts/`; if a script is re-run, re-check them.

## Framing and data

**What exactly is predicted, and when?** At **T0 (ticket creation, before triage)**: (1) will the ticket escalate, (2) how many hours to resolution. Anything created later (priority, category assigned at triage, CSAT, sentiment, final disposition) is not a valid input. → `reports/problem_framing.md`, `reports/leakage_register.md`.

**Which fields did you drop and how do you know nothing leaked?** The register classifies every raw column (feature / target / post-outcome / triage-assigned / sensitive / identifier). `assert_raw_columns_allowed` and `assert_features_valid` fail the run if a forbidden field appears. We also *demonstrate* the cost of leakage: adding CSAT lifts validation PR-AUC by 0.18 and resolution time by 0.23 (`artifacts/ml_classification/leakage_demonstration.csv`).

**How did you split?** By time (train < Sep 2025, validation Sep–Oct, test Nov–Dec) with customer-grouped CV inside train, so the model is evaluated on the future, not on shuffled neighbours. The test set is evaluated once after everything is frozen (`test_evaluations_performed: 1`).

**What was wrong with the data?** Blanks/NaNs, exact duplicates, inconsistent categories, outliers, imbalance, templated noisy text, subgroup imbalance and leakage columns; every treatment and its count is in `artifacts/data_audit/`. The audit asserts exactly ten shards / 1,000,000 rows.

## Classical ML

**Why is logistic regression as good as gradient boosting?** The signal is largely additive (previous tickets, order value, issue type). CV PR-AUC 0.573 (LR) vs 0.569 (default HGB) vs 0.573 (tuned HGB), differences ≈ 1 sd. We say so; we kept the tuned HGB for calibration (ECE 0.004) and the cost-threshold workflow, not because it "won".

**Why threshold 0.24 and not 0.5?** A missed escalation costs 3× an unnecessary review (assumption, stated up-front). Minimising expected cost on validation gives 0.24: recall 0.776 / precision 0.429 on the frozen test, flagging 55 % of tickets. At 0.5 the model would miss 67 % of escalations. → `threshold_table_validation.csv`.

**Is the model good?** Honest answer: modest. PR-AUC 0.572 vs base rate 0.307 (lift ≈ 1.4× in precision at the chosen operating point). Useful for queue ordering, not for automation; it is weakest on first-contact customers (67 % of their escalations missed). → `reports/classical_ml_report.md`.

**Regression?** Test MAE 3.49 h vs 6.72 h for the median baseline (−48 %), RMSE 11.4 h (heavy tail), R² 0.31. Ridge ties HGB. 90 % prediction interval covers 0.898 on test. → same report.

## Segmentation, DL, CV

**Are the clusters real?** Silhouette 0.186, split-half ARI 0.78: soft groupings, stable enough to describe, not "natural species". Personas describe associations only and exclude sensitive attributes. → `reports/segmentation_report.md`.

*(Tabular NN and vision answers: see `reports/tabular_nn_report.md`, `reports/vision_report.md`.)*

## NLP and transformers

**Show attention.** `reports/attention_explained.md` recomputes DistilBERT's Q/K/V by hand from the `q_lin/k_lin/v_lin` weights, checks the manual scores match the library's attention to numerical tolerance, and contrasts the contextual, position-aware representation with TF-IDF's order-free sparse vector.

**Did you really fine-tune?** Yes: DistilBERT on the 10,000-row subset, custom training loop, CPU, held-out comparison with TF-IDF on identical rows (`reports/transformer_finetuning.md`, `artifacts/transformer_finetune/training_log.csv`). Text is templated, so both models sit near the ceiling; the report says what that does and does not prove.

## LLM, RAG, agent

**How do you keep LLM output safe?** One schema per task; JSON extraction → Pydantic validation → one repair turn → deterministic fallback; an `order_id` not present in the message is deleted and routed to review. Prompts are versioned with hashed manifests and a regression test. → `reports/llm_benchmark.md`.

**What did the benchmark cost and why local models?** No API key was available, so we ran real local Qwen2.5 instruct models and report measured latency and token counts; cost is an *estimated compute cost* with its assumption stated, never an invented API price. The Anthropic provider exists and is stub-tested only.

**How does RAG avoid hallucination?** Extractive, cited answers (policy id + version); abstain below a calibrated score; conflicts (21-day marketplace vs 30-day standard) flagged for human review; injection text in retrieved docs is dropped. Weak spots: security-worded questions missed `account_takeover` twice. → `reports/rag_report.md`.

**Why a rule-based agent planner?** Reproducible, testable tool selection with a hard approval boundary; an LLM planner can be swapped in behind the same tool schemas and gate. The agent never executes refunds/cancellations/account changes; approved actions stay `executed: false`. → `reports/agent_report.md`.

## Engineering and responsibility

**What happens when torch, the model file or the LLM is unavailable?** Dense retrieval degrades to TF-IDF (precise, lower recall), the escalation endpoint returns a controlled 503, the LLM provider falls back to rules. A clean Python 3.13 environment with only `requirements-ci.txt` passes all 74 tests.

**Fairness?** Sensitive attributes are not inputs; subgroup recall/FPR/calibration with CIs in `reports/subgroup_report.md`. Synthetic data, so findings describe the pipeline, not real customers.

**What would you do next?** Hybrid (BM25 + dense) retrieval for security wording; real-traffic drift monitoring (PSI) with a metrics backend; SSO-backed approver identity; an LLM planner behind the existing gate; calibrate the cost ratio with the support team.
