# Evaluation rubric, guardrails and failure checks

This page is the single place that maps *what could go wrong* to *the guardrail* and *the test or script that proves it*. Risk scoring lives in `reports/risk_register.md`; subgroup results in `reports/subgroup_report.md`.

## 1. Evaluation rubric (what "good" means per component)

| Component | Primary measure | Pass bar (set before the run) | Where measured |
|---|---|---|---|
| Escalation classifier | PR-AUC + recall at the cost-based threshold | PR-AUC ≥ 0.50, ROC-AUC ≥ 0.70, recall ≥ 0.60 on the frozen test | `reports/classical_ml_report.md` |
| Resolution-time regression | MAE vs median baseline; 90 % interval coverage | MAE ≥ 30 % better; coverage within ±3 pts of 0.90 | same |
| Segmentation | silhouette, split-half stability (ARI), no tiny clusters | ARI ≥ 0.7, every cluster ≥ 3 % | `reports/segmentation_report.md` |
| Tabular NN / CNN / transfer | held-out metric vs classical or from-scratch baseline, with learning curves | reported, not thresholded | `reports/tabular_nn_report.md`, `reports/vision_report.md` |
| Text classifiers | macro-F1 on identical held-out rows, clean and noisy | reported with both clean and corrupted text | `reports/nlp_classical.md`, `reports/transformer_finetuning.md` |
| Structured LLM output | schema-valid rate (first attempt and after repair), task checks | ≥ 90 % valid after repair/fallback | `reports/llm_benchmark.md`, `tests/unit/test_llm_structured.py` |
| RAG | expected source cited; abstention on unsupported probes; conflict flagged | reported per question set | `reports/rag_report.md` |
| Agent | correct tool set, approval decision, zero executed actions | 100 % on the 15 cases | `reports/agent_report.md` |
| API | validation errors are 4xx, never 5xx; no stack traces; degrade to 503 per component | all integration tests green in a clean environment | `tests/integration/` |

## 2. Guardrails and the evidence that they work

| Risk | Guardrail | Evidence |
|---|---|---|
| **Target / post-outcome leakage** into the model | field-level register; `assert_raw_columns_allowed` and `assert_features_valid` raise `LeakageError` | `tests/unit/test_leakage_and_features.py`; `artifacts/ml_classification/leakage_demonstration.csv` (CSAT alone adds 0.18 PR-AUC) |
| Using the test set to choose things | temporal split; the test call is made once after freezing (recorded as `test_evaluations_performed: 1`) | `heldout_test_selected_model.json` in both ML artifact folders |
| LLM returns malformed / off-schema JSON | JSON extraction → Pydantic validation → **one repair turn** → deterministic rules fallback; never raises | `tests/unit/test_llm_structured.py` (malformed, repair, fallback, provider error) |
| LLM invents an order id | grounding check: an `order_id` not present in the message is removed, `needs_human_review` forced, flag `order_id_not_in_message` | `test_hallucinated_order_id_is_removed`, benchmark check `no_invented_order` (P05) |
| Prompt silently edited | prompt manifest with content hashes, regression test fails on drift | `tests/unit/test_prompt_regression.py::test_prompt_manifest_matches_files` |
| Secret / credential collection | `secret_guard` task, `contains_secret_request`, RAG/agent refusal, `redact()` on every logged string, forbidden log keys dropped | `test_privacy_and_logging.py`, P06 in the benchmark, RAG `OX03` probe |
| Unsupported policy answer (hallucination) | extractive answers with citations only; calibrated abstention; "No answer … insufficient policy evidence" | `reports/rag_report.md` (11/12 unanswerable probes abstained), `tests/unit/test_rag.py`, starter RAG contract |
| Conflicting policies | day-window conflict detector (21 vs 30 days) + `rag_conflict` rule → human review | `test_conflicting_return_windows_detected`, RAG04/RAG07 |
| Prompt injection (user text or retrieved text) | override phrases detected and ignored (RAG + agent), instruction-like sentences dropped from retrieved quotes, tools exposed only through strict schemas | `test_retrieved_injection_is_dropped_from_quotes`, `test_prompt_injection_does_not_bypass_approval` |
| Unauthorised or irreversible action | agent tools are read-only/advisory; refunds, cancellations, account changes are **proposals** held by `ApprovalGate` that need a named human and are never executed | `tests/unit/test_agent.py` (7 tests), `test_agent_approval_flow_never_executes` (API) |
| Wrong decision on a safety report | safety keywords always escalate with approved-guidance text only, no diagnosis | agent A08, `test_safety_report_escalates` |
| Path traversal / arbitrary file read via the vision endpoint | path must resolve inside `data/images` with an image suffix; 400/404 otherwise, validated *before* any model import | `test_vision_blocks_path_traversal` (4 payloads) |
| Stack-trace / secret leakage in API errors | handler returns generic `internal error`; component failures are `503` with a controlled message; health reports only `api_key_configured: bool` | `tests/integration/test_api_end_to_end.py` |
| Component unavailable in the deployment (no torch, sklearn version mismatch) | dense→TF-IDF retrieval fallback, escalation model 503 with message, LLM provider → rules fallback | clean-environment CI run (Python 3.13, `requirements-ci.txt`: 73 tests pass) |
| Unfair performance across groups | sensitive attributes excluded from inputs; recall/FPR/calibration by group with Wilson CIs and small-group flags | `reports/subgroup_report.md` |
| Secrets or raw data committed | `.gitignore`, `.env.example`, `scripts/repo_guard.py` (secret patterns, forbidden paths, size) as a CI step | `repo guard OK` in CI |

## 3. Known failure modes we measured and chose not to hide

* The escalation model misses **67 %** of true escalations from customers with no previous tickets and flags ~100 % of non-escalations among customers with ≥ 5 previous tickets (`reports/classical_ml_report.md`).
* RAG retrieval missed `account_takeover` for two security-worded questions (RAG13, OA07) — a safety-relevant miss; hybrid retrieval is the proposed fix.
* The rules LLM provider mislabels 5 of 20 real tickets in the benchmark, mostly vague wording (see `reports/llm_benchmark.md`); it is a safety fallback, not a quality baseline.
* The agent planner is rule-based and English-only.

## 4. Release gate (all must be true before publishing)

1. `pytest tests` green in a clean Python 3.13 venv with `requirements-ci.txt` (done: 73 passed).
2. `python scripts/repo_guard.py` clean. 3. GitHub Actions `ci` green on the pushed commit. 4. Every checklist item in `reports/evidence_map.md` has a file path that exists.
5. No metric appears in a report that is not produced by a script in `scripts/` (numbers are copied from `artifacts/`).
