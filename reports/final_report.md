# OmniSupport AI — final report

## 1. What was built

An end-to-end prototype for a retail-support operation, built from the student pack only (no instructor material was read or used). It covers data audit and leakage control, classical and deep models, NLP and transformer fine-tuning, an LLM layer with validated outputs, cited policy RAG, a human-gated agent, an API and UI, CI and monitoring. Requirement-to-evidence mapping: `reports/evidence_map.md`. Architecture: `reports/architecture.md`. Operations: `docs/runbook.md`.

## 2. Results by epic

| Epic | Completed work | Evidence | Validation | Open risk |
|---|---|---|---|---|
| 0 Foundations | Problem framing, decision log (D01–D16), risk register, repo skeleton, seeds, `.env.example`, repo guard | `reports/problem_framing.md`, `decision_log.md` | `scripts/repo_guard.py` | Environment differs from the pinned baseline (D15) |
| 1 Data/EDA/features | 10 shards = 1,000,000 rows asserted; 4,000 duplicates removed; T0 contract and leakage register enforced in code | `artifacts/data_audit/`, `reports/leakage_register.md`, `eda_report.md` | leakage tests | Synthetic data |
| 2 Classical ML | Escalation: PR-AUC 0.572, recall 0.776 at 0.24. Duration: MAE 3.49 h, interval coverage 0.898 | `reports/classical_ml_report.md` | Temporal test used once, grouped CV | Moderate signal; the cost ratio is an assumption |
| 3 Segmentation, tabular NN | k=6, silhouette 0.186. NN PR-AUC 0.5727, no gain over classical; overfit probe shows a +0.18 loss gap | `segmentation_report.md`, `tabular_nn_report.md` | seeded re-run reproduces val loss | Clusters are weak |
| 4 Vision | CNN and ResNet18 both 1.0 clean; robustness and data-size curves separate them | `vision_report.md` | 3 seeds | 72 synthetic test images |
| 5 NLP | BoW/TF-IDF tie at 1.0; semantic search p@5 0.63 vs 0.55; attention worked example; 2 pretrained pipelines; DistilBERT fine-tune vs TF-IDF (TF-IDF better on noisy text) | `nlp_classical.md`, `attention_explained.md`, `pretrained_pipelines.md`, `transformer_finetuning.md` | held-out test | Templated text caps all scores |
| 6 LLM engineering | 8 versioned prompts, Pydantic contracts, repair + fallback + policy floor, 30-request × 3-provider benchmark | `llm_benchmark.md` | regression and unit tests | Hosted API never called; benchmark rows pre-date the policy floor |
| 7 RAG | Section chunking chosen by comparison; calibrated abstention; conflict, injection and secret guards; index metadata | `rag_report.md` | 14/15 cited hits, 11/12 abstentions | 2 account-takeover misses; TF-IDF fallback abstains on 60 % of questions |
| 8 Agent | Strict tool schemas, one retry, approval gate, traces | `agent_report.md` | 15/15 correct, 0 executed | Deterministic planner only; 20 % partial refund is an assumption |
| 9 Guardrails/QA | Evaluation suite, subgroup analysis, risk register | `evaluation_and_guardrails.md`, `subgroup_report.md`, `risk_register.md` | 74 tests, clean venv | Limited attributes |
| 10 Integration/ops | FastAPI, Gradio, Docker, Actions, redacted logs, monitoring summary | `app/`, `deployment/`, `.github/`, `monitoring/` | 47-request workload via the app | Docker and hosted CI unobserved before push |
| 11 Final | README, runbook, viva notes, this report | – | – | – |

## 3. Honest findings

1. **More model did not help the tabular tasks.** Logistic regression, gradient boosting and the neural net sit within 0.001 PR-AUC; the deployed model is chosen for cost and explainability.
2. **Perfect text scores are a property of the data.** The messages are templated, so BoW, TF-IDF and DistilBERT all score 1.0. Under typo noise, TF-IDF (0.991) beat the fine-tuned DistilBERT (0.902).
3. **Raw small LLMs are unsafe for approval decisions.** Both Qwen models said a GBP 250 refund needed no approval. The code-level policy floor now overrides this; a unit test covers it. The benchmark table shows the raw behaviour, not the corrected one.
4. **Vision ranking needs stress tests.** A clean test cannot separate the two vision models; corruptions can. Neither is uniformly better.
5. **Abstention has costs both ways.** 11/12 unsupported probes abstained, but two safety-relevant misses remain, and the TF-IDF fallback refuses 60 % of answerable questions.

## 4. Known limitations

* Synthetic, templated data: ceiling effects in text and vision, and no external validity claims.
* Docker image not built locally (no daemon); hosted CI not yet observed.
* Anthropic provider tested only with a stub client; cost figures are estimates from an assumed hourly rate ($0.34), not an API bill.
* The rules provider was written while the public prompt cases were visible, so its 100 % on those is optimistic. The 20 real-ticket accuracy (75 %) is the fairer, still indicative, number.
* Local pickles were produced with scikit-learn 1.8.0; CI pins 1.9.0 (the escalation model loads with a version warning).
* Fairness analysis covers age band, language, accessibility and region only.
* Agent policy numbers (20 % partial refund, GBP 100 approval threshold) must be confirmed by the business.

## 5. Reproduce

See `README.md` ("Setup", "Verify without the data", "Reproduce the evidence").

## 6. Release status

The local git history is committed. Publication to a public GitHub repository is deliberately held until the account, repository name and visibility are confirmed by the owner.
