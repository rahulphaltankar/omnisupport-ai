# Evidence map

Every requirement → where it is implemented, the command that regenerates the evidence, the evidence file, and the headline result. Paths are relative to the repository root. "Verified" means the artefact exists and was produced by the listed command in this repository; limits are stated in the last column and again in `reports/final_report.md`.

| # | Requirement | Implementation | Command | Evidence | Result / limit |
|---|---|---|---|---|---|
| 1 | Ingest and audit exactly ten shards = 1,000,000 rows | `scripts/01_data_audit.py`, `src/omnisupport/data/` | `python scripts/01_data_audit.py` | `artifacts/data_audit/`, `reports/eda_report.md` | 10 shards, 1,000,000 rows asserted; 4,000 exact duplicates removed before splitting; raw data is not committed |
| 2a | Cleaning, features | `src/omnisupport/data/{cleaning,features}.py` | same | `artifacts/data_audit/cleaning_*` | Winsorised at train p99.9 inside the pipeline |
| 2b | EDA | `scripts/02_eda.py` | `python scripts/02_eda.py` | `reports/eda_report.md`, `artifacts/eda/` | Decision-oriented findings, no causal claims |
| 2c | Leakage audit, T0 targets | `src/omnisupport/data/leakage.py` (`assert_raw_columns_allowed`, `assert_features_valid`) | `pytest tests/unit/test_leakage_and_features.py` | `reports/leakage_register.md` | T0 = ticket creation; forbidden fields enforced in code and tested |
| 3a | Regression: baselines, CV, tuning, held-out | `scripts/04_ml_regression.py` | `python scripts/04_ml_regression.py` | `artifacts/ml_regression/`, `reports/classical_ml_report.md` | Test MAE 3.49 h vs median baseline (−48 %), RMSE 11.38, R² 0.31; 90 % interval coverage 0.898 |
| 3b | Classification: baselines, CV, tuning, threshold, held-out | `scripts/03_ml_classification.py` | `python scripts/03_ml_classification.py` | `artifacts/ml_classification/`, `reports/classical_ml_report.md` | Test PR-AUC 0.572, ROC-AUC 0.734, recall 0.776, precision 0.429 at cost-based threshold 0.24 (FN:FP = 3:1, an assumption) |
| 3c | Business interpretation | `reports/classical_ml_report.md`, `reports/problem_framing.md` | – | same | Moderate signal; use as a triage aid, not an automatic decision |
| 4 | Segmentation with cautious personas | `scripts/05_segmentation.py` | `python scripts/05_segmentation.py` | `reports/segmentation_report.md`, `artifacts/segmentation/` | KMeans k=6; silhouette 0.186 (weak separation), bootstrap ARI 0.784; personas are descriptive |
| 5 | Tabular NN, learning curves, checkpoints | `scripts/06_tabular_nn.py` | `python scripts/06_tabular_nn.py` | `reports/tabular_nn_report.md`, `artifacts/tabular_nn/`, `models/tabular_nn/best.pt` | Test PR-AUC 0.5727; does **not** beat classical (0.5720) |
| 6 | CNN vs transfer learning | `scripts/07_vision.py` | `python scripts/07_vision.py` | `reports/vision_report.md`, `artifacts/vision/` | Both 1.0 on clean 72-image test; ResNet18 more robust to blur/rotation/dark, CNN to heavy noise; 10 % data: CNN F1 0.52 vs ResNet 0.99. Synthetic images, small test |
| 7a | BoW/TF-IDF, semantic search | `scripts/08_nlp_classical.py` | `python scripts/08_nlp_classical.py` | `reports/nlp_classical.md`, `artifacts/nlp_classical/` | All combinations macro-F1 1.0 (templated text); search p@5 MiniLM 0.63 vs TF-IDF 0.55 (20 queries) |
| 7b | Attention Q/K/V visualisation | `scripts/09_attention.py` | `python scripts/09_attention.py` | `reports/attention_explained.md`, `artifacts/attention/` | Toy worked example + real DistilBERT layer-0 head-0 Q/K/V recomputed manually + positional-information demo + TF-IDF contrast |
| 7c | Two pretrained pipelines | `scripts/11_pretrained_pipelines.py` | `python scripts/11_pretrained_pipelines.py` | `reports/pretrained_pipelines.md` | Sentiment 20 ms/msg; zero-shot accuracy 0.38 |
| 7d | Real HF fine-tune vs TF-IDF | `scripts/10_finetune_transformer.py` | `python scripts/10_finetune_transformer.py` | `reports/transformer_finetuning.md`, `artifacts/transformer_finetune/` | DistilBERT trained 20.8 min on CPU. Clean test: both 1.0. Noisy text: DistilBERT 0.902 vs TF-IDF 0.991 — TF-IDF wins; templated text caveat |
| 8a | Versioned prompts | `prompts/*.v1.txt`, `prompts/manifest.json` | `pytest tests/unit/test_prompt_regression.py` | `prompts/manifest.json` (content hashes) | Version `prompts-v1.0.0` |
| 8b | Pydantic structured outputs | `src/omnisupport/llm/` | `pytest tests/unit/test_llm_structured.py` | tests | Validate → one repair → rules fallback → policy floor; never raises |
| 8c | ≥20-request benchmark | `scripts/12_llm_benchmark.py` | `python scripts/12_llm_benchmark.py` | `reports/llm_benchmark.md`, `artifacts/llm_benchmark/` | 30 requests × 3 providers (rules, Qwen 0.5B, Qwen 1.5B), real latency and token counts; cost is an **estimate** from an assumed hourly rate; no hosted API exercised |
| 9 | Policy RAG: chunking comparison, sources, evaluation, index metadata, conflict/unsupported, abstention | `src/omnisupport/rag/pipeline.py`, `scripts/13_rag_eval.py` | `python scripts/13_rag_eval.py` | `reports/rag_report.md`, `artifacts/rag/`, `models/rag/index_metadata.json` | Section chunking chosen; cited-source hit 14/15 on public questions; 11/12 unanswerable probes abstained; two account-takeover cases missed (RAG13, OA07) |
| 10 | Safe agent: strict schemas, missing info, tool failure, traces, approval | `src/omnisupport/agent/` | `python scripts/14_agent_eval.py` | `reports/agent_report.md`, `artifacts/agent/{agent_eval.csv,traces.json}` | 15/15 tool selection and approval decisions correct; no action ever executed |
| 11a | Evaluation, guardrails, tests | `tests/`, `reports/evaluation_and_guardrails.md` | `pytest tests` | same | 74 tests passed in a clean Python 3.13 venv with `requirements-ci.txt` |
| 11b | Subgroup analysis | `scripts/15_subgroup_analysis.py` | `python scripts/15_subgroup_analysis.py` | `reports/subgroup_report.md` | Generated; gaps reported with sample sizes |
| 11c | Risk register | `reports/risk_register.md` | – | same | R01–R18 plus final-phase additions, owners, residual risk |
| 12a | FastAPI / Gradio | `app/api/main.py`, `app/ui/app.py` | `uvicorn app.api.main:app` | `tests/integration/test_api_end_to_end.py` | Nine endpoints; optional-dependency endpoints return 503 not 500 |
| 12b | Docker | `deployment/Dockerfile`, `docker-compose.yml` | `docker build -f deployment/Dockerfile .` | files | Not built locally (no daemon); built and smoke-tested in CI |
| 12c | GitHub Actions CI | `.github/workflows/tests.yml` | – | workflow | Test job (compile, repo guard, pytest) and Docker build+smoke job. Local clean-venv run passed; hosted CI run passed (test + docker jobs) |
| 12d | Structured, privacy-aware logs | `src/omnisupport/serving/monitoring.py`, `src/omnisupport/utils/privacy.py` | `pytest tests/unit/test_privacy_and_logging.py` | `monitoring/sample_request_log.jsonl` | Redaction + pseudonymised ids; no raw message text |
| 12e | Populated monitoring | `scripts/16_monitoring_workload.py` | `python scripts/16_monitoring_workload.py` | `monitoring/monitoring_summary.csv`, `endpoint_breakdown.json`, `docs/monitoring_and_operations.md` | 47 requests, avg 780.5 ms, 2 failures; generated by real calls through the app, not hand-written |
| 13 | README, architecture, runbook, evidence map, viva | `README.md`, `reports/architecture.md`, `docs/runbook.md`, this file, `docs/viva_notes.md`, `reports/final_report.md` | – | – | – |

## Quality gates

| Gate | Evidence |
|---|---|
| No leakage | `reports/leakage_register.md`; enforced in `src/omnisupport/data/leakage.py`; tests |
| Held-out evaluation | Temporal test split evaluated once; thresholds chosen on validation |
| No instructor or hidden assets | `.gitignore` excludes them; `scripts/repo_guard.py` runs in CI; they were never read |
| No secrets | `.env.example` only; `scripts/repo_guard.py` scans for key shapes |
| Real fine-tuning | `artifacts/transformer_finetune/training_log.csv` |
| Real monitoring | `monitoring/monitoring_summary.csv` |
| Safe agent boundary | `ApprovalGate`; `executed` is always False; `reports/agent_report.md` |
| Tests and CI | Local clean-venv pass; hosted CI pending first push |
