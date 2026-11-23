# OmniSupport AI - Assessment-Ready Project Plan

## Delivery objective

Build a reproducible, safe retail-support AI prototype that demonstrates the complete course syllabus and provides visible evidence for every grading criterion. The target is the **Excellent** band: correct methodology, defensible decisions, reproducible runs, evaluation artefacts, safety controls, and a concise live demonstration that the team can explain.

## Non-negotiable quality gates

These are the first things to protect because they can invalidate otherwise strong work.

1. **No prediction-time leakage.** Establish the timestamp and fields available at each prediction point; exclude post-outcome columns from predictive features. Record the audit and enforce it in feature-building code.
2. **Honest evaluation.** Split before fitting transforms; prevent duplicate/group contamination; keep held-out data untouched until the final comparison.
3. **Real transformer fine-tuning.** Run actual training on the supplied 10,000-row subset (or document a genuine hardware-limited smaller sample) and compare on a held-out set against TF-IDF.
4. **Observable production evidence.** CI must run and monitoring must contain real output, not only a plan.
5. **Safe agent operation.** Local read tools only by default; sensitive/state-changing actions require explicit human approval, structured traces, and safe failure behaviour.
6. **Assessment integrity.** Never use instructor-only or hidden-evaluation assets in the submitted implementation, and do not hard-code answers.

## Working cadence and definition of done

Use a small vertical slice early: data -> one model -> API/UI -> test -> monitoring. Thereafter complete the epics below in order, keeping each story demonstrably runnable.

**Definition of done for every story**

- Code/script is reproducible from a documented command and fixed seed where relevant.
- Inputs, outputs, assumptions and limitations are recorded.
- At least one automated test or deterministic validation covers the changed contract where practical.
- Result tables/figures are saved under a clear `artifacts/` or `reports/` path and referenced in the README/report.
- No secrets, raw primary data, hidden instructor assets, or generated cache/model weights are committed unless explicitly required.

## Suggested repository structure

```text
src/                 reusable application and pipeline code
scripts/             executable data/model/evaluation entry points
notebooks/           concise, reproducible investigation narratives
tests/               unit, contract and regression tests
artifacts/           generated metrics, figures and small evaluation tables
reports/             final report, risk register, architecture and demo notes
models/              metadata/checkpoints (large files ignored or externally stored)
monitoring/          real monitoring summary and sample structured logs
```

## Jira-style backlog / WBS

Priority: **P0** = cannot achieve a high grade without it; **P1** = material score/evidence gain; **P2** = polish after the core is robust.

### EPIC 0 - Foundations, scope and reproducibility (P0)

**Story OS-01 - Establish an assessment-ready project skeleton**  
*As an assessor, I can install, run and inspect the project without guesswork.*

- Create the repository structure above, `.gitignore`, environment template, configuration pattern and deterministic seed utility.
- Preserve the supplied dependency baseline; document any version change and its reason.
- Write an initial README with setup, data placement, run order, architecture placeholder, and limitations.
- Add a `make`/script-style runbook or equivalent Windows-friendly commands.

**Acceptance evidence:** clean setup works; secrets are environment variables only; README run path works on a fresh environment.

**Story OS-02 - Define the product and measurement contract**  
*As a retail support lead, I need each component tied to a decision and a measurable outcome.*

- Define users: support operations lead, customer-support agent, and reviewer/approver.
- Select clearly prediction-time-valid targets for one regression and one classification task.
- Record prediction timestamp, business decision, metric targets, assumptions, risks, exclusions and system boundary.
- Publish metric definitions: e.g. MAE/RMSE for regression; PR-AUC, recall, precision, F1 and threshold-based cost for classification.

**Acceptance evidence:** `reports/problem_framing.md` has target definitions, success thresholds and a decision-to-component map.

### EPIC 1 - Data quality, EDA and features (P0)

**Story DATA-01 - Audit all primary data reproducibly**  
*As an engineer, I can prove the data pipeline reads exactly the ten primary Parquet shards and 1,000,000 rows.*

- Implement shard discovery with an explicit pattern; validate shard count and total row count.
- Profile schema, nulls and blanks, duplicate candidates, inconsistent categories, outliers, target imbalance and date ranges.
- Keep supporting data separate from the primary count.
- Save a compact audit table and data-quality figures.

**Acceptance evidence:** automated row-count assertion; `artifacts/data_audit.*`; data dictionary/assumption updates.

**Story DATA-02 - Create a leakage-safe cleaning and feature pipeline**  
*As a modeller, I can create the same valid features for train, test and production requests.*

- Define cleaning rules with before/after counts and rationale.
- Write a field-level leakage register: available-before-prediction, excluded-post-outcome, and allowed fields.
- Build numerical, categorical, temporal, interaction and text-derived features in fit/transform pipelines.
- Split data before learnt imputation/encoding/scaling; use temporal/group split if justified by the operational setting.

**Acceptance evidence:** leakage register; transform tests; saved feature schema; no fitted preprocessing on held-out data.

**Story DATA-03 - Produce decision-driving EDA**  
*As a stakeholder, I can see why the modelling choices are appropriate.*

- Create only useful plots: target distributions, missingness, key segment comparisons, relationships, outliers and imbalance.
- For every figure, write one observation and one modelling/business implication.

**Acceptance evidence:** concise EDA report with labelled figures, conclusions and limitations.

### EPIC 2 - Classical ML and evaluation (P0)

**Story ML-01 - Deliver a regression decision model**

- Establish naive and simple linear/tree baselines.
- Compare at least two appropriate pipelines; use MAE/RMSE and residual/error slices.
- Translate error into an operational limitation or use case.

**Story ML-02 - Deliver a classification prioritisation model**

- Establish a dummy baseline plus multiple suitable models.
- Handle class imbalance only inside train folds; report discrimination and class-aware metrics.
- Select and justify an operating threshold using a support-business trade-off.

**Story ML-03 - Validate and tune honestly**

- Use cross-validation and tune one pipeline with a documented search space.
- Produce confusion matrix, calibration/probability discussion if relevant, error analysis and subgroup slices.
- Freeze final held-out results once; do not repeatedly optimise on the test set.

**Acceptance evidence:** comparison table, CV results, chosen model rationale, threshold curve/table, held-out evaluation, and reproducible saved pipeline.

### EPIC 3 - Segmentation and tabular deep learning (P1)

**Story UNSUP-01 - Build actionable, cautious segmentation**

- Define the segmentation unit (customer or ticket) and features; scale appropriately.
- Compare a small range of cluster counts with silhouette/inertia and stability/business plausibility checks.
- Create descriptive personas as correlations, never causal claims.

**Acceptance evidence:** cluster-selection plot, profile table, cautious persona report and limitations.

**Story DL-01 - Implement a tabular neural-network experiment**

- Create a correct tensor/data-loader flow and train/validation loop.
- Compare at least two optimiser/regularisation settings; log learning curves and discuss overfitting.
- Save a checkpoint and metadata (seed, feature version, hyperparameters, metrics).

**Acceptance evidence:** learning curves, checkpoint, comparison table and explicit classical-vs-NN conclusion.

### EPIC 4 - Return-image computer vision (P1)

**Story CV-01 - Train and evaluate a baseline CNN**

- Build stratified train/validation/test splits; inspect labels and class counts.
- Train a basic CNN with augmentation; record training history.
- Report class-aware precision/recall/F1, confusion matrix and representative correct/error examples.

**Story CV-02 - Compare transfer learning**

- Fine-tune or use a pretrained vision backbone consistent with the course.
- Hold evaluation protocol constant against the baseline CNN; compare performance, training cost and failure patterns.

**Acceptance evidence:** both model runs, augmentation specification, class metrics, confusion matrices, error gallery and recommendation.

### EPIC 5 - NLP, attention and transformers (P0, 9-point rubric area)

**Story NLP-01 - Build classical text understanding and semantic search**

- Compare BoW and TF-IDF; train/evaluate a ticket classifier using leakage-safe splits.
- Generate embeddings and build semantic retrieval over support text; show good, weak and ambiguous retrieval examples.

**Story NLP-02 - Demonstrate attention correctly (Session 20)**

- Choose one realistic support message.
- Explain and visualise/work through query, key, value, scaled self-attention and positional information.
- Contrast contextual attention representations with sparse TF-IDF on the same message, including limitations.

**Story NLP-03 - Apply pretrained transformer pipelines**

- Demonstrate two suitable tasks (for example sentiment/classification and summarisation/extraction), with sample inputs, outputs and limitations.

**Story NLP-04 - Fine-tune a transformer (Session 22)**

- Use `transformer_finetune_10000.parquet`; define reproducible stratified train/validation/test partitions.
- Fine-tune a small Hugging Face sequence classifier for an appropriate label; log hardware, duration, seed, hyperparameters and learning/evaluation metrics.
- Compare the fine-tuned model fairly with TF-IDF on the same held-out examples; inspect errors and cost/latency trade-offs.
- If constrained, record the real limitation, selected smaller sample and completed training evidence.

**Acceptance evidence:** attention artifact, pipeline examples, training logs, saved model metadata, held-out TF-IDF comparison, error analysis and explanation suitable for viva.

### EPIC 6 - LLM application engineering (P0)

**Story LLM-01 - Create prompt library and safe structured outputs**

- Version prompts for extraction, classification and summarisation; define inputs, system constraints, examples and expected output.
- Validate Pydantic schemas; handle malformed/refused/unsupported responses without exceptions or invented certainty.
- Write prompt regression cases and failure analysis.

**Story LLM-02 - Benchmark performance (Session 25)**

- Execute at least 20 representative requests through the actual configured model/API path.
- Log request category, prompt/model version, latency, estimated input/output tokens, estimated cost, success/failure and output-quality review.
- Report aggregates and discuss a concrete trade-off: quality vs latency/cost/context/streaming/key handling/UX.

**Acceptance evidence:** immutable benchmark CSV with >=20 rows, summary table/chart, token/cost estimation method, prompt tests and trade-off decision.

### EPIC 7 - Grounded policy RAG (P0)

**Story RAG-01 - Build evidence-first retrieval and answers**

- Ingest all allowed policy/SOP documents with document/version metadata.
- Compare at least two chunk size/overlap approaches on retrieval quality and latency.
- Use embeddings and vector storage; return answer, sources, confidence/no-answer reason and retrieval trace.
- Implement abstention for unsupported or conflicting policy, rather than answering from general knowledge.

**Story RAG-02 - Evaluate RAG under normal and failure cases**

- Run supplied public RAG questions plus explicitly designed unsupported/conflicting cases.
- Score retrieval/grounding, citation accuracy, answer correctness, abstention behaviour and latency.

**Acceptance evidence:** chunking comparison, source-visible answers, evaluation table, no-answer cases and versioned index metadata.

### EPIC 8 - Controlled support agent (P0)

**Story AGENT-01 - Implement reliable local tool contracts**

- Implement read-only lookup-order, refund-calculation and return-eligibility tools against the supplied local stores.
- Publish strict schemas, validate arguments and normalise known error responses.
- Add deterministic unit/contract tests, including invalid IDs and missing data.

**Story AGENT-02 - Build a safe agent loop**

- Require missing information before calling a tool; do not infer customer/order identifiers.
- Capture tool selection, arguments, results, failures and final response in a trace.
- Gate sensitive actions behind explicit human approval; default to a proposed action, not execution.
- Run public agent cases and additional adversarial/error cases.

**Acceptance evidence:** tool schemas/tests, case results, human-approval flow, trace examples and failure-safe behaviour.

### EPIC 9 - Guardrails, responsible AI and quality assurance (P0)

**Story QA-01 - Build a cross-system evaluation suite**

- Consolidate ML, vision, NLP, LLM, RAG and agent metrics into one scorecard.
- Add regression tests for model/API/RAG/agent contracts; retain starter tests and extend them for implemented behaviour.
- Include hallucination, prompt-injection, unsupported-policy, tool-error and unsafe-action cases.

**Story GOV-01 - Document responsible-AI controls**

- Create a risk register: privacy/PII, bias/subgroups, safety, hallucination, retrieval staleness, automation risk, monitoring and mitigations/owners.
- Conduct meaningful subgroup analysis where dataset fields permit; state sample-size limits.
- Document data minimisation, retention, access controls, human oversight, auditability and escalation.

**Acceptance evidence:** passing expanded tests, evaluation scorecard, risk register, subgroup report and evidence of guardrail failures being handled.

### EPIC 10 - Product integration and production evidence (P0)

**Story INT-01 - Expose the integrated support workflow**

- Select a coherent minimal FastAPI and/or Streamlit/Gradio experience: ticket prediction, policy question with sources, and safe support-agent assistance.
- Validate requests with schemas; expose controlled 4xx-style/user-friendly errors for invalid or unsafe requests.
- Keep long inference responsive: status messages, timeouts and graceful fallbacks.

**Story OPS-01 - Package, test and observe the prototype**

- Make Docker build/run documentation work using the supplied Docker assets.
- Keep `.github/workflows/tests.yml` operational and add a CI status/evidence capture.
- Emit structured logs that exclude PII and include component/model/prompt versions.
- Produce actual `monitoring/monitoring_summary.csv` (or dashboard/log equivalent) with request count, average latency, failures, RAG no-answer rate, agent tool failures, model version and prompt version.
- Document alert thresholds, feedback loop, rollback/versioning and incident response.

**Acceptance evidence:** successful container command, passing CI evidence, structured log sample, populated monitoring summary, incident/feedback document.

### EPIC 11 - Final submission, demonstration and viva (P0)

**Story FINAL-01 - Publish a coherent technical narrative**

- Finalise README: problem, architecture, setup, run order, evidence map, component limitations, deployment and monitoring.
- Create an architecture diagram showing data, models, RAG, agent approval boundary, API/UI, logs and monitoring.
- Write a concise final report that links each claim to a saved result.

**Story FINAL-02 - Rehearse demonstrable outcomes**

- Prepare a 5-8 minute demo: business problem -> one predictive insight -> image/text capability -> sourced RAG -> safe agent -> monitoring/CI -> trade-offs.
- Prepare a viva sheet for data leakage, split strategy, metric/model choice, attention Q/K/V, fine-tuning, RAG source/abstention, agent approval, CI failure and monitoring signals.
- Run the supplied validation/checklist before packaging. Exclude instructor materials, hidden evaluation assets, raw data and secrets from the submission archive.

**Acceptance evidence:** final evidence map, recorded/demo runbook, completed checklist, clean fresh-clone smoke test and submission ZIP/repository review.

## Recommended execution sequence

| Sprint | Goal | Must exit with |
|---|---|---|
| 0 | Foundation and framing | OS-01, OS-02; clean environment and evidence map |
| 1 | Data truth | DATA-01 to DATA-03; leakage register and EDA |
| 2 | Core predictive slice | ML-01 to ML-03 plus basic API contract/tests |
| 3 | Advanced modelling | UNSUP-01, DL-01, CV-01/02 |
| 4 | High-value NLP | NLP-01 to NLP-04, especially attention and real fine-tuning |
| 5 | Generative systems | LLM-01/02, RAG-01/02, AGENT-01/02 |
| 6 | Quality and product | QA-01, GOV-01, INT-01, OPS-01 |
| 7 | Submission hardening | FINAL-01/02, clean-clone rehearsal and evidence audit |

If time is constrained, do not replace P0 evidence with more models. First complete an end-to-end, tested, monitored baseline, then improve experimental depth and presentation.

## Assessment evidence map (100 points)

| Rubric area | Points | Primary proof |
|---|---:|---|
| Problem framing | 5 | scope/target/metric document |
| Data quality | 6 | audit, cleaning pipeline, leakage register |
| EDA and features | 5 | annotated figures and valid feature pipeline |
| Classical ML | 8 | baselines/comparison and business choice |
| Evaluation/tuning | 6 | honest CV, threshold and error analysis |
| Clustering | 4 | validation and cautious personas |
| Deep learning | 6 | learning curves, comparison, checkpoint |
| Computer vision | 6 | CNN vs transfer-learning metrics/errors |
| NLP/attention/transformers | 9 | TF-IDF/search, QKV visual, pipelines, fine-tuning comparison |
| Prompt/LLM performance | 7 | schema/tests and >=20-request benchmark |
| RAG | 7 | chunk comparison, sourced evaluation, abstention |
| Agent/tools | 7 | schemas, traces, missing input/failure/approval cases |
| Evaluation/guardrails/testing | 6 | regression suite and failure evidence |
| Responsible AI | 5 | risk register, subgroup and oversight evidence |
| API/UI | 4 | working validated integrated flow |
| Deployment/CI/monitoring | 5 | Docker, CI, logs and populated metrics |
| Code/docs/reasoning | 4 | maintainability, report, architecture, viva readiness |

## Final pre-submission gate

- [ ] Every acceptance-evidence item above exists and is linked from the README/report.
- [ ] All public starter tests, new tests, and CI pass from a clean environment.
- [ ] Data row count is exactly 1,000,000 and supporting assets were not counted as primary data.
- [ ] Every headline metric can be regenerated; all splits, seeds and configuration are recorded.
- [ ] Attention explanation, actual fine-tuning run, 20+ LLM benchmark rows, and populated monitoring output are present.
- [ ] RAG visibly cites sources and abstains when evidence is absent/conflicting.
- [ ] Agent asks for missing details, handles tool failures and never executes sensitive actions without approval.
- [ ] README instructions have been tested by someone/fresh environment; Docker and CI proof are included.
- [ ] Submission contains no raw primary data/images unless requested, no secrets, no hidden instructor assets, and no hard-coded evaluation answers.

---

## Delivery status (appended at completion)

Status per story; evidence paths are in `reports/evidence_map.md`.

| Story | Status | Note |
|---|---|---|
| OS-01, OS-02 | Done | README, runbook, framing, decision log |
| DATA-01..03 | Done | Audit, leakage-safe pipeline, EDA |
| ML-01..03 | Done | Moderate signal, honest test |
| UNSUP-01, DL-01 | Done | NN gives no gain |
| CV-01, CV-02 | Done | Synthetic images, 72-image test |
| NLP-01..04 | Done | Templated text caveat; fine-tune is real (CPU) |
| LLM-01, LLM-02 | Done | Hosted API not exercised |
| RAG-01, RAG-02 | Done | Two takeover misses documented |
| AGENT-01, AGENT-02 | Done | Deterministic planner |
| QA-01, GOV-01 | Done | 74 tests; risk register R01–R24 |
| INT-01 | Done | API and UI |
| OPS-01 | Partly verified | Monitoring and tests done; Docker build and hosted CI pending |
| FINAL-01, FINAL-02 | Done locally | Public release awaits owner confirmation |
