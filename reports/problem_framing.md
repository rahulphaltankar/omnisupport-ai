# Problem framing — OmniSupport AI

## 1. Context
OmniSupport is an online retailer taking support requests through web chat, email, mobile app and phone. Operations leaders want an
AI-assisted layer that (a) forecasts which tickets will be hard, (b) understands ticket text and return images, (c) answers policy
questions with evidence, and (d) lets agents use *read-only* tools safely, with a human approving anything that changes money or data.
Everything here is a prototype on **synthetic** data; no real customer is affected.

## 2. Users and the decision each one makes

| User | Decision supported | Component | Why a model helps | Cost of a wrong output |
|---|---|---|---|---|
| Support operations lead | Which new tickets to route to the specialist queue first; how many specialist hours to staff | Escalation classifier + resolution-time regressor | Escalations are 31% of tickets and take ~9h longer; early routing shortens waits | Missed escalation = slow resolution, unhappy customer; false alarm = wasted specialist time |
| Customer-support agent | What the policy says and what to do next | Policy RAG with citations + agent tools | 40 overlapping policies; agents quote them inconsistently | Wrong policy quoted = wrong refund/return promise |
| Returns team | Whether a returned item photo shows damage / wrong item | CNN / transfer-learning classifier | Photos arrive unlabelled at volume | Wrong class sends a damaged item back to stock (or wrongly rejects a customer) |
| Customer-service leader | Drivers of resolution time; recurring demand segments | EDA + clustering | Plans staffing and process fixes | Over-claiming causality leads to wrong investment |
| Approver (senior agent / payments-risk) | Approve or reject proposed refunds > GBP 100 | Agent approval gate | Keeps humans accountable for money | Unauthorised refund / fraud exposure |
| Engineer / on-call | Is the service healthy and which version is live | API, logs, monitoring | — | Silent regression |

## 3. Prediction timing (the single most important framing choice)
**T0 = the moment a ticket is created, before any human has read it.** At T0 the system knows the customer profile, the order, the
channel, the timestamp, the customer's own title/message and whether they attached a file. It does **not** know the triage label,
priority, the later sentiment review, refunds, CSAT, 7-day status or final disposition. The full field-by-field register is in
[`leakage_register.md`](leakage_register.md) and is enforced in code (`omnisupport.data.leakage`).

## 4. Targets and metrics

| Task | Target | Type | Primary metric | Secondary | Baseline to beat | Success threshold (pre-registered) |
|---|---|---|---|---|---|---|
| Classification | `escalated` (31.0% positive) | binary | **PR-AUC** (class-aware) | ROC-AUC, recall at chosen threshold, precision, F1, expected-cost, calibration (Brier) | Dummy prior (PR-AUC = base rate 0.31) | PR-AUC ≥ 0.50 and ROC-AUC ≥ 0.70 on the held-out test, with recall ≥ 0.60 at the selected threshold |
| Regression | `resolution_time_hours` (median 23h, heavy tail) | continuous | **MAE** (hours) | RMSE, median-AE, error by slice, 90% interval coverage | Predict train median (MAE 6.8h) | ≥ 30% MAE reduction vs median baseline |
| Text classification | `issue_category` (10 classes) from the message | multi-class | **macro-F1** | accuracy, per-class P/R, robustness to noise | Majority class | macro-F1 ≥ 0.90 (templated text, see risks) |
| Vision | return-image class (4 classes) | multi-class | **macro-F1** on a stratified held-out test | confusion matrix, per-class recall | Majority class (25%) | macro-F1 ≥ 0.80 |
| Segmentation | ticket segments | unsupervised | silhouette + stability (ARI) + business plausibility | — | k=2 | silhouette > 0.2 and ARI > 0.7 across seeds |
| RAG | grounded answers | retrieval + generation | **retrieval hit@3** against hinted source; **abstention accuracy** | MRR, answer-fact coverage, citation accuracy, latency | Lexical TF-IDF retrieval | hit@3 ≥ 0.9; 100% abstention on unsupported probes |
| Agent | safe tool use | rule-checked | **policy-violation count = 0** | tool-selection accuracy, missing-info handling, failure handling | — | 0 unapproved sensitive actions; all 8 public cases pass |
| LLM app | structured output | pipeline | **schema validity rate** | latency, tokens, estimated cost, task quality | — | ≥ 90% valid after repair/fallback; ≥ 20 benchmarked requests |
| Operations | service | system | request count, mean latency, failures | RAG no-answer rate, agent tool failures, versions | — | populated monitoring summary from a real workload |

The thresholds are written **before** the final test run; results report pass/fail honestly even when a threshold is missed.

## 5. Business-cost framing for the escalation threshold
A missed escalation (false negative) delays a hard ticket; a false alarm (false positive) spends specialist triage time on a ticket
that would have been fine. With explicitly stated, adjustable assumptions (FN cost = 3 units, FP cost = 1 unit, because a late
specialist hand-off compounds into repeat contacts) the threshold is chosen on the **validation** period by minimising expected cost
and is then frozen before touching the test period. The cost ratio is a parameter, not a finding, and the threshold curve is saved so
operations can re-pick it.

## 6. In scope / out of scope
**In scope:** analysis of the 1,000,000-record extract; two T0 predictive models; segmentation; tabular NN; return-image classifier;
text classification, semantic search, attention demonstration, pretrained pipelines and a fine-tuned transformer; versioned prompts with
validated structured output and a ≥20-request benchmark; policy RAG; a local read-only agent with approval gates; FastAPI + Gradio demo;
Docker; GitHub Actions; structured privacy-aware logs and a monitoring summary.

**Out of scope:** real payments, real customer data, irreversible external actions, live carrier integrations, production-scale
vector stores, multilingual support (the `preferred_language` field is used for fairness evaluation only), model retraining
automation, causal claims, legal advice.

## 7. Assumptions (A#) and dependencies
- **A1** All data is synthetic; relationships are generator properties and may not hold in reality.
- **A2** Tickets are independent enough for a temporal split; customer overlap across periods is measured (~96% of later-period customers also appear in training) and reported rather than hidden.
- **A3** `delivery_delay_days` is the delay observed when the ticket is opened. If it were the *final* delay it would be partly post-outcome; its text mirror ("It is around N days beyond…") agrees with the field (r≈0.83 on rows that state it).
- **A4** `attachment_available` reflects what the customer attached at creation.
- **A5** `issue_category` and `priority` are assigned by humans at triage, hence excluded at T0.
- **A6** The 15,000-order agent store is the system of record for tools; it has no seller/channel field, so marketplace-specific rules route to manual review.
- **A7** No LLM API key is available in the build environment; LLM-dependent evidence uses real local instruction models (cost = estimated compute cost). The Anthropic provider path is implemented and unit-tested with a stub client but **not** exercised live.
- **Dependencies:** pinned `requirements.txt` baseline (Python 3.12 reference); locally the work ran on Python 3.14 with slightly older library builds, so exact decimals can differ by environment (documented in `decision_log.md`).

## 8. Data boundaries
- **Primary data:** exactly `data/raw/support_records_part_001..010.parquet` = 1,000,000 rows (validated in `omnisupport.data.loader`).
- **Supporting, never counted as primary:** `dataset_preview.csv`, `subsets/transformer_finetune_10000.parquet`, `images/` (480), `agent_store/*.csv`, `knowledge_base/*.md`, `evaluation/*.json`.
- **Not used:** any instructor-only or hidden-evaluation material; none is present in this repository.
- **Not committed:** raw shards, images and the fine-tuning subset (see `.gitignore`); the small synthetic agent store and knowledge base are committed because the agent, RAG, Docker image and CI tests need them.

## 9. Risk themes (full register in `risk_register.md`)
Leakage → T0 register + guards + leak-demonstration; overfitting → temporal hold-out touched once; subgroup harm → evaluation-only
sensitive attributes with confidence intervals; hallucination → extractive grounded answers + abstention; unsafe automation → proposals
only, human approval; privacy → redacted, pseudonymised structured logs; staleness → versioned prompts/indices/models with rollback.
