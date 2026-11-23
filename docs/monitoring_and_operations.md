# Monitoring and operations

Course-sized, local, and real: the numbers in `monitoring/monitoring_summary.csv` are produced by
`python scripts/16_monitoring_workload.py`, which drives a representative workload through the actual API and exports the counters kept by
`src/omnisupport/serving/monitoring.py`. Nothing in the CSV is typed by hand.

## What is recorded (and what is not)

| Signal (CSV column) | Source | Regression it detects | Alert threshold (proposed) | First response |
|---|---|---|---|---|
| `request_count` | every API call that reaches a handler | traffic drop (upstream outage) or spike (abuse / loop) | ±50% vs same hour last week | check gateway, rate-limit |
| `average_latency_ms` | wall time per handler | slow model load, CPU starvation, retrieval index too large | > 2x baseline for 15 min | check host CPU, cold start, switch to lighter backend |
| `failure_count` | handler raised 4xx(non-validation)/5xx | dependency failure, bad deploy | > 2% of requests | roll back last release |
| `rag_no_answer_rate` | RAG abstentions / RAG requests | knowledge gap (rate up) or over-confident retrieval (rate collapses toward 0 on unsupported probes) | outside baseline band ±10 pts | review unanswered questions, update policy docs, re-calibrate threshold |
| `agent_tool_failure_count` | tool exceptions (after one retry) | order store/outage, schema drift in a tool | any sustained non-zero | page owner, agent already degrades safely |
| `model_version` | `config.MODEL_VERSION` | silent model swap / stale artifact | changes without a release note | block / investigate |
| `prompt_version` | `config.PROMPT_VERSION` | prompt edited without a version bump | change without matching manifest hash (CI test) | revert prompt |

Per-endpoint request/failure/latency is in `monitoring/endpoint_breakdown.json` and `GET /monitoring/summary`.

**Privacy.** The request log (`logs/requests.jsonl`, sample in `monitoring/sample_request_log.jsonl`) keeps only: timestamp, a random request id, endpoint, ok flag, latency,
model/prompt versions and a few non-text counters (e.g. text length). Raw customer text, order ids, names, emails and secrets are never written; any free-text field passes through
`redact()` and keys such as `api_key`/`authorization`/`password` are dropped (`tests/unit/test_privacy_and_logging.py`, `test_monitoring.py`).
Pseudonymous identifiers use a salted hash (`pseudonymise`). Application logs go to stdout/stderr as JSON so a log shipper can collect them without touching disk in the read-only container.

## Quality monitoring beyond counters

* **Escalation model**: weekly compare mean predicted probability with the observed escalation rate (calibration gap; baseline ECE 0.004 on validation) and recall at the chosen threshold on resolved tickets. A gap > 0.03 or recall drop > 5 points triggers retraining review.
* **Drift**: population stability index on `delivery_delay_days`, `order_value`, `previous_ticket_count`, and channel/category mix; PSI > 0.2 is a review trigger.
* **LLM**: schema-valid-first-attempt rate and fallback rate (`StructuredResult.status`); a rise in `fallback`/`repaired` after a prompt or model change is the earliest sign of regression. The prompt regression tests run in CI.
* **RAG**: re-run `scripts/13_rag_eval.py` after any knowledge-base change; hit@k and the unsupported-probe abstention rate must not fall below the committed baseline.
* **Agent**: approval-required rate; any trace where a state-changing action has `executed: true` is a Sev-1 (it must always be false in this prototype).

## Feedback loop

Agents can mark an AI suggestion as wrong via the (stubbed) `/agent/approve` decision (reject + note). Rejected proposals and "no answer" RAG questions are exported weekly to the
policy owner (new FAQ / policy gaps) and to the ML owner (labelled hard cases for the next validation set, never for tuning on the frozen test set).

## Incident response (short runbook)

1. **Detect**: alert from a threshold above or a user report. 2. **Contain**: set `LLM_PROVIDER=rules` (offline deterministic path) and/or disable the endpoint at the gateway; the API
degrades to 503 per component instead of failing as a whole. 3. **Diagnose**: `GET /monitoring/summary`, `logs/requests.jsonl`, agent trace (`trace` field) and the approval ledger `logs/approvals.jsonl`.
4. **Recover**: roll back the container image tag; model/prompt/index versions are all in `/health`. 5. **Review**: add a regression test (prompt case, RAG probe or agent case) before closing.

## Rollback and versioning

* Code + prompts + index metadata + model metadata are versioned together (`PROJECT_VERSION`, `PROMPT_VERSION`, `INDEX_VERSION`, `MODEL_VERSION`, `AGENT_VERSION` in `src/omnisupport/config.py`; surfaced by `/health`).
* `prompts/manifest.json` stores a content hash per prompt; CI fails if a prompt changes without a manifest update.
* `models/rag/index_metadata.json` stores per-document SHA-256 prefixes, chunking strategy, embedding model and the calibrated abstention threshold.
* Rollback = redeploy the previous image tag (`docker run omnisupport-ai:<previous>`). No database migrations exist.

## Known gaps (honest limits)

* Counters are in-process (reset on restart); for production use a metrics backend (Prometheus/OpenTelemetry). The CSV export is the course-level equivalent.
* No real traffic: the workload is the public evaluation set plus own probes, so latency numbers describe this laptop-class CPU, not production.
* Anthropic provider path is implemented and unit-tested with a stub client but was **not** exercised against the live API (no key available); its cost is therefore not reported.
