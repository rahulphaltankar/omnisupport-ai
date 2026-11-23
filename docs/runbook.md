# Runbook: reproduce, run, demo, operate

All commands run from the repository root (the folder containing `pyproject.toml`). PowerShell shown; replace `$env:PYTHONPATH="src"` with `export PYTHONPATH=src` on Linux/macOS.

## 0. Environment

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements-ci.txt          # API + tests only (CI set, ~1 min)
pip install -r requirements-serve.txt       # + torch, transformers, FAISS, Gradio (for dense RAG, vision, UI, the experiments)
$env:PYTHONPATH = "src"
copy .env.example .env                      # optional; the defaults need no key
```

The pinned baseline is Python 3.12 (`requirements*.txt`); the experiments in this repository were produced on Python 3.14 with slightly older library builds (see `run_metadata.json` in each artifact folder for exact versions). CI and the clean-environment test use Python 3.12 / 3.13 with `requirements-ci.txt`.

## 1. Data (not committed)

Place the supplied files under `data/` as described in `data/README.md` (`raw/support_records_part_001..010.parquet`, `subsets/transformer_finetune_10000.parquet`, `images/`, `image_labels.csv`; `agent_store/` is committed). Scripts that need raw data fail fast with a clear message if it is missing, and the tests that need it skip.

## 2. Reproduce the evidence (in this order)

| Step | Command | Writes |
|---|---|---|
| Audit + cleaning + leakage register | `python scripts/01_data_audit.py` | `artifacts/data_audit/`, `reports/leakage_register.md`, `data/processed/clean.parquet` |
| EDA | `python scripts/02_eda.py` | `artifacts/eda/`, `reports/eda_report.md` |
| Classification (frozen test, once) | `python scripts/03_ml_classification.py` | `artifacts/ml_classification/`, `models/serving/escalation_*` |
| Regression | `python scripts/04_ml_regression.py` | `artifacts/ml_regression/` |
| Segmentation | `python scripts/05_segmentation.py` | `artifacts/segmentation/`, `reports/segmentation_report.md` |
| Tabular NN | `python scripts/06_tabular_nn.py` | `artifacts/tabular_nn/`, `models/tabular_nn/*.pt` |
| Vision | `python scripts/07_vision.py` | `artifacts/vision/`, `models/vision/` |
| BoW vs TF-IDF, semantic search | `python scripts/08_nlp_classical.py` | `artifacts/nlp_classical/`, `reports/nlp_classical.md` |
| Attention Q/K/V | `python scripts/09_attention.py` | `artifacts/attention/`, `reports/attention_explained.md` |
| Real DistilBERT fine-tune (~10–25 min CPU) | `python scripts/10_finetune_transformer.py` | `artifacts/transformer_finetune/`, `reports/transformer_finetuning.md` |
| Pretrained pipelines | `python scripts/11_pretrained_pipelines.py` | `reports/pretrained_pipelines.md` |
| LLM benchmark (30 requests × 3 providers) | `python scripts/12_llm_benchmark.py` | `artifacts/llm_benchmark/` |
| RAG chunking + calibration + evaluation | `python scripts/13_rag_eval.py` | `artifacts/rag/`, `models/rag/index_metadata.json` |
| Agent evaluation | `python scripts/14_agent_eval.py` | `artifacts/agent/` |
| Subgroup analysis | `python scripts/15_subgroup_analysis.py` | `artifacts/subgroups/`, `reports/subgroup_report.md` |
| Monitoring workload | `python scripts/16_monitoring_workload.py` | `monitoring/monitoring_summary.csv`, `endpoint_breakdown.json`, `sample_request_log.jsonl` |

`scripts/run_chain.sh 04 05 06 …` runs several of these sequentially and logs to `logs/`.

## 3. Test

```powershell
pytest -q tests                  # unit + integration + starter contracts
python scripts/repo_guard.py     # secrets / raw data / images / oversized files
```

## 4. Run the service

```powershell
uvicorn app.api.main:app --port 8000      # API docs at http://127.0.0.1:8000/docs
python app/ui/app.py                      # Gradio UI at http://127.0.0.1:7860
```

Smoke test:

```powershell
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/rag/ask -H "content-type: application/json" -d '{"question":"Who must approve refunds above GBP 100?"}'
curl -X POST http://127.0.0.1:8000/agent/run -H "content-type: application/json" -d '{"request":"Issue a GBP 240 refund immediately for order ORD00001003."}'
```

Docker:

```powershell
docker build -f deployment/Dockerfile -t omnisupport-ai .
docker run --rm -p 8000:8000 omnisupport-ai
# or: docker compose -f deployment/docker-compose.yml up --build
```

Switching the LLM provider (default `rules`, offline): set `LLM_PROVIDER=local_hf` and `MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct` in `.env`, or `LLM_PROVIDER=anthropic` with `LLM_API_KEY` and a model id (and `LLM_PRICE_IN_PER_MTOK` / `LLM_PRICE_OUT_PER_MTOK` if you want a cost estimate). Keys are read from the environment only and are never logged or returned by `/health`.

## 5. Five-minute demo (script)

1. `GET /health` → versions (`model`, `prompt`, `index`, `agent`). 2. **Extraction**: `POST /tickets/extract` with *"My order ORD00001234 is 6 days late…"* → validated JSON; then with *"My parcel is late again"* → `order_id: null` + `missing_information`.
3. **RAG**: refund-timing question → cited answer with policy version; *"teleporting an order to Mars"* → "No answer: insufficient policy evidence". 4. **Conflict**: marketplace-vs-standard return window question → conflict flagged, human review.
5. **Agent**: "Check order ORD00001001…" (trace shows one tool call); "Is this item still eligible?" (asks for the order id, no tool); "Issue a GBP 240 refund…" (approval required, `executed: false`); approve with an empty/"agent" approver (rejected) and with a named human (recorded, still not executed).
6. **Escalation score**: `POST /tickets/predict`. 7. **Monitoring**: `GET /monitoring/summary` and `monitoring/monitoring_summary.csv`. 8. Open `reports/evidence_map.md` to show where each requirement is proved.

## 6. Operating notes

See `docs/monitoring_and_operations.md` (signals, thresholds, incident response, rollback).
