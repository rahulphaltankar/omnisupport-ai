# OmniSupport AI

A connected, leakage-safe retail-support AI prototype: ticket escalation and resolution-time prediction, segmentation, tabular and vision deep learning, NLP and transformer fine-tuning, a validated LLM layer, a cited policy RAG with abstention, a human-gated support agent, an API/UI, CI and real monitoring output.

Everything below was produced by the scripts in this repository. Numbers are quoted from `artifacts/` and `reports/`; limitations are stated next to them. The supplied assignment README is preserved in `docs/assignment_brief_readme.md`.

> Raw data, images and large model weights are **not** committed (see `data/README.md`). Experiments need the supplied data; the API, tests and CI do not.

## Headline results

| Area | Result | Caveat |
|---|---|---|
| Escalation classifier (T0 features, temporal test) | PR-AUC 0.572, ROC-AUC 0.734, recall 0.776, precision 0.429 at threshold 0.24 | Moderate signal; threshold uses an assumed FN:FP cost of 3:1 |
| Resolution-time regression | MAE 3.49 h (−48 % vs median baseline), R² 0.31; 90 % interval coverage 0.898 | Heavy-tailed target |
| Tabular NN | PR-AUC 0.5727 | No better than classical models |
| Segmentation (k=6) | silhouette 0.186, ARI 0.784 | Weak separation; personas are descriptive |
| Vision (CNN vs ResNet18) | both 1.0 on clean test; ResNet18 more robust to blur/rotation/darkness; CNN F1 0.52 vs ResNet 0.99 with 10 % data | 72 synthetic test images |
| NLP text classification | TF-IDF/BoW and DistilBERT 1.0 on clean test; noisy text: TF-IDF 0.991 vs DistilBERT 0.902 | Templated text; ceiling effect |
| LLM benchmark (30 requests/model) | rules 0.6 ms, 83 %; Qwen2.5-0.5B 11.7 s, 90 %; Qwen2.5-1.5B 38.2 s, 93 % | Cost is an estimate; no hosted API exercised; both Qwen models failed the refund-approval check before the policy floor was added; the table is the re-run with the floor |
| RAG | cited-source hit 14/15; 11/12 unanswerable probes abstained | Two account-takeover cases missed |
| Agent | 15/15 tool and approval decisions correct; 0 actions executed | Deterministic planner |
| Tests | 74 passed in a clean Python 3.13 venv | Hosted CI and Docker build+smoke passed on GitHub (run 45193973913) |

## Repository layout

```text
src/omnisupport/   data, ml, nlp, vision, llm, rag, agent, serving, utils
scripts/           numbered experiment entry points (01 … 16) and repo_guard.py
starter_code/      supplied contract files, now thin adapters over src/
app/               FastAPI (app/api) and Gradio (app/ui)
prompts/           versioned prompts + manifest of content hashes
knowledge_base/    supplied policy documents (RAG corpus)
artifacts/         generated metrics, tables, figures
reports/           problem framing, per-area reports, risk register, decision log, evidence map, final report
docs/              runbook, monitoring and operations, viva notes
monitoring/        populated monitoring summary and sample structured log
tests/             unit, integration and supplied starter tests
deployment/        Dockerfile, docker-compose
```

Start with `reports/final_report.md`, then `reports/evidence_map.md` (requirement → code → command → evidence), `reports/architecture.md`, `docs/runbook.md` and `docs/viva_notes.md`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ci.txt        # API + tests
pip install -r requirements-serve.txt     # + torch, transformers, FAISS, Gradio (experiments, dense RAG, vision, UI)
$env:PYTHONPATH = "src"
copy .env.example .env                    # optional; defaults need no key
```

Linux/macOS: `export PYTHONPATH=src`. The pinned baseline is Python 3.12; the experiments were run on Python 3.14 with slightly different library builds (decision D15).

## Verify without the data

```powershell
python scripts/repo_guard.py     # no secrets, raw data or instructor assets in the tree
pytest tests                     # data-dependent tests skip with an explicit reason
```

## Run the services

```powershell
uvicorn app.api.main:app --port 8000       # http://localhost:8000/docs
python app/ui/app.py                        # Gradio UI
docker build -f deployment/Dockerfile -t omnisupport .
docker run -p 8000:8000 omnisupport
```

Endpoints: `/health`, `/tickets/predict`, `/tickets/extract`, `/rag/ask`, `/agent/run`, `/agent/approve`, `/vision/predict`, `/monitoring/summary`, `/meta/categories`. The default LLM provider is the offline `rules` provider; set `LLM_PROVIDER` and the keys documented in `.env.example` to use a model. Keys are read from the environment only. Free-tier hosted models work through `LLM_PROVIDER=openai_compat` (any OpenAI-style endpoint such as Mistral La Plateforme or Groq; see `.env.example`). No paid service is required anywhere in this project.

## Reproduce the evidence

Place the supplied data as described in `data/README.md`, then from the repository root:

```powershell
python scripts/01_data_audit.py
python scripts/02_eda.py
python scripts/03_ml_classification.py
python scripts/04_ml_regression.py
python scripts/05_segmentation.py
python scripts/06_tabular_nn.py
python scripts/07_vision.py
python scripts/08_nlp_classical.py
python scripts/09_attention.py
python scripts/10_finetune_transformer.py     # real DistilBERT fine-tune, ~20 min on CPU
python scripts/11_pretrained_pipelines.py
python scripts/12_llm_benchmark.py            # downloads two small Qwen models
python scripts/13_rag_eval.py
python scripts/14_agent_eval.py
python scripts/15_subgroup_analysis.py
python scripts/16_monitoring_workload.py
```

Each script writes under `artifacts/`, `reports/` or `monitoring/` and records seeds and library versions in `run_metadata.json`. Full timings and outputs per step are in `docs/runbook.md`.

## Safety design in one paragraph

Prediction is defined at ticket creation (T0), and fields that only exist after triage or resolution are rejected in code. Sensitive attributes are never features. LLM output must pass a Pydantic schema, one repair turn and a deterministic fallback, and a policy floor may only make safety fields stricter. RAG answers are extractive, cited and abstain when evidence is weak, conflicting or when a secret is requested. The agent only proposes; anything that changes state goes through a named-human approval gate and nothing is executed. Logs redact personal data and pseudonymise identifiers.

## Known limitations

* Data is synthetic and templated: text-classification and vision scores are ceiling-bound and do not predict real-world accuracy.
* Classification and regression signal is moderate; the model is a triage aid.
* LLM cost figures are estimates from an assumed compute rate; the hosted-model path (`anthropic` provider) is covered by stub tests only because no key was available.
* The 20 % partial-refund rule in the agent is an assumed placeholder that needs business confirmation.
* The Docker image was built and smoke-tested only in hosted CI, not locally.
* Fairness checks cover the recorded attributes only; small subgroups have wide intervals.

## Run the benchmark against a free hosted model (Mistral)

1. Create a free account at https://console.mistral.ai, choose the free "Experiment" plan, and create an API key under "API keys".
2. In PowerShell, from the repository root: `powershell -ExecutionPolicy Bypass -File scripts\run_mistral_benchmark.ps1`. It asks for the key (hidden), runs the 30 requests and writes `artifacts\llm_benchmark_mistral\`.
