"""Central configuration: paths, versions, seeds and environment handling.

Secrets are read from environment variables only (optionally loaded from a local,
git-ignored ``.env``). Nothing in this module ever prints or persists a secret.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is a supplied dependency, but keep imports optional for slim images
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(ROOT / ".env", override=False)

# ---------------------------------------------------------------- versions ---
# Every artefact that can regress is versioned explicitly (see reports/model_registry.md).
PROJECT_VERSION = "1.0.0"
PROMPT_VERSION = "prompts-v1.0.0"
MODEL_VERSION = "omnisupport-models-v1.0.0"
INDEX_VERSION = "policy-index-v1.0.0"
AGENT_VERSION = "agent-v1.0.0"

SEED = 42

# ------------------------------------------------------------------- paths ---
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"          # git-ignored cache
SUBSET_PATH = DATA_DIR / "subsets" / "transformer_finetune_10000.parquet"
IMAGE_LABELS = DATA_DIR / "image_labels.csv"
AGENT_STORE = DATA_DIR / "agent_store"
KNOWLEDGE_BASE = ROOT / "knowledge_base"
EVAL_DIR = ROOT / "evaluation"
PROMPTS_DIR = ROOT / "prompts"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
MONITORING = ROOT / "monitoring"
LOG_DIR = ROOT / "logs"                          # git-ignored

SHARD_GLOB = "support_records_part_*.parquet"
EXPECTED_SHARDS = 10
EXPECTED_ROWS = 1_000_000

# Operational policy constants (sourced from knowledge_base/09_refund_limits.md).
FRONTLINE_REFUND_LIMIT_GBP = 100.0
PAYMENTS_RISK_REFUND_LIMIT_GBP = 500.0


def artifact_dir(name: str) -> Path:
    p = ARTIFACTS / name
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class LLMSettings:
    """LLM provider settings. The API key is never logged or serialised."""

    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "rules").lower())
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", ""))
    timeout_s: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_S", "60")))
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_NEW_TOKENS", "200")))

    @property
    def api_key(self) -> str | None:  # property so it is not part of repr/asdict
        key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if key and key.startswith("replace_with"):
            return None
        return key

    def public_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "timeout_s": self.timeout_s,
            "max_new_tokens": self.max_new_tokens,
            "api_key_configured": bool(self.api_key),
        }
