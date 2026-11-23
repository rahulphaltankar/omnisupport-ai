"""Course-sized monitoring: thread-safe in-process counters + privacy-aware JSONL request log + CSV summary export.

Signals (each maps to a failure mode, see docs/monitoring_and_operations.md):
request_count, average_latency_ms, failure_count, rag_no_answer_rate, agent_tool_failure_count, model_version, prompt_version.
Raw user text is never stored: only a pseudonymous request id, endpoint, status, latency and component outcomes.
"""
from __future__ import annotations

import csv
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from omnisupport.config import LOG_DIR, MODEL_VERSION, MONITORING, PROMPT_VERSION
from omnisupport.utils.logging_utils import sanitise_fields

SUMMARY_FIELDS = ["request_count", "average_latency_ms", "failure_count", "rag_no_answer_rate", "agent_tool_failure_count", "model_version", "prompt_version"]


@dataclass
class MonitoringSummary:
    request_count: int = 0
    average_latency_ms: float = 0.0
    failure_count: int = 0
    rag_no_answer_rate: float = 0.0
    agent_tool_failure_count: int = 0
    model_version: str = MODEL_VERSION
    prompt_version: str = PROMPT_VERSION


class Monitor:
    def __init__(self, log_path: Path | None = None):
        self._lock = threading.Lock()
        self.log_path = log_path if log_path is not None else LOG_DIR / "requests.jsonl"
        self.reset()

    def reset(self):
        with self._lock:
            self.n = self.fail = self.rag_n = self.rag_no = self.agent_tool_fail = 0
            self.lat_sum = 0.0
            self.by_endpoint: dict[str, dict] = {}

    def record(self, endpoint: str, latency_ms: float, ok: bool = True, *, rag_no_answer: bool | None = None, agent_tool_failures: int = 0, extra: dict | None = None) -> str:
        rid = uuid.uuid4().hex[:12]
        with self._lock:
            self.n += 1
            self.lat_sum += latency_ms
            self.fail += 0 if ok else 1
            if rag_no_answer is not None:
                self.rag_n += 1
                self.rag_no += int(rag_no_answer)
            self.agent_tool_fail += agent_tool_failures
            e = self.by_endpoint.setdefault(endpoint, {"n": 0, "failures": 0, "lat_sum": 0.0})
            e["n"] += 1; e["failures"] += 0 if ok else 1; e["lat_sum"] += latency_ms
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "request_id": rid, "endpoint": endpoint, "ok": ok, "latency_ms": round(latency_ms, 2), "model_version": MODEL_VERSION, "prompt_version": PROMPT_VERSION,
               **sanitise_fields(extra or {})}
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError:
            pass
        return rid

    def summary(self) -> MonitoringSummary:
        with self._lock:
            return MonitoringSummary(self.n, round(self.lat_sum / self.n, 2) if self.n else 0.0, self.fail, round(self.rag_no / self.rag_n, 4) if self.rag_n else 0.0, self.agent_tool_fail)

    def endpoint_table(self) -> dict:
        with self._lock:
            return {k: {"requests": v["n"], "failures": v["failures"], "avg_latency_ms": round(v["lat_sum"] / v["n"], 2)} for k, v in self.by_endpoint.items()}

    def export(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path else MONITORING / "monitoring_summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = asdict(self.summary())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader(); w.writerow({k: row[k] for k in SUMMARY_FIELDS})
        return path


MONITOR = Monitor()
