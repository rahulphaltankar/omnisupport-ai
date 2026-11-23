"""Reproducibility helpers: seeding, run metadata, deterministic JSON writing."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnisupport.config import ROOT, SEED


def set_seed(seed: int = SEED) -> int:
    """Seed python, numpy and (if installed) torch for repeatable runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:  # pragma: no cover
        pass
    return seed


def _lib_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "sklearn", "torch", "torchvision", "transformers", "pydantic", "fastapi", "faiss"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            continue
    return out


def _git_sha() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def _json_default(o: Any):
    try:
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return o.as_posix()
    return str(o)


def write_json(path: Path | str, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path | str, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


class RunRecord:
    """Context manager that writes ``run_metadata.json`` for an experiment.

    Captures seed, configuration, split method, model version, library versions,
    wall-clock duration and free-form limitations so every artefact is traceable.
    """

    def __init__(self, out_dir: Path, name: str, *, seed: int = SEED, config: dict | None = None,
                 split_method: str = "n/a", model_version: str | None = None, limitations: list[str] | None = None):
        self.out_dir = Path(out_dir)
        self.meta: dict[str, Any] = {
            "run": name,
            "seed": seed,
            "config": config or {},
            "split_method": split_method,
            "model_version": model_version,
            "limitations": limitations or [],
            "extra": {},
        }
        self._t0 = 0.0

    def __enter__(self) -> "RunRecord":
        self._t0 = time.perf_counter()
        self.meta["started_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        set_seed(self.meta["seed"])
        return self

    def add(self, **kw) -> None:
        self.meta["extra"].update(kw)

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.meta["duration_s"] = round(time.perf_counter() - self._t0, 2)
        self.meta["status"] = "failed" if exc_type else "completed"
        self.meta["libraries"] = _lib_versions()
        self.meta["git_commit"] = _git_sha()
        self.meta["hardware"] = {"cpu_count": os.cpu_count(), "platform": platform.platform(), "gpu": _gpu()}
        write_json(self.out_dir / "run_metadata.json", self.meta)
        return False


def _gpu() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)"
    except Exception:
        return "unknown"


def main_guard() -> None:  # pragma: no cover
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
