"""Structured (JSON-lines) logging with privacy-aware field handling."""
from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from omnisupport.config import LOG_DIR
from omnisupport.utils.privacy import redact

_FORBIDDEN_KEYS = {"api_key", "authorization", "password", "token", "secret"}
_lock = threading.Lock()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage(), 500),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(sanitise_fields(extra))
        return json.dumps(payload, default=str)


def sanitise_fields(fields: dict) -> dict:
    """Drop forbidden keys and redact any string value. Never log raw user text."""
    clean = {}
    for k, v in fields.items():
        if k.lower() in _FORBIDDEN_KEYS:
            continue
        clean[k] = redact(v, 200) if isinstance(v, str) else v
    return clean


def get_logger(name: str = "omnisupport", log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if getattr(logger, "_omni_configured", False):
        return logger
    logger.setLevel(level)
    logger.propagate = False
    fmt = JsonFormatter()
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file or (LOG_DIR / "omnisupport.jsonl"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:  # read-only filesystem (e.g. hardened container): stderr only
        pass
    logger._omni_configured = True  # type: ignore[attr-defined]
    return logger


def append_jsonl(path: Path | str, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sanitise_fields(record), default=str) + "\n")
