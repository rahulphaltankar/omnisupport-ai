"""Release guard: fails if the repository would publish secrets, raw primary data/images, instructor-only material or oversized files.

Usage:  python scripts/repo_guard.py            (scans git-tracked files if inside a git repo, else every non-ignored file)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
SECRET_PATTERNS = {
    "anthropic/openai-style key": re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}"),
    "huggingface token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "github token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    "aws access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "assigned api key": re.compile(r"(?i)\b(?:api[_-]?key|secret|token)\s*[=:]\s*['\"][A-Za-z0-9_\-]{24,}['\"]"),
}
FORBIDDEN_PATH = [r"(^|/)Instructor", r"(^|/)hidden_evaluation", r"(^|/)data/raw/", r"(^|/)data/images/", r"(^|/)data/subsets/", r"(^|/)data/processed/", r"(^|/)\.env$", r"\.parquet$", r"\.(zip|tar|gz)$",
                  r"\.(pt|pth|bin|safetensors|ckpt)$"]
ALLOWED_EXCEPTIONS = {"models/tabular_nn/"}  # small committed checkpoints (< 5 MB)
SKIP_SUFFIX = {".png", ".jpg", ".jpeg", ".joblib", ".pt", ".pdf", ".ico"}


def candidate_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
        return [ROOT / p for p in out if p]
    except Exception:
        return [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


def main() -> int:
    problems = []
    for p in candidate_files():
        rel = p.relative_to(ROOT).as_posix()
        if not p.exists():
            continue
        if rel == "scripts/repo_guard.py":
            continue
        if any(re.search(f, rel) for f in FORBIDDEN_PATH) and not any(rel.startswith(a) for a in ALLOWED_EXCEPTIONS):
            problems.append(f"forbidden path: {rel}")
            continue
        if p.stat().st_size > MAX_BYTES:
            problems.append(f"file > 5MB: {rel}")
        if p.suffix.lower() in SKIP_SUFFIX:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pat in SECRET_PATTERNS.items():
            m = pat.search(text)
            if m and not (rel == ".env.example"):
                problems.append(f"possible {name} in {rel}: {m.group(0)[:8]}...")
    if problems:
        print("REPO GUARD FAILED:\n  " + "\n  ".join(problems))
        return 1
    print(f"repo guard OK ({len(candidate_files())} files scanned: no secrets, raw data, images, instructor material or oversized files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
