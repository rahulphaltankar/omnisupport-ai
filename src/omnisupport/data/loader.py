"""Reproducible loading and validation of the ten primary Parquet shards.

The *primary* dataset is exactly ``data/raw/support_records_part_001..010.parquet``
(10 x 100,000 = 1,000,000 rows). The preview CSV, the transformer subset, images and
the agent store are supporting assets and are never counted towards the primary total.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from omnisupport.config import EXPECTED_ROWS, EXPECTED_SHARDS, RAW_DIR, SHARD_GLOB

SHARD_RE = re.compile(r"^support_records_part_(\d{3})\.parquet$")
ROWS_PER_SHARD = EXPECTED_ROWS // EXPECTED_SHARDS


class DataValidationError(AssertionError):
    """Raised when the primary dataset does not match the documented contract."""


def discover_shards(raw_dir: Path | str = RAW_DIR) -> list[Path]:
    """Explicit-pattern shard discovery; exactly ten contiguous shards are required."""
    raw_dir = Path(raw_dir)
    paths = sorted(raw_dir.glob(SHARD_GLOB))
    names = [p.name for p in paths]
    bad = [n for n in names if not SHARD_RE.match(n)]
    if bad:
        raise DataValidationError(f"Unexpected shard file names: {bad}")
    if len(paths) != EXPECTED_SHARDS:
        raise DataValidationError(f"Expected {EXPECTED_SHARDS} primary shards in {raw_dir}, found {len(paths)}")
    idx = [int(SHARD_RE.match(n).group(1)) for n in names]
    if idx != list(range(1, EXPECTED_SHARDS + 1)):
        raise DataValidationError(f"Shard numbering is not contiguous 001..{EXPECTED_SHARDS:03d}: {idx}")
    return paths


def shard_row_counts(paths: list[Path]) -> dict[str, int]:
    """Row counts from Parquet footers only (no data read)."""
    import pyarrow.parquet as pq

    return {p.name: pq.ParquetFile(p).metadata.num_rows for p in paths}


def validate_counts(counts: dict[str, int]) -> int:
    """Pure validation of the row-count contract (testable without the data files)."""
    if len(counts) != EXPECTED_SHARDS:
        raise DataValidationError(f"Expected {EXPECTED_SHARDS} shards, got {len(counts)}")
    total = sum(counts.values())
    if total != EXPECTED_ROWS:
        raise DataValidationError(f"Expected exactly {EXPECTED_ROWS:,} rows, got {total:,}")
    wrong = {k: v for k, v in counts.items() if v != ROWS_PER_SHARD}
    if wrong:
        raise DataValidationError(f"Shards not at {ROWS_PER_SHARD:,} rows: {wrong}")
    return total


def validate_schemas(paths: list[Path]) -> list[str]:
    import pyarrow.parquet as pq

    ref = pq.ParquetFile(paths[0]).schema_arrow
    for p in paths[1:]:
        s = pq.ParquetFile(p).schema_arrow
        if s.names != ref.names:
            raise DataValidationError(f"Schema mismatch in {p.name}")
    return list(ref.names)


def iter_shards(columns: list[str] | None = None, raw_dir: Path | str = RAW_DIR):
    for p in discover_shards(raw_dir):
        yield p.name, pd.read_parquet(p, columns=columns)


def load_primary(columns: list[str] | None = None, raw_dir: Path | str = RAW_DIR, validate: bool = True) -> pd.DataFrame:
    """Load all ten shards, assert the exact 1,000,000-row contract, tag the source shard."""
    paths = discover_shards(raw_dir)
    if validate:
        validate_counts(shard_row_counts(paths))
        validate_schemas(paths)
    frames = []
    for p in paths:
        df = pd.read_parquet(p, columns=columns)
        df["_shard"] = int(SHARD_RE.match(p.name).group(1))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if validate and columns is None and len(out) != EXPECTED_ROWS:
        raise DataValidationError(f"Loaded {len(out):,} rows, expected {EXPECTED_ROWS:,}")
    return out
