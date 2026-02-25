from __future__ import annotations

import os
from pathlib import Path


VALID_BACKENDS = {"parquet", "hybrid", "duckdb"}


def get_data_backend() -> str:
    raw = str(os.getenv("QUANT_DATA_BACKEND", "parquet")).strip().lower()
    if raw not in VALID_BACKENDS:
        return "parquet"
    return raw


def is_duckdb_enabled() -> bool:
    return get_data_backend() in {"hybrid", "duckdb"}


def resolve_duckdb_path(base_dir: Path) -> Path:
    raw = str(os.getenv("QUANT_DUCKDB_PATH", "")).strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = base_dir / p
        return p
    return base_dir / "data" / "quant.db"

