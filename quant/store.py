from __future__ import annotations

from pathlib import Path
import pandas as pd

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def symbol_paths(base_dir: Path, symbol: str) -> tuple[Path, Path]:
    raw_dir = base_dir / "data" / "raw" / "market_daily"
    clean_dir = base_dir / "data" / "clean" / "market_daily"
    ensure_dir(raw_dir)
    ensure_dir(clean_dir)
    raw_csv = raw_dir / f"{symbol}.csv"
    clean_parquet = clean_dir / f"{symbol}.parquet"
    return raw_csv, clean_parquet

def read_existing_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

def write_outputs(raw_csv: Path, clean_parquet: Path, df: pd.DataFrame) -> None:
    # Raw CSV (easy debugging)
    df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    # Clean parquet (fast analytics)
    df.to_parquet(clean_parquet, index=False)
