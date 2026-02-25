from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant.data_backend import resolve_duckdb_path
from quant.data_manager import DataManager


def _load_source_df(base_dir: Path) -> pd.DataFrame:
    agg_path = base_dir / "data" / "clean" / "market_daily_all.parquet"
    if agg_path.exists():
        return pd.read_parquet(agg_path)

    per_symbol_dir = base_dir / "data" / "clean" / "market_daily"
    files = sorted(per_symbol_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            "No market source found. Expected data/clean/market_daily_all.parquet "
            "or data/clean/market_daily/*.parquet"
        )
    chunks: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_parquet(f)
        if df is not None and not df.empty:
            chunks.append(df)
    if not chunks:
        raise RuntimeError("All source parquet files are empty.")
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate market_daily parquet data into DuckDB.")
    ap.add_argument("--base-dir", default=".", help="project root")
    ap.add_argument("--db-path", default="", help="duckdb path (default: data/quant.db or QUANT_DUCKDB_PATH)")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    db_path = Path(args.db_path).resolve() if str(args.db_path).strip() else resolve_duckdb_path(base_dir)

    src = _load_source_df(base_dir)
    mgr = DataManager(db_path=db_path)
    try:
        n = mgr.upsert_market_daily(src)
    finally:
        mgr.close()

    print("DuckDB migration done.")
    print(f"db   : {db_path}")
    print(f"rows : {n}")


if __name__ == "__main__":
    main()

