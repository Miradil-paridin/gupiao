from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_market_daily_all(base_dir: Path) -> pd.DataFrame:
    """
    Read all per-symbol parquet files from:
      data/clean/market_daily/*.parquet
    Merge them into one dataframe and write:
      data/clean/market_daily_all.parquet
    """
    in_dir = base_dir / "data" / "clean" / "market_daily"
    out_path = base_dir / "data" / "clean" / "market_daily_all.parquet"

    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {in_dir}")

    dfs: list[pd.DataFrame] = []
    for fp in files:
        df = pd.read_parquet(fp)
        if df is None or df.empty:
            continue
        # Normalize date type
        df["date"] = pd.to_datetime(df["date"]).dt.date
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No data loaded (all files empty?)")

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = (
        all_df.drop_duplicates(["symbol", "date"], keep="last")
             .sort_values(["symbol", "date"])
             .reset_index(drop=True)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(out_path, index=False)

    return all_df
