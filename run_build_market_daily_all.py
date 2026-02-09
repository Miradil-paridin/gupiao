from __future__ import annotations

from pathlib import Path
from quant.aggregate import build_market_daily_all

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    all_df = build_market_daily_all(base_dir)

    print("\nDone.")
    print(f"Rows: {len(all_df):,}")
    print("Date range:", all_df["date"].min(), "->", all_df["date"].max())
    print("Symbols:", sorted(all_df["symbol"].unique().tolist()))

if __name__ == "__main__":
    main()
