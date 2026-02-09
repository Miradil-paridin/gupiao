# run_fetch_evidence.py
from __future__ import annotations

from datetime import date
from pathlib import Path
import pandas as pd

from quant.fetch_evidence import run_fetch_evidence, EvidenceConfig


def guess_symbols_from_market_daily(base_dir: Path) -> list[str]:
    # try to locate your merged daily file
    cand_dir = base_dir / "data" / "market" / "daily"
    if not cand_dir.exists():
        raise FileNotFoundError(f"Cannot find {cand_dir}. Please check your data folder.")

    # pick newest parquet/csv that contains a 'symbol' column
    files = sorted(
        list(cand_dir.glob("*.parquet")) + list(cand_dir.glob("*.csv")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for f in files:
        try:
            if f.suffix == ".parquet":
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)
            if "symbol" in df.columns:
                syms = sorted(df["symbol"].dropna().astype(str).unique().tolist())
                return syms
        except Exception:
            continue
    raise RuntimeError("Could not find a market daily file containing a 'symbol' column.")


def main():
    base_dir = Path(__file__).resolve().parent
    as_of = date.today()  # or manually set: date(2026, 1, 30)

    symbols = guess_symbols_from_market_daily(base_dir)
    print("Symbols:", symbols)

    cfg = EvidenceConfig(
        cninfo_days=180,
        max_news_items=60,
        max_reports_items=80,
        pause_seconds=0.7,
    )

    manifest_path = run_fetch_evidence(symbols=symbols, base_dir=base_dir, as_of=as_of, cfg=cfg)
    print(f"Evidence manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
