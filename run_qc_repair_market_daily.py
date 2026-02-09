from __future__ import annotations

from pathlib import Path

import yaml

from quant.qc_repair import qc_repair_market_daily
from quant.logger import setup_logger


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    
    # Setup logging
    log_file = base_dir / "data" / "logs" / "qc_repair.log"
    logger = setup_logger("quant", log_file=log_file)

    with open(base_dir / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    adjust = cfg.get("market_data", {}).get("adjust", "qfq")
    
    # Get provider from config
    providers_cfg = cfg.get("providers", {})
    provider_name = providers_cfg.get("primary", "akshare")
    
    logger.info(f"Starting QC repair with provider: {provider_name}")

    all_df = qc_repair_market_daily(
        base_dir=base_dir,
        adjust=adjust,
        refetch_if_invalid_over=10,  # refetch symbols with >=10 invalid rows
        provider_name=provider_name,
    )

    logger.info("QC repair completed")
    print("\nDone.")
    if not all_df.empty:
        print(f"Rows: {len(all_df):,}")
        print("Date range:", all_df["date"].min(), "->", all_df["date"].max())
        print("Symbols:", sorted(all_df["symbol"].unique().tolist()))


if __name__ == "__main__":
    main()
