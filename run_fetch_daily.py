from __future__ import annotations

from pathlib import Path
import os

import yaml

from quant.fetch_daily import fetch_daily_for_watchlist
from quant.logger import setup_logger

# --- hard disable proxies for this process ---
for k in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
]:
    os.environ.pop(k, None)

os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""
os.environ["NO_PROXY"] = "*"


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    # Setup logging
    log_file = base_dir / "data" / "logs" / "fetch_daily.log"
    logger = setup_logger("quant", log_file=log_file)

    # 优先读 config_v31.yaml
    cfg_path = base_dir / "config.yaml"
    for name in ["config_v31.yaml", "config.yaml"]:
        p = base_dir / name
        if p.exists():
            cfg_path = p
            break
    print(f"📂 配置: {cfg_path.name}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    codes = cfg["watchlist"]
    md = cfg["market_data"]

    # Provider configuration — 兼容两种格式
    providers_cfg = cfg.get("providers", {})

    # 格式1: daily: [baostock, akshare, sina]
    if "daily" in providers_cfg and isinstance(providers_cfg["daily"], list):
        daily_list = providers_cfg["daily"]
        primary_provider = daily_list[0] if daily_list else "baostock"
        fallback_providers = daily_list[1:] if len(daily_list) > 1 else []
    # 格式2: primary: baostock, fallback: [akshare]
    else:
        primary_provider = providers_cfg.get("primary", "baostock")
        fallback_providers = providers_cfg.get("fallback", [])

    logger.info(f"Starting daily data fetch for {len(codes)} symbols")
    logger.info(f"Primary provider: {primary_provider}, Fallback: {fallback_providers}")
    print(f"📊 股票数: {len(codes)}")
    print(f"🔌 数据源: {primary_provider} (主) + {fallback_providers} (备)")

    combined = fetch_daily_for_watchlist(
        base_dir=base_dir,
        codes=codes,
        start_date=md.get("start_date", "2015-01-01"),
        end_date=md.get("end_date", None),
        adjust=md.get("adjust", "qfq"),
        overwrite=bool(md.get("overwrite", False)),
        primary_provider=primary_provider,
        fallback_providers=fallback_providers,
    )

    logger.info("Fetch completed")
    print("\nDone.")
    print(f"Combined rows: {len(combined):,}")
    if not combined.empty:
        print("Date range:", combined["date"].min(), "→", combined["date"].max())
        print("Symbols:", sorted(combined["symbol"].unique().tolist()))


if __name__ == "__main__":
    main()