from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

import yaml

from quant.fetch_daily import fetch_daily_for_watchlist
from quant.logger import setup_logger

# Hard-disable proxies for this process.
for k in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
]:
    os.environ.pop(k, None)

os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

# Force UTF-8 console I/O on Windows to avoid Unicode print failures.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _configure_socket_timeout() -> float:
    raw = os.getenv("FETCH_SOCKET_TIMEOUT", "30")
    try:
        timeout_s = float(raw)
    except Exception:
        timeout_s = 30.0
    timeout_s = max(5.0, timeout_s)
    socket.setdefaulttimeout(timeout_s)
    return timeout_s


def _resolve_config_path(base_dir: Path, config_arg: str = "") -> Path:
    explicit = str(config_arg or "").strip()
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = base_dir / p
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        return p

    env_cfg = str(os.getenv("PIPELINE_CONFIG", "")).strip()
    if env_cfg:
        p = Path(env_cfg)
        if not p.is_absolute():
            p = base_dir / p
        if p.exists():
            return p

    for name in ["config.yaml", "config_v31.yaml"]:
        p = base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError("no config file found (expected config.yaml or config_v31.yaml)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch daily market data for watchlist.")
    ap.add_argument("--config", default="", help="config path (default: config.yaml, fallback: config_v31.yaml)")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent

    log_file = base_dir / "data" / "logs" / "fetch_daily.log"
    logger = setup_logger("quant", log_file=log_file)

    socket_timeout = _configure_socket_timeout()
    logger.info(f"Global socket timeout set: {socket_timeout:.1f}s")

    cfg_path = _resolve_config_path(base_dir, config_arg=args.config)
    print(f"Config: {cfg_path.name}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    codes = cfg.get("watchlist", [])
    md = cfg.get("market_data", {}) or {}

    providers_cfg = cfg.get("providers", {}) or {}
    if "daily" in providers_cfg and isinstance(providers_cfg["daily"], list):
        daily_list = providers_cfg["daily"]
        primary_provider = daily_list[0] if daily_list else "baostock"
        fallback_providers = daily_list[1:] if len(daily_list) > 1 else []
    else:
        primary_provider = providers_cfg.get("primary", "baostock")
        fallback_providers = providers_cfg.get("fallback", [])

    logger.info(f"Starting daily data fetch for {len(codes)} symbols")
    logger.info(f"Primary provider: {primary_provider}, Fallback: {fallback_providers}")
    print(f"Socket timeout: {socket_timeout:.1f}s")
    print(f"Symbols: {len(codes)}")
    print(f"Providers: {primary_provider} + {fallback_providers}")

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
        print("Date range:", combined["date"].min(), "->", combined["date"].max())
        print("Symbols:", sorted(combined["symbol"].unique().tolist()))


if __name__ == "__main__":
    main()
