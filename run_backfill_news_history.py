from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv


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


def _load_watchlist_codes(base_dir: Path, watchlist_file: str) -> list[str]:
    p = Path(watchlist_file)
    if not p.is_absolute():
        p = base_dir / p
    if not p.exists():
        raise FileNotFoundError(f"watchlist file not found: {p}")

    text = p.read_text(encoding="utf-8", errors="ignore")
    raw_codes = re.findall(r"(\d{6})", text)
    if not raw_codes:
        try:
            obj = yaml.safe_load(text) or {}
        except Exception:
            obj = {}
        if isinstance(obj, dict):
            wl = obj.get("watchlist", []) or []
        elif isinstance(obj, list):
            wl = obj
        else:
            wl = []
        raw_codes = [str(c).split(".")[0] for c in wl if str(c).strip()]

    out = sorted(set(str(c).zfill(6) for c in raw_codes if str(c).strip()))
    return out


def _trading_days_from_data(base_dir: Path, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> list[pd.Timestamp]:
    candidates = [
        base_dir / "data" / "features" / "features_daily.parquet",
        base_dir / "data" / "clean" / "market_daily_all.parquet",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["date"])
        except Exception:
            try:
                df = pd.read_parquet(p)
            except Exception:
                continue
        if "date" not in df.columns or df.empty:
            continue
        s = pd.to_datetime(df["date"], errors="coerce").dropna()
        s = s[(s >= start_dt) & (s <= end_dt)]
        days = sorted(s.dt.normalize().unique().tolist())
        if days:
            return [pd.Timestamp(d) for d in days]
    return []


def _build_dates(base_dir: Path, start_date: str, end_date: str, trading_days_only: bool) -> list[date]:
    start_dt = pd.to_datetime(start_date, errors="coerce")
    end_dt = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt):
        raise ValueError(f"invalid date range: {start_date} ~ {end_date}")
    if start_dt > end_dt:
        raise ValueError(f"start_date > end_date: {start_date} > {end_date}")

    dates: list[pd.Timestamp] = []
    if trading_days_only:
        dates = _trading_days_from_data(base_dir, start_dt, end_dt)
    if not dates:
        dates = list(pd.bdate_range(start=start_dt, end=end_dt))
    return [d.date() for d in dates]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill daily news history for sentiment factor research")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end-date", default="", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--config", default="", help="config path (default: config.yaml, fallback: config_v31.yaml)")
    ap.add_argument(
        "--watchlist",
        default="watchlist_cache.yaml",
        help="watchlist file used for NEWS_SYMBOLS (default: watchlist_cache.yaml if exists)",
    )
    ap.add_argument("--force", action="store_true", help="refetch even if manifest already exists")
    ap.add_argument("--max-days", type=int, default=0, help="limit processed dates (0 means no limit)")
    ap.add_argument("--max-symbols", type=int, default=0, help="limit watchlist symbols (0 means no limit)")
    ap.add_argument("--latest-first", action="store_true", help="process latest dates first")
    ap.add_argument("--sleep-sec", type=float, default=0.0, help="sleep between dates to reduce pressure")
    ap.add_argument(
        "--all-calendar-days",
        action="store_true",
        help="use all weekdays range instead of market trading days from local data",
    )
    return ap.parse_args()


def main() -> None:
    load_dotenv()
    base_dir = Path(__file__).resolve().parent
    args = _parse_args()

    end_date = args.end_date.strip() or date.today().isoformat()
    cfg_path = _resolve_config_path(base_dir, args.config)

    watchlist_codes: list[str] = []
    watchlist_arg = str(args.watchlist or "").strip()
    if watchlist_arg:
        try:
            watchlist_codes = _load_watchlist_codes(base_dir, watchlist_arg)
        except FileNotFoundError:
            # fallback to config watchlist if explicit file not found
            watchlist_codes = []

    dates = _build_dates(
        base_dir=base_dir,
        start_date=args.start_date,
        end_date=end_date,
        trading_days_only=not bool(args.all_calendar_days),
    )
    if bool(args.latest_first):
        dates = list(reversed(dates))
    if args.max_days and args.max_days > 0:
        dates = dates[: int(args.max_days)]

    if not dates:
        print("No dates to backfill.")
        return

    if watchlist_codes and int(args.max_symbols) > 0:
        watchlist_codes = watchlist_codes[: int(args.max_symbols)]

    env_base = os.environ.copy()
    env_base["PYTHONIOENCODING"] = "utf-8"
    env_base["PIPELINE_CONFIG"] = str(cfg_path)

    print("=" * 60)
    print("News History Backfill")
    print("=" * 60)
    print(f"Config: {cfg_path.name}")
    print(f"Range : {dates[0]} -> {dates[-1]} ({len(dates)} days)")
    if bool(args.latest_first):
        print("Order : latest-first")
    print(f"Force : {'yes' if args.force else 'no'}")
    if watchlist_codes:
        print(f"Pool  : watchlist {len(watchlist_codes)} symbols ({watchlist_arg})")
    else:
        print("Pool  : fallback to config watchlist")

    news_root = base_dir / "data" / "news"
    ok = 0
    skipped = 0
    failed = 0

    for idx, d in enumerate(dates, start=1):
        as_of = d.isoformat()
        manifest = news_root / as_of / "manifest.json"
        if manifest.exists() and (not args.force):
            skipped += 1
            print(f"[{idx}/{len(dates)}] {as_of} skip (exists)")
            continue

        env_run = env_base.copy()
        env_run["NEWS_AS_OF"] = as_of
        if watchlist_codes:
            env_run["NEWS_SYMBOLS"] = ",".join(watchlist_codes)

        cmd = [sys.executable, str(base_dir / "run_fetch_news.py"), "--as-of", as_of]
        print(f"[{idx}/{len(dates)}] {as_of} fetch ...")
        proc = subprocess.run(cmd, cwd=str(base_dir), env=env_run, text=True)
        if proc.returncode == 0:
            ok += 1
        else:
            failed += 1
            print(f"  ⚠ failed: {as_of} (exit={proc.returncode})")

        if args.sleep_sec > 0:
            time.sleep(float(args.sleep_sec))

    print("\nSummary")
    print(f"  ok={ok} | skipped={skipped} | failed={failed} | total={len(dates)}")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
