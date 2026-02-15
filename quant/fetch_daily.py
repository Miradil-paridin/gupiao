from __future__ import annotations

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import pandas as pd
from tqdm import tqdm

from .symbols import normalize_symbol, code_only
from .normalize import normalize_daily_data
from .store import symbol_paths, read_existing_parquet, write_outputs
from .providers import get_provider, ProviderError
from .logger import get_logger

logger = get_logger("quant.fetch_daily")

RECENT_UPDATE_DAYS = 3

# ── 新增：防限速配置 ──
MAX_WORKERS = 2          # 增量更新并发数
FETCH_DELAY = 1.0        # 每次请求后等待1秒
MAX_RETRIES = 2          # 单只股票最多重试2次（provider层已有5次重试）
RETRY_DELAY = 3.0        # 重试间隔3秒


def _parse_date(s: str | None) -> dt.date | None:
    if s is None:
        return None
    return dt.date.fromisoformat(str(s))


def _today() -> dt.date:
    return dt.date.today()


def _cached_fetch(
    provider_name: str,
    code6: str,
    start: dt.date,
    end: dt.date,
    adjust: str,
) -> pd.DataFrame:
    """不用lru_cache了 — 大股票池会撑爆内存"""
    provider = get_provider(provider_name)
    return provider.fetch_daily(code6=code6, start=start, end=end, adjust=adjust)


def _fetch_with_fallback(
    providers: Sequence[str],
    code6: str,
    start: dt.date,
    end: dt.date,
    adjust: str,
) -> pd.DataFrame:
    """Try fetching from providers in order with retry."""
    errors = []

    for provider_name in providers:
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(FETCH_DELAY)  # 防限速
                df = _cached_fetch(
                    provider_name,
                    code6=code6,
                    start=start,
                    end=end,
                    adjust=adjust,
                )
                if df is not None and not df.empty:
                    return df
                else:
                    logger.warning(f"Empty data from {provider_name} for {code6}")
                    break  # 空数据不重试，换provider
            except ProviderError as e:
                logger.warning(f"Provider {provider_name} failed for {code6} (attempt {attempt+1}): {e}")
                errors.append(f"{provider_name}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))  # 递增等待
            except Exception as e:
                logger.warning(f"Unexpected error from {provider_name} for {code6} (attempt {attempt+1}): {e}")
                errors.append(f"{provider_name}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

    raise ProviderError(
        f"All providers failed for {code6}. Errors: {'; '.join(errors)}"
    )


def _compute_fetch_start(
    start: dt.date,
    end: dt.date,
    existing: pd.DataFrame,
    overwrite: bool,
) -> dt.date:
    if overwrite or existing.empty:
        return start
    recent_start = end - dt.timedelta(days=RECENT_UPDATE_DAYS - 1)
    return max(start, recent_start)


def _fetch_one_symbol(
    base_dir: Path,
    code: str,
    start: dt.date,
    end: dt.date,
    adjust: str,
    overwrite: bool,
    providers: Sequence[str],
) -> pd.DataFrame | None:
    symbol = normalize_symbol(code)
    raw_csv, clean_parquet = symbol_paths(base_dir, symbol)

    existing = read_existing_parquet(clean_parquet)
    fetch_start = _compute_fetch_start(start, end, existing, overwrite)

    if fetch_start > end:
        logger.info(f"{symbol} is up-to-date, skipping fetch")
        return existing

    code6 = code_only(symbol)

    try:
        raw_df = _fetch_with_fallback(
            providers=providers,
            code6=code6,
            start=fetch_start,
            end=end,
            adjust=adjust,
        )
        new_df = normalize_daily_data(raw_df, symbol=symbol)
    except ProviderError as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        if not existing.empty:
            return existing
        return None

    if existing.empty:
        out = new_df
    else:
        out = pd.concat([existing, new_df], ignore_index=True)
        out = (
            out.drop_duplicates(["symbol", "date"], keep="last")
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )

    # Persist
    write_outputs(raw_csv, clean_parquet, out)
    return out


def fetch_daily_for_watchlist(
    base_dir: Path,
    codes: list[str],
    start_date: str,
    end_date: str | None,
    adjust: str = "qfq",
    overwrite: bool = False,
    primary_provider: str = "akshare",
    fallback_providers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch & persist per-symbol parquet/csv, also return combined dataframe.
    Incremental by default. Supports multiple data sources with fallback.
    降低并发 + 加延时 + 加重试，防止被BaoStock/AKShare限速。
    """
    start = _parse_date(start_date) or dt.date(2010, 1, 1)
    end = _parse_date(end_date) or _today()

    provider_names = [primary_provider] + (fallback_providers or [])
    providers: list[str] = []

    for name in provider_names:
        try:
            get_provider(name)
            providers.append(name)
        except ValueError:
            logger.warning(f"Skipping unknown provider: {name}")

    if not providers:
        raise ValueError("No valid providers configured")

    logger.info(f"Using providers: {providers}")

    # ── 统计新股票和已有股票 ──
    new_codes = []
    existing_codes = []
    for code in codes:
        symbol = normalize_symbol(code)
        _, clean_parquet = symbol_paths(base_dir, symbol)
        existing = read_existing_parquet(clean_parquet)
        if existing.empty:
            new_codes.append(code)
        else:
            existing_codes.append(code)

    total_new = len(new_codes)
    total_existing = len(existing_codes)
    print(f"  📊 已有数据: {total_existing} 只 (增量更新)")
    print(f"  📊 新股票: {total_new} 只 (全量抓取，较慢)")
    if total_new > 100:
        est_minutes = total_new * 2 / 60  # 每只约2秒
        print(f"  ⏱️ 预计新股票耗时: {est_minutes:.0f}-{est_minutes*2:.0f} 分钟")

    all_out: list[pd.DataFrame] = []
    failed: list[str] = []

    # ── 先抓已有股票（增量更新，串行）──
    if existing_codes:
        print(f"\n  🔄 增量更新 {total_existing} 只...")
        for code in tqdm(existing_codes, desc="增量更新"):
            try:
                out = _fetch_one_symbol(
                    base_dir, code, start, end,
                    adjust, overwrite, providers,
                )
                if out is not None:
                    all_out.append(out)
                else:
                    failed.append(code)
            except Exception as e:
                logger.error(f"Failed {code}: {e}")
                failed.append(code)

    # ── 再抓新股票（慢，全量，串行防限速）──
    if new_codes:
        print(f"\n  📥 全量抓取 {total_new} 只新股票（串行，防限速）...")
        success_count = 0
        fail_count = 0
        for i, code in enumerate(tqdm(new_codes, desc="全量抓取")):
            try:
                out = _fetch_one_symbol(
                    base_dir, code, start, end,
                    adjust, overwrite, providers,
                )
                if out is not None:
                    all_out.append(out)
                    success_count += 1
                else:
                    failed.append(code)
                    fail_count += 1
            except Exception as e:
                logger.error(f"Failed {code}: {e}")
                failed.append(code)
                fail_count += 1

            # 每50只打印一次进度
            if (i + 1) % 50 == 0:
                print(f"     进度: {i+1}/{total_new} | 成功: {success_count} | 失败: {fail_count}")

        print(f"  ✅ 全量抓取完成: 成功 {success_count} | 失败 {fail_count}")

    # ── 统计 ──
    if failed:
        print(f"\n  ⚠️ 失败: {len(failed)} 只")
        if len(failed) <= 20:
            print(f"     {failed}")
        else:
            print(f"     前20只: {failed[:20]}")
        # 保存失败列表，方便重试
        failed_path = base_dir / "data" / "logs" / "fetch_failed.txt"
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.write_text("\n".join(failed), encoding="utf-8")
        print(f"     失败列表已保存: {failed_path}")

    if not all_out:
        return pd.DataFrame()

    combined = pd.concat(all_out, ignore_index=True)
    combined = (
        combined.drop_duplicates(["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    return combined
