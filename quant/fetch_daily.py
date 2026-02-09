from __future__ import annotations

import datetime as dt
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


def _parse_date(s: str | None) -> dt.date | None:
    if s is None:
        return None
    return dt.date.fromisoformat(str(s))


def _today() -> dt.date:
    return dt.date.today()


@lru_cache(maxsize=128)
def _cached_fetch(
    provider_name: str,
    code6: str,
    start: dt.date,
    end: dt.date,
    adjust: str,
) -> pd.DataFrame:
    provider = get_provider(provider_name)
    return provider.fetch_daily(code6=code6, start=start, end=end, adjust=adjust)


def _fetch_with_fallback(
    providers: Sequence[str],
    code6: str,
    start: dt.date,
    end: dt.date,
    adjust: str,
) -> pd.DataFrame:
    """
    Try fetching from providers in order until one succeeds.
    
    Args:
        providers: List of providers to try
        code6: 6-digit stock code
        start: Start date
        end: End date
        adjust: Price adjustment type
    
    Returns:
        Raw DataFrame from first successful provider
    
    Raises:
        ProviderError: If all providers fail
    """
    errors = []
    
    for provider_name in providers:
        try:
            df = _cached_fetch(
                provider_name,
                code6=code6,
                start=start,
                end=end,
                adjust=adjust,
            )
            if df is not None and not df.empty:
                logger.info(f"Successfully fetched {code6} from {provider_name}")
                return df
            else:
                logger.warning(f"Empty data from {provider_name} for {code6}")
        except ProviderError as e:
            logger.warning(f"Provider {provider_name} failed for {code6}: {e}")
            errors.append(f"{provider_name}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error from {provider_name} for {code6}: {e}")
            errors.append(f"{provider_name}: {e}")
    
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
        # Already up-to-date
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
    
    Args:
        base_dir: Project base directory
        codes: List of stock codes to fetch
        start_date: Start date (ISO format)
        end_date: End date (ISO format, None = today)
        adjust: Price adjustment type (qfq/hfq/"")
        overwrite: If True, refetch all data ignoring existing
        primary_provider: Primary data provider name
        fallback_providers: List of fallback provider names
    
    Returns:
        Combined DataFrame of all fetched data
    """
    start = _parse_date(start_date) or dt.date(2010, 1, 1)
    end = _parse_date(end_date) or _today()

    # Initialize providers
    provider_names = [primary_provider] + (fallback_providers or [])
    providers: list[str] = []
    
    for name in provider_names:
        try:
            get_provider(name)
            providers.append(name)
        except ValueError as e:
            logger.warning(f"Skipping unknown provider: {name}")
    
    if not providers:
        raise ValueError("No valid providers configured")
    
    logger.info(f"Using providers: {providers}")

    all_out: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                _fetch_one_symbol,
                base_dir,
                code,
                start,
                end,
                adjust,
                overwrite,
                providers,
            )
            for code in codes
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fetching daily bars",
        ):
            out = future.result()
            if out is not None:
                all_out.append(out)

    if not all_out:
        return pd.DataFrame()

    combined = pd.concat(all_out, ignore_index=True)
    combined = combined.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)
    return combined
