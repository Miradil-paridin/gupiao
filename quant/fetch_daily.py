from __future__ import annotations

import datetime as dt
import importlib.util
import os
import time
from pathlib import Path
from typing import Sequence

import pandas as pd
from tqdm import tqdm

from .logger import get_logger
from .normalize import normalize_daily_data
from .providers import ProviderError, get_provider
from .store import read_existing_market_daily, symbol_paths, write_outputs
from .symbols import code_only, normalize_symbol

logger = get_logger("quant.fetch_daily")

RECENT_UPDATE_DAYS = 3
FETCH_DELAY = float(os.getenv("FETCH_DELAY_SECONDS", "1.0"))
MAX_RETRIES = max(1, int(os.getenv("FETCH_PROVIDER_ATTEMPTS", "2")))
RETRY_DELAY = float(os.getenv("FETCH_RETRY_DELAY_SECONDS", "3.0"))
PROGRESS_EVERY = max(1, int(os.getenv("FETCH_PROGRESS_EVERY", "20")))
PROVIDER_CIRCUIT_BREAKER = max(1, int(os.getenv("FETCH_PROVIDER_CIRCUIT_BREAKER", "12")))

PROVIDER_IMPORTS: dict[str, str] = {
    "baostock": "baostock",
    "akshare": "akshare",
}


def _parse_date(s: str | None) -> dt.date | None:
    if s is None:
        return None
    return dt.date.fromisoformat(str(s))


def _today() -> dt.date:
    return dt.date.today()


def _is_missing_dependency_error(msg: str) -> bool:
    text = (msg or "").lower()
    return any(
        token in text
        for token in ("not installed", "no module named", "modulenotfounderror")
    )


def _provider_dependency_available(provider_name: str) -> bool:
    module_name = PROVIDER_IMPORTS.get(provider_name)
    if not module_name:
        return True
    return importlib.util.find_spec(module_name) is not None


def _resolve_available_providers(provider_names: Sequence[str]) -> list[str]:
    providers: list[str] = []
    seen: set[str] = set()
    for name in provider_names:
        if name in seen:
            continue
        seen.add(name)
        try:
            get_provider(name)
        except ValueError:
            logger.warning(f"Skipping unknown provider: {name}")
            continue
        if not _provider_dependency_available(name):
            mod_name = PROVIDER_IMPORTS.get(name, name)
            logger.warning(f"Skipping provider {name}: dependency {mod_name} is not installed")
            continue
        providers.append(name)
    return providers


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
    provider_failures: dict[str, int],
    circuit_open: set[str],
) -> pd.DataFrame:
    """Try fetching from providers in order with retry and circuit breaker."""
    errors: list[str] = []

    for provider_name in providers:
        if provider_name in circuit_open:
            errors.append(f"{provider_name}: circuit-open")
            continue

        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(FETCH_DELAY)
                df = _cached_fetch(
                    provider_name,
                    code6=code6,
                    start=start,
                    end=end,
                    adjust=adjust,
                )
                if df is not None and not df.empty:
                    provider_failures[provider_name] = 0
                    return df
                logger.warning(f"Empty data from {provider_name} for {code6}")
                errors.append(f"{provider_name}: empty data")
                break
            except ProviderError as e:
                err = str(e)
                errors.append(f"{provider_name}: {err}")
                provider_failures[provider_name] = provider_failures.get(provider_name, 0) + 1
                logger.warning(
                    f"Provider {provider_name} failed for {code6} (attempt {attempt + 1}/{MAX_RETRIES}): {err}"
                )

                if _is_missing_dependency_error(err):
                    circuit_open.add(provider_name)
                    logger.error(f"Disabling provider {provider_name} due to missing dependency: {err}")
                    break
                if provider_failures[provider_name] >= PROVIDER_CIRCUIT_BREAKER:
                    circuit_open.add(provider_name)
                    logger.error(
                        f"Circuit open for provider {provider_name} after "
                        f"{provider_failures[provider_name]} failures"
                    )
                    break
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
            except Exception as e:
                err = str(e)
                errors.append(f"{provider_name}: {err}")
                provider_failures[provider_name] = provider_failures.get(provider_name, 0) + 1
                logger.warning(
                    f"Unexpected error from {provider_name} for {code6} "
                    f"(attempt {attempt + 1}/{MAX_RETRIES}): {err}"
                )
                if provider_failures[provider_name] >= PROVIDER_CIRCUIT_BREAKER:
                    circuit_open.add(provider_name)
                    logger.error(
                        f"Circuit open for provider {provider_name} after "
                        f"{provider_failures[provider_name]} failures"
                    )
                    break
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

    raise ProviderError(f"All providers failed for {code6}. Errors: {'; '.join(errors)}")


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


def _normalize_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize date column to pandas Timestamp to avoid mixed datetime/date sorting errors.
    """
    if df is None or df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out


def _fetch_one_symbol(
    base_dir: Path,
    code: str,
    start: dt.date,
    end: dt.date,
    adjust: str,
    overwrite: bool,
    providers: Sequence[str],
    provider_failures: dict[str, int],
    circuit_open: set[str],
) -> pd.DataFrame | None:
    symbol = normalize_symbol(code)
    raw_csv, clean_parquet = symbol_paths(base_dir, symbol)

    existing = read_existing_market_daily(
        base_dir=base_dir,
        symbol=symbol,
        parquet_path=clean_parquet,
    )
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
            provider_failures=provider_failures,
            circuit_open=circuit_open,
        )
        new_df = normalize_daily_data(raw_df, symbol=symbol)
        new_df = _normalize_date_col(new_df)
        existing = _normalize_date_col(existing)
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

    write_outputs(raw_csv, clean_parquet, out, base_dir=base_dir, symbol=symbol)
    return out


def _print_batch_progress(
    batch_name: str,
    idx: int,
    total: int,
    success_count: int,
    fail_count: int,
    start_ts: float,
) -> None:
    elapsed = time.time() - start_ts
    avg = elapsed / max(idx, 1)
    eta = max(total - idx, 0) * avg
    print(
        f"    [{batch_name}] {idx}/{total} | ok={success_count} fail={fail_count} "
        f"| elapsed={elapsed:.1f}s eta~{eta:.1f}s"
    )


def _run_batch(
    batch_name: str,
    base_dir: Path,
    codes: list[str],
    start: dt.date,
    end: dt.date,
    adjust: str,
    overwrite: bool,
    providers: Sequence[str],
    provider_failures: dict[str, int],
    circuit_open: set[str],
) -> tuple[list[pd.DataFrame], list[str], int, int]:
    outputs: list[pd.DataFrame] = []
    failed: list[str] = []
    if not codes:
        return outputs, failed, 0, 0

    ok_count = 0
    fail_count = 0
    batch_start = time.time()
    total = len(codes)

    for idx, code in enumerate(tqdm(codes, desc=batch_name), start=1):
        try:
            out = _fetch_one_symbol(
                base_dir=base_dir,
                code=code,
                start=start,
                end=end,
                adjust=adjust,
                overwrite=overwrite,
                providers=providers,
                provider_failures=provider_failures,
                circuit_open=circuit_open,
            )
            if out is not None:
                outputs.append(out)
                ok_count += 1
            else:
                failed.append(code)
                fail_count += 1
        except Exception as e:
            logger.error(f"Failed {code}: {e}")
            failed.append(code)
            fail_count += 1

        if idx == total or idx % PROGRESS_EVERY == 0:
            _print_batch_progress(
                batch_name=batch_name,
                idx=idx,
                total=total,
                success_count=ok_count,
                fail_count=fail_count,
                start_ts=batch_start,
            )

    return outputs, failed, ok_count, fail_count


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
    Incremental by default. Supports multi-source fallback.
    """
    start = _parse_date(start_date) or dt.date(2010, 1, 1)
    end = _parse_date(end_date) or _today()

    provider_names = [primary_provider] + (fallback_providers or [])
    providers = _resolve_available_providers(provider_names)
    if not providers:
        raise ValueError("No valid providers configured after dependency checks")

    logger.info(f"Using providers: {providers}")
    print(f"  Data providers active: {providers}")
    if len(providers) < len(set(provider_names)):
        print("  Some configured providers were skipped due to unknown name or missing dependency.")

    new_codes: list[str] = []
    existing_codes: list[str] = []
    for code in codes:
        symbol = normalize_symbol(code)
        _, clean_parquet = symbol_paths(base_dir, symbol)
        existing = read_existing_market_daily(
            base_dir=base_dir,
            symbol=symbol,
            parquet_path=clean_parquet,
        )
        if existing.empty:
            new_codes.append(code)
        else:
            existing_codes.append(code)

    total_new = len(new_codes)
    total_existing = len(existing_codes)
    print(f"  Existing symbols: {total_existing} (incremental)")
    print(f"  New symbols: {total_new} (full fetch)")
    if total_new > 100:
        est_minutes = total_new * 2 / 60
        print(f"  Estimated new-symbol time: {est_minutes:.0f}-{est_minutes * 2:.0f} minutes")

    all_out: list[pd.DataFrame] = []
    failed: list[str] = []
    provider_failures = {name: 0 for name in providers}
    circuit_open: set[str] = set()

    if existing_codes:
        print(f"\n  Incremental update for {total_existing} symbols ...")
        out1, failed1, ok1, fail1 = _run_batch(
            batch_name="incremental",
            base_dir=base_dir,
            codes=existing_codes,
            start=start,
            end=end,
            adjust=adjust,
            overwrite=overwrite,
            providers=providers,
            provider_failures=provider_failures,
            circuit_open=circuit_open,
        )
        all_out.extend(out1)
        failed.extend(failed1)
        print(f"  Incremental done: ok={ok1} fail={fail1}")

    if new_codes:
        print(f"\n  Full fetch for {total_new} new symbols ...")
        out2, failed2, ok2, fail2 = _run_batch(
            batch_name="full-fetch",
            base_dir=base_dir,
            codes=new_codes,
            start=start,
            end=end,
            adjust=adjust,
            overwrite=overwrite,
            providers=providers,
            provider_failures=provider_failures,
            circuit_open=circuit_open,
        )
        all_out.extend(out2)
        failed.extend(failed2)
        print(f"  Full fetch done: ok={ok2} fail={fail2}")

    if failed:
        print(f"\n  Failed symbols: {len(failed)}")
        if len(failed) <= 20:
            print(f"    {failed}")
        else:
            print(f"    first 20: {failed[:20]}")
        failed_path = base_dir / "data" / "logs" / "fetch_failed.txt"
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.write_text("\n".join(failed), encoding="utf-8")
        print(f"    saved failed list: {failed_path}")

    if circuit_open:
        print(f"\n  Circuit opened providers: {sorted(circuit_open)}")

    if not all_out:
        return pd.DataFrame()

    combined = pd.concat(all_out, ignore_index=True)
    combined = (
        combined.drop_duplicates(["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    return combined
