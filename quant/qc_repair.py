from __future__ import annotations

import datetime as dt
from pathlib import Path
import pandas as pd

from .store import ensure_dir
from .providers import get_provider, ProviderError
from .normalize import normalize_daily_data
from .logger import get_logger

logger = get_logger("quant.qc_repair")


def _is_invalid_row(df: pd.DataFrame) -> pd.Series:
    """Invalid if OHLC is NaN or <= 0."""
    invalid = pd.Series(False, index=df.index)
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            invalid |= df[c].isna() | (df[c] <= 0)
    return invalid


def _load_symbol_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df is None or df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def qc_repair_market_daily(
    base_dir: Path,
    adjust: str = "qfq",
    refetch_if_invalid_over: int = 10,
    provider_name: str = "akshare",
) -> pd.DataFrame:
    """
    QC & repair per-symbol daily bars.

    - Reads from: data/clean/market_daily/*.parquet
    - Writes to: data/clean_qc/market_daily/*.parquet + CSV
    - For symbols with many invalid rows, refetches full range using configured provider.
    - Drops any remaining invalid rows.
    - Writes combined: data/clean_qc/market_daily_all_qc.parquet
    
    Args:
        base_dir: Project base directory
        adjust: Price adjustment type (qfq/hfq/"")
        refetch_if_invalid_over: Refetch if invalid rows exceed this threshold
        provider_name: Data provider to use for refetching
    """
    src_dir = base_dir / "data" / "clean" / "market_daily"
    out_dir = base_dir / "data" / "clean_qc" / "market_daily"
    out_dir_all = base_dir / "data" / "clean_qc"
    log_dir = base_dir / "data" / "logs"
    ensure_dir(out_dir)
    ensure_dir(out_dir_all)
    ensure_dir(log_dir)

    # Initialize provider
    try:
        provider = get_provider(provider_name)
    except ValueError:
        logger.warning(f"Unknown provider {provider_name}, falling back to akshare")
        provider = get_provider("akshare")

    files = sorted(src_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {src_dir}")

    lines: list[str] = []
    all_dfs: list[pd.DataFrame] = []

    for fp in files:
        symbol = fp.stem
        df = _load_symbol_df(fp)
        if df.empty:
            lines.append(f"{symbol}\tEMPTY (kept empty)")
            continue

        invalid_mask = _is_invalid_row(df)
        invalid_count = int(invalid_mask.sum())

        # Refetch if too many invalid rows (e.g. Moutai)
        if invalid_count >= refetch_if_invalid_over:
            date_min = min(df["date"])
            date_max = max(df["date"])
            code6 = symbol.split(".")[0]

            try:
                logger.info(f"Refetching {symbol} due to {invalid_count} invalid rows")
                raw = provider.fetch_daily(code6=code6, start=date_min, end=date_max, adjust=adjust)
                df2 = normalize_daily_data(raw, symbol=symbol)
                df2["date"] = pd.to_datetime(df2["date"]).dt.date

                invalid2 = _is_invalid_row(df2)
                invalid2_count = int(invalid2.sum())

                lines.append(
                    f"{symbol}\trefetch\tinvalid_before={invalid_count}\tinvalid_after={invalid2_count}\t"
                    f"range={date_min}->{date_max}"
                )
                df = df2
                invalid_mask = invalid2
                invalid_count = invalid2_count
            except (ProviderError, Exception) as e:
                logger.error(f"Refetch failed for {symbol}: {e}")
                lines.append(f"{symbol}\trefetch_failed\tinvalid={invalid_count}\terr={e!r}")

        # Drop any remaining invalid rows
        if invalid_count > 0:
            df = df.loc[~invalid_mask].copy()
            lines.append(f"{symbol}\tdropped_invalid_rows={invalid_count}")

        df = df.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)

        # Write QC outputs
        out_parquet = out_dir / f"{symbol}.parquet"
        out_csv = out_dir / f"{symbol}.csv"
        df.to_parquet(out_parquet, index=False)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        all_dfs.append(df)

    # Combined
    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)
        all_df = all_df.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)
        all_df.to_parquet(out_dir_all / "market_daily_all_qc.parquet", index=False)
    else:
        all_df = pd.DataFrame()

    # Log
    today = dt.date.today().isoformat()
    log_path = log_dir / f"qc_repair_{today}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nQC repair log written to: {log_path}")
    print(f"QC combined parquet: {out_dir_all / 'market_daily_all_qc.parquet'}")
    return all_df
