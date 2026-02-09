from __future__ import annotations

import pandas as pd

# Column mapping for different data sources (Chinese -> English)
COLUMN_MAPPINGS = {
    # AkShare columns
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_chg",
    "涨跌额": "chg",
    "换手率": "turnover",
    # BaoStock columns
    "turn": "turnover",
    "pctChg": "pct_chg",
    # Sina columns
    "day": "date",
}

# Standard output columns
EXPECTED_COLS = [
    "date", "open", "high", "low", "close",
    "volume", "amount", "turnover", "amplitude", "pct_chg", "chg",
]

# Required columns (must be present after normalization)
REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]


def normalize_daily_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Standardize daily market data from any provider to our schema.
    
    Output columns:
        symbol, date, open, high, low, close, volume, amount, turnover, amplitude, pct_chg, chg
    
    Args:
        df: Raw DataFrame from any provider
        symbol: Canonical symbol (e.g., "600519.SH")
    
    Returns:
        Normalized DataFrame with standard schema
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol"] + EXPECTED_COLS)

    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Apply column mappings
    df = df.rename(columns=COLUMN_MAPPINGS)

    # Keep only known columns if present
    keep = [c for c in EXPECTED_COLS if c in df.columns]
    out = df[keep].copy()

    # Ensure required columns exist
    missing_required = [c for c in REQUIRED_COLS if c not in out.columns]
    if missing_required:
        raise ValueError(f"Missing required columns after normalization: {missing_required}")

    # Parse date
    out["date"] = pd.to_datetime(out["date"]).dt.date

    # Add symbol
    out.insert(0, "symbol", symbol)

    # Ensure numeric types where possible
    num_cols = [c for c in out.columns if c not in ("symbol", "date")]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Sort and dedupe
    out = out.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)
    return out


# Backward compatibility alias
def normalize_akshare_daily(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Backward compatible alias for normalize_daily_data.
    
    Deprecated: Use normalize_daily_data instead.
    """
    return normalize_daily_data(df, symbol)
