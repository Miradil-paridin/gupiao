from __future__ import annotations

import datetime as dt
from pathlib import Path
import pandas as pd


def _try_get_trading_calendar() -> set[dt.date] | None:
    """
    Try to load China A-share trading calendar via AkShare.
    If it fails, return None (validator will skip missing-day checks).
    """
    try:
        import akshare as ak  # local import so validator still works if missing
        cal = ak.tool_trade_date_hist_sina()

        # Known possible column names across versions
        for col in ["trade_date", "日期", "date"]:
            if col in cal.columns:
                dates = pd.to_datetime(cal[col]).dt.date
                return set(dates.tolist())

        # Fallback: first column
        dates = pd.to_datetime(cal.iloc[:, 0]).dt.date
        return set(dates.tolist())
    except Exception:
        return None


def validate_market_daily(base_dir: Path) -> pd.DataFrame:
    """
    Validate each symbol parquet:
      data/clean/market_daily/<symbol>.parquet

    Checks:
      - duplicate (symbol, date)
      - invalid price/volume (<=0, NaN)
      - missing trading days (if calendar available)

    Writes:
      data/logs/validation_YYYY-MM-DD.txt
    Returns:
      summary dataframe
    """
    in_dir = base_dir / "data" / "clean_qc" / "market_daily"
    log_dir = base_dir / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {in_dir}")

    cal = _try_get_trading_calendar()

    rows: list[dict] = []
    lines: list[str] = []

    for fp in files:
        df = pd.read_parquet(fp)
        symbol = fp.stem

        if df is None or df.empty:
            rows.append({
                "symbol": symbol,
                "rows": 0,
                "date_min": None,
                "date_max": None,
                "dup_rows": 0,
                "invalid_rows": 0,
                "missing_trade_days": None if cal is None else 0,
            })
            lines.append(f"{symbol}\tEMPTY")
            continue

        df["date"] = pd.to_datetime(df["date"]).dt.date

        # duplicates
        dup_mask = df.duplicated(["symbol", "date"], keep=False)
        dup_rows = int(dup_mask.sum())

        # invalid rows: any of these critical fields missing or non-positive
        critical = []
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                critical.append(c)

        invalid_mask = pd.Series(False, index=df.index)
        for c in critical:
            invalid_mask |= df[c].isna()
        # prices must be > 0, volume >= 0 (some days might have 0 volume if suspended)
        for c in ["open", "high", "low", "close"]:
            if c in df.columns:
                invalid_mask |= (df[c] <= 0)

        if "volume" in df.columns:
            invalid_mask |= (df["volume"] < 0)

        invalid_rows = int(invalid_mask.sum())

        date_min = min(df["date"])
        date_max = max(df["date"])

        missing_trade_days = None
        if cal is not None:
            # expected trading days within range
            expected = {d for d in cal if date_min <= d <= date_max}
            actual = set(df["date"].tolist())
            missing = expected - actual
            missing_trade_days = len(missing)

        rows.append({
            "symbol": symbol,
            "rows": int(len(df)),
            "date_min": date_min,
            "date_max": date_max,
            "dup_rows": dup_rows,
            "invalid_rows": invalid_rows,
            "missing_trade_days": missing_trade_days,
        })

        msg = (
            f"{symbol}\trows={len(df)}\t"
            f"range={date_min}->{date_max}\t"
            f"dup={dup_rows}\tinvalid={invalid_rows}"
        )
        if cal is not None:
            msg += f"\tmissing_trade_days={missing_trade_days}"
        lines.append(msg)

    summary = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)

    # Write log
    today = dt.date.today().isoformat()
    log_path = log_dir / f"validation_{today}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nValidation log written to: {log_path}")
    return summary
