from __future__ import annotations

import datetime as dt
import akshare as ak
import pandas as pd

def _fmt_yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")

def fetch_akshare_daily(
    code6: str,
    start: dt.date,
    end: dt.date,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    Fetch A-share daily bars from AkShare (Eastmoney-backed endpoint in many setups).
    Returns raw dataframe with Chinese columns (usually).
    """
    # AkShare expects code like "600519" (not "600519.SH")
    df = ak.stock_zh_a_hist(
        symbol=code6,
        period="daily",
        start_date=_fmt_yyyymmdd(start),
        end_date=_fmt_yyyymmdd(end),
        adjust=adjust or "",
    )
    return df
