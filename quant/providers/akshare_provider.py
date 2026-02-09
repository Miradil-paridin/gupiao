"""
AkShare data provider (Eastmoney backend).

Fixed issues:
1. Added post-fetch date filtering (AkShare sometimes ignores start_date)
2. Increased retry attempts and delays for network stability
3. Added request rate limiting to avoid disconnections
4. Better error handling for common network errors
"""
from __future__ import annotations
import datetime as dt
import time
import pandas as pd
from .base import DataProvider, ProviderError, retry
from ..logger import get_logger

logger = get_logger("quant.providers.akshare")

# Rate limiting: minimum seconds between requests
_last_request_time: float = 0
MIN_REQUEST_INTERVAL = 0.8  # seconds


def _rate_limit():
    """Ensure minimum interval between requests to avoid being blocked."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        sleep_time = MIN_REQUEST_INTERVAL - elapsed
        time.sleep(sleep_time)
    _last_request_time = time.time()


class AkShareProvider(DataProvider):
    """
    Data provider using AkShare library.
    Backend: Eastmoney (东方财富)
    """

    @property
    def name(self) -> str:
        return "akshare"

    @retry(
        max_attempts=5,      # Increased from 3
        delay=2.0,           # Increased from 1.0
        backoff=2.0,
        exceptions=(Exception,),
    )
    def fetch_daily(
        self,
        code6: str,
        start: dt.date,
        end: dt.date,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """
        Fetch daily bars from AkShare.

        Args:
            code6: 6-digit stock code
            start: Start date
            end: End date
            adjust: qfq/hfq/""

        Returns:
            DataFrame with standardized columns
        """
        try:
            import akshare as ak
        except ImportError:
            raise ProviderError("akshare not installed. Run: pip install akshare")

        logger.debug(f"Fetching {code6} from AkShare: {start} -> {end}")

        # Apply rate limiting
        _rate_limit()

        try:
            df = ak.stock_zh_a_hist(
                symbol=code6,
                period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust=adjust or "",
            )
        except ConnectionError as e:
            raise ProviderError(f"Connection error for {code6}: {e}") from e
        except TimeoutError as e:
            raise ProviderError(f"Timeout for {code6}: {e}") from e
        except Exception as e:
            error_str = str(e).lower()
            # Identify common network errors for better logging
            if any(x in error_str for x in ['remotedisconnected', 'connection', 'proxy', 'timeout', 'reset']):
                raise ProviderError(f"Network error for {code6}: {e}") from e
            raise ProviderError(f"AkShare fetch failed for {code6}: {e}") from e

        if df is None or df.empty:
            logger.warning(f"No data returned for {code6} from AkShare")
            return pd.DataFrame()

        # Normalize column names (AkShare returns Chinese column names)
        col_map = {
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
        }
        df = df.rename(columns=col_map)

        # Ensure required columns exist
        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ProviderError(f"AkShare response missing columns: {missing}")

        # ============================================================
        # FIX: Force date filtering (AkShare sometimes ignores start_date)
        # ============================================================
        df["date"] = pd.to_datetime(df["date"]).dt.date
        original_len = len(df)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        filtered_len = len(df)

        if original_len != filtered_len:
            logger.info(
                f"Date filtered {code6}: {original_len} -> {filtered_len} rows "
                f"(removed {original_len - filtered_len} out-of-range rows)"
            )

        if df.empty:
            logger.warning(f"No data for {code6} in date range {start} -> {end}")
            return pd.DataFrame()

        # Convert date back to string for downstream compatibility
        df["date"] = df["date"].astype(str)

        return df