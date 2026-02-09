"""
BaoStock data provider (证券宝).
http://baostock.com/

Fixed issues:
1. Added post-fetch date filtering
2. Increased retry attempts
3. Better session management (login/logout)
4. Handle empty string values from BaoStock
"""
from __future__ import annotations
import datetime as dt
import time
import pandas as pd
from .base import DataProvider, ProviderError, retry
from ..logger import get_logger

logger = get_logger("quant.providers.baostock")

# Rate limiting
_last_request_time: float = 0
MIN_REQUEST_INTERVAL = 0.3  # BaoStock is more stable, shorter interval


def _rate_limit():
    """Ensure minimum interval between requests."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


class BaoStockProvider(DataProvider):
    """
    Data provider using BaoStock library.
    Free and stable A-share data source.
    """

    @property
    def name(self) -> str:
        return "baostock"

    def _code_to_baostock(self, code6: str) -> str:
        """Convert 6-digit code to BaoStock format (sh.600519 or sz.000001)."""
        if code6.startswith("6"):
            return f"sh.{code6}"
        else:
            return f"sz.{code6}"

    def _adjust_to_baostock(self, adjust: str) -> str:
        """
        Convert adjust type to BaoStock adjustflag.
        BaoStock: 1=后复权, 2=前复权, 3=不复权
        """
        if adjust == "qfq":
            return "2"
        elif adjust == "hfq":
            return "1"
        else:
            return "3"

    @retry(max_attempts=5, delay=2.0, backoff=2.0, exceptions=(Exception,))
    def fetch_daily(
        self,
        code6: str,
        start: dt.date,
        end: dt.date,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """
        Fetch daily bars from BaoStock.

        Args:
            code6: 6-digit stock code
            start: Start date
            end: End date
            adjust: qfq/hfq/""

        Returns:
            DataFrame with standardized columns
        """
        try:
            import baostock as bs
        except ImportError:
            raise ProviderError("baostock not installed. Run: pip install baostock")

        bs_code = self._code_to_baostock(code6)
        adjustflag = self._adjust_to_baostock(adjust)

        logger.debug(f"Fetching {code6} from BaoStock: {start} -> {end}")

        # Apply rate limiting
        _rate_limit()

        # Login to BaoStock
        lg = bs.login()
        if lg.error_code != "0":
            raise ProviderError(f"BaoStock login failed: {lg.error_msg}")

        try:
            rs = bs.query_history_k_data_plus(
                code=bs_code,
                fields="date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag=adjustflag,
            )

            if rs.error_code != "0":
                raise ProviderError(f"BaoStock query failed: {rs.error_msg}")

            # Collect results
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                logger.warning(f"No data returned for {code6} from BaoStock")
                return pd.DataFrame()

            df = pd.DataFrame(data_list, columns=rs.fields)

        finally:
            # Always logout
            try:
                bs.logout()
            except Exception:
                pass  # Ignore logout errors

        # ============================================================
        # FIX: Handle empty strings from BaoStock
        # BaoStock returns "" for missing values, need to handle before numeric conversion
        # ============================================================
        df = df.replace("", pd.NA)

        # Normalize column names
        col_map = {
            "turn": "turnover",
            "pctChg": "pct_chg",
        }
        df = df.rename(columns=col_map)

        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ============================================================
        # FIX: Force date filtering (same as akshare)
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

        # ============================================================
        # FIX: Remove rows with all NaN prices (suspended stocks)
        # ============================================================
        price_cols = ["open", "high", "low", "close"]
        df = df.dropna(subset=price_cols, how="all")

        return df