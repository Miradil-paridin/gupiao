"""
Sina Finance data provider (新浪财经).
Uses public API endpoints.

Note: Sina API does NOT support price adjustment (复权).
For accurate backtesting, use BaoStock or AkShare instead.
"""
from __future__ import annotations
import datetime as dt
import re
import time
from urllib.parse import urlencode
import pandas as pd
from .base import DataProvider, ProviderError, retry
from ..logger import get_logger

logger = get_logger("quant.providers.sina")

# Rate limiting
_last_request_time: float = 0
MIN_REQUEST_INTERVAL = 0.5


def _rate_limit():
    """Ensure minimum interval between requests."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


class SinaProvider(DataProvider):
    """
    Data provider using Sina Finance API.
    Free public API, good for real-time and historical data.

    ⚠️ IMPORTANT: Sina does NOT support price adjustment (复权).
    For backtesting with adjusted prices, use BaoStock or AkShare.

    Note: Sina API has some limitations:
    - May not support very old historical data
    - Rate limiting may apply
    - NO adjust support (always returns unadjusted prices)
    """

    @property
    def name(self) -> str:
        return "sina"

    def _code_to_sina(self, code6: str) -> str:
        """Convert 6-digit code to Sina format (sh600519 or sz000001)."""
        if code6.startswith("6"):
            return f"sh{code6}"
        else:
            return f"sz{code6}"

    @retry(max_attempts=5, delay=2.0, backoff=2.0, exceptions=(Exception,))
    def fetch_daily(
        self,
        code6: str,
        start: dt.date,
        end: dt.date,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """
        Fetch daily bars from Sina Finance.

        Args:
            code6: 6-digit stock code
            start: Start date
            end: End date
            adjust: qfq/hfq/"" (⚠️ Sina does NOT support adjustment!)

        Returns:
            DataFrame with standardized columns

        Note:
            Sina's historical data API is less comprehensive than AkShare/BaoStock.
            For full historical data with adjustment, use other providers.
        """
        try:
            import requests
        except ImportError:
            raise ProviderError("requests not installed. Run: pip install requests")

        sina_code = self._code_to_sina(code6)

        logger.debug(f"Fetching {code6} from Sina Finance: {start} -> {end}")

        # Warn about adjustment
        if adjust and adjust != "":
            logger.warning(
                f"Sina provider does NOT support {adjust} adjustment. "
                "Data will be UNADJUSTED. For accurate backtesting, use baostock or akshare."
            )

        # Apply rate limiting
        _rate_limit()

        # Use Sina's money.finance API for historical data
        url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"

        # Calculate roughly how many data points we need
        days_diff = (end - start).days
        data_count = min(days_diff + 100, 10000)  # Buffer for non-trading days

        params = {
            "symbol": sina_code,
            "scale": "240",  # Daily (240 minutes = 1 day)
            "ma": "no",
            "datalen": data_count,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            text = response.text

            if not text or text == "null":
                logger.warning(f"No data returned for {code6} from Sina")
                return pd.DataFrame()

            # Parse the JSON-like response
            data = self._parse_sina_response(text)

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)

        except requests.RequestException as e:
            raise ProviderError(f"Sina fetch failed for {code6}: {e}") from e

        # Normalize column names
        col_map = {
            "day": "date",
        }
        df = df.rename(columns=col_map)

        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ============================================================
        # Date filtering (consistent with other providers)
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

        return df.reset_index(drop=True)

    def _parse_sina_response(self, text: str) -> list[dict]:
        """
        Parse Sina's JSON-like response format.

        The response is JavaScript-style, not valid JSON:
        [{day:"2024-01-01",open:"100.00",...}, ...]
        """
        import json

        try:
            # Try direct JSON parse first
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Convert JavaScript-style to valid JSON
        # Add quotes around keys
        text = re.sub(r'(\w+):', r'"\1":', text)
        # Replace single quotes with double quotes
        text = text.replace("'", '"')

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Sina response: {e}")
            return []