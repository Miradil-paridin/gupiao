"""
BaoStock data provider.
http://baostock.com/

修复：全局单次 login，不再每次请求都 login/logout
"""
from __future__ import annotations
import datetime as dt
import time
import threading
import atexit
import pandas as pd
from .base import DataProvider, ProviderError, retry
from ..logger import get_logger

logger = get_logger("quant.providers.baostock")

# Rate limiting
_last_request_time: float = 0
MIN_REQUEST_INTERVAL = 0.5

# ── 全局 BaoStock 会话管理 ──
_bs_lock = threading.Lock()
_bs_logged_in = False


def _ensure_login():
    """确保 BaoStock 已登录（全局只登录一次）"""
    global _bs_logged_in
    if _bs_logged_in:
        return
    with _bs_lock:
        if _bs_logged_in:
            return
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            raise ProviderError(f"BaoStock login failed: {lg.error_msg}")
        _bs_logged_in = True
        # 设置 socket 超时，防止 recv 卡死
        try:
            # baostock 内部用的是 socketutil 模块
            from baostock.util import socketutil
            if hasattr(socketutil, 'default_socket') and socketutil.default_socket is not None:
                socketutil.default_socket.settimeout(30)
                logger.info("BaoStock socket timeout set to 30s")
            else:
                logger.warning("Could not find baostock socket to set timeout")
        except Exception as e:
            logger.warning(f"Failed to set socket timeout: {e}")
        logger.info("BaoStock login success (global session)")


def _global_logout():
    """程序退出时登出"""
    global _bs_logged_in
    if _bs_logged_in:
        try:
            import baostock as bs
            bs.logout()
            logger.info("BaoStock logout success")
        except Exception:
            pass
        _bs_logged_in = False


# 注册退出时自动登出
atexit.register(_global_logout)


def _rate_limit():
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


class BaoStockProvider(DataProvider):

    @property
    def name(self) -> str:
        return "baostock"

    def _code_to_baostock(self, code6: str) -> str:
        if code6.startswith("6"):
            return f"sh.{code6}"
        else:
            return f"sz.{code6}"

    def _adjust_to_baostock(self, adjust: str) -> str:
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
        try:
            import baostock as bs
        except ImportError:
            raise ProviderError("baostock not installed. Run: pip install baostock")

        bs_code = self._code_to_baostock(code6)
        adjustflag = self._adjust_to_baostock(adjust)

        logger.debug(f"Fetching {code6} from BaoStock: {start} -> {end}")

        _rate_limit()
        _ensure_login()

        try:
            rs = bs.query_history_k_data_plus(
                code=bs_code,
                fields="date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag=adjustflag,
            )
        except Exception as e:
            # socket 断了/乱码/超时 → 重置登录状态，让 retry 重连
            global _bs_logged_in
            logger.warning(f"BaoStock query exception for {code6}: {e}, resetting session")
            try:
                bs.logout()
            except Exception:
                pass
            _bs_logged_in = False
            raise ProviderError(f"BaoStock connection error for {code6}: {e}")

        if rs.error_code != "0":
            # 会话可能过期，重新登录再试一次
            _bs_logged_in = False
            try:
                bs.logout()
            except Exception:
                pass
            _ensure_login()
            try:
                rs = bs.query_history_k_data_plus(
                    code=bs_code,
                    fields="date,open,high,low,close,volume,amount,turn,pctChg",
                    start_date=start.strftime("%Y-%m-%d"),
                    end_date=end.strftime("%Y-%m-%d"),
                    frequency="d",
                    adjustflag=adjustflag,
                )
            except Exception as e2:
                _bs_logged_in = False
                raise ProviderError(f"BaoStock reconnect failed for {code6}: {e2}")
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

        # Handle empty strings
        df = df.replace("", pd.NA)

        # Normalize column names
        df = df.rename(columns={"turn": "turnover", "pctChg": "pct_chg"})

        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Date filtering
        df["date"] = pd.to_datetime(df["date"]).dt.date
        original_len = len(df)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        filtered_len = len(df)

        if original_len != filtered_len:
            logger.info(f"Date filtered {code6}: {original_len} -> {filtered_len} rows")

        if df.empty:
            logger.warning(f"No data for {code6} in date range {start} -> {end}")
            return pd.DataFrame()

        df["date"] = df["date"].astype(str)

        # Remove suspended stocks
        price_cols = ["open", "high", "low", "close"]
        df = df.dropna(subset=price_cols, how="all")

        return df