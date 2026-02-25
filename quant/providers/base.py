"""
Base class and utilities for data providers.

Enhanced with:
1. Better retry logic for network errors
2. Jitter to avoid thundering herd
3. More specific exception handling
"""
from __future__ import annotations
import datetime as dt
import random
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, TypeVar
import pandas as pd
from ..logger import get_logger

logger = get_logger("quant.providers")

T = TypeVar("T")

NON_RETRYABLE_KEYWORDS = (
    "not installed",
    "no module named",
    "modulenotfounderror",
    "unknown provider",
)


class ProviderError(Exception):
    """Exception raised when a provider fails to fetch data."""
    pass


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    jitter: float = 0.5,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator with exponential backoff and jitter.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        jitter: Random jitter factor (0-1) to add to delay

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    error_str = str(e).lower()
                    if any(k in error_str for k in NON_RETRYABLE_KEYWORDS):
                        logger.error(f"{func.__name__} non-retryable error: {e}")
                        raise ProviderError(str(e)) from e

                    # Check if it's a network-related error
                    is_network_error = any(x in error_str for x in [
                        'remotedisconnected', 'connection', 'proxy',
                        'timeout', 'reset', 'eof', 'broken pipe',
                        'network', 'socket'
                    ])

                    if attempt < max_attempts:
                        # Add jitter to avoid thundering herd
                        jittered_delay = current_delay * (1 + random.uniform(-jitter, jitter))

                        if is_network_error:
                            logger.warning(
                                f"{func.__name__} network error (attempt {attempt}/{max_attempts}): {e}. "
                                f"Retrying in {jittered_delay:.1f}s..."
                            )
                        else:
                            logger.warning(
                                f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                                f"Retrying in {jittered_delay:.1f}s..."
                            )

                        time.sleep(jittered_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )

            raise ProviderError(
                f"{func.__name__} failed after {max_attempts} attempts"
            ) from last_exception

        return wrapper
    return decorator


class DataProvider(ABC):
    """
    Abstract base class for market data providers.

    All providers must implement:
    - name: Provider identifier
    - fetch_daily: Fetch daily OHLCV data
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass

    @abstractmethod
    def fetch_daily(
        self,
        code6: str,
        start: dt.date,
        end: dt.date,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for a stock.

        Args:
            code6: 6-digit stock code (e.g., "600519")
            start: Start date
            end: End date
            adjust: Price adjustment type:
                - "qfq": Forward adjusted (前复权)
                - "hfq": Backward adjusted (后复权)
                - "": No adjustment (不复权)

        Returns:
            DataFrame with columns:
                date, open, high, low, close, volume, amount (optional)

        Raises:
            ProviderError: If fetch fails after retries
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
