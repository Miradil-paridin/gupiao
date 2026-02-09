"""
Multi-source data providers for A-share market data.

Supported providers:
- akshare: AkShare (Eastmoney backend)
- baostock: BaoStock (证券宝)
- sina: Sina Finance (新浪财经)
"""
from __future__ import annotations

from .base import DataProvider, ProviderError
from .akshare_provider import AkShareProvider
from .baostock_provider import BaoStockProvider
from .sina_provider import SinaProvider

__all__ = [
    "DataProvider",
    "ProviderError",
    "AkShareProvider",
    "BaoStockProvider",
    "SinaProvider",
    "get_provider",
    "PROVIDERS",
]

PROVIDERS: dict[str, type[DataProvider]] = {
    "akshare": AkShareProvider,
    "baostock": BaoStockProvider,
    "sina": SinaProvider,
}


def get_provider(name: str) -> DataProvider:
    """
    Get a data provider instance by name.
    
    Args:
        name: Provider name (akshare, baostock, sina)
    
    Returns:
        DataProvider instance
    
    Raises:
        ValueError: If provider name is unknown
    """
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name]()
