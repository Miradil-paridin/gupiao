"""
Quant package for A-share market data fetching, processing, and backtesting.

Modules:
- providers: Multi-source data providers (AkShare, BaoStock, Sina)
- fetch_daily: Daily data fetching with fallback support
- normalize: Data normalization to standard schema
- validate: Data quality validation
- qc_repair: Data quality control and repair
- features: Feature engineering
- backtest: Strategy backtesting
- logger: Centralized logging
"""
from .logger import setup_logger, get_logger
from .providers import get_provider, PROVIDERS, ProviderError

__all__ = [
    "setup_logger",
    "get_logger", 
    "get_provider",
    "PROVIDERS",
    "ProviderError",
]
