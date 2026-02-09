"""
Multi-source news providers for A-share market news and announcements.

Supported providers:
- eastmoney: 东方财富 (Eastmoney)
- sina: 新浪财经 (Sina Finance)
- cls: 财联社 (CLS)
- ths: 同花顺 (10jqka)
"""
from __future__ import annotations

from .base import NewsProvider, NewsItem, NewsProviderError
from .eastmoney_provider import EastmoneyNewsProvider
from .sina_provider import SinaNewsProvider
from .cls_provider import CLSNewsProvider
from .ths_provider import THSNewsProvider

__all__ = [
    "NewsProvider",
    "NewsItem",
    "NewsProviderError",
    "EastmoneyNewsProvider",
    "SinaNewsProvider",
    "CLSNewsProvider",
    "THSNewsProvider",
    "get_news_provider",
    "NEWS_PROVIDERS",
]

NEWS_PROVIDERS: dict[str, type[NewsProvider]] = {
    "eastmoney": EastmoneyNewsProvider,
    "sina": SinaNewsProvider,
    "cls": CLSNewsProvider,
    "ths": THSNewsProvider,
}


def get_news_provider(name: str) -> NewsProvider:
    """
    Get a news provider instance by name.
    
    Args:
        name: Provider name (eastmoney, sina, cls, ths)
    
    Returns:
        NewsProvider instance
    
    Raises:
        ValueError: If provider name is unknown
    """
    if name not in NEWS_PROVIDERS:
        raise ValueError(f"Unknown news provider: {name}. Available: {list(NEWS_PROVIDERS.keys())}")
    return NEWS_PROVIDERS[name]()
