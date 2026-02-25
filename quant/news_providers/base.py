"""
Base class and utilities for news providers.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from functools import wraps
from typing import Callable, TypeVar, Optional

from ..logger import get_logger

logger = get_logger("quant.news_providers")

T = TypeVar("T")


class NewsProviderError(Exception):
    """Exception raised when a news provider fails to fetch data."""
    pass


@dataclass
class NewsItem:
    """
    Standardized news item structure.
    All news providers should convert their output to this format.
    """
    title: str                          # 新闻标题
    content: str                        # 新闻内容/摘要
    publish_time: Optional[datetime]    # 发布时间
    source: str                         # 来源 (provider name)
    url: Optional[str] = None           # 原文链接
    stock_codes: list[str] = field(default_factory=list)  # 相关股票代码
    keywords: list[str] = field(default_factory=list)     # 关键词
    sentiment: Optional[str] = None     # 情感: positive/negative/neutral
    category: Optional[str] = None      # 分类: announcement/news/research
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "content": self.content,
            "publish_time": self.publish_time.isoformat() if self.publish_time else None,
            "source": self.source,
            "url": self.url,
            "stock_codes": self.stock_codes,
            "keywords": self.keywords,
            "sentiment": self.sentiment,
            "category": self.category,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "NewsItem":
        """Create from dictionary."""
        pub_time = d.get("publish_time")
        if pub_time and isinstance(pub_time, str):
            try:
                pub_time = datetime.fromisoformat(pub_time)
            except ValueError:
                pub_time = None
        
        return cls(
            title=d.get("title", ""),
            content=d.get("content", ""),
            publish_time=pub_time,
            source=d.get("source", ""),
            url=d.get("url"),
            stock_codes=d.get("stock_codes", []),
            keywords=d.get("keywords", []),
            sentiment=d.get("sentiment"),
            category=d.get("category"),
        )


def retry_news(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator with exponential backoff for news fetching.
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
                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
            
            raise NewsProviderError(
                f"{func.__name__} failed after {max_attempts} attempts"
            ) from last_exception
        
        return wrapper
    return decorator


class NewsProvider(ABC):
    """
    Abstract base class for news providers.
    
    All providers must implement:
    - name: Provider identifier
    - fetch_stock_news: Fetch news for a specific stock
    - fetch_market_news: Fetch general market news (optional)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass
    
    @abstractmethod
    def fetch_stock_news(
        self,
        code6: str,
        max_items: int = 50,
    ) -> list[NewsItem]:
        """
        Fetch news for a specific stock.
        
        Args:
            code6: 6-digit stock code (e.g., "600519")
            max_items: Maximum number of news items to return
        
        Returns:
            List of NewsItem objects
        
        Raises:
            NewsProviderError: If fetch fails after retries
        """
        pass
    
    def fetch_market_news(
        self,
        max_items: int = 100,
    ) -> list[NewsItem]:
        """
        Fetch general market news (not stock-specific).
        
        Args:
            max_items: Maximum number of news items to return
        
        Returns:
            List of NewsItem objects
        
        Note:
            Not all providers support this. Default implementation returns empty list.
        """
        return []
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
