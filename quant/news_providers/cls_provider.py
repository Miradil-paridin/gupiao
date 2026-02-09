"""
CLS (财联社) news provider.
财联社是专业的财经快讯和深度报道平台。
Uses AkShare for data fetching.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from .base import NewsProvider, NewsItem, NewsProviderError, retry_news
from ..logger import get_logger

logger = get_logger("quant.news_providers.cls")


class CLSNewsProvider(NewsProvider):
    """
    News provider using CLS (财联社).
    Provides real-time financial news flashes and market updates.
    
    CLS is known for:
    - Fast breaking news (电报/快讯)
    - A-share market focus
    - Professional financial journalism
    """
    
    @property
    def name(self) -> str:
        return "cls"
    
    def _parse_datetime(self, s: str) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if not s:
            return None
        
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ]
        
        s = str(s).strip()
        
        for fmt in formats:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        
        return None
    
    @retry_news(max_attempts=3, delay=1.0, backoff=2.0)
    def fetch_stock_news(
        self,
        code6: str,
        max_items: int = 50,
    ) -> list[NewsItem]:
        """
        Fetch stock-related news from CLS.
        
        Note: CLS doesn't have a direct stock-specific news API via AkShare,
        so we fetch general news and filter by stock code mention.
        
        Args:
            code6: 6-digit stock code
            max_items: Maximum items to return
        
        Returns:
            List of NewsItem objects
        """
        # CLS doesn't provide direct stock-specific news via AkShare
        # We'll fetch telegraph/flash news and try to match
        all_news = self.fetch_market_news(max_items=200)
        
        # Filter by stock code mention in title or content
        stock_news = []
        for item in all_news:
            if code6 in item.title or code6 in item.content:
                item.stock_codes = [code6]
                stock_news.append(item)
        
        logger.info(f"Found {len(stock_news)} CLS news items mentioning {code6}")
        return stock_news[:max_items]
    
    @retry_news(max_attempts=3, delay=1.0, backoff=2.0)
    def fetch_market_news(
        self,
        max_items: int = 100,
    ) -> list[NewsItem]:
        """
        Fetch general market news/telegraph from CLS via AkShare.
        
        Args:
            max_items: Maximum items to return
        
        Returns:
            List of NewsItem objects
        """
        try:
            import akshare as ak
        except ImportError:
            raise NewsProviderError("akshare not installed")
        
        logger.debug("Fetching telegraph news from CLS")
        
        items = []
        
        # Try fetching CLS telegraph (财联社电报)
        try:
            df = ak.stock_info_global_cls()
            
            if df is not None and not df.empty:
                for _, row in df.head(max_items).iterrows():
                    # Columns may include: 时间, 内容, 标题
                    title = str(row.get("标题", row.get("title", "")))
                    content = str(row.get("内容", row.get("content", "")))
                    time_str = str(row.get("时间", row.get("time", "")))
                    
                    # If no title, use first part of content
                    if not title and content:
                        title = content[:50] + "..." if len(content) > 50 else content
                    
                    pub_time = self._parse_datetime(time_str)
                    
                    items.append(NewsItem(
                        title=title,
                        content=content[:500] if content else "",
                        publish_time=pub_time or datetime.now(),
                        source=self.name,
                        stock_codes=[],
                        category="telegraph",
                    ))
                
                logger.info(f"Fetched {len(items)} telegraph items from CLS")
        except Exception as e:
            logger.warning(f"CLS telegraph fetch failed: {e}")
        
        # Also try fetching CLS depth articles if available
        try:
            df_depth = ak.stock_info_global_em()  # This might be different source but good fallback
            
            if df_depth is not None and not df_depth.empty:
                for _, row in df_depth.head(max_items - len(items)).iterrows():
                    title = str(row.get("标题", row.get("title", "")))
                    content = str(row.get("内容", row.get("content", "")))
                    
                    if not title:
                        continue
                    
                    items.append(NewsItem(
                        title=title,
                        content=content[:500] if content else "",
                        publish_time=datetime.now(),
                        source=f"{self.name}_global",
                        stock_codes=[],
                        category="market",
                    ))
        except Exception as e:
            logger.debug(f"CLS depth fetch failed (may not be available): {e}")
        
        return items[:max_items]
    
    @retry_news(max_attempts=3, delay=1.0, backoff=2.0)
    def fetch_telegraph(
        self,
        max_items: int = 50,
    ) -> list[NewsItem]:
        """
        Fetch CLS telegraph/flash news (电报/快讯).
        These are real-time short news items.
        
        Args:
            max_items: Maximum items to return
        
        Returns:
            List of NewsItem objects
        """
        try:
            import akshare as ak
        except ImportError:
            raise NewsProviderError("akshare not installed")
        
        logger.debug("Fetching CLS telegraph (电报)")
        
        try:
            df = ak.stock_info_global_cls()
            
            if df is None or df.empty:
                return []
            
            items = []
            for _, row in df.head(max_items).iterrows():
                content = str(row.get("内容", ""))
                time_str = str(row.get("时间", ""))
                
                # Telegraph items are usually short, so content is the main thing
                title = content[:80] + "..." if len(content) > 80 else content
                
                pub_time = self._parse_datetime(time_str)
                
                items.append(NewsItem(
                    title=title,
                    content=content,
                    publish_time=pub_time or datetime.now(),
                    source=self.name,
                    stock_codes=[],
                    category="telegraph",
                ))
            
            logger.info(f"Fetched {len(items)} telegraph items from CLS")
            return items
            
        except Exception as e:
            logger.warning(f"CLS telegraph fetch failed: {e}")
            return []
