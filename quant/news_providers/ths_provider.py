"""
THS (同花顺/10jqka) news provider.
同花顺是国内领先的金融数据和分析平台。
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from .base import NewsProvider, NewsItem, NewsProviderError, retry_news
from ..logger import get_logger

logger = get_logger("quant.news_providers.ths")


class THSNewsProvider(NewsProvider):
    """
    News provider using THS (同花顺).
    Provides stock news, market analysis, and financial data.
    
    THS is known for:
    - Comprehensive A-share coverage
    - Real-time market data
    - Individual stock analysis reports
    """
    
    @property
    def name(self) -> str:
        return "ths"
    
    def _parse_datetime(self, s: str) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if not s:
            return None
        
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%m-%d %H:%M",
        ]
        
        s = str(s).strip()
        
        for fmt in formats:
            try:
                dt = datetime.strptime(s, fmt)
                # If year not specified, use current year
                if dt.year == 1900:
                    dt = dt.replace(year=datetime.now().year)
                return dt
            except ValueError:
                continue
        
        return None
    
    def _code_to_ths(self, code6: str) -> str:
        """Convert 6-digit code to THS format."""
        # THS uses just the 6-digit code
        return code6
    
    @retry_news(max_attempts=3, delay=1.0, backoff=2.0)
    def fetch_stock_news(
        self,
        code6: str,
        max_items: int = 50,
    ) -> list[NewsItem]:
        """
        Fetch stock news from THS (同花顺).
        
        Args:
            code6: 6-digit stock code
            max_items: Maximum items to return
        
        Returns:
            List of NewsItem objects
        """
        try:
            import requests
        except ImportError:
            raise NewsProviderError("requests not installed")
        
        logger.debug(f"Fetching news for {code6} from THS")
        
        # THS stock news page
        url = f"http://stockpage.10jqka.com.cn/{code6}/news/"
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "http://stockpage.10jqka.com.cn/",
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.encoding = "gbk"
            html = response.text
        except requests.RequestException as e:
            logger.warning(f"THS fetch failed for {code6}: {e}")
            # Fall back to AkShare method
            return self._fetch_via_akshare(code6, max_items)
        
        items = []
        
        # Parse HTML - THS news list typically in <ul class="newslist"> or similar
        # Pattern to find news items
        patterns = [
            r'<a[^>]*href="(http[^"]*)"[^>]*>([^<]+)</a>\s*<span[^>]*>([^<]*)</span>',
            r'<a[^>]*href="([^"]+)"[^>]*title="([^"]+)"[^>]*>',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html)
            for match in matches[:max_items]:
                if len(match) >= 2:
                    url_match = match[0]
                    title = match[1].strip()
                    date_str = match[2].strip() if len(match) > 2 else ""
                    
                    if not title or len(title) < 3:
                        continue
                    
                    # Filter out non-news links
                    if "10jqka.com.cn" not in url_match and not url_match.startswith("http"):
                        continue
                    
                    pub_time = self._parse_datetime(date_str)
                    
                    items.append(NewsItem(
                        title=title,
                        content="",
                        publish_time=pub_time or datetime.now(),
                        source=self.name,
                        url=url_match,
                        stock_codes=[code6],
                        category="news",
                    ))
            
            if items:
                break
        
        # If no items found, try AkShare fallback
        if not items:
            items = self._fetch_via_akshare(code6, max_items)
        
        logger.info(f"Fetched {len(items)} news items for {code6} from THS")
        return items[:max_items]
    
    def _fetch_via_akshare(self, code6: str, max_items: int) -> list[NewsItem]:
        """Fallback method using AkShare."""
        try:
            import akshare as ak
            
            # Try individual stock news
            df = ak.stock_news_em(symbol=code6)
            
            if df is None or df.empty:
                return []
            
            items = []
            for _, row in df.head(max_items).iterrows():
                title = str(row.get("新闻标题", ""))
                content = str(row.get("新闻内容", ""))[:300]
                time_str = str(row.get("发布时间", ""))
                
                pub_time = self._parse_datetime(time_str)
                
                items.append(NewsItem(
                    title=title,
                    content=content,
                    publish_time=pub_time or datetime.now(),
                    source=f"{self.name}_fallback",
                    stock_codes=[code6],
                    category="news",
                ))
            
            return items
        except Exception as e:
            logger.debug(f"AkShare fallback failed: {e}")
            return []
    
    @retry_news(max_attempts=3, delay=1.0, backoff=2.0)
    def fetch_market_news(
        self,
        max_items: int = 100,
    ) -> list[NewsItem]:
        """
        Fetch general market news from THS.
        
        Args:
            max_items: Maximum items to return
        
        Returns:
            List of NewsItem objects
        """
        try:
            import requests
        except ImportError:
            raise NewsProviderError("requests not installed")
        
        logger.debug("Fetching market news from THS")
        
        # THS market news/financial news page
        url = "http://news.10jqka.com.cn/cjzx_list/"
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.encoding = "gbk"
            html = response.text
        except requests.RequestException as e:
            logger.warning(f"THS market news fetch failed: {e}")
            return []
        
        items = []
        
        # Parse news list
        pattern = r'<a[^>]*href="(http[^"]+10jqka[^"]+)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, html)
        
        for url_match, title in matches[:max_items]:
            title = title.strip()
            if not title or len(title) < 5:
                continue
            
            items.append(NewsItem(
                title=title,
                content="",
                publish_time=datetime.now(),
                source=self.name,
                url=url_match,
                stock_codes=[],
                category="market",
            ))
        
        logger.info(f"Fetched {len(items)} market news from THS")
        return items[:max_items]
    
    @retry_news(max_attempts=3, delay=1.0, backoff=2.0)
    def fetch_stock_notices(
        self,
        code6: str,
        max_items: int = 30,
    ) -> list[NewsItem]:
        """
        Fetch stock announcements/notices from THS.
        
        Args:
            code6: 6-digit stock code
            max_items: Maximum items to return
        
        Returns:
            List of NewsItem objects with category='announcement'
        """
        try:
            import requests
        except ImportError:
            raise NewsProviderError("requests not installed")
        
        logger.debug(f"Fetching notices for {code6} from THS")
        
        # THS announcements page
        url = f"http://stockpage.10jqka.com.cn/{code6}/announce/"
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.encoding = "gbk"
            html = response.text
        except requests.RequestException as e:
            logger.warning(f"THS notices fetch failed for {code6}: {e}")
            return []
        
        items = []
        
        # Parse announcements
        pattern = r'<a[^>]*href="([^"]+)"[^>]*title="([^"]+)"[^>]*>'
        matches = re.findall(pattern, html)
        
        for url_match, title in matches[:max_items]:
            title = title.strip()
            if not title:
                continue
            
            items.append(NewsItem(
                title=title,
                content="",
                publish_time=datetime.now(),
                source=self.name,
                url=url_match if url_match.startswith("http") else f"http:{url_match}",
                stock_codes=[code6],
                category="announcement",
            ))
        
        logger.info(f"Fetched {len(items)} notices for {code6} from THS")
        return items[:max_items]
