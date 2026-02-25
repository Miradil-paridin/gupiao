"""
Sina Finance (新浪财经) news provider.
Uses web scraping and AkShare for data fetching.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from .base import NewsProvider, NewsItem, NewsProviderError, retry_news
from ..logger import get_logger

logger = get_logger("quant.news_providers.sina")


class SinaNewsProvider(NewsProvider):
    """
    News provider using Sina Finance.
    Provides stock news, market news, and financial headlines.
    """
    
    @property
    def name(self) -> str:
        return "sina"
    
    def _parse_datetime(self, s: str) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if not s:
            return None
        
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y年%m月%d日 %H:%M",
            "%Y年%m月%d日",
            "%m月%d日 %H:%M",
        ]
        
        s = str(s).strip()
        
        for fmt in formats:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        
        # Try to extract date components
        match = re.search(r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})", s)
        if match:
            try:
                return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                pass
        
        return None
    
    def _code_to_sina(self, code6: str) -> str:
        """Convert 6-digit code to Sina format."""
        if code6.startswith("6"):
            return f"sh{code6}"
        else:
            return f"sz{code6}"

    def _extract_date_from_url(self, url: str) -> Optional[datetime]:
        if not url:
            return None
        s = str(url)
        m = re.search(r"(20\d{2}-\d{2}-\d{2})", s)
        if m:
            return self._parse_datetime(m.group(1))
        m = re.search(r"(20\d{6})", s)
        if m:
            raw = m.group(1)
            return self._parse_datetime(f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}")
        return None

    def _is_noise_title(self, title: str) -> bool:
        t = str(title or "").strip()
        if not t:
            return True
        noise_keywords = (
            "友情链接", "联系我们", "关于我们", "免责声明", "法律声明", "隐私",
            "意见反馈", "留言板", "常见问题", "举报", "广告服务", "投稿", "APP下载",
            "contact us", "privacy", "terms", "feedback",
        )
        tl = t.lower()
        return any((k in t) or (k in tl) for k in noise_keywords)
    
    @retry_news(max_attempts=3, delay=1.0, backoff=2.0)
    def fetch_stock_news(
        self,
        code6: str,
        max_items: int = 50,
    ) -> list[NewsItem]:
        """
        Fetch stock news from Sina Finance.
        
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
        
        sina_code = self._code_to_sina(code6)
        logger.debug(f"Fetching news for {code6} from Sina Finance")
        
        # Sina stock news API
        url = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllNewsStock/symbol/{sina_code}.phtml"
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://finance.sina.com.cn/",
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.encoding = "gb2312"
            html = response.text
        except requests.RequestException as e:
            raise NewsProviderError(f"Sina fetch failed for {code6}: {e}") from e
        
        items = []
        
        # Parse HTML to extract news items
        # Pattern: <a target="_blank" href="URL">TITLE</a></br>DATE
        pattern = r'<a[^>]*target="_blank"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, html)
        
        for url_match, title in matches[:max_items]:
            title = title.strip()
            if not title or len(title) < 6 or self._is_noise_title(title):
                continue
            
            pub_time = self._extract_date_from_url(url_match)
            
            items.append(NewsItem(
                title=title,
                content="",  # Would need to fetch full article
                publish_time=pub_time,
                source=self.name,
                url=url_match if url_match.startswith("http") else f"https:{url_match}",
                stock_codes=[code6],
                category="news",
            ))
        
        # Fallback: try AkShare if web scraping returns few results
        if len(items) < 5:
            items.extend(self._fetch_via_akshare(code6, max_items - len(items)))
        
        logger.info(f"Fetched {len(items)} news items for {code6} from Sina")
        return items[:max_items]
    
    def _fetch_via_akshare(self, code6: str, max_items: int) -> list[NewsItem]:
        """Fallback method using AkShare's Sina news interface."""
        try:
            import akshare as ak
            # Some AkShare versions have sina news functions
            df = ak.stock_news_em(symbol=code6)  # Actually from Eastmoney, but works as fallback
            
            if df is None or df.empty:
                return []
            
            items = []
            for _, row in df.head(max_items).iterrows():
                title = str(row.get("新闻标题", ""))
                content = str(row.get("新闻内容", ""))[:300]
                time_str = str(row.get("发布时间", row.get("publish_time", "")))
                pub_time = self._parse_datetime(time_str)
                if not title or title.lower() == "nan":
                    continue
                
                items.append(NewsItem(
                    title=title,
                    content=content,
                    publish_time=pub_time,
                    source="sina_fallback",
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
        Fetch general market news from Sina Finance.
        
        Args:
            max_items: Maximum items to return
        
        Returns:
            List of NewsItem objects
        """
        try:
            import requests
        except ImportError:
            raise NewsProviderError("requests not installed")
        
        logger.debug("Fetching market news from Sina Finance")
        
        # Sina finance news feed
        url = "https://finance.sina.com.cn/roll/index.d.html?cid=56592&page=1"
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.encoding = "utf-8"
            html = response.text
        except requests.RequestException as e:
            logger.warning(f"Sina market news fetch failed: {e}")
            return []
        
        items = []
        
        # Parse news items
        pattern = r'<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, html)
        
        for url_match, title in matches[:max_items]:
            title = title.strip()
            if not title or len(title) < 5:
                continue
            if self._is_noise_title(title):
                continue
            if "finance.sina.com.cn" not in url_match and "sina.com.cn" not in url_match:
                continue
            
            items.append(NewsItem(
                title=title,
                content="",
                publish_time=self._extract_date_from_url(url_match),
                source=self.name,
                url=url_match,
                stock_codes=[],
                category="market",
            ))
        
        logger.info(f"Fetched {len(items)} market news from Sina")
        return items[:max_items]
