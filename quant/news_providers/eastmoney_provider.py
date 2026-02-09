"""
Eastmoney (东方财富) news provider.
Uses AkShare for data fetching.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from .base import NewsProvider, NewsItem, NewsProviderError, retry_news
from ..logger import get_logger

logger = get_logger("quant.news_providers.eastmoney")


class EastmoneyNewsProvider(NewsProvider):
    """
    News provider using Eastmoney via AkShare.
    Provides individual stock news and research reports.
    """
    
    @property
    def name(self) -> str:
        return "eastmoney"
    
    def _parse_datetime(self, s: str) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if not s:
            return None
        
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(str(s).strip(), fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse datetime: {s}")
        return None
    
    @retry_news(max_attempts=3, delay=1.0, backoff=2.0)
    def fetch_stock_news(
        self,
        code6: str,
        max_items: int = 50,
    ) -> list[NewsItem]:
        """
        Fetch stock news from Eastmoney via AkShare.
        
        Args:
            code6: 6-digit stock code
            max_items: Maximum items to return
        
        Returns:
            List of NewsItem objects
        """
        try:
            import akshare as ak
        except ImportError:
            raise NewsProviderError("akshare not installed. Run: pip install akshare")
        
        logger.debug(f"Fetching news for {code6} from Eastmoney")
        
        try:
            df = ak.stock_news_em(symbol=code6)
        except Exception as e:
            msg = str(e)
            # AkShare may raise regex errors on some environments
            if "invalid escape sequence: \\u" in msg or "Invalid regular expression" in msg:
                logger.warning(
                    f"Eastmoney news fetch failed for {code6} due to regex issue: {e}. "
                    "Skipping Eastmoney for this symbol."
                )
                return []
            raise NewsProviderError(f"Eastmoney news fetch failed for {code6}: {e}") from e
        
        if df is None or df.empty:
            logger.warning(f"No news returned for {code6} from Eastmoney")
            return []
        
        items = []
        for _, row in df.head(max_items).iterrows():
            # Column names may vary: 新闻标题, 新闻内容, 发布时间, 文章来源, 新闻链接
            title = str(row.get("新闻标题", row.get("title", "")))
            content = str(row.get("新闻内容", row.get("content", "")))
            pub_time_str = str(row.get("发布时间", row.get("publish_time", "")))
            url = str(row.get("新闻链接", row.get("url", ""))) or None
            
            pub_time = self._parse_datetime(pub_time_str)
            
            items.append(NewsItem(
                title=title,
                content=content[:500] if content else "",  # Truncate long content
                publish_time=pub_time or datetime.now(),
                source=self.name,
                url=url if url and url != "nan" else None,
                stock_codes=[code6],
                category="news",
            ))
        
        logger.info(f"Fetched {len(items)} news items for {code6} from Eastmoney")
        return items
    
    @retry_news(max_attempts=3, delay=1.0, backoff=2.0)
    def fetch_research_reports(
        self,
        code6: str,
        max_items: int = 30,
    ) -> list[NewsItem]:
        """
        Fetch research reports from Eastmoney via AkShare.
        
        Args:
            code6: 6-digit stock code
            max_items: Maximum items to return
        
        Returns:
            List of NewsItem objects (category='research')
        """
        try:
            import akshare as ak
        except ImportError:
            raise NewsProviderError("akshare not installed")
        
        logger.debug(f"Fetching research reports for {code6} from Eastmoney")
        
        try:
            df = ak.stock_research_report_em(symbol=code6)
        except Exception as e:
            logger.warning(f"Research report fetch failed for {code6}: {e}")
            return []
        
        if df is None or df.empty:
            return []
        
        items = []
        for _, row in df.head(max_items).iterrows():
            title = str(row.get("报告名称", row.get("title", "")))
            org = str(row.get("机构名称", ""))
            date_str = str(row.get("日期", ""))
            
            pub_time = self._parse_datetime(date_str)
            
            # Build content from available fields
            content_parts = []
            if org:
                content_parts.append(f"机构: {org}")
            if row.get("研究员"):
                content_parts.append(f"研究员: {row['研究员']}")
            
            items.append(NewsItem(
                title=title,
                content="; ".join(content_parts) if content_parts else "",
                publish_time=pub_time or datetime.now(),
                source=self.name,
                stock_codes=[code6],
                category="research",
            ))
        
        logger.info(f"Fetched {len(items)} research reports for {code6} from Eastmoney")
        return items
