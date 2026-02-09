"""
News aggregator module.
Aggregates news from multiple providers, deduplicates, and prepares for AI consumption.
"""
from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Iterable, Optional

from .news_providers import (
    NewsProvider, NewsItem, NewsProviderError,
    get_news_provider, NEWS_PROVIDERS
)
from .logger import get_logger

logger = get_logger("quant.news_aggregator")


@dataclass
class AggregatedNews:
    """
    Aggregated and processed news for a symbol or market.
    """
    symbol: Optional[str]
    as_of: date
    items: list[NewsItem] = field(default_factory=list)
    summary: str = ""
    sentiment_score: float = 0.0  # -1 to 1
    key_topics: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "item_count": len(self.items),
            "items": [item.to_dict() for item in self.items],
            "summary": self.summary,
            "sentiment_score": self.sentiment_score,
            "key_topics": self.key_topics,
        }


def _compute_hash(title: str, content: str) -> str:
    """Compute hash for deduplication."""
    text = f"{title}:{content[:100]}"
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _simple_sentiment(text: str) -> str:
    """
    Simple rule-based sentiment analysis for Chinese financial text.
    Returns: positive, negative, or neutral
    """
    positive_words = [
        "涨", "上涨", "上升", "增长", "利好", "突破", "新高", "暴涨", "大涨",
        "盈利", "增收", "超预期", "看好", "买入", "推荐", "强势", "回暖",
        "复苏", "景气", "扩张", "增持", "加仓", "资金流入"
    ]
    
    negative_words = [
        "跌", "下跌", "下降", "下滑", "利空", "跌破", "新低", "暴跌", "大跌",
        "亏损", "减收", "不及预期", "看空", "卖出", "减持", "弱势", "疲软",
        "衰退", "萎缩", "清仓", "资金流出", "风险", "警告", "处罚"
    ]
    
    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


def _extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
    """
    Extract key financial terms from text.
    """
    # Common financial keywords to look for
    keyword_patterns = [
        r"业绩[预告|快报|公告]?",
        r"[季|年]报",
        r"分红",
        r"送股",
        r"增发",
        r"回购",
        r"减持",
        r"增持",
        r"并购",
        r"重组",
        r"IPO",
        r"定增",
        r"解禁",
        r"质押",
        r"违规",
        r"处罚",
        r"研发",
        r"产能",
        r"订单",
        r"合同",
        r"中标",
    ]
    
    found = []
    for pattern in keyword_patterns:
        if re.search(pattern, text):
            match = re.search(pattern, text)
            if match:
                found.append(match.group())
    
    return list(set(found))[:max_keywords]


class NewsAggregator:
    """
    Aggregates news from multiple providers and processes them.
    """
    
    def __init__(
        self,
        provider_names: list[str] | None = None,
        dedupe: bool = True,
        max_items_per_provider: int = 30,
    ):
        """
        Initialize the aggregator.
        
        Args:
            provider_names: List of provider names to use (default: all)
            dedupe: Whether to deduplicate news items
            max_items_per_provider: Max items to fetch from each provider
        """
        self.provider_names = provider_names or list(NEWS_PROVIDERS.keys())
        self.dedupe = dedupe
        self.max_items_per_provider = max_items_per_provider
        
        # Initialize providers
        self.providers: list[NewsProvider] = []
        for name in self.provider_names:
            try:
                self.providers.append(get_news_provider(name))
            except ValueError as e:
                logger.warning(f"Skipping unknown provider: {name}")
    
    def fetch_stock_news(
        self,
        code6: str,
        max_total: int = 100,
    ) -> list[NewsItem]:
        """
        Fetch and aggregate news for a specific stock from all providers.
        
        Args:
            code6: 6-digit stock code
            max_total: Maximum total items to return
        
        Returns:
            Deduplicated list of NewsItem objects
        """
        all_items: list[NewsItem] = []
        seen_hashes: set[str] = set()
        
        for provider in self.providers:
            try:
                items = provider.fetch_stock_news(code6, self.max_items_per_provider)
                
                for item in items:
                    if self.dedupe:
                        h = _compute_hash(item.title, item.content)
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)
                    
                    # Add sentiment if not already set
                    if not item.sentiment:
                        item.sentiment = _simple_sentiment(item.title + " " + item.content)
                    
                    # Extract keywords if not already set
                    if not item.keywords:
                        item.keywords = _extract_keywords(item.title + " " + item.content)
                    
                    all_items.append(item)
                    
            except NewsProviderError as e:
                logger.warning(f"Provider {provider.name} failed for {code6}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error from {provider.name} for {code6}: {e}")
        
        # Sort by publish time (newest first)
        all_items.sort(key=lambda x: x.publish_time or datetime.min, reverse=True)
        
        logger.info(f"Aggregated {len(all_items)} news items for {code6}")
        return all_items[:max_total]
    
    def fetch_market_news(
        self,
        max_total: int = 200,
    ) -> list[NewsItem]:
        """
        Fetch and aggregate general market news from all providers.
        
        Args:
            max_total: Maximum total items to return
        
        Returns:
            Deduplicated list of NewsItem objects
        """
        all_items: list[NewsItem] = []
        seen_hashes: set[str] = set()
        
        for provider in self.providers:
            try:
                items = provider.fetch_market_news(self.max_items_per_provider)
                
                for item in items:
                    if self.dedupe:
                        h = _compute_hash(item.title, item.content)
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)
                    
                    if not item.sentiment:
                        item.sentiment = _simple_sentiment(item.title + " " + item.content)
                    
                    all_items.append(item)
                    
            except NewsProviderError as e:
                logger.warning(f"Provider {provider.name} failed for market news: {e}")
            except Exception as e:
                logger.error(f"Unexpected error from {provider.name} for market news: {e}")
        
        all_items.sort(key=lambda x: x.publish_time or datetime.min, reverse=True)
        
        logger.info(f"Aggregated {len(all_items)} market news items")
        return all_items[:max_total]
    
    def aggregate_for_symbol(
        self,
        code6: str,
        as_of: date | None = None,
        max_items: int = 50,
    ) -> AggregatedNews:
        """
        Create an aggregated news bundle for a symbol.
        
        Args:
            code6: 6-digit stock code
            as_of: Reference date (default: today)
            max_items: Maximum items to include
        
        Returns:
            AggregatedNews object with processed news
        """
        as_of = as_of or date.today()
        items = self.fetch_stock_news(code6, max_items)
        
        # Filter to recent items (last 7 days)
        cutoff = datetime.combine(as_of - timedelta(days=7), datetime.min.time())
        recent_items = [
            item for item in items
            if item.publish_time and item.publish_time >= cutoff
        ]
        
        # Calculate sentiment score
        sentiment_map = {"positive": 1, "negative": -1, "neutral": 0}
        if recent_items:
            scores = [sentiment_map.get(item.sentiment, 0) for item in recent_items]
            avg_sentiment = sum(scores) / len(scores)
        else:
            avg_sentiment = 0.0
        
        # Collect all keywords
        all_keywords = []
        for item in recent_items:
            all_keywords.extend(item.keywords)
        
        # Count keyword frequency
        keyword_counts = {}
        for kw in all_keywords:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        
        top_keywords = sorted(keyword_counts.keys(), key=lambda k: keyword_counts[k], reverse=True)[:5]
        
        # Build summary
        summary_parts = []
        if recent_items:
            summary_parts.append(f"最近7天共{len(recent_items)}条新闻")
            pos_count = sum(1 for i in recent_items if i.sentiment == "positive")
            neg_count = sum(1 for i in recent_items if i.sentiment == "negative")
            summary_parts.append(f"正面{pos_count}条,负面{neg_count}条")
            if top_keywords:
                summary_parts.append(f"热点话题:{','.join(top_keywords)}")
        else:
            summary_parts.append("最近7天无相关新闻")
        
        return AggregatedNews(
            symbol=code6,
            as_of=as_of,
            items=recent_items,
            summary="; ".join(summary_parts),
            sentiment_score=avg_sentiment,
            key_topics=top_keywords,
        )


def run_fetch_all_news(
    symbols: Iterable[str],
    base_dir: Path,
    as_of: date | None = None,
    provider_names: list[str] | None = None,
    max_items_per_symbol: int = 50,
    include_market_news: bool = True,
) -> Path:
    """
    Fetch news for all symbols and save to disk.
    
    Args:
        symbols: List of symbols (e.g., ["600519.SH", "000921.SZ"])
        base_dir: Project base directory
        as_of: Reference date (default: today)
        provider_names: List of provider names to use
        max_items_per_symbol: Max news items per symbol
        include_market_news: Whether to also fetch market-wide news
    
    Returns:
        Path to the manifest file
    """
    as_of = as_of or date.today()
    
    out_dir = base_dir / "data" / "news" / as_of.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    aggregator = NewsAggregator(provider_names=provider_names)
    
    manifest = {
        "as_of": as_of.isoformat(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "providers": aggregator.provider_names,
        "symbols": [],
        "market_news": None,
    }
    
    # Fetch per-symbol news
    for sym in symbols:
        code6 = sym.split(".")[0]
        
        try:
            agg_news = aggregator.aggregate_for_symbol(code6, as_of, max_items_per_symbol)
            
            # Save to file
            sym_file = out_dir / f"{sym}.json"
            with open(sym_file, "w", encoding="utf-8") as f:
                json.dump(agg_news.to_dict(), f, ensure_ascii=False, indent=2)
            
            manifest["symbols"].append({
                "symbol": sym,
                "code6": code6,
                "item_count": len(agg_news.items),
                "sentiment_score": agg_news.sentiment_score,
                "key_topics": agg_news.key_topics,
                "file": str(sym_file),
            })
            
            logger.info(f"Saved news for {sym}: {len(agg_news.items)} items")
            
        except Exception as e:
            logger.error(f"Failed to fetch news for {sym}: {e}")
            manifest["symbols"].append({
                "symbol": sym,
                "error": str(e),
            })
    
    # Fetch market news
    if include_market_news:
        try:
            market_items = aggregator.fetch_market_news(max_total=100)
            
            market_file = out_dir / "_market.json"
            with open(market_file, "w", encoding="utf-8") as f:
                json.dump({
                    "as_of": as_of.isoformat(),
                    "item_count": len(market_items),
                    "items": [item.to_dict() for item in market_items],
                }, f, ensure_ascii=False, indent=2)
            
            manifest["market_news"] = {
                "item_count": len(market_items),
                "file": str(market_file),
            }
            
            logger.info(f"Saved market news: {len(market_items)} items")
            
        except Exception as e:
            logger.error(f"Failed to fetch market news: {e}")
            manifest["market_news"] = {"error": str(e)}
    
    # Save manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    logger.info(f"News manifest saved to: {manifest_path}")
    return manifest_path
