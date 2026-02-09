from __future__ import annotations

import json
import os
import re
from pathlib import Path
from datetime import date, datetime
from typing import Optional, Any

import pandas as pd

from .logger import get_logger

logger = get_logger("quant.briefs")


# ----------------------------
# Helpers
# ----------------------------
def _read_json_any(path: Path) -> Any:
    """
    Read .json or .jsonl with best-effort decoding.
    - .json: returns object
    - .jsonl: returns list[dict]
    """
    if path.suffix.lower() == ".jsonl":
        items = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
                except Exception:
                    continue
        return items

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def _pick_news_file(news_dir: Path, symbol: str) -> Optional[Path]:
    """
    Prefer symbol.json, fallback symbol.jsonl
    """
    p1 = news_dir / f"{symbol}.json"
    if p1.exists():
        return p1
    p2 = news_dir / f"{symbol}.jsonl"
    if p2.exists():
        return p2
    return None


def _build_blacklist() -> list[str]:
    raw = os.getenv("NEWS_TITLE_BLACKLIST", "").strip()
    if not raw:
        # 兜底：就算没配 env，也过滤最常见的导航垃圾
        raw = "友情链接|联系我们|关于我们|免责声明|隐私|法律声明|内容合作|运营许可|站点地图|客服|广告服务|投稿|APP下载"
    return [x.strip() for x in raw.split("|") if x.strip()]


def _is_bad_title(title: str, blacklist: list[str], min_len: int = 6) -> bool:
    if not title:
        return True
    t = str(title).strip()
    if len(t) < min_len:
        return True
    if "�" in t:  # 乱码替换符
        return True
    for b in blacklist:
        if b and b in t:
            return True
    return False


def _normalize_news_items(news_data: Any) -> list[dict]:
    """
    Accept:
      - dict with items/data/news/list -> list[dict]
      - list[dict]
      - dict mapping symbol -> list[dict]  (rare)
    Return list[dict]
    """
    if news_data is None:
        return []

    if isinstance(news_data, list):
        return [x for x in news_data if isinstance(x, dict)]

    if isinstance(news_data, dict):
        for k in ("items", "data", "news", "list"):
            v = news_data.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        # symbol -> list map
        if news_data and all(isinstance(v, list) for v in news_data.values()):
            out = []
            for v in news_data.values():
                out.extend([x for x in v if isinstance(x, dict)])
            return out

    return []


def _sentiment_of_item(it: dict) -> float:
    """
    Convert common sentiment representations to a numeric score in [-1, 1].
    """
    v = it.get("sentiment")
    if v is None:
        v = it.get("sentiment_label") or it.get("label")

    if isinstance(v, (int, float)):
        # assume already a score
        vv = float(v)
        if vv > 1:
            vv = 1
        if vv < -1:
            vv = -1
        return vv

    if isinstance(v, str):
        s = v.lower()
        if "pos" in s or "正" in s:
            return 0.6
        if "neg" in s or "负" in s:
            return -0.6
        if "neu" in s or "中" in s:
            return 0.0

    return 0.0


def _overall_sentiment(score: float) -> str:
    if score > 0.3:
        return "positive"
    if score < -0.3:
        return "negative"
    return "neutral"


def _summarize_news(news_data: Any) -> dict:
    """
    Create a summary of news data suitable for AI consumption.
    Compatible with dict/list/jsonl outputs.
    """
    if not news_data:
        return {
            "has_news": False,
            "summary": "无相关新闻",
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "key_topics": [],
            "headlines": [],
            "item_count": 0,
        }

    blacklist = _build_blacklist()
    min_len = int(os.getenv("NEWS_MIN_TITLE_LEN", "6"))

    items = _normalize_news_items(news_data)

    # Filter bad titles again (double safety)
    cleaned = []
    for it in items:
        title = (it.get("title") or it.get("headline") or "").strip()
        if _is_bad_title(title, blacklist, min_len=min_len):
            continue
        cleaned.append(it)

    # headlines: top 5 (best-effort: prefer most recent if "time"/"datetime"/"date" exists)
    def _dt_key(it: dict) -> str:
        for k in ("datetime", "time", "date", "publish_time", "pub_time", "created_at"):
            v = it.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    cleaned_sorted = sorted(cleaned, key=_dt_key, reverse=True)

    headlines = []
    for it in cleaned_sorted[:5]:
        title = (it.get("title") or it.get("headline") or "").strip()
        if not title:
            continue
        sentiment_label = it.get("sentiment")
        if sentiment_label is None:
            sentiment_label = _overall_sentiment(_sentiment_of_item(it))
        headlines.append(
            {
                "title": title[:100],
                "sentiment": sentiment_label if isinstance(sentiment_label, str) else _overall_sentiment(_sentiment_of_item(it)),
                "source": (it.get("source") or it.get("provider") or "").strip(),
            }
        )

    # sentiment_score: use existing if present, else compute average
    sentiment_score = 0.0
    if isinstance(news_data, dict) and isinstance(news_data.get("sentiment_score"), (int, float)):
        sentiment_score = float(news_data.get("sentiment_score", 0.0))
    else:
        if cleaned:
            sentiment_score = sum(_sentiment_of_item(it) for it in cleaned) / max(1, len(cleaned))
        else:
            sentiment_score = 0.0

    overall = _overall_sentiment(sentiment_score)

    summary_text = ""
    if isinstance(news_data, dict):
        summary_text = str(news_data.get("summary", "") or "")

    return {
        "has_news": len(cleaned) > 0,
        "item_count": len(cleaned),
        "summary": summary_text if summary_text else ("有新闻但未提供摘要" if cleaned else "无相关新闻"),
        "sentiment": overall,
        "sentiment_score": float(sentiment_score),
        "key_topics": (news_data.get("key_topics", []) if isinstance(news_data, dict) else []),
        "headlines": headlines,
    }


def _reason_row(row: pd.Series) -> list[str]:
    reasons = []

    # Trend / momentum
    if row.get("trend_up", 0) == 1:
        reasons.append(f"Trend: close above MA20 (ma_dist_20={row['ma_dist_20']:.2%}).")
    else:
        reasons.append(f"Trend: close below MA20 (ma_dist_20={row['ma_dist_20']:.2%}).")

    if row.get("ret_20d") is not None and pd.notna(row["ret_20d"]):
        reasons.append(f"Momentum(20d): {row['ret_20d']:.2%}.")
    if row.get("ret_60d") is not None and pd.notna(row["ret_60d"]):
        reasons.append(f"Momentum(60d): {row['ret_60d']:.2%}.")

    # Risk
    if row.get("risk_high", 0) == 1:
        reasons.append(f"Risk: high volatility (vol_20d={row['vol_20d']:.2f}).")
    else:
        reasons.append(f"Risk: vol_20d={row['vol_20d']:.2f}, atr%={row['atr_pct']:.2%}.")

    # Liquidity/attention proxy
    if row.get("vol_ratio_20") is not None and pd.notna(row["vol_ratio_20"]):
        reasons.append(f"Volume surprise: vol_ratio_20={row['vol_ratio_20']:.2f}.")

    # Action rationale
    action = row["action"]
    if action == "INVEST_MORE":
        reasons.append("Action rationale: top-ranked among the basket with positive trend and acceptable risk.")
    elif action == "REDUCE":
        reasons.append("Action rationale: strong trend/momentum but volatility is high; reduce/avoid adding aggressively.")
    elif action == "WITHDRAW":
        reasons.append("Action rationale: trend is down with weak momentum and/or score is very poor; exit/avoid.")
    elif action == "LEAST":
        reasons.append("Action rationale: relatively weak score vs peers (not a forced withdraw).")
    else:
        reasons.append("Action rationale: mixed signals; hold/observe.")

    return reasons


def _pick_latest_news_dir(news_root: Path, as_of: str) -> Optional[Path]:
    """
    Choose latest news dir <= as_of. If none, fallback to the latest available.
    """
    if not news_root.exists():
        return None
    dirs = [p for p in news_root.iterdir() if p.is_dir()]
    if not dirs:
        return None

    def parse_name(p: Path) -> Optional[date]:
        try:
            return datetime.strptime(p.name, "%Y-%m-%d").date()
        except Exception:
            return None

    target = None
    try:
        target = datetime.strptime(as_of, "%Y-%m-%d").date()
    except Exception:
        target = None

    dated = [(p, parse_name(p)) for p in dirs]
    dated = [(p, d) for (p, d) in dated if d is not None]
    if not dated:
        # no parseable date dirs, use lexicographic fallback
        return sorted(dirs, key=lambda p: p.name)[-1]

    if target is not None:
        candidates = [p for (p, d) in dated if d <= target]
        if candidates:
            return sorted(candidates, key=lambda p: p.name)[-1]

    # fallback: latest overall
    return sorted([p for (p, d) in dated], key=lambda p: p.name)[-1]


def build_ai_briefs(
    ranking_csv: Path,
    out_dir: Path,
    base_dir: Path | None = None,
    include_news: bool = True,
    include_all_bundle: bool = True,
) -> Path:
    df = pd.read_csv(ranking_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    as_of = df["date"].iloc[0].isoformat()

    day_dir = out_dir / as_of
    day_dir.mkdir(parents=True, exist_ok=True)

    # Try to find news data
    news_dir = None
    if include_news and base_dir:
        expected = base_dir / "data" / "news" / as_of
        if expected.exists():
            news_dir = expected
        else:
            news_root = base_dir / "data" / "news"
            picked = _pick_latest_news_dir(news_root, as_of)
            if picked:
                logger.warning(
                    f"News directory not found for {as_of}. Falling back to: {picked.name}"
                )
                news_dir = picked
            else:
                logger.warning(f"News directory not found: {expected}")
                news_dir = None

    bundle = {
        "as_of": as_of,
        "universe_size": int(len(df)),
        "has_news_data": news_dir is not None,
        "universe": [],
    }

    for _, row in df.iterrows():
        symbol = str(row["symbol"])

        # Load news data for this symbol
        news_summary = {"has_news": False, "summary": "无新闻数据", "sentiment": "neutral", "sentiment_score": 0.0, "headlines": []}
        if news_dir:
            p = _pick_news_file(news_dir, symbol)
            if p:
                try:
                    news_data = _read_json_any(p)
                except Exception as e:
                    logger.warning(f"Failed to load news for {symbol}: {e}")
                    news_data = None
            else:
                news_data = None
            news_summary = _summarize_news(news_data)

        brief = {
            "symbol": symbol,
            "as_of": as_of,
            "action": row["action"],
            "rank": int(row["rank"]),
            "score": float(row["score"]),
            "snapshot": {
                "close": float(row["close"]),
                "ma_dist_20": float(row["ma_dist_20"]),
                "ret_20d": float(row["ret_20d"]),
                "ret_60d": float(row["ret_60d"]),
                "vol_20d": float(row["vol_20d"]),
                "atr_pct": float(row["atr_pct"]),
                "vol_ratio_20": float(row["vol_ratio_20"]),
            },
            "flags": {
                "trend_up": int(row.get("trend_up", 0)),
                "mom_bad": int(row.get("mom_bad", 0)),
                "risk_high": int(row.get("risk_high", 0)),
            },
            "news": news_summary,
            "reasons": _reason_row(row),
            "notes": [
                "Includes quant signals (price/volume) and news sentiment.",
                "Use as a decision aid; not financial advice.",
            ],
        }

        with open(day_dir / f"{symbol}.json", "w", encoding="utf-8") as f:
            json.dump(brief, f, ensure_ascii=False, indent=2)

        bundle["universe"].append(brief)

    # Market news summary
    if news_dir:
        market_news_file_json = news_dir / "_market.json"
        market_news_file_jsonl = news_dir / "_market.jsonl"
        market_data = None
        try:
            if market_news_file_json.exists():
                market_data = _read_json_any(market_news_file_json)
            elif market_news_file_jsonl.exists():
                market_data = _read_json_any(market_news_file_jsonl)
        except Exception as e:
            logger.warning(f"Failed to load market news: {e}")
            market_data = None

        if market_data:
            market_sum = _summarize_news(market_data)
            # only keep compact headlines
            bundle["market_news"] = {
                "item_count": market_sum.get("item_count", 0),
                "sentiment": market_sum.get("sentiment", "neutral"),
                "sentiment_score": market_sum.get("sentiment_score", 0.0),
                "headlines": market_sum.get("headlines", [])[:5],
            }

    if include_all_bundle:
        all_path = day_dir / "ALL.json"
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)
        return all_path

    return day_dir
