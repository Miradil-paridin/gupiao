# quant/fetch_evidence.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

try:
    import akshare as ak
except Exception as e:
    raise RuntimeError("AkShare is required. Activate .venv then pip install akshare") from e


# ----------------------------
# helpers
# ----------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def ymd(d: date | datetime | str) -> str:
    if isinstance(d, str):
        return d
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    return d.strftime("%Y-%m-%d")


def code6_from_symbol(symbol: str) -> str:
    # "600519.SH" -> "600519"
    return symbol.split(".")[0].strip()


def em_prefix_symbol(symbol: str) -> str:
    # "688981.SH" -> "sh688981"; "002460.SZ" -> "sz002460"
    code6, mkt = symbol.split(".")
    mkt = mkt.upper()
    if mkt == "SH":
        return f"sh{code6}"
    if mkt == "SZ":
        return f"sz{code6}"
    # fallback
    return code6


def last_quarter_end(as_of: date) -> str:
    # returns YYYYMMDD of the most recent completed quarter end
    y = as_of.year
    m = as_of.month
    if m <= 3:
        return f"{y-1}1231"
    if m <= 6:
        return f"{y}0331"
    if m <= 9:
        return f"{y}0630"
    return f"{y}0930"


def safe_to_jsonl(df: pd.DataFrame, path: Path) -> int:
    if df is None or df.empty:
        path.write_text("", encoding="utf-8")
        return 0
    records = df.to_dict(orient="records")
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def sleep_polite(i: int, base: float = 0.6) -> None:
    # simple jitterless pacing; enough to avoid hammering
    time.sleep(base + 0.15 * (i % 3))


@dataclass
class EvidenceConfig:
    # evidence window
    cninfo_days: int = 120          # pull last ~120 days of公告
    max_news_items: int = 60        # keep top N recent news
    max_reports_items: int = 80     # keep top N research reports (often huge)
    # pacing
    pause_seconds: float = 0.7


# ----------------------------
# fetchers (AkShare)
# ----------------------------
def fetch_news_em(code6: str) -> pd.DataFrame:
    # 东方财富指定个股新闻
    df = ak.stock_news_em(symbol=code6)  # columns: 新闻标题/新闻内容/发布时间/文章来源/新闻链接...
    return df


def fetch_cninfo_disclosures(code6: str, start_date: str, end_date: str, category: str = "") -> pd.DataFrame:
    # 巨潮资讯公告查询
    df = ak.stock_zh_a_disclosure_report_cninfo(
        symbol=code6,
        market="沪深京",
        keyword="",
        category=category,
        start_date=start_date,
        end_date=end_date,
    )
    return df


def fetch_research_reports_em(code6: str) -> pd.DataFrame:
    # 东方财富研究报告-个股研报（含 PDF 链接）
    df = ak.stock_research_report_em(symbol=code6)
    return df


def fetch_top10_shareholders_em(symbol_with_prefix: str, quarter_end_yyyymmdd: str) -> pd.DataFrame:
    # 十大股东（需要 sh/sz 前缀 + 季度末日期）
    df = ak.stock_gdfx_top_10_em(symbol=symbol_with_prefix, date=quarter_end_yyyymmdd)
    return df


def fetch_shareholder_counts_em(quarter_end_yyyymmdd: str) -> pd.DataFrame:
    # 股东户数（一次返回全市场，需过滤）
    df = ak.stock_zh_a_gdhs(symbol=quarter_end_yyyymmdd)
    return df


# ----------------------------
# main pipeline
# ----------------------------
def run_fetch_evidence(
    symbols: Iterable[str],
    base_dir: Path,
    as_of: date,
    cfg: Optional[EvidenceConfig] = None,
) -> Path:
    cfg = cfg or EvidenceConfig()

    out_day_dir = ensure_dir(base_dir / "data" / "evidence" / ymd(as_of))
    manifest = {
        "as_of": ymd(as_of),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": [],
        "cross": {},
    }

    # compute CNINFO window
    end = as_of.strftime("%Y%m%d")
    start_dt = as_of - pd.Timedelta(days=cfg.cninfo_days)
    start = pd.Timestamp(start_dt).strftime("%Y%m%d")

    q_end = last_quarter_end(as_of)

    # 1) shareholder counts (cross file)
    try:
        gdhs_all = fetch_shareholder_counts_em(q_end)
        keep_codes = {code6_from_symbol(s) for s in symbols}
        gdhs_keep = gdhs_all[gdhs_all["代码"].astype(str).isin(keep_codes)].copy()
        cross_dir = ensure_dir(out_day_dir / "_cross")
        gdhs_path = cross_dir / f"shareholder_counts_{q_end}.csv"
        gdhs_keep.to_csv(gdhs_path, index=False, encoding="utf-8-sig")
        manifest["cross"]["shareholder_counts"] = str(gdhs_path)
        manifest["cross"]["shareholder_counts_rows"] = int(gdhs_keep.shape[0])
    except Exception as e:
        manifest["cross"]["shareholder_counts_error"] = repr(e)

    # 2) per-symbol items
    for i, sym in enumerate(symbols):
        code6 = code6_from_symbol(sym)
        sym_dir = ensure_dir(out_day_dir / sym)

        sym_meta = {
            "symbol": sym,
            "code6": code6,
            "paths": {},
            "counts": {},
            "errors": {},
            "quarter_end": q_end,
        }

        # news
        try:
            news = fetch_news_em(code6)
            if not news.empty and "发布时间" in news.columns:
                # normalize & keep recent items
                news = news.copy()
                news["发布时间"] = news["发布时间"].astype(str)
                news = news.head(cfg.max_news_items)
            p = sym_dir / "news_em.jsonl"
            sym_meta["counts"]["news_em"] = safe_to_jsonl(news, p)
            sym_meta["paths"]["news_em"] = str(p)
        except Exception as e:
            sym_meta["errors"]["news_em"] = repr(e)

        sleep_polite(i, base=cfg.pause_seconds)

        # CNINFO disclosures
        try:
            cn = fetch_cninfo_disclosures(code6, start_date=start, end_date=end, category="")
            p = sym_dir / "cninfo.jsonl"
            sym_meta["counts"]["cninfo"] = safe_to_jsonl(cn, p)
            sym_meta["paths"]["cninfo"] = str(p)
        except Exception as e:
            sym_meta["errors"]["cninfo"] = repr(e)

        sleep_polite(i + 1, base=cfg.pause_seconds)

        # research reports
        try:
            rr = fetch_research_reports_em(code6)
            if not rr.empty and "日期" in rr.columns:
                rr = rr.copy()
                rr["日期"] = rr["日期"].astype(str)
                rr = rr.head(cfg.max_reports_items)
            p = sym_dir / "research_em.jsonl"
            sym_meta["counts"]["research_em"] = safe_to_jsonl(rr, p)
            sym_meta["paths"]["research_em"] = str(p)
        except Exception as e:
            sym_meta["errors"]["research_em"] = repr(e)

        sleep_polite(i + 2, base=cfg.pause_seconds)

        # top10 shareholders snapshot
        try:
            top10 = fetch_top10_shareholders_em(em_prefix_symbol(sym), q_end)
            p = sym_dir / "top10_shareholders.json"
            if top10 is None or top10.empty:
                p.write_text("[]", encoding="utf-8")
                sym_meta["counts"]["top10_shareholders"] = 0
            else:
                p.write_text(top10.to_json(orient="records", force_ascii=False), encoding="utf-8")
                sym_meta["counts"]["top10_shareholders"] = int(top10.shape[0])
            sym_meta["paths"]["top10_shareholders"] = str(p)
        except Exception as e:
            sym_meta["errors"]["top10_shareholders"] = repr(e)

        manifest["symbols"].append(sym_meta)

    # write manifest
    manifest_path = out_day_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
