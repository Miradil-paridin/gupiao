"""
Fetch news from multiple sources for all watchlist stocks.
+ optional QC cleanup after fetch (drop placeholders/garbled/duplicates/too-old).
+ keep original JSON structure when writing back (dict stays dict, list stays list).
+ 支持 NEWS_SYMBOLS 环境变量：run_all_daily.py 传入时只抓信号前N的股票
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from datetime import date, datetime, timedelta

import yaml
from dotenv import load_dotenv

from quant.news_aggregator import run_fetch_all_news
from quant.logger import setup_logger


# ----------------------------
# Helpers
# ----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _maybe_disable_proxy() -> None:
    if os.getenv("DISABLE_PROXY", "0").strip() in ("1", "true", "True", "YES", "yes"):
        for k in [
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
        ]:
            os.environ.pop(k, None)
        os.environ["NO_PROXY"] = "*"


def _to_symbol(code: str) -> str:
    code = str(code).strip()
    if "." in code:
        return code.upper()

    if code.startswith(("60", "68")):
        return f"{code}.SH"
    if code.startswith(("00", "02", "30")):
        return f"{code}.SZ"
    if code.startswith(("43", "83", "87", "88")):
        return f"{code}.BJ"
    if code.startswith("6"):
        return f"{code}.SH"
    return f"{code}.SZ"


def _read_text_utf8(path: Path) -> str:
    b = path.read_bytes()
    return b.decode("utf-8", errors="replace")


def _write_text_utf8(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _parse_dt_any(s: str) -> datetime | None:
    if not s:
        return None
    s = str(s).strip().replace("/", "-")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    if m:
        try:
            return datetime.fromisoformat(m.group(1))
        except Exception:
            return None
    return None


def _item_time(item: dict) -> datetime | None:
    for k in ["datetime", "time", "date", "publish_time", "pub_time", "pubDate", "created_at", "createdAt"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            dt = _parse_dt_any(v)
            if dt:
                return dt
    return None


def _build_blacklist() -> list[str]:
    raw = os.getenv("NEWS_TITLE_BLACKLIST", "").strip()
    if not raw:
        raw = "友情链接|联系我们|关于我们|免责声明|隐私|法律声明|内容合作|运营许可|站点地图|客服|广告服务|投稿|APP下载"
    return [x.strip() for x in raw.split("|") if x.strip()]


def _is_bad_title(title: str, blacklist: list[str], min_len: int, drop_garbled: bool) -> bool:
    if not title:
        return True
    t = title.strip()
    if len(t) < min_len:
        return True
    if drop_garbled and "�" in t:
        return True
    for b in blacklist:
        if b and b in t:
            return True
    return False


# ----------------------------
# JSON structure-preserving loader/dumper
# ----------------------------
def _load_items_from_file(path: Path):
    """
    Return (items, kind, root_obj, items_key)
    - kind: 'jsonl' or 'json'
    - root_obj: original JSON root (dict/list) for json; None for jsonl
    - items_key:
        * if root is dict and items stored under a key, return that key
        * if root is list, None
        * if root is dict mapping symbol -> list, '__symbol_map__'
    """
    if path.suffix.lower() == ".jsonl":
        items = []
        for line in _read_text_utf8(path).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
            except Exception:
                continue
        return items, "jsonl", None, None

    # .json
    try:
        root = json.loads(_read_text_utf8(path))
    except Exception:
        return [], "json", None, None

    if isinstance(root, list):
        return [x for x in root if isinstance(x, dict)], "json", root, None

    if isinstance(root, dict):
        for k in ["items", "data", "news", "list"]:
            v = root.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)], "json", root, k

        # dict mapping symbol -> list
        if root and all(isinstance(v, list) for v in root.values()):
            # we will handle this case separately in QC (keep mapping)
            return [], "json", root, "__symbol_map__"

        # fallback: treat dict itself as one item
        return [root], "json", root, None

    return [], "json", None, None


def _dump_items_to_file(path: Path, items: list[dict], kind: str, root_obj=None, items_key=None) -> None:
    if kind == "jsonl":
        lines = [json.dumps(x, ensure_ascii=False) for x in items]
        _write_text_utf8(path, "\n".join(lines) + ("\n" if lines else ""))
        return

    # json: keep original structure if possible
    if isinstance(root_obj, dict) and items_key and items_key not in (None, "__symbol_map__"):
        root_obj[items_key] = items
        _write_text_utf8(path, json.dumps(root_obj, ensure_ascii=False, indent=2))
        return

    # root list or unknown -> write list
    _write_text_utf8(path, json.dumps(items, ensure_ascii=False, indent=2))


# ----------------------------
# QC Cleaner
# ----------------------------
def _qc_clean_news_folder(news_dir: Path, as_of: date, logger) -> Path | None:
    enable = os.getenv("NEWS_QC_ENABLE", "1").strip() in ("1", "true", "True", "YES", "yes")
    if not enable:
        logger.info("NEWS_QC_ENABLE=0, skip QC.")
        return None

    max_age_days = int(os.getenv("NEWS_MAX_AGE_DAYS", "7"))
    min_len = int(os.getenv("NEWS_MIN_TITLE_LEN", "6"))
    drop_garbled = os.getenv("NEWS_DROP_GARBLED", "1").strip() in ("1", "true", "True", "YES", "yes")
    backup_raw = os.getenv("NEWS_QC_BACKUP_RAW", "1").strip() in ("1", "true", "True", "YES", "yes")
    blacklist = _build_blacklist()

    cutoff_dt = datetime.combine(as_of, datetime.min.time()) - timedelta(days=max_age_days)

    raw_dir = news_dir / "_raw"
    if backup_raw:
        _ensure_dir(raw_dir)

    report = {
        "as_of": str(as_of),
        "news_dir": str(news_dir),
        "max_age_days": max_age_days,
        "min_title_len": min_len,
        "drop_garbled": drop_garbled,
        "blacklist_count": len(blacklist),
        "files": [],
    }

    files = []
    for p in news_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name.lower().startswith("manifest"):
            continue
        if p.name.lower().endswith(".log"):
            continue
        if p.suffix.lower() not in (".json", ".jsonl"):
            continue
        if "_raw" in p.parts:
            continue
        files.append(p)

    if not files:
        logger.info("No news files found for QC.")
        return None

    logger.info(f"QC cleaning {len(files)} news files...")

    for p in files:
        items, kind, root_obj, items_key = _load_items_from_file(p)

        # Special: symbol->list map, keep mapping
        if isinstance(root_obj, dict) and items_key == "__symbol_map__":
            before_total = 0
            after_total = 0
            dropped_total = {"bad_title": 0, "too_old": 0, "dup": 0}

            new_map = {}
            for sym, lst in root_obj.items():
                if not isinstance(lst, list):
                    continue
                before_total += len(lst)

                kept = []
                seen = set()
                dropped = {"bad_title": 0, "too_old": 0, "dup": 0}

                for it in lst:
                    if not isinstance(it, dict):
                        continue
                    title = (it.get("title") or it.get("headline") or "").strip()
                    source = (it.get("source") or it.get("provider") or "").strip()
                    dt = _item_time(it)

                    if _is_bad_title(title, blacklist, min_len, drop_garbled):
                        dropped["bad_title"] += 1
                        continue
                    if dt and dt < cutoff_dt:
                        dropped["too_old"] += 1
                        continue
                    key = (title, source, dt.isoformat() if dt else "")
                    if key in seen:
                        dropped["dup"] += 1
                        continue
                    seen.add(key)
                    kept.append(it)

                after_total += len(kept)
                for k in dropped_total:
                    dropped_total[k] += dropped[k]

                new_map[sym] = kept

            # backup
            if backup_raw:
                rel = p.relative_to(news_dir)
                backup_path = raw_dir / rel
                _ensure_dir(backup_path.parent)
                if not backup_path.exists():
                    shutil.copy2(p, backup_path)

            # write back map
            _write_text_utf8(p, json.dumps(new_map, ensure_ascii=False, indent=2))

            report["files"].append(
                {
                    "file": str(p),
                    "kind": "json(symbol_map)",
                    "before": before_total,
                    "after": after_total,
                    "dropped": dropped_total,
                }
            )
            continue

        before = len(items)

        kept = []
        seen = set()
        dropped = {"bad_title": 0, "too_old": 0, "dup": 0}

        for it in items:
            title = (it.get("title") or it.get("headline") or "").strip()
            source = (it.get("source") or it.get("provider") or "").strip()
            dt = _item_time(it)

            if _is_bad_title(title, blacklist, min_len, drop_garbled):
                dropped["bad_title"] += 1
                continue
            if dt and dt < cutoff_dt:
                dropped["too_old"] += 1
                continue

            key = (title, source, dt.isoformat() if dt else "")
            if key in seen:
                dropped["dup"] += 1
                continue
            seen.add(key)
            kept.append(it)

        after = len(kept)

        # backup raw
        if backup_raw:
            rel = p.relative_to(news_dir)
            backup_path = raw_dir / rel
            _ensure_dir(backup_path.parent)
            if not backup_path.exists():
                shutil.copy2(p, backup_path)

        # overwrite cleaned (preserve structure)
        _dump_items_to_file(p, kept, kind, root_obj=root_obj, items_key=items_key)

        report["files"].append(
            {
                "file": str(p),
                "kind": kind,
                "before": before,
                "after": after,
                "dropped": dropped,
            }
        )

    qc_path = news_dir / "qc_report.json"
    _write_text_utf8(qc_path, json.dumps(report, ensure_ascii=False, indent=2))
    logger.info(f"QC report written: {qc_path}")
    return qc_path


# ----------------------------
# Main
# ----------------------------
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="", help="YYYY-MM-DD. default=today")
    return ap.parse_args()


def main() -> None:
    load_dotenv()
    _maybe_disable_proxy()
    args = _parse_args()

    base_dir = Path(__file__).resolve().parent

    log_dir = base_dir / "data" / "logs"
    _ensure_dir(log_dir)
    log_file = log_dir / "fetch_news.log"
    logger = setup_logger("quant", log_file=log_file)

    # 优先读 config_v31.yaml
    cfg_path = base_dir / "config.yaml"
    for name in ["config_v31.yaml", "config.yaml"]:
        p = base_dir / name
        if p.exists():
            cfg_path = p
            break
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 优先从环境变量读取（run_all_daily.py 传入的 top N 信号股）
    news_symbols_env = os.getenv("NEWS_SYMBOLS", "").strip()
    if news_symbols_env:
        codes = [c.strip() for c in news_symbols_env.split(",") if c.strip()]
        logger.info(f"📰 NEWS_SYMBOLS: 只抓 {len(codes)} 只信号股的新闻")
        print(f"📰 精准模式: 只抓 {len(codes)} 只信号股的新闻")
    else:
        codes = cfg.get("watchlist", [])
        logger.info(f"📰 全量模式: 抓 {len(codes)} 只股票的新闻")
        print(f"📰 全量模式: 抓 {len(codes)} 只股票的新闻")
    symbols = [_to_symbol(c) for c in codes if str(c).strip()]

    news_cfg = cfg.get("news", {})
    provider_names = news_cfg.get("providers", ["eastmoney", "cls", "sina", "ths"])
    max_items = int(news_cfg.get("max_items_per_symbol", 50))
    include_market = bool(news_cfg.get("include_market_news", True))

    cfg_max_age = news_cfg.get("max_age_days")
    if cfg_max_age is not None and not os.getenv("NEWS_MAX_AGE_DAYS"):
        os.environ["NEWS_MAX_AGE_DAYS"] = str(int(cfg_max_age))

    if args.as_of.strip():
        as_of = datetime.strptime(args.as_of.strip(), "%Y-%m-%d").date()
    elif os.getenv("NEWS_AS_OF", "").strip():
        as_of = datetime.strptime(os.getenv("NEWS_AS_OF").strip(), "%Y-%m-%d").date()
    else:
        as_of = date.today()

    logger.info(f"Fetching news for {len(symbols)} symbols")
    logger.info(f"Providers: {provider_names}")
    logger.info(f"Max items per symbol: {max_items}, include market news: {include_market}")
    logger.info(f"As of: {as_of}, NEWS_MAX_AGE_DAYS: {os.getenv('NEWS_MAX_AGE_DAYS', '7')}")

    manifest_path = run_fetch_all_news(
        symbols=symbols,
        base_dir=base_dir,
        as_of=as_of,
        provider_names=provider_names,
        max_items_per_symbol=max_items,
        include_market_news=include_market,
    )

    news_dir = Path(manifest_path).parent if manifest_path else (base_dir / "data" / "news" / str(as_of))
    qc_path = None
    try:
        if news_dir.exists():
            qc_path = _qc_clean_news_folder(news_dir, as_of=as_of, logger=logger)
    except Exception as e:
        logger.exception(f"QC failed (ignored): {e}")

    print("\nDone.")
    print(f"News manifest saved to: {manifest_path}")
    if qc_path:
        print(f"News QC report saved to: {qc_path}")
        print(f"Raw backups (if enabled) under: {news_dir / '_raw'}")


if __name__ == "__main__":
    main()