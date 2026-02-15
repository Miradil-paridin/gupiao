"""
将 watchlist_cache.yaml 的股票池同步到 config.yaml / config_v31.yaml

使用: python sync_watchlist.py
"""
from pathlib import Path
import yaml


def main():
    base_dir = Path(__file__).resolve().parent

    # 读取缓存
    cache_path = base_dir / "watchlist_cache.yaml"
    if not cache_path.exists():
        print("❌ watchlist_cache.yaml 不存在！请先运行 run_update_watchlist.py")
        return

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = yaml.safe_load(f) or {}

    symbols = cache.get("watchlist", [])
    meta = cache.get("meta", {})
    print(f"📂 缓存: {len(symbols)} 只股票 (更新于 {meta.get('updated', '未知')})")

    # 更新到所有配置文件
    for name in ["config_v31.yaml", "config.yaml"]:
        cfg_path = base_dir / name
        if not cfg_path.exists():
            continue

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        old_count = len(cfg.get("watchlist", []))
        cfg["watchlist"] = symbols

        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

        print(f"  ✓ {name}: {old_count} → {len(symbols)} 只")

    print(f"\n✅ 同步完成! 后续运行 run_fetch_daily.py 即可抓取新股票池数据")


if __name__ == "__main__":
    main()