"""
将 watchlist_cache.yaml 的股票池同步到配置文件。

默认优先级:
1) --config 显式指定
2) PIPELINE_CONFIG 环境变量
3) config.yaml
4) config_v31.yaml

使用:
  python sync_watchlist.py
  python sync_watchlist.py --config config_v31.yaml
  python sync_watchlist.py --all-configs
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def _enable_windows_utf8_console() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except Exception:
        pass


def _resolve_config_path(base_dir: Path, config_arg: str = "") -> Path:
    explicit = str(config_arg or "").strip()
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = base_dir / p
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        return p

    env_cfg = str(os.getenv("PIPELINE_CONFIG", "")).strip()
    if env_cfg:
        p = Path(env_cfg)
        if not p.is_absolute():
            p = base_dir / p
        if p.exists():
            return p

    for name in ["config.yaml", "config_v31.yaml"]:
        p = base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError("no config file found (expected config.yaml or config_v31.yaml)")


def _target_config_paths(base_dir: Path, config_arg: str = "", all_configs: bool = False) -> list[Path]:
    if all_configs:
        paths = [base_dir / "config.yaml", base_dir / "config_v31.yaml"]
        return [p for p in paths if p.exists()]
    return [_resolve_config_path(base_dir, config_arg=config_arg)]


def main() -> None:
    _enable_windows_utf8_console()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="Sync watchlist_cache.yaml into project config.")
    ap.add_argument("--config", default="", help="target config path")
    ap.add_argument("--all-configs", action="store_true", help="sync both config.yaml and config_v31.yaml if present")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent

    cache_path = base_dir / "watchlist_cache.yaml"
    if not cache_path.exists():
        print("[错误] watchlist_cache.yaml 不存在！请先运行 run_update_watchlist.py")
        return

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = yaml.safe_load(f) or {}

    symbols = cache.get("watchlist", [])
    meta = cache.get("meta", {})
    print(f"[缓存] {len(symbols)} 只股票 (更新于 {meta.get('updated', '未知')})")

    targets = _target_config_paths(base_dir, config_arg=args.config, all_configs=bool(args.all_configs))
    if not targets:
        print("[错误] 未找到可同步的配置文件")
        return

    for cfg_path in targets:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        old_count = len(cfg.get("watchlist", []))
        cfg["watchlist"] = symbols

        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

        print(f"  [OK] {cfg_path.name}: {old_count} → {len(symbols)} 只")

    print("\n[完成] 同步完成! 后续运行 run_fetch_daily.py 即可抓取新股票池数据")


if __name__ == "__main__":
    main()
