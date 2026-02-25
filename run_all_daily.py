"""
一键启动 A股量化日度流程（v3.2 优化版）

功能：
- 多数据源行情抓取（并行+缓存+增量更新）
- 特征工程（含涨停+TDX月线指标）
- 策略信号生成（v3策略）
- 新闻聚合（仅抓取信号排名前N的股票，非全覆盖）
- AI研报生成（优化版，含可视化+质量评估）

日常流程：
  python run_all_daily.py           # 正常运行
  python run_all_daily.py --fast    # 快速模式（跳过新闻+图表）

股票池更新（每周/月）：
  python run_update_watchlist.py
  python sync_watchlist.py
  python run_fetch_daily.py         # 全量拉取新股票池
  python run_build_market_daily_all.py
  python run_build_features_daily.py

支持多模型：通过 .env 中的 LLM_PROVIDER 切换 deepseek / mimo
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml
from dotenv import load_dotenv


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


# Force UTF-8 console I/O on Windows to avoid emoji/Unicode print failures.
_enable_windows_utf8_console()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def run_step(
    base_dir: Path,
    script_name: str,
    env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> None:
    """运行单个步骤"""
    script_path = base_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"🚀 RUN: {script_name}")
    print(f"{'='*60}")

    start_time = time.time()
    proc = subprocess.run(cmd, cwd=str(base_dir), env=env, text=True)
    elapsed = time.time() - start_time

    if proc.returncode != 0:
        raise RuntimeError(f"Step failed: {script_name} (exit={proc.returncode})")

    print(f"✅ 完成 {script_name} (耗时: {elapsed:.1f}秒)")


def read_latest_rank_as_of(base_dir: Path) -> str | None:
    """读取最新信号日期"""
    p = base_dir / "data" / "signals" / "latest_daily_rank.csv"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            first = next(reader, None)
            if not first:
                return None
            d = (first.get("date") or "").strip()
            if not d:
                return None
            d = d.replace("/", "-")
            datetime.fromisoformat(d)
            return d
    except Exception:
        return None


def _first_day_with_month_offset(d: date, month_offset: int) -> date:
    """Return the first day of month after applying month offset to d's month."""
    month_index = d.year * 12 + (d.month - 1) + int(month_offset)
    y = month_index // 12
    m = (month_index % 12) + 1
    return date(y, m, 1)


def compute_trade_journal_window(as_of_text: str, months: int = 3) -> tuple[str, str]:
    """
    Build rolling natural-month window for trade journal.
    Example:
      as_of=2026-03-xx, months=3 -> start=2026-01-01, end=2026-03-xx
      as_of=2026-04-xx, months=3 -> start=2026-02-01, end=2026-04-xx
    """
    m = max(int(months), 1)
    try:
        end_dt = datetime.fromisoformat(str(as_of_text).strip()).date()
    except Exception:
        end_dt = date.today()
    start_dt = _first_day_with_month_offset(end_dt, -(m - 1))
    return start_dt.isoformat(), end_dt.isoformat()


def _parse_iso_date(s: str) -> date | None:
    t = str(s or "").strip()
    if not t:
        return None
    try:
        return datetime.fromisoformat(t).date()
    except Exception:
        return None


def resolve_config_path(base_dir: Path, config_arg: str = "") -> Path:
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


def load_config(cfg_path: Path) -> dict:
    """加载配置文件。"""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    print(f"  📂 配置: {cfg_path.name}")
    return cfg


def _clamp01(v: object, default: float | None = None) -> float | None:
    try:
        x = float(v)
    except Exception:
        return default
    if x != x:  # NaN
        return default
    return float(min(1.0, max(0.0, x)))


def load_strategy_process_summary(base_dir: Path) -> dict | None:
    p = base_dir / "data" / "backtests" / "strategy_process_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def derive_gate_controls(summary: dict | None) -> tuple[str, str, float | None]:
    if not summary:
        return "unknown", "hold", None

    trade_gate = summary.get("trade_gate", {}) or {}
    trade_status = str(trade_gate.get("status", "")).strip().lower()
    trade_action = str(trade_gate.get("action", "")).strip().lower()
    trade_cap = _clamp01(trade_gate.get("position_cap"), default=None)
    if trade_status in {"pass", "warn", "fail"}:
        if trade_status == "fail" and trade_cap is None:
            trade_cap = 0.0
        if trade_action == "":
            trade_action = "stop" if trade_status == "fail" else "normal"
        return trade_status, trade_action, trade_cap

    gate_status = str((summary.get("quality_gate", {}) or {}).get("status", "unknown")).strip().lower()
    monitor = summary.get("failure_monitor", {}) or {}
    monitor_action = str(monitor.get("action", "hold")).strip().lower()
    suggested_cap = _clamp01(monitor.get("suggested_position_cap"), default=None)

    if gate_status == "fail" or monitor_action == "stop":
        return gate_status, monitor_action, 0.0
    return gate_status, monitor_action, suggested_cap


def get_news_symbols(base_dir: Path, top_n: int = 30) -> list[str]:
    """
    获取需要抓新闻的股票列表：
    - 只取信号排名前 top_n 的股票
    - 而非全部450只覆盖
    """
    rank_path = base_dir / "data" / "signals" / "latest_daily_rank.csv"
    if not rank_path.exists():
        return []

    try:
        import pandas as pd
        df = pd.read_csv(rank_path)
        # 取排名靠前的或action=INVEST_MORE的
        if "rank" in df.columns:
            df = df.sort_values("rank").head(top_n)
        elif "score" in df.columns:
            df = df.sort_values("score", ascending=False).head(top_n)
        else:
            df = df.head(top_n)

        symbols = df["symbol"].astype(str).tolist() if "symbol" in df.columns else []
        return symbols
    except Exception:
        return []


def get_llm_provider() -> str:
    """获取当前配置的 LLM 提供商"""
    return os.getenv("LLM_PROVIDER", "deepseek").lower()


def preflight_runtime(base_dir: Path) -> None:
    """Check interpreter and optional dependencies before running pipeline."""
    print(f"🐍 Python: {sys.executable}")

    venv_python = base_dir / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        try:
            using_venv = Path(sys.executable).resolve() == venv_python.resolve()
        except Exception:
            using_venv = str(Path(sys.executable)).lower() == str(venv_python).lower()
        if not using_venv:
            print(f"⚠️ 当前未使用项目虚拟环境: {venv_python}")
            print("   建议先执行: .\\.venv\\Scripts\\Activate.ps1")

    if importlib.util.find_spec("baostock") is None:
        print("⚠️ 未检测到 baostock，沪深300基准对照将自动降级。")
        print(f"   安装命令: \"{sys.executable}\" -m pip install baostock")


def print_banner():
    """打印启动横幅"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🚀 A股智能量化交易系统 v3.2                              ║
║                                                              ║
║     多因子选股 · 涨停回调 · 主力控盘 · 智能风控 · AI研报     ║
║     450只动态股票池 · 新闻精准覆盖 · 两阶段选股架构          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_summary(
    provider: str,
    inferred_as_of: str,
    as_of: str,
    total_time: float,
    steps_run: list[str],
    n_watchlist: int = 0,
    n_news_symbols: int = 0,
):
    """打印运行总结"""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                      ✅ ALL DONE                             ║
╠══════════════════════════════════════════════════════════════╣
║  LLM Provider  : {provider.upper():<43} ║
║  股票池        : {str(n_watchlist) + ' 只':<43} ║
║  新闻覆盖      : {str(n_news_symbols) + ' 只 (信号前N)':<43} ║
║  信号日期      : {(inferred_as_of or 'N/A'):<43} ║
║  新闻日期      : {as_of:<43} ║
║  总耗时        : {total_time:.1f} 秒{' '*(39-len(f'{total_time:.1f}'))} ║
║  运行步骤      : {len(steps_run)} 个{' '*(40-len(str(len(steps_run))))} ║
╚══════════════════════════════════════════════════════════════╝
""")


def main() -> None:
    load_dotenv()
    base_dir = Path(__file__).resolve().parent
    total_start = time.time()
    steps_run = []
    preflight_runtime(base_dir)

    print_banner()

    # 显示当前使用的 LLM
    provider = get_llm_provider()
    print(f"🤖 LLM Provider: {provider.upper()}")
    print(f"📅 运行日期: {date.today().isoformat()}")

    ap = argparse.ArgumentParser(description="一键运行 A股量化日度流程")
    ap.add_argument("--qc", action="store_true", help="运行数据质量修复")
    ap.add_argument("--skip-news", action="store_true", help="跳过新闻抓取")
    ap.add_argument("--skip-briefs", action="store_true", help="跳过AI briefs生成")
    ap.add_argument("--skip-ai", action="store_true", help="跳过AI研报")
    ap.add_argument("--skip-md", action="store_true", help="跳过Markdown日报")
    ap.add_argument("--skip-health-card", action="store_true", help="跳过策略健康诊断卡")
    ap.add_argument("--skip-gate-calibration", action="store_true", help="跳过闸门阈值校准报告")
    ap.add_argument("--skip-backtest-report", action="store_true", help="跳过回测HTML报告")
    ap.add_argument("--skip-charts", action="store_true", help="跳过图表生成（加速）")
    ap.add_argument("--fast", action="store_true", help="快速模式（跳过新闻+图表+AI增强）")
    ap.add_argument("--config", default="", help="config path (default: config.yaml, fallback: config_v31.yaml)")
    ap.add_argument("--as-of", default="", help="指定日期 (YYYY-MM-DD)")
    ap.add_argument("--disable-proxy", action="store_true", help="禁用代理")
    ap.add_argument("--news-top-n", type=int, default=30, help="新闻只抓信号排名前N的股票 (默认30)")
    ap.add_argument("--news-backfill-start", default="", help="新闻历史回补开始日期 YYYY-MM-DD（可选）")
    ap.add_argument("--news-backfill-end", default="", help="新闻历史回补结束日期 YYYY-MM-DD（默认=as_of）")
    ap.add_argument("--news-backfill-watchlist", default="", help="新闻历史回补股票池文件（默认 watchlist_cache.yaml）")
    ap.add_argument("--news-backfill-force", action="store_true", help="新闻历史回补忽略已存在日期，强制重抓")
    ap.add_argument("--news-backfill-max-days", type=int, default=0, help="新闻历史回补最多处理天数（0=不限制）")
    ap.add_argument("--news-backfill-max-symbols", type=int, default=0, help="新闻历史回补最多股票数（0=不限制）")
    ap.add_argument("--news-backfill-latest-first", action="store_true", help="新闻历史回补按最新日期优先")
    ap.add_argument("--skip-paper-trade", action="store_true", help="跳过 paper trading 记账")
    ap.add_argument("--paper-sync-portfolio", action="store_true", help="将 paper 持仓同步到 data/portfolio.yaml")
    ap.add_argument("--paper-initial-cash", type=float, default=None, help="首次初始化 paper 资金")
    ap.add_argument("--paper-ignore-gate", action="store_true", help="ignore strategy gate and run paper as-is")
    ap.add_argument("--skip-backtest", action="store_true", help="skip run_backtest_strategy_v3.py quick backtest")
    ap.add_argument(
        "--backtest-watchlist",
        default="",
        help="watchlist file for quick backtest (auto: watchlist_cache.yaml -> backtest_watchlist.yaml)",
    )
    ap.add_argument("--backtest-top-n", type=int, default=300, help="dynamic top_n for quick backtest")
    ap.add_argument("--backtest-walk-forward", action="store_true", help="enable walk-forward in quick backtest")
    ap.add_argument("--skip-strategy-process", action="store_true", help="skip strategy governance pipeline")
    ap.add_argument("--strategy-monitor-days", type=int, default=60, help="monitor window for strategy process")
    ap.add_argument("--strategy-monitor-short-days", type=int, default=20, help="short monitor window for strategy process")
    ap.add_argument("--strategy-warn-cap-min", type=float, default=0.60, help="warn gate cap lower bound")
    ap.add_argument("--strategy-warn-cap-max", type=float, default=0.80, help="warn gate cap upper bound")
    ap.add_argument("--strategy-gate-max-avg-turnover", type=float, default=0.45, help="gate max avg turnover")
    ap.add_argument("--strategy-gate-max-slippage-drag-pct", type=float, default=15.0, help="gate max slippage drag pct")
    ap.add_argument(
        "--strategy-overfit-max-combos",
        type=int,
        default=4,
        help="max combos for overfit CV diagnostics (effective min: 4)",
    )
    ap.add_argument(
        "--strategy-overfit-cv-splits",
        type=int,
        default=3,
        help="CV splits for overfit diagnostics (effective min: 3)",
    )
    ap.add_argument("--strategy-gate-max-pbo", type=float, default=0.45, help="gate max accepted PBO")
    ap.add_argument("--strategy-gate-min-dsr", type=float, default=0.55, help="gate min accepted DSR")
    ap.add_argument(
        "--strategy-gate-profile",
        choices=["production", "research"],
        default="production",
        help="strategy gate profile",
    )
    ap.add_argument(
        "--strategy-gate-min-wf-mean-sharpe",
        type=float,
        default=None,
        help="override strategy gate min WF mean sharpe",
    )
    ap.add_argument(
        "--strategy-gate-min-wf-sharpe-ok-ratio",
        type=float,
        default=None,
        help="override strategy gate min WF sharpe-ok ratio",
    )
    ap.add_argument(
        "--strategy-gate-min-wf-folds-required",
        type=int,
        default=None,
        help="override strategy gate min required WF folds",
    )
    ap.add_argument("--strategy-max-combos", type=int, default=2, help="max combos for strategy stability zone")
    args = ap.parse_args()

    cfg_path = resolve_config_path(base_dir, args.config)
    cfg = load_config(cfg_path)
    if cfg_path.name == "config.yaml" and (base_dir / "config_v31.yaml").exists():
        print("  ℹ 检测到 config_v31.yaml；当前默认使用 config.yaml。可用 --config config_v31.yaml 切换。")
    ai_cfg = cfg.get("ai", {}) or {}
    include_news_cfg = bool(ai_cfg.get("include_news", True))
    n_watchlist = len(cfg.get("watchlist", []))

    # 从config读取新闻覆盖数
    news_top_n = args.news_top_n
    news_cfg = cfg.get("news", {}) or {}
    if news_cfg.get("news_top_n"):
        news_top_n = int(news_cfg["news_top_n"])
    news_backfill_cfg = news_cfg.get("backfill", {}) if isinstance(news_cfg.get("backfill", {}), dict) else {}

    print(f"📊 股票池: {n_watchlist} 只")
    print(f"📰 新闻覆盖: 信号前 {news_top_n} 只 + 市场新闻")

    # 快速模式
    if args.fast:
        args.skip_news = True
        args.skip_charts = True
        args.skip_backtest = True
        args.skip_backtest_report = True
        args.skip_strategy_process = True
        print("⚡ 快速模式: 跳过新闻+图表+回测报告")

    # 环境变量
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PIPELINE_CONFIG"] = str(cfg_path)
    if args.disable_proxy:
        env["DISABLE_PROXY"] = "1"

    # ========================================
    # 1) 基础量化流程
    # ========================================
    print("\n" + "="*60)
    print("📊 阶段一：数据抓取与特征工程")
    print("="*60)

    run_step(base_dir, "run_fetch_daily.py", env=env)
    steps_run.append("数据抓取")

    if args.qc:
        run_step(base_dir, "run_qc_repair_market_daily.py", env=env)
        steps_run.append("数据质量修复")

    run_step(base_dir, "run_build_market_daily_all.py", env=env)
    steps_run.append("市场数据构建")

    run_step(base_dir, "run_build_features_daily.py", env=env)
    steps_run.append("特征工程")

    run_step(base_dir, "run_make_daily_rank.py", env=env)
    steps_run.append("信号生成")

    # 从最新 ranking 推断交易日
    inferred_as_of = read_latest_rank_as_of(base_dir)
    as_of = args.as_of.strip() or inferred_as_of or date.today().isoformat()

    # 可选：新闻历史回补（手动或按配置周任务触发）
    backfill_start = str(args.news_backfill_start or "").strip()
    backfill_end = str(args.news_backfill_end or "").strip() or as_of
    auto_backfill = False
    if not backfill_start:
        auto_enabled = bool(news_backfill_cfg.get("enabled", False))
        auto_weekday = int(news_backfill_cfg.get("weekday", 0))
        auto_lookback = max(1, int(news_backfill_cfg.get("lookback_days", 7)))
        as_of_dt = _parse_iso_date(as_of)
        if auto_enabled and as_of_dt and as_of_dt.weekday() == auto_weekday:
            backfill_start = (as_of_dt - timedelta(days=auto_lookback)).isoformat()
            auto_backfill = True

    if backfill_start:
        backfill_script = base_dir / "run_backfill_news_history.py"
        if backfill_script.exists():
            backfill_watchlist = str(args.news_backfill_watchlist or "").strip()
            if not backfill_watchlist:
                backfill_watchlist = str(news_backfill_cfg.get("watchlist", "watchlist_cache.yaml")).strip()
            backfill_max_days = int(args.news_backfill_max_days) if int(args.news_backfill_max_days) > 0 else int(news_backfill_cfg.get("max_days", 0) or 0)
            backfill_max_symbols = int(args.news_backfill_max_symbols) if int(args.news_backfill_max_symbols) > 0 else int(news_backfill_cfg.get("max_symbols", 0) or 0)
            backfill_latest_first = bool(news_backfill_cfg.get("latest_first", False)) or bool(args.news_backfill_latest_first)
            backfill_args = [
                "--start-date", backfill_start,
                "--end-date", backfill_end,
                "--config", str(cfg_path),
            ]
            if backfill_watchlist:
                backfill_args.extend(["--watchlist", backfill_watchlist])
            if bool(args.news_backfill_force):
                backfill_args.append("--force")
            if backfill_max_days > 0:
                backfill_args.extend(["--max-days", str(backfill_max_days)])
            if backfill_max_symbols > 0:
                backfill_args.extend(["--max-symbols", str(backfill_max_symbols)])
            if backfill_latest_first:
                backfill_args.append("--latest-first")

            trigger = "auto-weekly" if auto_backfill else "manual"
            print(f"\n🗂️ 新闻历史回补: {trigger} | {backfill_start} -> {backfill_end}")
            run_step(base_dir, "run_backfill_news_history.py", env=env, extra_args=backfill_args)
            steps_run.append("news-backfill")
        else:
            print("⏭️ 跳过: run_backfill_news_history.py (文件不存在)")

    # ========================================
    # 2) 新闻抓取（只抓信号前N的股票）
    # ========================================
    print("\n" + "="*60)
    print("📰 阶段二：新闻与资讯")
    print("="*60)

    n_news_symbols = 0
    do_news = (not args.skip_news) and include_news_cfg
    if do_news:
        # 获取需要抓新闻的股票
        news_symbols = get_news_symbols(base_dir, top_n=news_top_n)
        n_news_symbols = len(news_symbols)

        env2 = env.copy()
        env2["NEWS_AS_OF"] = as_of
        # 将新闻股票列表传给新闻脚本（通过环境变量）
        if news_symbols:
            env2["NEWS_SYMBOLS"] = ",".join(news_symbols)
            print(f"  📰 新闻覆盖: {n_news_symbols} 只信号股 + 市场新闻")

        run_step(base_dir, "run_fetch_news.py", env=env2)
        steps_run.append("新闻抓取")
    else:
        print("⏭️ 跳过: run_fetch_news.py")

    if not args.skip_briefs:
        run_step(base_dir, "run_build_ai_briefs.py", env=env)
        steps_run.append("AI Briefs")
    else:
        print("⏭️ 跳过: run_build_ai_briefs.py")

    # Refresh backtest stats/equity before report generation.
    pipeline_watchlist = args.backtest_watchlist.strip()
    if not pipeline_watchlist:
        for cand in ["watchlist_cache.yaml", "backtest_watchlist.yaml"]:
            if (base_dir / cand).exists():
                pipeline_watchlist = cand
                break

    if not args.skip_backtest and (base_dir / "run_backtest_strategy_v3.py").exists():
        bt_start = str(
            (cfg.get("backtest", {}) or {}).get("start_date")
            or (cfg.get("market_data", {}) or {}).get("start_date")
            or "2020-01-01"
        )

        bt_args = [
            "--base-dir", ".",
            "--config", str(cfg_path),
            "--start-date", bt_start,
            "--dynamic-watchlist",
            "--dynamic-top-n", str(int(args.backtest_top_n)),
        ]
        if pipeline_watchlist:
            bt_args.extend(["--watchlist", pipeline_watchlist])
        if args.backtest_walk_forward:
            bt_args.extend([
                "--walk-forward",
                "--wf-train-days", "504",
                "--wf-test-days", "126",
                "--wf-step-days", "126",
            ])

        run_step(base_dir, "run_backtest_strategy_v3.py", env=env, extra_args=bt_args)
        steps_run.append("quick-backtest")
    elif args.skip_backtest:
        print("⏭️ 跳过: run_backtest_strategy_v3.py")
    else:
        print("⏭️ 跳过: run_backtest_strategy_v3.py (文件不存在)")

    gate_status = "unknown"
    monitor_action = "hold"
    suggested_cap: float | None = None
    strategy_summary_path = base_dir / "data" / "backtests" / "strategy_process_summary.json"

    if not args.skip_strategy_process and (base_dir / "tools" / "strategy_process_pipeline.py").exists():
        sp_args = [
            "--base-dir", ".",
            "--config", str(cfg_path),
            "--monitor-long-days", str(int(args.strategy_monitor_days)),
            "--monitor-short-days", str(int(args.strategy_monitor_short_days)),
            "--warn-cap-min", str(float(args.strategy_warn_cap_min)),
            "--warn-cap-max", str(float(args.strategy_warn_cap_max)),
            "--gate-max-avg-turnover", str(float(args.strategy_gate_max_avg_turnover)),
            "--gate-max-slippage-drag-pct", str(float(args.strategy_gate_max_slippage_drag_pct)),
            "--overfit-max-combos", str(int(args.strategy_overfit_max_combos)),
            "--overfit-cv-splits", str(int(args.strategy_overfit_cv_splits)),
            "--gate-max-pbo", str(float(args.strategy_gate_max_pbo)),
            "--gate-min-dsr", str(float(args.strategy_gate_min_dsr)),
            "--gate-profile", str(args.strategy_gate_profile),
            "--max-combos", str(int(args.strategy_max_combos)),
        ]
        if args.strategy_gate_min_wf_mean_sharpe is not None:
            sp_args.extend(["--gate-min-wf-mean-sharpe", str(float(args.strategy_gate_min_wf_mean_sharpe))])
        if args.strategy_gate_min_wf_sharpe_ok_ratio is not None:
            sp_args.extend(
                ["--gate-min-wf-sharpe-ok-ratio", str(float(args.strategy_gate_min_wf_sharpe_ok_ratio))]
            )
        if args.strategy_gate_min_wf_folds_required is not None:
            sp_args.extend(
                ["--gate-min-wf-folds-required", str(int(args.strategy_gate_min_wf_folds_required))]
            )
        if pipeline_watchlist:
            sp_args.extend(["--watchlist", pipeline_watchlist])
        run_step(base_dir, "tools/strategy_process_pipeline.py", env=env, extra_args=sp_args)
        steps_run.append("strategy-process")
    elif args.skip_strategy_process:
        print("⏭️ 跳过: tools/strategy_process_pipeline.py")
    else:
        print("⏭️ 跳过: tools/strategy_process_pipeline.py (文件不存在)")

    summary = load_strategy_process_summary(base_dir)
    gate_status, monitor_action, suggested_cap = derive_gate_controls(summary)
    cap_text = f"{suggested_cap:.2f}" if suggested_cap is not None else "N/A"
    print(f"🛡️ Strategy gate: status={gate_status}, monitor={monitor_action}, cap={cap_text}")

    if not args.skip_paper_trade and (base_dir / "run_paper_trading.py").exists():
        paper_args: list[str] = ["--config", str(cfg_path)]
        if args.paper_sync_portfolio:
            paper_args.append("--sync-portfolio")
        if args.paper_initial_cash is not None:
            paper_args.extend(["--initial-cash", str(float(args.paper_initial_cash))])
        if not args.paper_ignore_gate:
            if suggested_cap is not None:
                paper_args.extend(["--max-total-position-cap", f"{float(suggested_cap):.6f}"])
            if strategy_summary_path.exists():
                paper_args.extend(["--strategy-summary", str(strategy_summary_path)])
        else:
            print("⚠️ paper risk gating disabled by --paper-ignore-gate")
        run_step(base_dir, "run_paper_trading.py", env=env, extra_args=paper_args)
        steps_run.append("paper-trading")
    elif args.skip_paper_trade:
        print("⏭️ 跳过: run_paper_trading.py")
    else:
        print("⏭️ 跳过: run_paper_trading.py (文件不存在)")

    # ========================================
    # 3) 报告生成
    # ========================================
    print("\n" + "="*60)
    print("📝 阶段三：研报生成")
    print("="*60)

    # Markdown 日报（不走LLM）
    if not args.skip_md:
        if (base_dir / "run_generate_daily_report.py").exists():
            run_step(base_dir, "run_generate_daily_report.py", env=env)
            steps_run.append("Markdown日报")
        else:
            print("⏭️ 跳过: run_generate_daily_report.py (文件不存在)")
    else:
        print("⏭️ 跳过: run_generate_daily_report.py")

    # 策略健康诊断卡（自动化健康状态 + 建议动作）
    if not args.skip_health_card:
        if (base_dir / "run_generate_strategy_health_card.py").exists():
            run_step(base_dir, "run_generate_strategy_health_card.py", env=env, extra_args=["--as-of", as_of])
            steps_run.append("策略健康诊断卡")
        else:
            print("⏭️ 跳过: run_generate_strategy_health_card.py (文件不存在)")
    else:
        print("⏭️ 跳过: run_generate_strategy_health_card.py")

    # 闸门阈值校准（滚动窗口阈值-收益-回撤对照）
    if not args.skip_gate_calibration:
        if (base_dir / "run_generate_gate_calibration.py").exists():
            rolling_path = base_dir / "data" / "backtests" / "strategy_process_rolling.csv"
            if strategy_summary_path.exists() and rolling_path.exists():
                run_step(
                    base_dir,
                    "run_generate_gate_calibration.py",
                    env=env,
                    extra_args=["--as-of", as_of, "--config", str(cfg_path)],
                )
                steps_run.append("闸门阈值校准")
            else:
                print("⏭️ 跳过: run_generate_gate_calibration.py (缺少 strategy_process 输出文件)")
        else:
            print("⏭️ 跳过: run_generate_gate_calibration.py (文件不存在)")
    else:
        print("⏭️ 跳过: run_generate_gate_calibration.py")

    # AI研报（使用优化版 v2）
    if not args.skip_ai:
        candidate_scripts = ["run_ai_report_v2.py", "run_ai_report.py"]
        ai_report_script = next(
            (name for name in candidate_scripts if (base_dir / name).exists()),
            None,
        )

        if ai_report_script:
            ai_args = []
            if args.skip_charts:
                ai_args.append("--skip-charts")
            if args.fast:
                ai_args.append("--skip-ai")

            run_step(base_dir, ai_report_script, env=env, extra_args=ai_args)
            steps_run.append("AI研报")
        else:
            print("⏭️ 跳过: AI研报 (未找到 run_ai_report_v2.py/run_ai_report.py)")
    else:
        print("⏭️ 跳过: AI研报")

    # 每日信号仪表盘（HTML）
    if (base_dir / "run_generate_dashboard.py").exists():
        run_step(base_dir, "run_generate_dashboard.py", env=env)
        steps_run.append("信号仪表盘")
    else:
        print("⏭️ 跳过: run_generate_dashboard.py (文件不存在)")

    # AI研报仪表盘（HTML）
    if (base_dir / "run_generate_ai_dashboard.py").exists():
        run_step(base_dir, "run_generate_ai_dashboard.py", env=env)
        steps_run.append("AI研报仪表盘")
    else:
        print("⏭️ 跳过: run_generate_ai_dashboard.py (文件不存在)")

    if (base_dir / "run_generate_trade_journal_dashboard.py").exists():
        tj_start, tj_end = compute_trade_journal_window(as_of, months=3)
        print(f"  📘 交易日志区间: {tj_start} ~ {tj_end} (rolling 3 months)")
        run_step(
            base_dir,
            "run_generate_trade_journal_dashboard.py",
            env=env,
            extra_args=["--start-date", tj_start, "--end-date", tj_end],
        )
        steps_run.append("交易日志网页")
    else:
        print("⏭️ 跳过: run_generate_trade_journal_dashboard.py (文件不存在)")

    # 回测报告（HTML）
    if not args.skip_backtest_report:
        if (base_dir / "run_generate_report.py").exists():
            run_step(base_dir, "run_generate_report.py", env=env)
            steps_run.append("回测报告")
        else:
            print("⏭️ 跳过: run_generate_report.py (文件不存在)")
    else:
        print("⏭️ 跳过: 回测报告")

    # ========================================
    # 完成
    # ========================================
    total_time = time.time() - total_start

    print_summary(
        provider=provider,
        inferred_as_of=inferred_as_of,
        as_of=as_of,
        total_time=total_time,
        steps_run=steps_run,
        n_watchlist=n_watchlist,
        n_news_symbols=n_news_symbols,
    )

    # 显示输出文件
    print("📁 输出文件:")
    report_date = inferred_as_of or as_of
    output_files = [
        ("信号文件", "data/signals/latest_daily_rank.csv"),
        ("Markdown日报", f"data/reports/daily_report_{report_date}.md"),
        ("健康诊断卡", f"data/reports/strategy_health_card_{report_date}.md"),
        ("健康诊断卡(最新)", "data/reports/strategy_health_card_latest.md"),
        ("闸门校准报告", f"data/reports/strategy_gate_calibration_{report_date}.md"),
        ("闸门校准报告(最新)", "data/reports/strategy_gate_calibration_latest.md"),
        ("闸门校准汇总JSON", "data/backtests/strategy_gate_calibration.json"),
        ("AI研报", f"data/reports/ai_report_{report_date}.md"),
        ("AI研报(最新)", "out/latest_ai_report.md"),
        ("AI研报仪表盘", "out/ai_report_dashboard.html"),
        ("交易日志网页", "out/trade_journal_dashboard.html"),
        ("回测报告", "out/backtest_report.html"),
        ("信号仪表盘", "out/daily_dashboard.html"),
        ("图表目录", "out/charts/"),
    ]
    for name, path in output_files:
        full_path = base_dir / path
        exists = "✅" if full_path.exists() else "❌"
        print(f"   {exists} {name}: {path}")


if __name__ == "__main__":
    main()
