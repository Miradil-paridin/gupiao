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
import os
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv


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


def load_config(base_dir: Path) -> dict:
    """加载配置（优先 config_v31.yaml）"""
    for name in ["config_v31.yaml", "config.yaml"]:
        cfg_path = base_dir / name
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            print(f"  📂 配置: {name}")
            return cfg
    raise FileNotFoundError("config.yaml not found.")


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
    ap.add_argument("--skip-backtest-report", action="store_true", help="跳过回测HTML报告")
    ap.add_argument("--skip-charts", action="store_true", help="跳过图表生成（加速）")
    ap.add_argument("--fast", action="store_true", help="快速模式（跳过新闻+图表+AI增强）")
    ap.add_argument("--as-of", default="", help="指定日期 (YYYY-MM-DD)")
    ap.add_argument("--disable-proxy", action="store_true", help="禁用代理")
    ap.add_argument("--news-top-n", type=int, default=30, help="新闻只抓信号排名前N的股票 (默认30)")
    args = ap.parse_args()

    cfg = load_config(base_dir)
    ai_cfg = cfg.get("ai", {}) or {}
    include_news_cfg = bool(ai_cfg.get("include_news", True))
    n_watchlist = len(cfg.get("watchlist", []))

    # 从config读取新闻覆盖数
    news_top_n = args.news_top_n
    news_cfg = cfg.get("news", {}) or {}
    if news_cfg.get("news_top_n"):
        news_top_n = int(news_cfg["news_top_n"])

    print(f"📊 股票池: {n_watchlist} 只")
    print(f"📰 新闻覆盖: 信号前 {news_top_n} 只 + 市场新闻")

    # 快速模式
    if args.fast:
        args.skip_news = True
        args.skip_charts = True
        args.skip_backtest_report = True
        print("⚡ 快速模式: 跳过新闻+图表+回测报告")

    # 环境变量
    env = os.environ.copy()
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
        ("AI研报", f"data/reports/ai_report_{report_date}.md"),
        ("AI研报(最新)", "out/latest_ai_report.md"),
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