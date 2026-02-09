"""
一键启动 A股量化日度流程（v2 优化版）

功能：
- 多数据源行情抓取（并行+缓存+增量更新）
- 特征工程（含涨停+TDX月线指标）
- 策略信号生成（v3策略）
- 新闻聚合
- AI研报生成（优化版，含可视化+质量评估）

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
    """加载配置"""
    cfg_path = base_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError("config.yaml not found.")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_llm_provider() -> str:
    """获取当前配置的 LLM 提供商"""
    return os.getenv("LLM_PROVIDER", "deepseek").lower()


def print_banner():
    """打印启动横幅"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🚀 A股智能量化交易系统 v3.0                              ║
║                                                              ║
║     多因子选股 · 涨停回调 · 主力控盘 · 智能风控 · AI研报     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_summary(
    provider: str,
    inferred_as_of: str,
    as_of: str,
    total_time: float,
    steps_run: list[str],
):
    """打印运行总结"""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                      ✅ ALL DONE                             ║
╠══════════════════════════════════════════════════════════════╣
║  LLM Provider  : {provider.upper():<43} ║
║  信号日期      : {inferred_as_of or 'N/A':<43} ║
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
    ap.add_argument("--skip-charts", action="store_true", help="跳过图表生成（加速）")
    ap.add_argument("--fast", action="store_true", help="快速模式（跳过新闻+图表+AI增强）")
    ap.add_argument("--as-of", default="", help="指定日期 (YYYY-MM-DD)")
    ap.add_argument("--disable-proxy", action="store_true", help="禁用代理")
    args = ap.parse_args()

    cfg = load_config(base_dir)
    ai_cfg = cfg.get("ai", {}) or {}
    include_news_cfg = bool(ai_cfg.get("include_news", True))

    # 快速模式
    if args.fast:
        args.skip_news = True
        args.skip_charts = True
        print("⚡ 快速模式: 跳过新闻+图表")

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
    # 2) 新闻抓取
    # ========================================
    print("\n" + "="*60)
    print("📰 阶段二：新闻与资讯")
    print("="*60)

    do_news = (not args.skip_news) and include_news_cfg
    if do_news:
        env2 = env.copy()
        env2["NEWS_AS_OF"] = as_of
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
        # 优先使用 v2 版本
        ai_report_script = "run_ai_report_v2.py"
        if not (base_dir / ai_report_script).exists():
            ai_report_script = "run_ai_report.py"  # 降级到原版

        # 构建参数
        ai_args = []
        if args.skip_charts:
            ai_args.append("--skip-charts")
        if args.fast:
            ai_args.append("--skip-ai")  # 快速模式跳过LLM增强

        run_step(base_dir, ai_report_script, env=env, extra_args=ai_args)
        steps_run.append("AI研报")
    else:
        print("⏭️ 跳过: AI研报")

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
    )

    # 显示输出文件
    print("📁 输出文件:")
    output_files = [
        ("信号文件", "data/signals/latest_daily_rank.csv"),
        ("AI研报", "out/latest_ai_report.md"),
        ("图表目录", "out/charts/"),
    ]
    for name, path in output_files:
        full_path = base_dir / path
        exists = "✅" if full_path.exists() else "❌"
        print(f"   {exists} {name}: {path}")


if __name__ == "__main__":
    main()