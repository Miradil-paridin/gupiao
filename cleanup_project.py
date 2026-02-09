"""
项目清理脚本

清理冗余文件，优化项目结构
"""
from pathlib import Path
import shutil


def main():
    base_dir = Path(__file__).resolve().parent

    print("=" * 60)
    print("🧹 项目清理脚本")
    print("=" * 60)

    # 可安全删除的文件
    files_to_delete = [
        # 旧版本脚本（已被v3替代）
        "run_backtest_strategy_v2.py",
        "run_generate_chart_v2.py",
        "run_deepseek_report.py",  # 已被 run_ai_report.py 替代

        # 一次性验证脚本（可选）
        # "run_validate_data_sources.py",
        # "run_validate_market_daily.py",
        # "run_qlib_backtest.py",

        # 重复文件（检查后删除）
        # "quant/providers_akshare.py",  # 如果 providers/ 里已有
    ]

    # 可删除的目录
    dirs_to_delete = [
        # ".venv-qlib",  # 如果不用 Qlib
        # "__pycache__",  # Python缓存
    ]

    print("\n📋 将删除以下文件:")
    deleted_count = 0

    for f in files_to_delete:
        path = base_dir / f
        if path.exists():
            print(f"   🗑️ {f}")
            # path.unlink()  # 取消注释执行删除
            deleted_count += 1
        else:
            print(f"   ⏭️ {f} (不存在)")

    print(f"\n📁 将删除以下目录:")
    for d in dirs_to_delete:
        path = base_dir / d
        if path.exists():
            print(f"   🗑️ {d}/")
            # shutil.rmtree(path)  # 取消注释执行删除
        else:
            print(f"   ⏭️ {d}/ (不存在)")

    # 清理 __pycache__
    print("\n🔍 清理 __pycache__ 目录...")
    pycache_count = 0
    for pycache in base_dir.rglob("__pycache__"):
        print(f"   🗑️ {pycache.relative_to(base_dir)}")
        # shutil.rmtree(pycache)  # 取消注释执行删除
        pycache_count += 1

    print(f"\n" + "=" * 60)
    print(f"📊 清理统计:")
    print(f"   文件: {deleted_count} 个")
    print(f"   __pycache__: {pycache_count} 个")
    print("=" * 60)

    print("\n⚠️ 注意: 当前为预览模式，未实际删除文件")
    print("   如需执行删除，请编辑脚本取消注释 unlink/rmtree 行")

    # 显示推荐的项目结构
    print("\n" + "=" * 60)
    print("📁 推荐的项目结构:")
    print("=" * 60)

    recommended_structure = """
├── quant/                      # 核心模块
│   ├── providers/              # 数据源
│   ├── news_providers/         # 新闻源
│   ├── features.py             # 特征工程
│   ├── signals.py              # 信号生成
│   ├── tdx_indicators.py       # 通达信指标
│   ├── market_regime.py        # 市场环境
│   ├── report_charts.py        # 研报图表
│   ├── report_generator.py     # 研报生成器
│   └── report_quality.py       # 质量评估
├── tools/                      # 工具脚本
│   ├── build_signal_history.py
│   ├── run_backtest_from_signal.py
│   └── report_metrics.py
├── data/                       # 数据目录
├── out/                        # 输出目录
├── run_all_daily.py            # 🚀 主入口
├── run_ai_report_v2.py         # AI研报
├── run_backtest_strategy_v3.py # 回测
├── config.yaml                 # 配置
├── .env                        # 环境变量
└── README.md                   # 说明文档
"""
    print(recommended_structure)


if __name__ == "__main__":
    main()