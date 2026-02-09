"""
研报可视化工具

生成各类图表：
- 价格走势图
- 技术指标图
- 行业热力图
- 收益曲线图
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

# 尝试导入绘图库
try:
    import matplotlib

    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib 未安装，图表功能不可用")

# 中文字体设置
if HAS_MATPLOTLIB:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False


def create_price_chart(
        df: pd.DataFrame,
        symbol: str,
        name: str = "",
        days: int = 30,
        output_path: Path = None,
) -> Optional[Path]:
    """
    创建价格走势图（含均线和成交量）

    Args:
        df: 包含 date, open, high, low, close, volume 的 DataFrame
        symbol: 股票代码
        name: 股票名称
        days: 显示天数
        output_path: 输出路径

    Returns:
        图片路径
    """
    if not HAS_MATPLOTLIB:
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days)

    if df.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # 价格图
    ax1.plot(df["date"], df["close"], color='#1f77b4', linewidth=2, label='收盘价')

    # 均线
    if len(df) >= 5:
        df["ma5"] = df["close"].rolling(5).mean()
        ax1.plot(df["date"], df["ma5"], color='#ff7f0e', linewidth=1,
                 linestyle='--', label='MA5', alpha=0.8)

    if len(df) >= 20:
        df["ma20"] = df["close"].rolling(20).mean()
        ax1.plot(df["date"], df["ma20"], color='#2ca02c', linewidth=1,
                 linestyle='--', label='MA20', alpha=0.8)

    # 填充涨跌区域
    ax1.fill_between(df["date"], df["close"].min() * 0.98, df["close"],
                     alpha=0.1, color='#1f77b4')

    ax1.set_title(f'{symbol} {name} 价格走势（近{days}天）', fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    # 成交量图
    colors = ['#ef5350' if df["close"].iloc[i] >= df["open"].iloc[i] else '#26a69a'
              for i in range(len(df))]
    ax2.bar(df["date"], df["volume"] / 1e6, color=colors, alpha=0.7, width=0.8)
    ax2.set_ylabel('成交量(百万)', fontsize=10)
    ax2.set_xlabel('日期', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    plt.tight_layout()

    if output_path is None:
        output_path = Path(f"price_{symbol}.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    return output_path


def create_indicator_chart(
        df: pd.DataFrame,
        symbol: str,
        name: str = "",
        days: int = 30,
        output_path: Path = None,
) -> Optional[Path]:
    """
    创建技术指标图（RSI、ATR、MACD风格）
    """
    if not HAS_MATPLOTLIB:
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days)

    if df.empty:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(12, 9),
                             gridspec_kw={'height_ratios': [2, 1, 1]})

    # 价格 + 布林带
    ax1 = axes[0]
    if len(df) >= 20:
        df["ma20"] = df["close"].rolling(20).mean()
        df["std20"] = df["close"].rolling(20).std()
        df["upper"] = df["ma20"] + 2 * df["std20"]
        df["lower"] = df["ma20"] - 2 * df["std20"]

        ax1.fill_between(df["date"], df["lower"], df["upper"],
                         alpha=0.2, color='#1f77b4', label='布林带')
        ax1.plot(df["date"], df["ma20"], color='#ff7f0e',
                 linewidth=1, label='MA20')

    ax1.plot(df["date"], df["close"], color='#1f77b4', linewidth=2, label='收盘价')
    ax1.set_title(f'{symbol} {name} 技术指标分析', fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # RSI
    ax2 = axes[1]
    if "rsi_14" in df.columns:
        rsi = df["rsi_14"]
    else:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

    ax2.plot(df["date"], rsi, color='#9467bd', linewidth=1.5)
    ax2.axhline(y=70, color='#ef5350', linestyle='--', alpha=0.7, label='超买(70)')
    ax2.axhline(y=30, color='#26a69a', linestyle='--', alpha=0.7, label='超卖(30)')
    ax2.fill_between(df["date"], 30, 70, alpha=0.1, color='gray')
    ax2.set_ylabel('RSI(14)', fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ATR / 波动率
    ax3 = axes[2]
    if "atr_14" in df.columns:
        atr = df["atr_14"]
        ax3.plot(df["date"], atr, color='#d62728', linewidth=1.5)
        ax3.set_ylabel('ATR(14)', fontsize=10)
    elif "vol_20d" in df.columns:
        vol = df["vol_20d"] * 100  # 转为百分比
        ax3.plot(df["date"], vol, color='#d62728', linewidth=1.5)
        ax3.set_ylabel('波动率(%)', fontsize=10)
    else:
        vol = df["close"].pct_change().rolling(20).std() * np.sqrt(252) * 100
        ax3.plot(df["date"], vol, color='#d62728', linewidth=1.5)
        ax3.set_ylabel('波动率(%)', fontsize=10)

    ax3.set_xlabel('日期', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    plt.tight_layout()

    if output_path is None:
        output_path = Path(f"indicator_{symbol}.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    return output_path


def create_industry_heatmap(
        industry_data: Dict[str, float],
        title: str = "行业涨跌幅热力图",
        output_path: Path = None,
) -> Optional[Path]:
    """
    创建行业热力图

    Args:
        industry_data: {行业名: 涨跌幅%}
        title: 图表标题
        output_path: 输出路径
    """
    if not HAS_MATPLOTLIB:
        return None

    if not industry_data:
        return None

    # 排序
    sorted_data = sorted(industry_data.items(), key=lambda x: x[1], reverse=True)
    industries = [x[0] for x in sorted_data]
    values = [x[1] for x in sorted_data]

    fig, ax = plt.subplots(figsize=(10, max(6, len(industries) * 0.4)))

    # 颜色映射
    colors = ['#ef5350' if v >= 0 else '#26a69a' for v in values]

    bars = ax.barh(industries, values, color=colors, alpha=0.8)

    # 添加数值标签
    for bar, val in zip(bars, values):
        x_pos = bar.get_width()
        ha = 'left' if val >= 0 else 'right'
        offset = 0.1 if val >= 0 else -0.1
        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                f'{val:+.2f}%', va='center', ha=ha, fontsize=9)

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('涨跌幅(%)', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if output_path is None:
        output_path = Path("industry_heatmap.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    return output_path


def create_equity_curve(
        equity_data: pd.DataFrame,
        benchmark_data: pd.DataFrame = None,
        title: str = "策略收益曲线",
        output_path: Path = None,
) -> Optional[Path]:
    """
    创建收益曲线图

    Args:
        equity_data: 包含 date, equity 的 DataFrame
        benchmark_data: 基准数据（可选）
        title: 图表标题
        output_path: 输出路径
    """
    if not HAS_MATPLOTLIB:
        return None

    equity_data = equity_data.copy()
    equity_data["date"] = pd.to_datetime(equity_data["date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # 净值曲线
    initial = equity_data["equity"].iloc[0]
    equity_data["nav"] = equity_data["equity"] / initial

    ax1.plot(equity_data["date"], equity_data["nav"],
             color='#1f77b4', linewidth=2, label='策略')

    if benchmark_data is not None and not benchmark_data.empty:
        benchmark_data = benchmark_data.copy()
        benchmark_data["date"] = pd.to_datetime(benchmark_data["date"])
        benchmark_initial = benchmark_data["equity"].iloc[0]
        benchmark_data["nav"] = benchmark_data["equity"] / benchmark_initial
        ax1.plot(benchmark_data["date"], benchmark_data["nav"],
                 color='#ff7f0e', linewidth=1.5, linestyle='--', label='沪深300')

    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('净值', fontsize=10)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 回撤曲线
    peak = equity_data["nav"].cummax()
    drawdown = (equity_data["nav"] / peak - 1) * 100

    ax2.fill_between(equity_data["date"], drawdown, 0,
                     color='#ef5350', alpha=0.5)
    ax2.plot(equity_data["date"], drawdown, color='#ef5350', linewidth=1)
    ax2.set_ylabel('回撤(%)', fontsize=10)
    ax2.set_xlabel('日期', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))

    plt.tight_layout()

    if output_path is None:
        output_path = Path("equity_curve.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    return output_path


def create_signal_summary_chart(
        signals: pd.DataFrame,
        output_path: Path = None,
) -> Optional[Path]:
    """
    创建信号汇总图
    """
    if not HAS_MATPLOTLIB:
        return None

    if signals.empty:
        return None

    # 统计各类信号数量
    action_counts = signals["action"].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 饼图
    colors = {
        'INVEST_MORE': '#26a69a',
        'HOLD': '#78909c',
        'REDUCE': '#ffb74d',
        'WITHDRAW': '#ef5350',
        'LEAST': '#7e57c2',
    }
    pie_colors = [colors.get(a, '#90a4ae') for a in action_counts.index]

    ax1.pie(action_counts.values, labels=action_counts.index,
            colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('信号分布', fontsize=12, fontweight='bold')

    # 得分分布
    if "score" in signals.columns:
        ax2.hist(signals["score"], bins=30, color='#1f77b4',
                 alpha=0.7, edgecolor='white')
        ax2.axvline(x=signals["score"].mean(), color='#ef5350',
                    linestyle='--', label=f'均值: {signals["score"].mean():.2f}')
        ax2.set_xlabel('得分', fontsize=10)
        ax2.set_ylabel('数量', fontsize=10)
        ax2.set_title('得分分布', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        output_path = Path("signal_summary.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    return output_path


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")

    df = pd.DataFrame({
        "date": dates,
        "symbol": "600519",
        "open": 1800 + np.cumsum(np.random.randn(60) * 10),
        "close": 1800 + np.cumsum(np.random.randn(60) * 10),
        "high": 1800 + np.cumsum(np.random.randn(60) * 10) + 20,
        "low": 1800 + np.cumsum(np.random.randn(60) * 10) - 20,
        "volume": np.random.randint(1000000, 5000000, 60),
    })
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    # 测试图表生成
    out_dir = Path("test_charts")
    out_dir.mkdir(exist_ok=True)

    print("生成价格走势图...")
    create_price_chart(df, "600519", "贵州茅台", output_path=out_dir / "price.png")

    print("生成技术指标图...")
    create_indicator_chart(df, "600519", "贵州茅台", output_path=out_dir / "indicator.png")

    print("生成行业热力图...")
    industry_data = {
        "白酒": 2.5,
        "银行": -0.8,
        "新能源": 1.2,
        "医药": -1.5,
        "科技": 3.1,
        "地产": -2.3,
    }
    create_industry_heatmap(industry_data, output_path=out_dir / "heatmap.png")

    print("生成收益曲线...")
    equity_df = pd.DataFrame({
        "date": dates,
        "equity": 100000 * (1 + np.cumsum(np.random.randn(60) * 0.01)),
    })
    create_equity_curve(equity_df, output_path=out_dir / "equity.png")

    print(f"图表已保存到 {out_dir}/")