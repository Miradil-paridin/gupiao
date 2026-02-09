"""
策略回测 + 网页报告生成

功能：
1. 使用当前策略进行历史回测
2. 生成美观的网页版报告（HTML）
3. 包含交互式图表

使用方法：
    python run_backtest_report.py
    python run_backtest_report.py --start-date 2023-01-01
    python run_backtest_report.py --top-k 10
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import yaml


@dataclass
class BacktestConfig:
    """回测配置"""
    # 选股
    top_k: int = 10

    # 入场条件
    pullback_window_start: int = 3
    pullback_window_end: int = 10
    pullback_min_pct: float = 0.05
    pullback_max_pct: float = 0.25
    tdx_min_score: float = 1.5

    # 风控
    stop_loss_pct: float = 0.08
    trailing_stop_pct: float = 0.10
    max_hold_days: int = 15

    # 仓位
    initial_capital: float = 100000.0
    max_single_weight: float = 0.15
    max_total_position: float = 0.80
    cost_bps: float = 15.0

    # 开关
    use_stop_loss: bool = True
    use_trailing_stop: bool = True
    use_time_stop: bool = True


@dataclass
class BacktestResult:
    """回测结果"""
    # 收益指标
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float

    # 交易统计
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_hold_days: float

    # 时间范围
    start_date: str
    end_date: str
    trading_days: int

    # 曲线数据
    equity_curve: pd.DataFrame
    monthly_returns: pd.DataFrame
    trade_log: pd.DataFrame


def load_config(base_dir: Path) -> BacktestConfig:
    """从 config.yaml 加载配置"""
    config_path = base_dir / "config.yaml"

    if not config_path.exists():
        print("⚠️ config.yaml 不存在，使用默认配置")
        return BacktestConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    strategy = cfg.get("strategy", {})
    risk = cfg.get("risk_control", {})

    return BacktestConfig(
        top_k=strategy.get("top_k", 10),
        pullback_window_start=strategy.get("pullback_window_start", 3),
        pullback_window_end=strategy.get("pullback_window_end", 10),
        pullback_min_pct=strategy.get("pullback_min_pct", 0.05),
        pullback_max_pct=strategy.get("pullback_max_pct", 0.25),
        tdx_min_score=strategy.get("tdx_min_score", 1.5),
        stop_loss_pct=risk.get("stop_loss_pct", 0.08),
        trailing_stop_pct=risk.get("trailing_stop_pct", 0.10),
        max_hold_days=risk.get("max_hold_days", 15),
        max_single_weight=strategy.get("max_single_weight", 0.15),
        max_total_position=strategy.get("max_total_position", 0.80),
        use_stop_loss=risk.get("use_stop_loss", True),
        use_trailing_stop=risk.get("use_trailing_stop", True),
        use_time_stop=risk.get("use_time_stop", True),
    )


def load_features(base_dir: Path, start_date: str) -> pd.DataFrame:
    """加载特征数据"""
    feats_path = base_dir / "data" / "features" / "features_daily.parquet"

    if not feats_path.exists():
        raise FileNotFoundError(f"特征文件不存在: {feats_path}\n请先运行 python run_build_features_daily.py")

    df = pd.read_parquet(feats_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.to_datetime(start_date)].copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    return df


def load_hs300(base_dir: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """加载沪深300数据"""
    hs300_path = base_dir / "data" / "index" / "hs300_daily.parquet"

    if not hs300_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(hs300_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.to_datetime(start_date)) &
            (df["date"] <= pd.to_datetime(end_date))].copy()
    df = df.sort_values("date").reset_index(drop=True)

    return df


def compute_signal_score(row: pd.Series, cfg: BacktestConfig) -> float:
    """计算信号得分"""
    score = 0.0

    # 基础因子
    score += row.get("ma_dist_20", 0) * 2.0
    score += row.get("ret_20d", 0) * 1.0
    score += row.get("ret_60d", 0) * 0.5
    score -= row.get("vol_20d", 0) * 1.0
    score -= row.get("atr_pct", 0) * 0.5
    score += row.get("vol_ratio_20", 0) * 0.3

    # TDX指标
    score += row.get("high30_breakout", 0) * 1.0
    score += row.get("main_force_strong", 0) * 1.5
    score += row.get("has_limit_up_30d", 0) * 0.5

    return score


def check_entry_conditions(row: pd.Series, cfg: BacktestConfig) -> bool:
    """检查入场条件"""
    # 条件1: 涨停回调
    days_since = row.get("days_since_limit_up", 999)
    pullback = row.get("pullback_pct", 0)
    volume_breakout = row.get("volume_breakout", 0)

    limit_up_entry = (
            cfg.pullback_window_start <= days_since <= cfg.pullback_window_end and
            cfg.pullback_min_pct <= pullback <= cfg.pullback_max_pct and
            volume_breakout == 1
    )

    # 条件2: TDX指标
    tdx_score = row.get("tdx_score", 0)
    high30 = row.get("high30_breakout", 0)
    main_force = row.get("main_force_strong", 0)

    tdx_entry = tdx_score >= cfg.tdx_min_score or (high30 == 1 and main_force == 1)

    # 条件3: 简单趋势
    ma_dist = row.get("ma_dist_20", 0)
    trend_entry = ma_dist > 0

    # 可交易性过滤
    one_line = row.get("one_line_board", 0)
    limit_up = row.get("limit_up_flag", 0)
    tradeable = one_line == 0 and limit_up == 0

    return tradeable and (limit_up_entry or tdx_entry or trend_entry)


def run_backtest(
        features: pd.DataFrame,
        cfg: BacktestConfig,
        start_date: str,
        end_date: str,
) -> BacktestResult:
    """运行回测"""
    dates = sorted(features["date"].unique())
    dates = [d for d in dates if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)]

    # 初始化
    capital = cfg.initial_capital
    positions = {}  # {symbol: {shares, entry_price, entry_date, high_price}}

    equity_records = []
    trade_records = []

    for date in dates:
        day_data = features[features["date"] == date].copy()

        if day_data.empty:
            continue

        # 计算当日组合价值
        portfolio_value = capital
        for symbol, pos in positions.items():
            sym_data = day_data[day_data["symbol"] == symbol]
            if not sym_data.empty:
                current_price = sym_data["close"].iloc[0]
                portfolio_value += pos["shares"] * current_price
                pos["current_price"] = current_price
                pos["high_price"] = max(pos["high_price"], current_price)
                pos["hold_days"] = (date - pos["entry_date"]).days

        # 记录净值
        equity_records.append({
            "date": date,
            "equity": portfolio_value,
            "capital": capital,
            "positions": len(positions),
        })

        # === 检查止损/止盈 ===
        symbols_to_close = []
        for symbol, pos in positions.items():
            if "current_price" not in pos:
                continue

            current_price = pos["current_price"]
            entry_price = pos["entry_price"]
            high_price = pos["high_price"]
            hold_days = pos["hold_days"]

            pnl_pct = current_price / entry_price - 1
            drawdown = current_price / high_price - 1

            close_reason = None

            # 硬止损
            if cfg.use_stop_loss and pnl_pct <= -cfg.stop_loss_pct:
                close_reason = "止损"
            # 移动止盈
            elif cfg.use_trailing_stop and drawdown <= -cfg.trailing_stop_pct:
                close_reason = "止盈"
            # 时间止损
            elif cfg.use_time_stop and hold_days >= cfg.max_hold_days:
                close_reason = "时间止损"

            if close_reason:
                symbols_to_close.append((symbol, close_reason))

        # 执行平仓
        for symbol, reason in symbols_to_close:
            pos = positions.pop(symbol)
            sell_price = pos["current_price"]
            shares = pos["shares"]

            # 扣除交易成本
            cost = sell_price * shares * cfg.cost_bps / 10000
            proceeds = sell_price * shares - cost
            capital += proceeds

            pnl = proceeds - pos["entry_price"] * shares
            pnl_pct = pnl / (pos["entry_price"] * shares)

            trade_records.append({
                "symbol": symbol,
                "entry_date": pos["entry_date"],
                "exit_date": date,
                "entry_price": pos["entry_price"],
                "exit_price": sell_price,
                "shares": shares,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "hold_days": pos["hold_days"],
                "exit_reason": reason,
            })

        # === 选股 ===
        # 计算得分
        day_data["score"] = day_data.apply(lambda r: compute_signal_score(r, cfg), axis=1)

        # 过滤入场条件
        day_data["eligible"] = day_data.apply(lambda r: check_entry_conditions(r, cfg), axis=1)
        eligible = day_data[day_data["eligible"]].copy()

        # 排序选Top K
        eligible = eligible.sort_values("score", ascending=False).head(cfg.top_k)

        # === 调仓 ===
        current_symbols = set(positions.keys())
        target_symbols = set(eligible["symbol"].tolist())

        # 卖出不在目标中的
        for symbol in current_symbols - target_symbols:
            if symbol in positions:
                pos = positions.pop(symbol)
                if "current_price" in pos:
                    sell_price = pos["current_price"]
                    shares = pos["shares"]
                    cost = sell_price * shares * cfg.cost_bps / 10000
                    proceeds = sell_price * shares - cost
                    capital += proceeds

                    pnl = proceeds - pos["entry_price"] * shares
                    pnl_pct = pnl / (pos["entry_price"] * shares)

                    trade_records.append({
                        "symbol": symbol,
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "entry_price": pos["entry_price"],
                        "exit_price": sell_price,
                        "shares": shares,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "hold_days": pos.get("hold_days", 0),
                        "exit_reason": "调仓",
                    })

        # 买入新标的
        new_symbols = target_symbols - current_symbols
        if new_symbols and capital > 0:
            # 计算可用资金
            available = capital * cfg.max_total_position
            per_stock = min(available / len(new_symbols), capital * cfg.max_single_weight)

            for symbol in new_symbols:
                sym_data = day_data[day_data["symbol"] == symbol]
                if sym_data.empty:
                    continue

                price = sym_data["close"].iloc[0]
                if price <= 0:
                    continue

                shares = int(per_stock / price / 100) * 100  # 整手
                if shares <= 0:
                    continue

                cost = price * shares * cfg.cost_bps / 10000
                total_cost = price * shares + cost

                if total_cost > capital:
                    continue

                capital -= total_cost
                positions[symbol] = {
                    "shares": shares,
                    "entry_price": price,
                    "entry_date": date,
                    "high_price": price,
                }

    # 最终平仓
    final_date = dates[-1] if dates else datetime.now()
    for symbol, pos in positions.items():
        if "current_price" in pos:
            sell_price = pos["current_price"]
            shares = pos["shares"]
            cost = sell_price * shares * cfg.cost_bps / 10000
            proceeds = sell_price * shares - cost
            capital += proceeds

            pnl = proceeds - pos["entry_price"] * shares
            pnl_pct = pnl / (pos["entry_price"] * shares)

            trade_records.append({
                "symbol": symbol,
                "entry_date": pos["entry_date"],
                "exit_date": final_date,
                "entry_price": pos["entry_price"],
                "exit_price": sell_price,
                "shares": shares,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "hold_days": pos.get("hold_days", 0),
                "exit_reason": "结束",
            })

    # 构建结果
    equity_df = pd.DataFrame(equity_records)
    trade_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()

    if equity_df.empty:
        raise ValueError("回测结果为空，请检查数据")

    # 计算指标
    equity_df["daily_return"] = equity_df["equity"].pct_change().fillna(0)

    final_equity = equity_df["equity"].iloc[-1]
    total_return = final_equity / cfg.initial_capital - 1

    trading_days = len(equity_df)
    annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0

    # 最大回撤
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = equity_df["equity"] / equity_df["peak"] - 1
    max_drawdown = equity_df["drawdown"].min()

    # 夏普比率
    daily_returns = equity_df["daily_return"]
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    # 胜率
    if not trade_df.empty:
        winning = (trade_df["pnl"] > 0).sum()
        total_trades = len(trade_df)
        win_rate = winning / total_trades if total_trades > 0 else 0
        avg_hold = trade_df["hold_days"].mean()
    else:
        winning = 0
        total_trades = 0
        win_rate = 0
        avg_hold = 0

    # 月度收益
    equity_df["month"] = equity_df["date"].dt.to_period("M")
    monthly = equity_df.groupby("month").agg({
        "equity": ["first", "last"],
        "daily_return": "sum",
    })
    monthly.columns = ["start_equity", "end_equity", "return"]
    monthly["return"] = monthly["end_equity"] / monthly["start_equity"] - 1
    monthly = monthly.reset_index()

    return BacktestResult(
        total_return=total_return,
        annual_return=annual_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        win_rate=win_rate,
        total_trades=total_trades,
        winning_trades=winning,
        losing_trades=total_trades - winning,
        avg_hold_days=avg_hold,
        start_date=str(dates[0].date()) if dates else "",
        end_date=str(dates[-1].date()) if dates else "",
        trading_days=trading_days,
        equity_curve=equity_df,
        monthly_returns=monthly,
        trade_log=trade_df,
    )


def generate_html_report(
        result: BacktestResult,
        cfg: BacktestConfig,
        hs300: pd.DataFrame,
        output_path: Path,
) -> Path:
    """生成网页版报告"""

    # 准备图表数据
    equity_data = result.equity_curve[["date", "equity", "drawdown"]].copy()
    equity_data["date"] = equity_data["date"].dt.strftime("%Y-%m-%d")
    equity_json = equity_data.to_json(orient="records")

    # 沪深300数据
    if not hs300.empty:
        hs300 = hs300.copy()
        hs300["date"] = hs300["date"].dt.strftime("%Y-%m-%d")
        initial_hs300 = hs300["close"].iloc[0]
        hs300["nav"] = hs300["close"] / initial_hs300
        hs300_json = hs300[["date", "nav"]].to_json(orient="records")
    else:
        hs300_json = "[]"

    # 月度收益数据
    monthly_data = result.monthly_returns.copy()
    monthly_data["month"] = monthly_data["month"].astype(str)
    monthly_json = monthly_data[["month", "return"]].to_json(orient="records")

    # 交易记录
    if not result.trade_log.empty:
        trades = result.trade_log.copy()
        trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.strftime("%Y-%m-%d")
        trades["exit_date"] = pd.to_datetime(trades["exit_date"]).dt.strftime("%Y-%m-%d")
        trades_json = trades.head(100).to_json(orient="records")  # 最多100条
    else:
        trades_json = "[]"

    # 配置信息
    config_info = asdict(cfg)

    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>策略回测报告</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .header p {{
            color: #888;
            font-size: 1.1em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .metric-value {{
            font-size: 2.2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-value.positive {{ color: #00d4ff; }}
        .metric-value.negative {{ color: #ff6b6b; }}
        .metric-value.neutral {{ color: #ffd93d; }}
        .metric-label {{
            color: #888;
            font-size: 0.95em;
        }}
        .chart-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .chart-title {{
            font-size: 1.3em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .chart {{
            height: 400px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section-title {{
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-left: 15px;
            border-left: 4px solid #00d4ff;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{
            background: rgba(0,212,255,0.1);
            font-weight: 600;
        }}
        tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
        .profit {{ color: #00d4ff; }}
        .loss {{ color: #ff6b6b; }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .config-item {{
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 10px;
        }}
        .config-key {{
            color: #888;
            font-size: 0.9em;
        }}
        .config-value {{
            font-size: 1.1em;
            font-weight: 600;
            color: #00d4ff;
        }}
        .footer {{
            text-align: center;
            padding: 30px 0;
            color: #666;
            border-top: 1px solid rgba(255,255,255,0.1);
            margin-top: 30px;
        }}
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 1.8em; }}
            .metric-value {{ font-size: 1.6em; }}
            .chart {{ height: 300px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 策略回测报告</h1>
            <p>回测区间: {result.start_date} ~ {result.end_date} | 交易日: {result.trading_days}天</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {'positive' if result.total_return > 0 else 'negative'}">{result.total_return * 100:.2f}%</div>
                <div class="metric-label">总收益率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if result.annual_return > 0 else 'negative'}">{result.annual_return * 100:.2f}%</div>
                <div class="metric-label">年化收益率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{result.max_drawdown * 100:.2f}%</div>
                <div class="metric-label">最大回撤</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if result.sharpe_ratio > 1 else 'neutral'}">{result.sharpe_ratio:.2f}</div>
                <div class="metric-label">夏普比率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if result.win_rate > 0.5 else 'neutral'}">{result.win_rate * 100:.1f}%</div>
                <div class="metric-label">胜率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{result.total_trades}</div>
                <div class="metric-label">总交易次数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value positive">{result.winning_trades}</div>
                <div class="metric-label">盈利次数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{result.avg_hold_days:.1f}天</div>
                <div class="metric-label">平均持仓</div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">📈 收益曲线</div>
            <div id="equityChart" class="chart"></div>
        </div>

        <div class="chart-container">
            <div class="chart-title">📉 回撤曲线</div>
            <div id="drawdownChart" class="chart"></div>
        </div>

        <div class="chart-container">
            <div class="chart-title">📊 月度收益</div>
            <div id="monthlyChart" class="chart"></div>
        </div>

        <div class="section">
            <h2 class="section-title">⚙️ 策略配置</h2>
            <div class="config-grid">
                <div class="config-item">
                    <div class="config-key">每日选股数</div>
                    <div class="config-value">{cfg.top_k}</div>
                </div>
                <div class="config-item">
                    <div class="config-key">止损阈值</div>
                    <div class="config-value">-{cfg.stop_loss_pct * 100:.0f}%</div>
                </div>
                <div class="config-item">
                    <div class="config-key">移动止盈</div>
                    <div class="config-value">-{cfg.trailing_stop_pct * 100:.0f}%</div>
                </div>
                <div class="config-item">
                    <div class="config-key">最大持仓天数</div>
                    <div class="config-value">{cfg.max_hold_days}天</div>
                </div>
                <div class="config-item">
                    <div class="config-key">单只最大仓位</div>
                    <div class="config-value">{cfg.max_single_weight * 100:.0f}%</div>
                </div>
                <div class="config-item">
                    <div class="config-key">最大总仓位</div>
                    <div class="config-value">{cfg.max_total_position * 100:.0f}%</div>
                </div>
                <div class="config-item">
                    <div class="config-key">回调范围</div>
                    <div class="config-value">{cfg.pullback_min_pct * 100:.0f}%-{cfg.pullback_max_pct * 100:.0f}%</div>
                </div>
                <div class="config-item">
                    <div class="config-key">TDX最低分</div>
                    <div class="config-value">{cfg.tdx_min_score}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">📋 交易记录（最近100条）</h2>
            <div style="overflow-x: auto;">
                <table id="tradesTable">
                    <thead>
                        <tr>
                            <th>股票</th>
                            <th>买入日期</th>
                            <th>卖出日期</th>
                            <th>买入价</th>
                            <th>卖出价</th>
                            <th>持仓天数</th>
                            <th>盈亏</th>
                            <th>盈亏%</th>
                            <th>退出原因</th>
                        </tr>
                    </thead>
                    <tbody id="tradesBody">
                    </tbody>
                </table>
            </div>
        </div>

        <div class="footer">
            <p>报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>⚠️ 本报告仅供参考，不构成投资建议</p>
        </div>
    </div>

    <script>
        // 数据
        const equityData = {equity_json};
        const hs300Data = {hs300_json};
        const monthlyData = {monthly_json};
        const tradesData = {trades_json};

        // 收益曲线图
        const equityChart = echarts.init(document.getElementById('equityChart'));
        const dates = equityData.map(d => d.date);
        const equity = equityData.map(d => (d.equity / {cfg.initial_capital} * 100).toFixed(2));
        const hs300Nav = hs300Data.map(d => (d.nav * 100).toFixed(2));
        const hs300Dates = hs300Data.map(d => d.date);

        equityChart.setOption({{
            tooltip: {{
                trigger: 'axis',
                backgroundColor: 'rgba(0,0,0,0.8)',
                borderColor: '#00d4ff',
                textStyle: {{ color: '#fff' }}
            }},
            legend: {{
                data: ['策略净值', '沪深300'],
                textStyle: {{ color: '#888' }},
                top: 10
            }},
            grid: {{
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            }},
            xAxis: {{
                type: 'category',
                data: dates,
                axisLine: {{ lineStyle: {{ color: '#444' }} }},
                axisLabel: {{ color: '#888' }}
            }},
            yAxis: {{
                type: 'value',
                name: '净值(%)',
                axisLine: {{ lineStyle: {{ color: '#444' }} }},
                axisLabel: {{ color: '#888' }},
                splitLine: {{ lineStyle: {{ color: 'rgba(255,255,255,0.1)' }} }}
            }},
            series: [
                {{
                    name: '策略净值',
                    type: 'line',
                    data: equity,
                    smooth: true,
                    lineStyle: {{ color: '#00d4ff', width: 2 }},
                    areaStyle: {{
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            {{ offset: 0, color: 'rgba(0,212,255,0.3)' }},
                            {{ offset: 1, color: 'rgba(0,212,255,0.05)' }}
                        ])
                    }}
                }},
                {{
                    name: '沪深300',
                    type: 'line',
                    data: hs300Nav,
                    smooth: true,
                    lineStyle: {{ color: '#ff6b6b', width: 1, type: 'dashed' }}
                }}
            ]
        }});

        // 回撤曲线
        const drawdownChart = echarts.init(document.getElementById('drawdownChart'));
        const drawdown = equityData.map(d => (d.drawdown * 100).toFixed(2));

        drawdownChart.setOption({{
            tooltip: {{
                trigger: 'axis',
                backgroundColor: 'rgba(0,0,0,0.8)',
                borderColor: '#ff6b6b'
            }},
            grid: {{
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            }},
            xAxis: {{
                type: 'category',
                data: dates,
                axisLine: {{ lineStyle: {{ color: '#444' }} }},
                axisLabel: {{ color: '#888' }}
            }},
            yAxis: {{
                type: 'value',
                name: '回撤(%)',
                axisLine: {{ lineStyle: {{ color: '#444' }} }},
                axisLabel: {{ color: '#888' }},
                splitLine: {{ lineStyle: {{ color: 'rgba(255,255,255,0.1)' }} }}
            }},
            series: [{{
                type: 'line',
                data: drawdown,
                smooth: true,
                lineStyle: {{ color: '#ff6b6b', width: 1 }},
                areaStyle: {{
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        {{ offset: 0, color: 'rgba(255,107,107,0.5)' }},
                        {{ offset: 1, color: 'rgba(255,107,107,0.1)' }}
                    ])
                }}
            }}]
        }});

        // 月度收益
        const monthlyChart = echarts.init(document.getElementById('monthlyChart'));
        const months = monthlyData.map(d => d.month);
        const returns = monthlyData.map(d => (d.return * 100).toFixed(2));
        const colors = returns.map(r => r >= 0 ? '#00d4ff' : '#ff6b6b');

        monthlyChart.setOption({{
            tooltip: {{
                trigger: 'axis',
                backgroundColor: 'rgba(0,0,0,0.8)'
            }},
            grid: {{
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            }},
            xAxis: {{
                type: 'category',
                data: months,
                axisLine: {{ lineStyle: {{ color: '#444' }} }},
                axisLabel: {{ color: '#888', rotate: 45 }}
            }},
            yAxis: {{
                type: 'value',
                name: '收益(%)',
                axisLine: {{ lineStyle: {{ color: '#444' }} }},
                axisLabel: {{ color: '#888' }},
                splitLine: {{ lineStyle: {{ color: 'rgba(255,255,255,0.1)' }} }}
            }},
            series: [{{
                type: 'bar',
                data: returns.map((v, i) => ({{
                    value: v,
                    itemStyle: {{ color: colors[i] }}
                }})),
                barWidth: '60%'
            }}]
        }});

        // 交易记录表
        const tbody = document.getElementById('tradesBody');
        tradesData.forEach(t => {{
            const pnlClass = t.pnl >= 0 ? 'profit' : 'loss';
            const row = `
                <tr>
                    <td>${{t.symbol}}</td>
                    <td>${{t.entry_date}}</td>
                    <td>${{t.exit_date}}</td>
                    <td>¥${{t.entry_price.toFixed(2)}}</td>
                    <td>¥${{t.exit_price.toFixed(2)}}</td>
                    <td>${{t.hold_days}}天</td>
                    <td class="${{pnlClass}}">¥${{t.pnl.toFixed(2)}}</td>
                    <td class="${{pnlClass}}">${{(t.pnl_pct * 100).toFixed(2)}}%</td>
                    <td>${{t.exit_reason}}</td>
                </tr>
            `;
            tbody.innerHTML += row;
        }});

        // 响应式
        window.addEventListener('resize', () => {{
            equityChart.resize();
            drawdownChart.resize();
            monthlyChart.resize();
        }});
    </script>
</body>
</html>
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="策略回测 + 网页报告生成")
    parser.add_argument("--start-date", default="2023-01-01", help="回测起始日期")
    parser.add_argument("--end-date", default="", help="回测结束日期（默认今天）")
    parser.add_argument("--top-k", type=int, default=0, help="覆盖 top_k 参数")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    print("=" * 60)
    print("📊 策略回测 + 网页报告生成")
    print("=" * 60)

    # 加载配置
    print("\n📂 加载配置...")
    cfg = load_config(base_dir)

    if args.top_k > 0:
        cfg.top_k = args.top_k

    print(f"   Top K: {cfg.top_k}")
    print(f"   止损: -{cfg.stop_loss_pct * 100:.0f}%")
    print(f"   止盈: -{cfg.trailing_stop_pct * 100:.0f}%")

    # 加载数据
    print("\n📂 加载数据...")
    features = load_features(base_dir, args.start_date)
    print(f"   数据行数: {len(features)}")
    print(f"   股票数: {features['symbol'].nunique()}")
    print(f"   日期范围: {features['date'].min().date()} ~ {features['date'].max().date()}")

    # 确定结束日期
    end_date = args.end_date or str(features["date"].max().date())

    # 加载沪深300
    print("\n📂 加载沪深300...")
    hs300 = load_hs300(base_dir, args.start_date, end_date)
    if not hs300.empty:
        print(f"   数据行数: {len(hs300)}")
    else:
        print("   ⚠️ 沪深300数据不可用")

    # 运行回测
    print("\n🚀 运行回测...")
    result = run_backtest(features, cfg, args.start_date, end_date)

    print(f"\n📊 回测结果:")
    print(f"   总收益: {result.total_return * 100:+.2f}%")
    print(f"   年化收益: {result.annual_return * 100:+.2f}%")
    print(f"   最大回撤: {result.max_drawdown * 100:.2f}%")
    print(f"   夏普比率: {result.sharpe_ratio:.2f}")
    print(f"   胜率: {result.win_rate * 100:.1f}%")
    print(f"   交易次数: {result.total_trades}")

    # 生成网页报告
    print("\n📝 生成网页报告...")
    output_path = base_dir / "out" / "backtest_report.html"
    generate_html_report(result, cfg, hs300, output_path)
    print(f"   ✅ 报告已保存: {output_path}")

    # 保存数据
    csv_path = base_dir / "out" / "backtest_equity.csv"
    result.equity_curve.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"   ✅ 净值曲线: {csv_path}")

    if not result.trade_log.empty:
        trades_path = base_dir / "out" / "backtest_trades.csv"
        result.trade_log.to_csv(trades_path, index=False, encoding="utf-8-sig")
        print(f"   ✅ 交易记录: {trades_path}")

    print("\n" + "=" * 60)
    print("✅ 完成！")
    print(f"   打开浏览器查看: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()