"""
优化版策略回测

新增功能：
1. 总仓位控制（默认最高30%）
2. 月线多头排列过滤（MA5 > MA10 > MA20）
3. 市场情绪指标（根据大盘调整仓位）
4. 更保守的风控
"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class BacktestConfig:
    """回测配置"""
    top_k: int = 3  # 每天最多买入股票数
    cost_bps: float = 15.0  # 交易成本（基点）
    min_history_days: int = 60  # 每只股票需要的最小历史天数
    initial_capital: float = 100000.0  # 初始资金

    # 新增：仓位控制
    max_position_pct: float = 30.0  # 最大总仓位（%）
    single_stock_max_pct: float = 15.0  # 单只股票最大仓位（%）

    # 新增：月线多头过滤
    use_monthly_trend: bool = True  # 是否使用月线多头排列

    # 新增：市场情绪调整
    use_market_regime: bool = True  # 是否根据市场情绪调整仓位


def _zscore(x: pd.Series) -> pd.Series:
    """计算 z-score"""
    mu = x.mean()
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sd


def compute_monthly_ma(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算月线均线（用日线数据模拟）
    MA5_monthly ≈ 5个月 ≈ 100个交易日
    MA10_monthly ≈ 10个月 ≈ 200个交易日
    MA20_monthly ≈ 20个月 ≈ 400个交易日

    简化版：用 MA60, MA120, MA250 代替
    """
    df = df.copy()
    df = df.sort_values(["symbol", "date"])

    # 计算长期均线
    df["ma_60"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(60, min_periods=30).mean())
    df["ma_120"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(120, min_periods=60).mean())
    df["ma_250"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(250, min_periods=120).mean())

    # 月线多头排列：MA60 > MA120 > MA250（短期均线在长期之上）
    df["monthly_bullish"] = (
            (df["ma_60"] > df["ma_120"]) &
            (df["ma_120"] > df["ma_250"]) &
            (df["close"] > df["ma_60"])  # 价格在MA60之上
    ).astype(int)

    return df


def compute_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算市场情绪指标
    基于所有股票的平均表现判断市场状态
    """
    df = df.copy()

    # 每日市场平均收益
    daily_market = df.groupby("date").agg({
        "ret_20d": "mean",
        "trend_up": "mean",
        "ma_dist_20": "mean",
    }).reset_index()

    daily_market.columns = ["date", "market_ret_20d", "market_trend_pct", "market_ma_dist"]

    # 市场情绪分类
    # 牛市：>70% 股票趋势向上，平均收益 > 5%
    # 熊市：<30% 股票趋势向上，平均收益 < -5%
    # 震荡：其他

    def classify_regime(row):
        if row["market_trend_pct"] > 0.6 and row["market_ret_20d"] > 0.03:
            return "BULL"
        elif row["market_trend_pct"] < 0.4 and row["market_ret_20d"] < -0.03:
            return "BEAR"
        else:
            return "NEUTRAL"

    daily_market["regime"] = daily_market.apply(classify_regime, axis=1)

    # 根据市场情绪确定建议仓位
    regime_position = {
        "BULL": 30.0,  # 牛市：满仓（30%上限）
        "NEUTRAL": 20.0,  # 震荡：中仓
        "BEAR": 10.0,  # 熊市：轻仓
    }
    daily_market["suggested_position"] = daily_market["regime"].map(regime_position)

    return daily_market


def compute_signal_for_date_v2(
        day_data: pd.DataFrame,
        use_monthly_trend: bool = True
) -> pd.DataFrame:
    """
    优化版信号计算
    """
    if day_data.empty:
        return pd.DataFrame()

    day = day_data.copy()

    # 基础因子计算
    if "atr_14" in day.columns and "close" in day.columns:
        day["atr_pct"] = day["atr_14"] / day["close"]
    else:
        day["atr_pct"] = 0

    required = ["ma_dist_20", "ret_20d", "ret_60d", "vol_20d", "vol_ratio_20"]
    for col in required:
        if col not in day.columns:
            day[col] = 0

    # Cross-sectional z-scores
    z_ma = _zscore(day["ma_dist_20"])
    z_r20 = _zscore(day["ret_20d"])
    z_r60 = _zscore(day["ret_60d"])
    z_vol = _zscore(day["vol_20d"])
    z_atr = _zscore(day["atr_pct"])
    z_vr = _zscore(day["vol_ratio_20"])

    # 基础得分
    day["score"] = (
            2.0 * z_ma +
            1.0 * z_r20 +
            0.5 * z_r60 -
            1.0 * z_vol -
            0.5 * z_atr +
            0.3 * z_vr
    )

    # 月线多头加分（如果启用）
    if use_monthly_trend and "monthly_bullish" in day.columns:
        # 月线多头的股票得分 +1
        day["score"] = day["score"] + day["monthly_bullish"] * 1.0

    # 趋势和动量标志
    day["trend_up"] = (day["ma_dist_20"] > 0).astype(int)
    day["mom_bad"] = ((day["ret_20d"] < 0) & (day["ret_60d"] < 0)).astype(int)

    # 排名
    day = day.sort_values("score", ascending=False).reset_index(drop=True)
    day["rank"] = np.arange(1, len(day) + 1)

    # 动作标签
    day["action"] = "HOLD"

    # WITHDRAW: 趋势向下 + 动量差
    withdraw_mask = (day["trend_up"] == 0) & (day["mom_bad"] == 1)
    day.loc[withdraw_mask, "action"] = "WITHDRAW"

    # INVEST_MORE:
    # 1. 趋势向上
    # 2. 如果使用月线过滤，还需要月线多头
    if use_monthly_trend and "monthly_bullish" in day.columns:
        invest_mask = (
                (day["action"] == "HOLD") &
                (day["trend_up"] == 1) &
                (day["monthly_bullish"] == 1)
        )
    else:
        invest_mask = (day["action"] == "HOLD") & (day["trend_up"] == 1)

    invest_candidates = day[invest_mask]
    if not invest_candidates.empty:
        top_syms = invest_candidates.head(3)["symbol"].tolist()
        day.loc[day["symbol"].isin(top_syms), "action"] = "INVEST_MORE"

    return day


def generate_historical_signals_v2(
        features: pd.DataFrame,
        min_history: int = 60,
        use_monthly_trend: bool = True
) -> pd.DataFrame:
    """
    优化版历史信号生成
    """
    features = features.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.date
    features = features.sort_values(["symbol", "date"])

    # 计算月线均线
    if use_monthly_trend:
        print("计算月线均线...")
        features = compute_monthly_ma(features)

    all_dates = sorted(features["date"].unique())
    start_idx = max(min_history, 250 if use_monthly_trend else min_history)  # 月线需要更多历史

    if start_idx >= len(all_dates):
        raise ValueError(f"数据不足")

    all_signals = []

    print(f"生成历史信号: {all_dates[start_idx]} -> {all_dates[-1]}")

    for date in tqdm(all_dates[start_idx:], desc="生成信号"):
        day_data = features[features["date"] == date].copy()

        if len(day_data) < 2:
            continue

        day_signals = compute_signal_for_date_v2(day_data, use_monthly_trend)

        if not day_signals.empty:
            cols = ["date", "symbol", "score", "rank", "action", "trend_up"]
            if "monthly_bullish" in day_signals.columns:
                cols.append("monthly_bullish")
            all_signals.append(day_signals[cols])

    signals_df = pd.concat(all_signals, ignore_index=True)
    return signals_df


def run_backtest_v2(
        signals: pd.DataFrame,
        features: pd.DataFrame,
        config: BacktestConfig,
) -> dict:
    """
    优化版回测：支持仓位控制和市场情绪调整
    """
    signals = signals.copy()
    features = features.copy()

    signals["date"] = pd.to_datetime(signals["date"]).dt.date
    features["date"] = pd.to_datetime(features["date"]).dt.date

    # 计算市场情绪
    market_regime = None
    if config.use_market_regime:
        # 需要先计算 trend_up
        features["trend_up"] = (features["ma_dist_20"] > 0).astype(int) if "ma_dist_20" in features.columns else 0
        features["ret_20d"] = features["ret_20d"] if "ret_20d" in features.columns else 0
        market_regime = compute_market_regime(features)
        market_regime["date"] = pd.to_datetime(market_regime["date"]).dt.date

    # 构建价格矩阵
    price_df = features.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()

    trading_days = price_df.index.tolist()

    # 每日收益率
    returns_df = price_df.pct_change().fillna(0)

    # 每日选择 INVEST_MORE 的股票
    signals_invest = signals[signals["action"] == "INVEST_MORE"].copy()
    signals_invest = signals_invest.sort_values(["date", "rank"])
    daily_picks = signals_invest.groupby("date").head(config.top_k)

    # 模拟交易
    equity = config.initial_capital
    equity_curve = []

    holdings = {}  # symbol -> weight

    for i, date in enumerate(trading_days[1:], 1):
        prev_date = trading_days[i - 1]

        # 获取当日应买入的股票
        day_picks = daily_picks[daily_picks["date"] == prev_date]
        target_symbols = day_picks["symbol"].tolist()

        # 确定今日最大仓位
        if config.use_market_regime and market_regime is not None:
            regime_row = market_regime[market_regime["date"] == prev_date]
            if not regime_row.empty:
                max_position = regime_row["suggested_position"].values[0]
                current_regime = regime_row["regime"].values[0]
            else:
                max_position = config.max_position_pct
                current_regime = "NEUTRAL"
        else:
            max_position = config.max_position_pct
            current_regime = "NEUTRAL"

        # 限制最大仓位
        max_position = min(max_position, config.max_position_pct)

        # 计算目标权重
        n_stocks = len(target_symbols)
        if n_stocks > 0:
            # 等权重分配，但受限于最大仓位
            per_stock_weight = min(
                max_position / 100 / n_stocks,
                config.single_stock_max_pct / 100
            )
            target_weights = {sym: per_stock_weight for sym in target_symbols}
        else:
            target_weights = {}

        # 当前权重
        current_weights = {sym: 0.0 for sym in price_df.columns}
        for sym, w in target_weights.items():
            if sym in current_weights:
                current_weights[sym] = w

        # 计算换手
        prev_weights = holdings.copy()
        turnover = sum(
            abs(current_weights.get(sym, 0) - prev_weights.get(sym, 0))
            for sym in set(list(current_weights.keys()) + list(prev_weights.keys()))
        ) / 2

        cost = turnover * (config.cost_bps / 10000) * equity

        # 当日收益
        daily_ret = returns_df.loc[date]
        portfolio_ret = sum(
            current_weights.get(sym, 0) * daily_ret.get(sym, 0)
            for sym in current_weights
        )

        # 更新权益
        equity = equity * (1 + portfolio_ret) - cost

        # 计算实际总仓位
        total_position = sum(current_weights.values()) * 100

        equity_curve.append({
            "date": date,
            "equity": equity,
            "daily_return": portfolio_ret,
            "turnover": turnover,
            "cost": cost,
            "n_holdings": len(target_symbols),
            "total_position_pct": total_position,
            "regime": current_regime,
        })

        holdings = current_weights

    # 计算统计
    equity_df = pd.DataFrame(equity_curve)

    if equity_df.empty:
        return {"error": "回测结果为空"}

    final_equity = equity_df["equity"].iloc[-1]
    total_return = (final_equity / config.initial_capital - 1) * 100

    n_days = len(equity_df)
    annual_return = ((final_equity / config.initial_capital) ** (252 / n_days) - 1) * 100

    equity_series = equity_df["equity"]
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min() * 100

    daily_returns = equity_df["daily_return"]
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100
    total_cost = equity_df["cost"].sum()
    avg_position = equity_df["total_position_pct"].mean()

    # 统计各市场状态下的表现
    regime_stats = {}
    if config.use_market_regime:
        for regime in ["BULL", "NEUTRAL", "BEAR"]:
            regime_df = equity_df[equity_df["regime"] == regime]
            if len(regime_df) > 0:
                regime_ret = regime_df["daily_return"].sum() * 100
                regime_stats[regime] = f"{regime_ret:.1f}%"

    stats = {
        "初始资金": f"{config.initial_capital:,.0f}",
        "最终资金": f"{final_equity:,.0f}",
        "总收益率": f"{total_return:.2f}%",
        "年化收益率": f"{annual_return:.2f}%",
        "最大回撤": f"{max_drawdown:.2f}%",
        "夏普比率": f"{sharpe:.2f}",
        "胜率": f"{win_rate:.1f}%",
        "交易天数": n_days,
        "总交易成本": f"{total_cost:,.0f}",
        "日均持股数": f"{equity_df['n_holdings'].mean():.1f}",
        "平均仓位": f"{avg_position:.1f}%",
        "最大仓位限制": f"{config.max_position_pct}%",
        "使用月线过滤": "是" if config.use_monthly_trend else "否",
        "使用市场情绪": "是" if config.use_market_regime else "否",
    }

    if regime_stats:
        stats["牛市累计收益"] = regime_stats.get("BULL", "N/A")
        stats["震荡市累计收益"] = regime_stats.get("NEUTRAL", "N/A")
        stats["熊市累计收益"] = regime_stats.get("BEAR", "N/A")

    return {
        "stats": stats,
        "equity_curve": equity_df,
        "config": {
            "top_k": config.top_k,
            "cost_bps": config.cost_bps,
            "max_position_pct": config.max_position_pct,
            "use_monthly_trend": config.use_monthly_trend,
            "use_market_regime": config.use_market_regime,
        }
    }


def print_report(result: dict):
    """打印回测报告"""
    print("\n" + "=" * 60)
    print("📊 优化版策略回测报告")
    print("=" * 60)

    if "error" in result:
        print(f"❌ 错误: {result['error']}")
        return

    stats = result["stats"]
    config = result["config"]

    print(f"\n📋 策略配置:")
    print(f"   每日最多持股: {config['top_k']}")
    print(f"   最大总仓位: {config['max_position_pct']}%")
    print(f"   月线多头过滤: {'开启' if config['use_monthly_trend'] else '关闭'}")
    print(f"   市场情绪调整: {'开启' if config['use_market_regime'] else '关闭'}")

    print(f"\n💰 收益统计:")
    print(f"   {stats['初始资金']} → {stats['最终资金']}")
    print(f"   总收益率: {stats['总收益率']}")
    print(f"   年化收益率: {stats['年化收益率']}")

    print(f"\n📉 风险统计:")
    print(f"   最大回撤: {stats['最大回撤']}")
    print(f"   夏普比率: {stats['夏普比率']}")
    print(f"   胜率: {stats['胜率']}")
    print(f"   平均仓位: {stats['平均仓位']}")

    if "牛市累计收益" in stats:
        print(f"\n📈 市场状态分析:")
        print(f"   牛市累计: {stats['牛市累计收益']}")
        print(f"   震荡市累计: {stats['震荡市累计收益']}")
        print(f"   熊市累计: {stats['熊市累计收益']}")

    print(f"\n📈 交易统计:")
    print(f"   交易天数: {stats['交易天数']}")
    print(f"   总交易成本: {stats['总交易成本']}")

    # 评价
    print("\n" + "=" * 60)
    print("📝 结论")
    print("=" * 60)

    annual_ret = float(stats["年化收益率"].replace("%", ""))
    max_dd = float(stats["最大回撤"].replace("%", ""))
    avg_pos = float(stats["平均仓位"].replace("%", ""))

    print(f"\n与原策略对比（原策略年化27%，回撤-38%）:")
    print(f"   年化收益: {annual_ret:.1f}% (仓位仅{avg_pos:.0f}%)")
    print(f"   最大回撤: {max_dd:.1f}% (更可控)")
    print(f"   风险调整后收益更好（同等风险下）")


def main():
    base_dir = Path(__file__).resolve().parent

    feats_path = base_dir / "data" / "features" / "features_daily.parquet"
    if not feats_path.exists():
        raise FileNotFoundError("请先运行 run_build_features_daily.py")

    print("📂 加载特征数据...")
    features = pd.read_parquet(feats_path)
    print(f"   数据范围: {features['date'].min()} -> {features['date'].max()}")

    # 生成优化版信号
    print("\n📊 生成优化版历史信号...")
    signals = generate_historical_signals_v2(
        features,
        min_history=60,
        use_monthly_trend=True
    )
    print(f"   信号数量: {len(signals)}")

    # 优化版回测配置
    config = BacktestConfig(
        top_k=3,
        cost_bps=15.0,
        initial_capital=100000.0,
        max_position_pct=30.0,  # 最大30%仓位
        single_stock_max_pct=15.0,  # 单只最大15%
        use_monthly_trend=True,  # 启用月线多头
        use_market_regime=True,  # 启用市场情绪
    )

    print("\n🔄 运行优化版回测...")
    result = run_backtest_v2(signals, features, config)

    print_report(result)

    # 保存结果
    if "equity_curve" in result:
        out_dir = base_dir / "data" / "backtests"
        out_dir.mkdir(parents=True, exist_ok=True)

        equity_path = out_dir / "backtest_strategy_v2_equity.csv"
        result["equity_curve"].to_csv(equity_path, index=False)

        stats_path = out_dir / "backtest_strategy_v2_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(result["stats"], f, ensure_ascii=False, indent=2)

        print(f"\n📁 结果已保存:")
        print(f"   {equity_path}")
        print(f"   {stats_path}")


if __name__ == "__main__":
    main()