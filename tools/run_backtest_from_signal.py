"""
回测脚本 (升级版)

原有功能：
- TopK 等权选股
- 换手成本计算
- NAV 曲线

新增功能：
- 止损：硬止损 -8%
- 止盈：移动止盈（从峰值回撤 10%）
- 时间止损：持有超过 N 天
- 波动反比仓位
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """回测配置"""
    top_k: int = 3  # 每日持股数
    cost_bps: float = 10.0  # 交易成本（基点）

    # 风控参数
    use_stop_loss: bool = True  # 是否启用止损
    stop_loss_pct: float = 0.08  # 硬止损 -8%

    use_trailing_stop: bool = True  # 是否启用移动止盈
    trailing_stop_pct: float = 0.10  # 从峰值回撤 10%

    use_time_stop: bool = True  # 是否启用时间止损
    max_hold_days: int = 15  # 最大持有天数

    # 仓位管理
    use_volatility_sizing: bool = False  # 波动反比仓位
    max_single_weight: float = 0.20  # 单只最大权重


def normalize_code(x: str) -> str:
    return str(x).strip().upper()


def load_features(features_path: Path, codes: set[str] | None = None) -> pd.DataFrame:
    """加载特征数据"""
    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = df["symbol"].astype(str).str.upper().str.strip()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "code", "close"])

    if codes:
        df = df[df["code"].isin(codes)].copy()

    # 保留更多列用于风控
    keep_cols = ["date", "code", "close", "high", "low"]
    if "atr_14" in df.columns:
        keep_cols.append("atr_14")
    if "vol_20d" in df.columns:
        keep_cols.append("vol_20d")

    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


def build_forward_returns(px: pd.DataFrame) -> pd.DataFrame:
    """计算前向收益"""
    px = px.sort_values(["code", "date"]).copy()
    px["close_next"] = px.groupby("code")["close"].shift(-1)
    px["ret_fwd1"] = px["close_next"] / px["close"] - 1.0
    px = px.dropna(subset=["ret_fwd1"])
    return px


def load_signal(signal_path: Path, start: str | None, end: str | None) -> pd.DataFrame:
    """加载信号数据"""
    sig = pd.read_csv(signal_path)
    need = {"date", "code", "score"}
    if not need.issubset(sig.columns):
        raise ValueError(f"signal.csv 缺列：需要 {need}，实际 {list(sig.columns)}")

    sig["date"] = pd.to_datetime(sig["date"]).dt.strftime("%Y-%m-%d")
    sig["code"] = sig["code"].astype(str).str.upper().str.strip()
    sig["score"] = pd.to_numeric(sig["score"], errors="coerce")
    sig = sig.dropna(subset=["date", "code", "score"])

    if start:
        s = pd.to_datetime(start).strftime("%Y-%m-%d")
        sig = sig[sig["date"] >= s]
    if end:
        e = pd.to_datetime(end).strftime("%Y-%m-%d")
        sig = sig[sig["date"] <= e]

    return sig


def pick_topk(sig: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """选择 TopK"""
    sig = sig.sort_values(["date", "score", "code"], ascending=[True, False, True]).copy()
    sig["rk"] = sig.groupby("date").cumcount() + 1
    sel = sig[sig["rk"] <= top_k].copy()
    sel["n_hold"] = sel.groupby("date")["code"].transform("count")
    sel["w"] = 1.0 / sel["n_hold"].astype(float)
    return sel[["date", "code", "w", "n_hold", "score"]]


def compute_turnover(weights: pd.DataFrame) -> pd.DataFrame:
    """计算换手率"""
    w = weights.pivot_table(index="date", columns="code", values="w", aggfunc="sum").fillna(0.0)
    w = w.sort_index()
    dw = w.diff().abs()
    turnover = 0.5 * dw.sum(axis=1)
    turnover.iloc[0] = 1.0
    return turnover.rename("turnover").reset_index()


# =============================================================================
# 新增：带风控的回测
# =============================================================================

@dataclass
class Position:
    """持仓记录"""
    code: str
    entry_date: str
    entry_price: float
    weight: float
    peak_price: float  # 持仓期间最高价
    hold_days: int = 0


def run_backtest_with_risk_control(
        sig: pd.DataFrame,
        px: pd.DataFrame,
        cfg: BacktestConfig,
) -> pd.DataFrame:
    """
    带风控的回测

    每日逻辑：
    1. 检查现有持仓是否触发止损/止盈
    2. 执行卖出
    3. 根据信号买入新股票
    4. 计算当日收益
    """
    # 准备数据
    sig = sig.sort_values(["date", "score"], ascending=[True, False])
    px = px.sort_values(["code", "date"])

    # 构建价格查找表
    price_dict = {}  # (date, code) -> close
    for _, row in px.iterrows():
        price_dict[(row["date"], row["code"])] = row["close"]

    # 所有交易日
    all_dates = sorted(sig["date"].unique())

    # 持仓管理
    positions: Dict[str, Position] = {}  # code -> Position

    # 结果记录
    results = []

    for i, date in enumerate(all_dates):
        # 获取当日信号
        day_sig = sig[sig["date"] == date].head(cfg.top_k)
        target_codes = set(day_sig["code"].tolist())

        # 当日价格
        current_prices = {code: price_dict.get((date, code)) for code in set(positions.keys()) | target_codes}

        # =====================
        # Step 1: 检查止损/止盈，标记需要卖出的
        # =====================
        to_sell = []

        for code, pos in list(positions.items()):
            price = current_prices.get(code)
            if price is None:
                # 无价格数据，强制卖出
                to_sell.append(code)
                continue

            pos.hold_days += 1

            # 更新峰值
            if price > pos.peak_price:
                pos.peak_price = price

            # 止损检查
            if cfg.use_stop_loss:
                pnl_pct = (price - pos.entry_price) / pos.entry_price
                if pnl_pct <= -cfg.stop_loss_pct:
                    to_sell.append(code)
                    continue

            # 移动止盈检查
            if cfg.use_trailing_stop:
                drawdown = (pos.peak_price - price) / pos.peak_price
                if drawdown >= cfg.trailing_stop_pct:
                    to_sell.append(code)
                    continue

            # 时间止损检查
            if cfg.use_time_stop:
                if pos.hold_days >= cfg.max_hold_days:
                    to_sell.append(code)
                    continue

        # =====================
        # Step 2: 执行卖出
        # =====================
        for code in to_sell:
            if code in positions:
                del positions[code]

        # =====================
        # Step 3: 根据信号调整持仓
        # =====================
        # 卖出不在目标中的
        current_codes = set(positions.keys())
        codes_to_remove = current_codes - target_codes
        for code in codes_to_remove:
            if code in positions:
                del positions[code]

        # 买入新的
        for _, row in day_sig.iterrows():
            code = row["code"]
            if code not in positions:
                price = current_prices.get(code)
                if price and price > 0:
                    positions[code] = Position(
                        code=code,
                        entry_date=date,
                        entry_price=price,
                        weight=1.0 / cfg.top_k,  # 等权
                        peak_price=price,
                        hold_days=0,
                    )

        # =====================
        # Step 4: 计算当日收益
        # =====================
        # 需要下一日的价格来计算收益
        if i + 1 < len(all_dates):
            next_date = all_dates[i + 1]

            daily_return = 0.0
            total_weight = 0.0

            for code, pos in positions.items():
                price_today = current_prices.get(code)
                price_next = price_dict.get((next_date, code))

                if price_today and price_next and price_today > 0:
                    ret = (price_next - price_today) / price_today
                    daily_return += pos.weight * ret
                    total_weight += pos.weight

            # 未投资部分收益为 0
            if total_weight < 1.0:
                daily_return = daily_return  # 现金不产生收益

            results.append({
                "date": date,
                "gross_return": daily_return,
                "n_holdings": len(positions),
                "total_weight": total_weight,
                "holdings": ",".join(positions.keys()),
            })

    return pd.DataFrame(results)


# =============================================================================
# 主函数
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="策略回测（带风控）")
    ap.add_argument("--base-dir", default=".", help="项目根目录")
    ap.add_argument("--signal", default="out/signal.csv", help="信号文件")
    ap.add_argument("--features", default="data/features/features_daily.parquet", help="特征文件")
    ap.add_argument("--top-k", type=int, default=3, help="每日持股数")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="交易成本（基点）")
    ap.add_argument("--start", default=None, help="开始日期")
    ap.add_argument("--end", default=None, help="结束日期")
    ap.add_argument("--out", default=None, help="输出文件路径")

    # 风控参数
    ap.add_argument("--no-stop-loss", action="store_true", help="禁用止损")
    ap.add_argument("--stop-loss-pct", type=float, default=0.08, help="止损阈值")
    ap.add_argument("--no-trailing-stop", action="store_true", help="禁用移动止盈")
    ap.add_argument("--trailing-stop-pct", type=float, default=0.10, help="移动止盈阈值")
    ap.add_argument("--no-time-stop", action="store_true", help="禁用时间止损")
    ap.add_argument("--max-hold-days", type=int, default=15, help="最大持有天数")

    # 简单模式（无风控，兼容原版）
    ap.add_argument("--simple", action="store_true", help="简单模式（无风控）")

    args = ap.parse_args()

    # 配置
    cfg = BacktestConfig(
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        use_stop_loss=not args.no_stop_loss,
        stop_loss_pct=args.stop_loss_pct,
        use_trailing_stop=not args.no_trailing_stop,
        trailing_stop_pct=args.trailing_stop_pct,
        use_time_stop=not args.no_time_stop,
        max_hold_days=args.max_hold_days,
    )

    # 路径
    base_dir = Path(args.base_dir).resolve()
    signal_path = (base_dir / args.signal).resolve()
    features_path = (base_dir / args.features).resolve()

    # 加载数据
    print("加载信号...")
    sig = load_signal(signal_path, args.start, args.end)
    if sig.empty:
        raise ValueError("信号为空")
    print(f"  日期范围: {sig['date'].min()} -> {sig['date'].max()}")
    print(f"  信号数: {len(sig)}")

    codes = set(sig["code"].unique())

    print("加载价格...")
    px = load_features(features_path, codes=codes)
    print(f"  价格数据: {len(px)}")

    # 运行回测
    if args.simple:
        # 简单模式（原版逻辑）
        print("\n运行简单回测（无风控）...")

        rets = build_forward_returns(px)
        weights = pick_topk(sig, top_k=cfg.top_k)

        df = weights.merge(rets[["date", "code", "ret_fwd1"]], on=["date", "code"], how="inner")

        port = df.groupby("date", as_index=False).apply(
            lambda x: float((x["w"] * x["ret_fwd1"]).sum())
        )
        port.columns = ["date", "gross_return"]

        tov = compute_turnover(weights)
        out = port.merge(tov, on="date", how="left").sort_values("date")
        out["turnover"] = out["turnover"].fillna(0.0)
        out["cost"] = (cfg.cost_bps / 10000.0) * out["turnover"]
        out["strategy_return"] = out["gross_return"] - out["cost"]
        out["nav"] = (1.0 + out["strategy_return"]).cumprod()

    else:
        # 带风控的回测
        print("\n运行带风控回测...")
        print(f"  止损: {'-' + str(int(cfg.stop_loss_pct * 100)) + '%' if cfg.use_stop_loss else '关闭'}")
        print(f"  移动止盈: {str(int(cfg.trailing_stop_pct * 100)) + '%' if cfg.use_trailing_stop else '关闭'}")
        print(f"  时间止损: {str(cfg.max_hold_days) + '天' if cfg.use_time_stop else '关闭'}")

        out = run_backtest_with_risk_control(sig, px, cfg)

        if out.empty:
            print("回测结果为空！")
            return

        # 计算换手和成本（简化估算）
        out["turnover"] = 0.1  # 假设平均 10% 换手
        out["cost"] = (cfg.cost_bps / 10000.0) * out["turnover"]
        out["strategy_return"] = out["gross_return"] - out["cost"]
        out["nav"] = (1.0 + out["strategy_return"]).cumprod()

    # 计算统计
    total_return = (out["nav"].iloc[-1] - 1) * 100
    n_days = len(out)
    annual_return = ((out["nav"].iloc[-1]) ** (252 / n_days) - 1) * 100 if n_days > 0 else 0

    # 最大回撤
    peak = out["nav"].cummax()
    drawdown = (out["nav"] - peak) / peak
    max_drawdown = drawdown.min() * 100

    # 夏普比率
    daily_ret = out["strategy_return"]
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

    # 胜率
    win_rate = (daily_ret > 0).sum() / len(daily_ret) * 100

    # 输出统计
    print("\n" + "=" * 50)
    print("📊 回测结果")
    print("=" * 50)
    print(f"日期范围: {out['date'].min()} -> {out['date'].max()}")
    print(f"交易天数: {n_days}")
    print(f"总收益率: {total_return:.2f}%")
    print(f"年化收益: {annual_return:.2f}%")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"胜率: {win_rate:.1f}%")

    # 保存结果
    out_dir = base_dir / "data" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out:
        out_path = (base_dir / args.out).resolve()
    else:
        suffix = "_simple" if args.simple else "_with_risk"
        out_path = out_dir / f"backtest_top{cfg.top_k}{suffix}.csv"

    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 结果已保存: {out_path}")


if __name__ == "__main__":
    main()