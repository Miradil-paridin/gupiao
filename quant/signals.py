"""
信号计算模块 (升级版)

原有功能：
- 因子得分计算 (score)
- 横截面排名 (rank)
- 动作标签 (INVEST_MORE, HOLD, REDUCE, WITHDRAW, LEAST)

新增功能：
- 市场环境过滤 (可选)
- 可交易性过滤 (一字板、涨停)
- 涨停回调入场 (eligible)
- 波动反比仓位分配 (target_weight)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class SignalConfig:
    """信号配置（兼容原版 + 新增参数）"""

    # === 原有参数 ===
    invest_more_n: int = 3  # 每日选股数
    least_n: int = 2  # 最差股票数
    withdraw_score_threshold: float = -0.5  # 撤退分数阈值
    risk_vol_20d_threshold: float = 0.55  # 高波动阈值

    # === 新增：入场过滤 ===
    use_tradeability_filter: bool = True  # 可交易性过滤
    use_limit_up_entry: bool = False  # 涨停回调入场（默认关闭，兼容旧版）

    # === 新增：涨停回调参数 ===
    pullback_window_start: int = 3  # 回调窗口开始
    pullback_window_end: int = 10  # 回调窗口结束
    pullback_min_pct: float = 0.05  # 最小回调 5%
    pullback_max_pct: float = 0.25  # 最大回调 25%
    volume_breakout_ratio: float = 1.3  # 放量倍数

    # === 新增：仓位管理 ===
    use_volatility_sizing: bool = False  # 波动反比仓位（默认关闭）
    max_single_weight: float = 0.20  # 单只最大权重 20%

    # === 新增：市场环境 ===
    use_market_regime: bool = False  # 市场环境过滤（默认关闭）


def _zscore(x: pd.Series) -> pd.Series:
    """计算横截面 z-score"""
    x = x.astype(float)
    mu = x.mean()
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sd


def load_features(base_dir: Path) -> pd.DataFrame:
    """加载特征数据"""
    p = base_dir / "data" / "features" / "features_daily.parquet"
    if not p.exists():
        raise FileNotFoundError("features_daily.parquet not found. Run run_build_features_daily.py first.")
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# =============================================================================
# 新增：可交易性过滤
# =============================================================================

def filter_tradeability(day: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    """
    可交易性过滤

    过滤规则：
    1. 一字板：high == low（买不进）
    2. 当日涨停：不追涨停
    3. 异常数据：close <= 0
    """
    day = day.copy()
    day["tradeable"] = True

    # 异常数据
    if "close" in day.columns:
        day.loc[day["close"] <= 0, "tradeable"] = False

    # 一字板
    if "one_line_board" in day.columns:
        day.loc[day["one_line_board"] == 1, "tradeable"] = False
    elif "high" in day.columns and "low" in day.columns:
        day.loc[day["high"] == day["low"], "tradeable"] = False

    # 当日涨停（不追）
    if "limit_up_flag" in day.columns:
        day.loc[day["limit_up_flag"] == 1, "tradeable"] = False

    # 当日接近涨停（不追）
    if "near_limit_up" in day.columns:
        day.loc[day["near_limit_up"] == 1, "tradeable"] = False

    return day


# =============================================================================
# 新增：涨停回调入场资格
# =============================================================================

def compute_eligible(day: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    """
    计算入场资格

    涨停回调入场条件：
    1. 近期有涨停（3-10天内）
    2. 回调幅度在 5%-25%
    3. 放量突破
    4. 近期低点不创新低
    """
    day = day.copy()
    day["eligible"] = False

    # 检查必需列
    required = ["days_since_limit_up", "pullback_pct", "volume_breakout", "price_above_ma5"]
    missing = [c for c in required if c not in day.columns]

    if missing:
        # 缺少涨停特征，使用简化版（趋势向上即可）
        if "ma_dist_20" in day.columns:
            day["eligible"] = day["ma_dist_20"] > 0
        else:
            day["eligible"] = True
        return day

    # 条件1：在回调窗口内
    in_window = (
            (day["days_since_limit_up"] >= cfg.pullback_window_start) &
            (day["days_since_limit_up"] <= cfg.pullback_window_end)
    )

    # 条件2：回调幅度合理
    pullback_ok = (
            (day["pullback_pct"] >= cfg.pullback_min_pct) &
            (day["pullback_pct"] <= cfg.pullback_max_pct)
    )

    # 条件3：放量突破 + 价格在MA5上方
    breakout_ok = (
            (day["volume_breakout"] == 1) &
            (day["price_above_ma5"] == 1)
    )

    # 条件4：不创新低
    if "no_new_low" in day.columns:
        no_new_low = day["no_new_low"] == 1
    else:
        no_new_low = pd.Series(True, index=day.index)

    # 综合条件
    eligible_mask = in_window & pullback_ok & breakout_ok & no_new_low
    day.loc[eligible_mask, "eligible"] = True

    return day


# =============================================================================
# 新增：仓位分配
# =============================================================================

def compute_weights(selected: pd.DataFrame, cfg: SignalConfig) -> Dict[str, float]:
    """
    计算目标仓位权重

    规则：
    1. 如果启用波动反比：权重 = 1/atr_pct
    2. 否则等权
    3. 限制单只最大权重
    """
    if selected.empty:
        return {}

    selected = selected.copy()
    n = len(selected)

    if cfg.use_volatility_sizing and "atr_pct" in selected.columns:
        # 波动反比
        atr = selected["atr_pct"].clip(lower=0.01)
        raw_w = 1.0 / atr
        raw_w = raw_w / raw_w.sum()
    else:
        # 等权
        raw_w = pd.Series(1.0 / n, index=selected.index)

    # 限制单只最大权重
    weights = raw_w.clip(upper=cfg.max_single_weight)

    # 归一化（如果有截断）
    if weights.sum() > 1.0:
        weights = weights / weights.sum()

    return dict(zip(selected["symbol"], weights))


# =============================================================================
# 主函数：计算每日排名（兼容原版 API）
# =============================================================================

def compute_daily_ranking(
        feats: pd.DataFrame,
        as_of: str | None = None,
        cfg: SignalConfig | None = None,
        market_can_open: bool = True,  # 新增：市场是否允许开仓
) -> pd.DataFrame:
    """
    计算单日横截面排名 + 动作标签

    Args:
        feats: 特征数据
        as_of: 日期 "YYYY-MM-DD"，None 则取最新
        cfg: 配置
        market_can_open: 市场是否允许开仓（新增）

    Returns:
        DataFrame with: date, symbol, score, rank, action, target_weight, ...
    """
    if cfg is None:
        cfg = SignalConfig()

    if as_of is None:
        d = feats["date"].max()
    else:
        d = pd.to_datetime(as_of).date()

    day = feats[feats["date"] == d].copy()

    if day.empty:
        raise ValueError(f"No rows for date: {d}")

    # =====================
    # 必需列检查
    # =====================
    need = [
        "symbol", "date", "close",
        "ma_dist_20", "ret_20d", "ret_60d",
        "vol_20d", "atr_14", "vol_ratio_20",
        "ma_20"
    ]
    miss = [c for c in need if c not in day.columns]
    if miss:
        raise ValueError(f"Missing feature columns: {miss}")

    # =====================
    # Step 1: 可交易性过滤（新增）
    # =====================
    if cfg.use_tradeability_filter:
        day = filter_tradeability(day, cfg)
        # 只对可交易的股票计算信号
        tradeable_day = day[day.get("tradeable", True) == True].copy()
    else:
        tradeable_day = day.copy()
        day["tradeable"] = True

    if tradeable_day.empty:
        # 全部不可交易，返回空结果
        day["score"] = 0
        day["rank"] = np.nan
        day["action"] = "WITHDRAW"
        day["target_weight"] = 0.0
        return day

    # =====================
    # Step 2: 计算因子得分
    # =====================
    tradeable_day["atr_pct"] = tradeable_day["atr_14"] / tradeable_day["close"]

    z_ma = _zscore(tradeable_day["ma_dist_20"])
    z_r20 = _zscore(tradeable_day["ret_20d"])
    z_r60 = _zscore(tradeable_day["ret_60d"])
    z_vol = _zscore(tradeable_day["vol_20d"])
    z_atr = _zscore(tradeable_day["atr_pct"])
    z_vr = _zscore(tradeable_day["vol_ratio_20"])

    tradeable_day["score"] = (
            2.0 * z_ma +
            1.0 * z_r20 +
            0.5 * z_r60 -
            1.0 * z_vol -
            0.5 * z_atr +
            0.3 * z_vr
    )

    # =====================
    # Step 3: 趋势/动量标记
    # =====================
    tradeable_day["trend_up"] = (tradeable_day["ma_dist_20"] > 0).astype(int)
    tradeable_day["mom_bad"] = (
            (tradeable_day["ret_20d"] < 0) & (tradeable_day["ret_60d"] < 0)
    ).astype(int)
    tradeable_day["risk_high"] = (tradeable_day["vol_20d"] >= cfg.risk_vol_20d_threshold).astype(int)

    # =====================
    # Step 4: 入场资格（新增）
    # =====================
    if cfg.use_limit_up_entry:
        tradeable_day = compute_eligible(tradeable_day, cfg)
    else:
        # 不使用涨停回调，趋势向上即有资格
        tradeable_day["eligible"] = tradeable_day["trend_up"] == 1

    # =====================
    # Step 5: 排名
    # =====================
    tradeable_day = tradeable_day.sort_values("score", ascending=False).reset_index(drop=True)
    tradeable_day["rank"] = np.arange(1, len(tradeable_day) + 1)

    # =====================
    # Step 6: 动作标签
    # =====================
    tradeable_day["action"] = "HOLD"

    # WITHDRAW: 趋势向下 + 动量差，或分数太低
    withdraw_mask = (
            ((tradeable_day["trend_up"] == 0) & (tradeable_day["mom_bad"] == 1)) |
            (tradeable_day["score"] <= cfg.withdraw_score_threshold)
    )
    tradeable_day.loc[withdraw_mask, "action"] = "WITHDRAW"

    # REDUCE: 高波动但趋势向上
    reduce_mask = (
            (tradeable_day["action"] != "WITHDRAW") &
            (tradeable_day["risk_high"] == 1) &
            (tradeable_day["trend_up"] == 1) &
            (tradeable_day["mom_bad"] == 0)
    )
    tradeable_day.loc[reduce_mask, "action"] = "REDUCE"

    # INVEST_MORE: 从有资格的候选中选 TopN
    if cfg.use_market_regime and not market_can_open:
        # 市场不允许开仓
        invest_syms = []
    else:
        invest_candidates = tradeable_day[
            (tradeable_day["action"] == "HOLD") &
            (tradeable_day["eligible"] == True)
            ].copy()
        invest_syms = invest_candidates.sort_values("score", ascending=False).head(cfg.invest_more_n)["symbol"].tolist()

    tradeable_day.loc[tradeable_day["symbol"].isin(invest_syms), "action"] = "INVEST_MORE"

    # LEAST: 最差的几只
    least_candidates = tradeable_day[tradeable_day["action"] == "HOLD"].sort_values("score", ascending=True)
    least_syms = least_candidates.head(cfg.least_n)["symbol"].tolist()
    tradeable_day.loc[tradeable_day["symbol"].isin(least_syms), "action"] = "LEAST"

    # =====================
    # Step 7: 目标权重（新增）
    # =====================
    selected = tradeable_day[tradeable_day["action"] == "INVEST_MORE"]
    weights = compute_weights(selected, cfg)

    tradeable_day["target_weight"] = tradeable_day["symbol"].map(weights).fillna(0.0)

    # =====================
    # Step 8: 合并回原数据
    # =====================
    # 把计算结果合并回完整的 day（包括不可交易的）
    merge_cols = ["symbol", "score", "rank", "action", "atr_pct", "trend_up", "mom_bad",
                  "risk_high", "eligible", "target_weight"]
    merge_cols = [c for c in merge_cols if c in tradeable_day.columns]

    day = day.merge(
        tradeable_day[["symbol"] + [c for c in merge_cols if c != "symbol"]],
        on="symbol",
        how="left",
        suffixes=("", "_new")
    )

    # 处理未计算的行（不可交易）
    day["action"] = day["action"].fillna("WITHDRAW")
    day["target_weight"] = day["target_weight"].fillna(0.0)
    day["score"] = day["score"].fillna(0.0)

    # =====================
    # 输出
    # =====================
    out_cols = [
        "date", "symbol",
        "close",
        "score", "rank", "action",
        "target_weight",
        "ma_dist_20", "ret_20d", "ret_60d",
        "vol_20d", "atr_pct", "vol_ratio_20",
        "trend_up", "mom_bad", "risk_high",
    ]

    # 新增列（如果存在）
    extra_cols = ["tradeable", "eligible", "limit_up_flag", "pullback_pct"]
    for c in extra_cols:
        if c in day.columns:
            out_cols.append(c)

    out_cols = [c for c in out_cols if c in day.columns]
    day = day[out_cols].sort_values("rank", na_position="last").reset_index(drop=True)

    return day


# =============================================================================
# 保存函数（保持不变）
# =============================================================================

def save_daily_ranking(base_dir: Path, ranking: pd.DataFrame) -> tuple[Path, Path]:
    out_dir = base_dir / "data" / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = pd.to_datetime(ranking["date"].iloc[0]).date().isoformat()
    dated_path = out_dir / f"daily_rank_{d}.csv"
    latest_path = out_dir / "latest_daily_rank.csv"
    ranking.to_csv(dated_path, index=False, encoding="utf-8-sig")
    ranking.to_csv(latest_path, index=False, encoding="utf-8-sig")
    return dated_path, latest_path


def export_qlib_signal_csv(base_dir: Path, ranking: pd.DataFrame) -> Path:
    """导出 Qlib 格式信号"""
    from pandas.errors import EmptyDataError

    out_path = base_dir / "out" / "signal.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ranking is None or ranking.empty:
        raise ValueError("ranking is empty.")

    for c in ("date", "symbol", "score"):
        if c not in ranking.columns:
            raise ValueError(f"ranking missing required column: {c}")

    df = ranking[["date", "symbol", "score"]].copy()
    df.rename(columns={"symbol": "code"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = df["code"].astype(str).str.strip().str.upper()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["date", "code", "score"])

    old_df = None
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            old = pd.read_csv(out_path)
            need = {"date", "code", "score"}
            if need.issubset(old.columns):
                old_df = old[["date", "code", "score"]].copy()
        except Exception:
            old_df = None

    if old_df is not None and not old_df.empty:
        all_df = pd.concat([old_df, df], ignore_index=True)
    else:
        all_df = df

    all_df = all_df.drop_duplicates(subset=["date", "code"], keep="last")
    all_df = all_df.sort_values(["date", "code"])
    all_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    return out_path


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent.parent

    print("加载特征...")
    feats = load_features(base_dir)
    print(f"日期范围: {feats['date'].min()} -> {feats['date'].max()}")

    # 测试原版配置
    print("\n=== 原版配置 ===")
    cfg_v1 = SignalConfig()
    ranking_v1 = compute_daily_ranking(feats, cfg=cfg_v1)
    print(ranking_v1[["symbol", "action", "rank", "score"]].head(10))

    # 测试升级版配置
    print("\n=== 升级版配置（含入场过滤）===")
    cfg_v2 = SignalConfig(
        use_tradeability_filter=True,
        use_limit_up_entry=True,
        use_volatility_sizing=True,
    )
    ranking_v2 = compute_daily_ranking(feats, cfg=cfg_v2)
    print(ranking_v2[["symbol", "action", "rank", "score", "target_weight"]].head(10))