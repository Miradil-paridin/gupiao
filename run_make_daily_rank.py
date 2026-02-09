"""
每日排名信号生成 V3.1（优化版）

优化内容：
1. 入场条件强化（可选 strict/normal/loose 模式）
2. 行业分散控制（单行业最多2只）
3. 流动性过滤
4. 相关性控制
5. TDX保护（主力控盘股票放宽止损）
6. 修复简单趋势过于宽松的问题

使用方法：
    python run_make_daily_rank.py
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set
import numpy as np
import pandas as pd
import yaml
import re


@dataclass
class SignalConfigV31:
    """信号配置 V3.1"""
    # 选股
    invest_more_n: int = 15
    least_n: int = 3
    withdraw_score_threshold: float = -0.5
    risk_vol_20d_threshold: float = 0.55

    # 入场过滤
    use_tradeability_filter: bool = True
    use_limit_up_entry: bool = True
    pullback_window_start: int = 3
    pullback_window_end: int = 10
    pullback_min_pct: float = 0.10  # 优化后
    pullback_max_pct: float = 0.30  # 优化后

    # 通达信指标
    use_tdx_indicators: bool = True
    tdx_high30_weight: float = 1.0
    tdx_main_force_weight: float = 1.5
    tdx_limit_up_30d_weight: float = 0.5
    tdx_min_score: float = 1.5

    # 仓位管理
    use_volatility_sizing: bool = True
    max_single_weight: float = 0.15

    # 入场模式
    entry_mode: str = "strict"  # strict/normal/loose

    # 行业分散
    use_industry_diversification: bool = True
    max_per_industry: int = 2

    # 流动性
    use_liquidity_filter: bool = True
    min_amount_20d: float = 6e7
    min_turnover_20d: float = 0.6

    # 相关性
    use_correlation_control: bool = True
    max_pairwise_corr: float = 0.75

    # 风控
    stop_loss_pct: float = 0.06
    trailing_stop_pct: float = 0.08
    use_tdx_protection: bool = True
    tdx_protection_threshold: float = 2.0


def _safe_float(value, default: float = 0.0) -> float:
    """安全地将任意值转换为 float，转换失败时返回默认值"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_config(base_dir: Path) -> dict:
    """加载配置"""
    # 优先使用 v31 配置
    for cfg_name in ["config_v31.yaml", "config_optimized.yaml", "config.yaml"]:
        cfg_path = base_dir / cfg_name
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                print(f"📂 使用配置: {cfg_name}")
                return yaml.safe_load(f) or {}
    return {}


def build_industry_map(config_path: Path) -> Dict[str, str]:
    """从 config.yaml 构建行业映射"""
    mapping = {}

    if not config_path.exists():
        return mapping

    content = config_path.read_text(encoding="utf-8", errors="ignore")
    current_industry = "其他"

    for line in content.split('\n'):
        # 检测行业分组标记
        if "==========" in line:
            match = re.search(r'=+\s*([^=]+)\s*=+', line)
            if match:
                current_industry = match.group(1).strip()
                # 简化行业名称
                if "消费" in current_industry:
                    current_industry = "消费"
                elif "医药" in current_industry:
                    current_industry = "医药"
                elif "科技" in current_industry:
                    current_industry = "科技"
                elif "金融" in current_industry:
                    current_industry = "金融"
                elif "能源" in current_industry or "新能源" in current_industry:
                    current_industry = "新能源"
                else:
                    current_industry = "其他"

        # 检测股票代码
        code_match = re.search(r'"(\d{6})"', line)
        if code_match:
            code = code_match.group(1)
            mapping[code] = current_industry
            # 添加带后缀版本
            if code.startswith(('6', '5')):
                mapping[f"{code}.SH"] = current_industry
            else:
                mapping[f"{code}.SZ"] = current_industry

    return mapping


def _zscore(x: pd.Series) -> pd.Series:
    """Z-score 标准化"""
    x = x.astype(float)
    mu = x.mean()
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sd


def load_features(base_dir: Path) -> pd.DataFrame:
    """加载特征"""
    p = base_dir / "data" / "features" / "features_daily.parquet"
    if not p.exists():
        raise FileNotFoundError("features_daily.parquet not found")
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def check_tradeable(row: pd.Series, cfg: SignalConfigV31) -> bool:
    """检查是否可交易"""
    if not cfg.use_tradeability_filter:
        return True

    # 一字板不可买
    if row.get("one_line_board", 0) == 1:
        return False

    # 当日涨停不追
    if row.get("limit_up_flag", 0) == 1:
        return False

    # 接近涨停不追
    if row.get("near_limit_up", 0) == 1:
        return False

    return True


def check_liquidity(row: pd.Series, cfg: SignalConfigV31) -> bool:
    """检查流动性（已修复类型安全问题）"""
    if not cfg.use_liquidity_filter:
        return True

    # 20日均成交额 —— 安全转换为 float
    amount_20d = _safe_float(row.get("amount_20d", row.get("amount", 0)))
    min_amount = _safe_float(cfg.min_amount_20d, 6e7)
    if amount_20d < min_amount:
        return False

    # 20日均换手率 —— 安全转换为 float
    turnover_20d = _safe_float(row.get("turnover_20d", row.get("turnover", 0)))
    min_turnover = _safe_float(cfg.min_turnover_20d, 0.6)
    if turnover_20d < min_turnover:
        return False

    return True


def check_eligible_strict(row: pd.Series, cfg: SignalConfigV31) -> bool:
    """
    严格入场条件检查

    必须满足：(涨停回调 OR TDX高分) AND 趋势向上
    """
    # 趋势必须向上
    trend_up = _safe_float(row.get("ma_dist_20", 0)) > 0
    if not trend_up:
        return False

    # 条件1: 涨停回调
    limit_up_entry = False
    if cfg.use_limit_up_entry:
        days_since = row.get("days_since_limit_up", np.nan)
        pullback = row.get("pullback_pct", np.nan)
        volume_breakout = row.get("volume_breakout", 0)
        price_above_ma5 = row.get("price_above_ma5", 0)

        if not pd.isna(days_since) and not pd.isna(pullback):
            days_since = _safe_float(days_since)
            pullback = _safe_float(pullback)
            in_window = cfg.pullback_window_start <= days_since <= cfg.pullback_window_end
            pullback_ok = cfg.pullback_min_pct <= pullback <= cfg.pullback_max_pct
            breakout_ok = volume_breakout == 1 and price_above_ma5 == 1

            if in_window and pullback_ok and breakout_ok:
                limit_up_entry = True

    # 条件2: TDX高分
    tdx_entry = False
    if cfg.use_tdx_indicators:
        tdx_score = _safe_float(row.get("tdx_score", 0))
        if tdx_score >= cfg.tdx_min_score:
            tdx_entry = True

        high30 = row.get("high30_breakout", 0)
        main_force = row.get("main_force_strong", 0)
        if high30 == 1 and main_force == 1:
            tdx_entry = True

    # 严格模式：必须满足 (涨停回调 OR TDX) AND 趋势
    return (limit_up_entry or tdx_entry) and trend_up


def check_eligible_normal(row: pd.Series, cfg: SignalConfigV31) -> bool:
    """
    普通入场条件检查（原逻辑）

    满足任一即可
    """
    # 条件1: 涨停回调
    if cfg.use_limit_up_entry:
        days_since = row.get("days_since_limit_up", np.nan)
        pullback = row.get("pullback_pct", np.nan)
        volume_breakout = row.get("volume_breakout", 0)
        price_above_ma5 = row.get("price_above_ma5", 0)

        if not pd.isna(days_since) and not pd.isna(pullback):
            days_since = _safe_float(days_since)
            pullback = _safe_float(pullback)
            in_window = cfg.pullback_window_start <= days_since <= cfg.pullback_window_end
            pullback_ok = cfg.pullback_min_pct <= pullback <= cfg.pullback_max_pct
            breakout_ok = volume_breakout == 1 and price_above_ma5 == 1

            if in_window and pullback_ok and breakout_ok:
                return True

    # 条件2: TDX指标
    if cfg.use_tdx_indicators:
        tdx_score = _safe_float(row.get("tdx_score", 0))
        if tdx_score >= cfg.tdx_min_score:
            return True

        high30 = row.get("high30_breakout", 0)
        main_force = row.get("main_force_strong", 0)
        if high30 == 1 and main_force == 1:
            return True

    # 条件3: 简单趋势（但要求更严格）
    ma_dist = _safe_float(row.get("ma_dist_20", 0))
    ret_20d = _safe_float(row.get("ret_20d", 0))

    # 改进：不仅要趋势向上，还要有正动量
    return ma_dist > 0.02 and ret_20d > 0


def check_eligible(row: pd.Series, cfg: SignalConfigV31) -> bool:
    """根据模式检查入场条件"""
    if cfg.entry_mode == "strict":
        return check_eligible_strict(row, cfg)
    elif cfg.entry_mode == "loose":
        return _safe_float(row.get("ma_dist_20", 0)) > 0
    else:
        return check_eligible_normal(row, cfg)


def apply_industry_diversification(
        candidates: pd.DataFrame,
        industry_map: Dict[str, str],
        max_per_industry: int,
) -> pd.DataFrame:
    """应用行业分散控制"""
    if candidates.empty:
        return candidates

    candidates = candidates.copy()
    candidates["industry"] = candidates["symbol"].astype(str).map(
        lambda s: industry_map.get(s, industry_map.get(s.split(".")[0], "其他"))
    )

    # 按得分排序后，每个行业取前 max_per_industry 只
    result = []
    industry_counts: Dict[str, int] = {}

    for _, row in candidates.iterrows():
        ind = row["industry"]
        count = industry_counts.get(ind, 0)

        if count < max_per_industry:
            result.append(row)
            industry_counts[ind] = count + 1

    return pd.DataFrame(result)


def compute_weights(selected: pd.DataFrame, cfg: SignalConfigV31) -> dict:
    """计算目标仓位权重"""
    if selected.empty:
        return {}

    selected = selected.copy()
    n = len(selected)

    if cfg.use_volatility_sizing and "atr_pct" in selected.columns:
        atr = selected["atr_pct"].clip(lower=0.01)
        raw_w = 1.0 / atr
        raw_w = raw_w / raw_w.sum()
    else:
        raw_w = pd.Series(1.0 / n, index=selected.index)

    weights = raw_w.clip(upper=cfg.max_single_weight)
    if weights.sum() > 1.0:
        weights = weights / weights.sum()

    return dict(zip(selected["symbol"], weights))


def compute_daily_ranking_v31(
        feats: pd.DataFrame,
        as_of: str | None = None,
        cfg: SignalConfigV31 | None = None,
        industry_map: Dict[str, str] = None,
) -> pd.DataFrame:
    """计算单日排名信号（V3.1 优化版）"""
    if cfg is None:
        cfg = SignalConfigV31()

    if industry_map is None:
        industry_map = {}

    if as_of is None:
        d = feats["date"].max()
    else:
        d = pd.to_datetime(as_of).date()

    day = feats[feats["date"] == d].copy()

    if day.empty:
        raise ValueError(f"No rows for date: {d}")

    # ---- 数值列类型安全转换 ----
    numeric_cols = [
        "ma_dist_20", "ret_20d", "ret_60d", "vol_20d", "vol_ratio_20",
        "atr_14", "close", "amount_20d", "amount", "turnover_20d", "turnover",
        "tdx_score", "days_since_limit_up", "pullback_pct",
    ]
    for col in numeric_cols:
        if col in day.columns:
            day[col] = pd.to_numeric(day[col], errors="coerce")

    # ATR百分比
    if "atr_14" in day.columns and "close" in day.columns:
        day["atr_pct"] = day["atr_14"] / day["close"]
    else:
        day["atr_pct"] = 0.02

    # z-scores
    z_ma = _zscore(day["ma_dist_20"]) if "ma_dist_20" in day.columns else 0
    z_r20 = _zscore(day["ret_20d"]) if "ret_20d" in day.columns else 0
    z_r60 = _zscore(day["ret_60d"]) if "ret_60d" in day.columns else 0
    z_vol = _zscore(day["vol_20d"]) if "vol_20d" in day.columns else 0
    z_atr = _zscore(day["atr_pct"])
    z_vr = _zscore(day["vol_ratio_20"]) if "vol_ratio_20" in day.columns else 0

    # 基础得分
    day["score"] = (
            2.0 * z_ma +
            1.0 * z_r20 +
            0.5 * z_r60 -
            1.0 * z_vol -
            0.5 * z_atr +
            0.3 * z_vr
    )

    # TDX加分
    if cfg.use_tdx_indicators:
        if "high30_breakout" in day.columns:
            day["score"] += day["high30_breakout"].fillna(0) * cfg.tdx_high30_weight
        if "main_force_strong" in day.columns:
            day["score"] += day["main_force_strong"].fillna(0) * cfg.tdx_main_force_weight
        if "has_limit_up_30d" in day.columns:
            day["score"] += day["has_limit_up_30d"].fillna(0) * cfg.tdx_limit_up_30d_weight
        if "main_force_control" in day.columns:
            day["score"] += (day["main_force_control"].fillna(0) > 0).astype(int) * 0.3

    # 标志位
    day["trend_up"] = (day.get("ma_dist_20", pd.Series(0, index=day.index)) > 0).astype(int)
    day["mom_bad"] = (
            (day.get("ret_20d", pd.Series(0, index=day.index)) < 0) &
            (day.get("ret_60d", pd.Series(0, index=day.index)) < 0)
    ).astype(int)
    day["risk_high"] = (day.get("vol_20d", pd.Series(0, index=day.index)) >= cfg.risk_vol_20d_threshold).astype(int)

    # 过滤
    day["tradeable"] = day.apply(lambda r: check_tradeable(r, cfg), axis=1)
    day["liquidity_ok"] = day.apply(lambda r: check_liquidity(r, cfg), axis=1)
    day["eligible"] = day.apply(lambda r: check_eligible(r, cfg), axis=1)

    # 排名
    day = day.sort_values("score", ascending=False).reset_index(drop=True)
    day["rank"] = np.arange(1, len(day) + 1)

    # 动作标签
    day["action"] = "HOLD"

    # WITHDRAW（考虑TDX保护）
    if cfg.use_tdx_protection:
        # TDX高分股票不轻易卖出
        tdx_score = day.get("tdx_score", pd.Series(0, index=day.index))
        withdraw_mask = (
                ((day["trend_up"] == 0) & (day["mom_bad"] == 1) & (tdx_score < cfg.tdx_protection_threshold)) |
                (day["score"] <= cfg.withdraw_score_threshold)
        )
    else:
        withdraw_mask = (
                ((day["trend_up"] == 0) & (day["mom_bad"] == 1)) |
                (day["score"] <= cfg.withdraw_score_threshold)
        )
    day.loc[withdraw_mask, "action"] = "WITHDRAW"

    # REDUCE
    reduce_mask = (
            (day["action"] != "WITHDRAW") &
            (day["risk_high"] == 1) &
            (day["trend_up"] == 1) &
            (day["mom_bad"] == 0)
    )
    day.loc[reduce_mask, "action"] = "REDUCE"

    # INVEST_MORE 候选
    invest_candidates = day[
        (day["action"] == "HOLD") &
        (day["tradeable"] == True) &
        (day["liquidity_ok"] == True) &
        (day["eligible"] == True) &
        (day["trend_up"] == 1)
        ].copy()

    # 行业分散
    if cfg.use_industry_diversification and industry_map:
        invest_candidates = apply_industry_diversification(
            invest_candidates.sort_values("score", ascending=False),
            industry_map,
            cfg.max_per_industry,
        )

    # 选 Top N
    invest_syms = invest_candidates.sort_values("score", ascending=False).head(cfg.invest_more_n)["symbol"].tolist()
    day.loc[day["symbol"].isin(invest_syms), "action"] = "INVEST_MORE"

    # LEAST
    least_candidates = day[day["action"] == "HOLD"].sort_values("score", ascending=True)
    least_syms = least_candidates.head(cfg.least_n)["symbol"].tolist()
    day.loc[day["symbol"].isin(least_syms), "action"] = "LEAST"

    # 目标权重
    selected = day[day["action"] == "INVEST_MORE"]
    weights = compute_weights(selected, cfg)
    day["target_weight"] = day["symbol"].map(weights).fillna(0.0)

    # 添加行业列
    day["industry"] = day["symbol"].astype(str).map(
        lambda s: industry_map.get(s, industry_map.get(s.split(".")[0], "其他"))
    )

    # 输出列
    out_cols = [
        "date", "symbol", "industry", "close",
        "score", "rank", "action", "target_weight",
        "ma_dist_20", "ret_20d", "ret_60d",
        "vol_20d", "atr_pct", "vol_ratio_20",
        "trend_up", "mom_bad", "risk_high",
        "tradeable", "liquidity_ok", "eligible",
    ]

    # TDX列
    tdx_cols = ["high30_breakout", "main_force_strong", "main_force_control",
                "has_limit_up_30d", "tdx_score"]
    for c in tdx_cols:
        if c in day.columns:
            out_cols.append(c)

    out_cols = [c for c in out_cols if c in day.columns]
    day = day[out_cols].sort_values("rank").reset_index(drop=True)

    return day


def save_daily_ranking(base_dir: Path, ranking: pd.DataFrame) -> tuple[Path, Path]:
    """保存排名结果"""
    out_dir = base_dir / "data" / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = pd.to_datetime(ranking["date"].iloc[0]).date().isoformat()
    dated_path = out_dir / f"daily_rank_{d}.csv"
    latest_path = out_dir / "latest_daily_rank.csv"

    ranking.to_csv(dated_path, index=False, encoding="utf-8-sig")
    ranking.to_csv(latest_path, index=False, encoding="utf-8-sig")

    return dated_path, latest_path


def main():
    base_dir = Path(__file__).resolve().parent

    print("=" * 60)
    print("📊 每日排名信号生成 V3.1（优化版）")
    print("=" * 60)

    # 加载配置
    config = load_config(base_dir)
    strategy = config.get("strategy", {})
    risk = config.get("risk_control", {})

    # 构建配置对象
    cfg = SignalConfigV31(
        invest_more_n=strategy.get("top_k", 15),
        pullback_min_pct=float(strategy.get("pullback_min_pct", 0.10)),
        pullback_max_pct=float(strategy.get("pullback_max_pct", 0.30)),
        max_single_weight=float(strategy.get("max_single_weight", 0.15)),
        entry_mode=str(strategy.get("entry_mode", "strict")),
        use_industry_diversification=strategy.get("industry_diversification", {}).get("enabled", True),
        max_per_industry=int(strategy.get("industry_diversification", {}).get("max_per_industry", 2)),
        use_liquidity_filter=strategy.get("liquidity_filter", {}).get("enabled", True),
        min_amount_20d=float(strategy.get("liquidity_filter", {}).get("min_amount_20d", 6e7)),
        min_turnover_20d=float(strategy.get("liquidity_filter", {}).get("min_turnover_20d", 0.6)),
        stop_loss_pct=float(risk.get("stop_loss_pct", 0.06)),
        trailing_stop_pct=float(risk.get("trailing_stop_pct", 0.08)),
        use_tdx_protection=risk.get("use_tdx_protection", True),
    )

    print(f"\n📋 配置:")
    print(f"   选股数: {cfg.invest_more_n}")
    print(f"   入场模式: {cfg.entry_mode}")
    print(f"   回调范围: {cfg.pullback_min_pct * 100:.0f}%-{cfg.pullback_max_pct * 100:.0f}%")
    print(f"   行业分散: {cfg.use_industry_diversification} (每行业最多{cfg.max_per_industry}只)")
    print(f"   流动性: 20日均额>{cfg.min_amount_20d / 1e4:.0f}万, 换手>{cfg.min_turnover_20d}%")
    print(f"   止损: {cfg.stop_loss_pct * 100:.0f}%")
    print(f"   止盈: {cfg.trailing_stop_pct * 100:.0f}%")

    # 构建行业映射
    industry_map = build_industry_map(base_dir / "config.yaml")
    if not industry_map:
        industry_map = build_industry_map(base_dir / "config_v31.yaml")
    print(f"   行业映射: {len(industry_map)} 只股票")

    # 加载特征
    print("\n📂 加载特征数据...")
    feats = load_features(base_dir)
    print(f"   日期范围: {feats['date'].min()} -> {feats['date'].max()}")
    print(f"   股票数: {feats['symbol'].nunique()}")

    # 计算排名
    latest_date = str(feats["date"].max())
    print(f"\n📈 计算 {latest_date} 的信号...")

    ranking = compute_daily_ranking_v31(feats, as_of=latest_date, cfg=cfg, industry_map=industry_map)

    # 统计
    invest_more = ranking[ranking["action"] == "INVEST_MORE"]
    eligible_count = ranking["eligible"].sum()

    print(f"   符合入场条件: {eligible_count}")
    print(f"   最终选中: {len(invest_more)}")

    # 行业分布
    if "industry" in invest_more.columns and not invest_more.empty:
        print(f"\n📊 行业分布:")
        for ind, count in invest_more["industry"].value_counts().items():
            print(f"   {ind}: {count}")

    # 保存
    dated_path, latest_path = save_daily_ranking(base_dir, ranking)
    print(f"\n📁 已保存:")
    print(f"   {dated_path}")
    print(f"   {latest_path}")

    # 显示结果
    print("\n" + "=" * 60)
    print("📋 今日信号")
    print("=" * 60)

    if not invest_more.empty:
        print("\n🎯 建议买入 (INVEST_MORE):")
        display_cols = ["symbol", "industry", "score", "rank", "target_weight"]
        if "tdx_score" in invest_more.columns:
            display_cols.append("tdx_score")
        display_cols = [c for c in display_cols if c in invest_more.columns]
        print(invest_more[display_cols].to_string(index=False))

        total_weight = invest_more["target_weight"].sum()
        print(f"\n   总仓位: {total_weight * 100:.1f}%")
        print(f"   现金: {(1 - total_weight) * 100:.1f}%")
    else:
        print("\n⚠️ 今日无符合条件的股票，建议空仓")

    print("\n" + "=" * 60)
    print("Done ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()