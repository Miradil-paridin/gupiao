"""
信号计算模块 (V3.1 统一版)

功能：
- 因子得分计算 (score)
- 横截面排名 (rank)
- 动作标签 (INVEST_MORE, HOLD, REDUCE, WITHDRAW, LEAST)
- 可交易性过滤 + 流动性过滤
- 严格/普通/宽松 入场模式
- 行业分散控制
- TDX 指标加分与保护
- 波动反比仓位分配
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import re

import numpy as np
import pandas as pd


@dataclass
class SignalConfig:
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
    pullback_min_pct: float = 0.10
    pullback_max_pct: float = 0.30
    use_dynamic_pullback: bool = False
    dynamic_pullback_min_atr_mult: float = 1.5
    dynamic_pullback_max_atr_mult: float = 4.0

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

    # 市场环境
    use_market_regime: bool = False

    # 行业分散
    use_industry_diversification: bool = True
    max_per_industry: int = 2

    # 流动性
    use_liquidity_filter: bool = True
    min_amount_20d: float = 6e7
    min_turnover_20d: float = 0.6

    # 风控
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


def _zscore(x: pd.Series) -> pd.Series:
    """Z-score 标准化"""
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


def build_industry_map(config_path: Path) -> Dict[str, str]:
    """从 config.yaml 构建行业映射"""
    mapping: Dict[str, str] = {}

    if not config_path.exists():
        return mapping

    content = config_path.read_text(encoding="utf-8", errors="ignore")
    current_industry = "其他"

    for line in content.split("\n"):
        if "==========" in line:
            match = re.search(r"=+\s*([^=]+)\s*=+", line)
            if match:
                current_industry = match.group(1).strip()
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

        code_match = re.search(r'"(\d{6})"', line)
        if code_match:
            code = code_match.group(1)
            mapping[code] = current_industry
            if code.startswith(("6", "5")):
                mapping[f"{code}.SH"] = current_industry
            else:
                mapping[f"{code}.SZ"] = current_industry

    return mapping


def filter_tradeability(day: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    """可交易性过滤"""
    day = day.copy()
    day["tradeable"] = True

    if "close" in day.columns:
        day.loc[day["close"] <= 0, "tradeable"] = False

    if "one_line_board" in day.columns:
        day.loc[day["one_line_board"] == 1, "tradeable"] = False
    elif "high" in day.columns and "low" in day.columns:
        day.loc[day["high"] == day["low"], "tradeable"] = False

    if "limit_up_flag" in day.columns:
        day.loc[day["limit_up_flag"] == 1, "tradeable"] = False

    if "near_limit_up" in day.columns:
        day.loc[day["near_limit_up"] == 1, "tradeable"] = False

    return day


def check_liquidity(row: pd.Series, cfg: SignalConfig) -> bool:
    """检查流动性"""
    if not cfg.use_liquidity_filter:
        return True

    amount_20d = _safe_float(row.get("amount_20d", row.get("amount", 0)))
    min_amount = _safe_float(cfg.min_amount_20d, 6e7)
    if amount_20d < min_amount:
        return False

    turnover_20d = _safe_float(row.get("turnover_20d", row.get("turnover", 0)))
    min_turnover = _safe_float(cfg.min_turnover_20d, 0.6)
    if turnover_20d < min_turnover:
        return False

    return True


def _get_pullback_bounds(row: pd.Series, cfg: SignalConfig) -> tuple[float, float]:
    """获取动态回调范围（可选根据 ATR 调整）。"""
    min_pct = cfg.pullback_min_pct
    max_pct = cfg.pullback_max_pct

    if not cfg.use_dynamic_pullback:
        return min_pct, max_pct

    atr_pct = _safe_float(row.get("atr_pct", 0))
    if atr_pct <= 0:
        return min_pct, max_pct

    dyn_min = atr_pct * cfg.dynamic_pullback_min_atr_mult
    dyn_max = atr_pct * cfg.dynamic_pullback_max_atr_mult
    min_pct = max(min_pct, dyn_min)
    max_pct = max(max_pct, dyn_max)
    if max_pct < min_pct:
        max_pct = min_pct
    return min_pct, max_pct


def check_eligible_strict(row: pd.Series, cfg: SignalConfig) -> bool:
    """严格入场条件检查"""
    trend_up = _safe_float(row.get("ma_dist_20", 0)) > 0
    if not trend_up:
        return False

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
            pb_min, pb_max = _get_pullback_bounds(row, cfg)
            pullback_ok = pb_min <= pullback <= pb_max
            breakout_ok = volume_breakout == 1 and price_above_ma5 == 1

            if in_window and pullback_ok and breakout_ok:
                limit_up_entry = True

    tdx_entry = False
    if cfg.use_tdx_indicators:
        tdx_score = _safe_float(row.get("tdx_score", 0))
        if tdx_score >= cfg.tdx_min_score:
            tdx_entry = True

        high30 = row.get("high30_breakout", 0)
        main_force = row.get("main_force_strong", 0)
        if high30 == 1 and main_force == 1:
            tdx_entry = True

    return (limit_up_entry or tdx_entry) and trend_up


def check_eligible_normal(row: pd.Series, cfg: SignalConfig) -> bool:
    """普通入场条件检查"""
    if cfg.use_limit_up_entry:
        days_since = row.get("days_since_limit_up", np.nan)
        pullback = row.get("pullback_pct", np.nan)
        volume_breakout = row.get("volume_breakout", 0)
        price_above_ma5 = row.get("price_above_ma5", 0)

        if not pd.isna(days_since) and not pd.isna(pullback):
            days_since = _safe_float(days_since)
            pullback = _safe_float(pullback)
            in_window = cfg.pullback_window_start <= days_since <= cfg.pullback_window_end
            pb_min, pb_max = _get_pullback_bounds(row, cfg)
            pullback_ok = pb_min <= pullback <= pb_max
            breakout_ok = volume_breakout == 1 and price_above_ma5 == 1

            if in_window and pullback_ok and breakout_ok:
                return True

    if cfg.use_tdx_indicators:
        tdx_score = _safe_float(row.get("tdx_score", 0))
        if tdx_score >= cfg.tdx_min_score:
            return True

        high30 = row.get("high30_breakout", 0)
        main_force = row.get("main_force_strong", 0)
        if high30 == 1 and main_force == 1:
            return True

    ma_dist = _safe_float(row.get("ma_dist_20", 0))
    ret_20d = _safe_float(row.get("ret_20d", 0))

    return ma_dist > 0.02 and ret_20d > 0


def check_eligible(row: pd.Series, cfg: SignalConfig) -> bool:
    """根据模式检查入场条件"""
    if cfg.entry_mode == "strict":
        return check_eligible_strict(row, cfg)
    if cfg.entry_mode == "loose":
        return _safe_float(row.get("ma_dist_20", 0)) > 0
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

    result = []
    industry_counts: Dict[str, int] = {}

    for _, row in candidates.iterrows():
        ind = row["industry"]
        count = industry_counts.get(ind, 0)

        if count < max_per_industry:
            result.append(row)
            industry_counts[ind] = count + 1

    return pd.DataFrame(result)


def compute_weights(selected: pd.DataFrame, cfg: SignalConfig) -> Dict[str, float]:
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


def compute_daily_ranking(
    feats: pd.DataFrame,
    as_of: str | None = None,
    cfg: SignalConfig | None = None,
    industry_map: Dict[str, str] | None = None,
    market_can_open: bool = True,
) -> pd.DataFrame:
    """计算单日排名信号（V3.1 统一版）"""
    if cfg is None:
        cfg = SignalConfig()

    if industry_map is None:
        industry_map = {}

    if as_of is None:
        d = feats["date"].max()
    else:
        d = pd.to_datetime(as_of).date()

    day = feats[feats["date"] == d].copy()

    if day.empty:
        raise ValueError(f"No rows for date: {d}")

    numeric_cols = [
        "ma_dist_20", "ret_20d", "ret_60d", "vol_20d", "vol_ratio_20",
        "atr_14", "close", "amount_20d", "amount", "turnover_20d", "turnover",
        "tdx_score", "days_since_limit_up", "pullback_pct",
    ]
    for col in numeric_cols:
        if col in day.columns:
            day[col] = pd.to_numeric(day[col], errors="coerce")

    if "atr_14" in day.columns and "close" in day.columns:
        day["atr_pct"] = day["atr_14"] / day["close"]
    else:
        day["atr_pct"] = 0.02

    z_ma = _zscore(day["ma_dist_20"]) if "ma_dist_20" in day.columns else 0
    z_r20 = _zscore(day["ret_20d"]) if "ret_20d" in day.columns else 0
    z_r60 = _zscore(day["ret_60d"]) if "ret_60d" in day.columns else 0
    z_vol = _zscore(day["vol_20d"]) if "vol_20d" in day.columns else 0
    z_atr = _zscore(day["atr_pct"])
    z_vr = _zscore(day["vol_ratio_20"]) if "vol_ratio_20" in day.columns else 0

    day["score"] = (
        2.0 * z_ma +
        1.0 * z_r20 +
        0.5 * z_r60 -
        1.0 * z_vol -
        0.5 * z_atr +
        0.3 * z_vr
    )

    if cfg.use_tdx_indicators:
        if "high30_breakout" in day.columns:
            day["score"] += day["high30_breakout"].fillna(0) * cfg.tdx_high30_weight
        if "main_force_strong" in day.columns:
            day["score"] += day["main_force_strong"].fillna(0) * cfg.tdx_main_force_weight
        if "has_limit_up_30d" in day.columns:
            day["score"] += day["has_limit_up_30d"].fillna(0) * cfg.tdx_limit_up_30d_weight
        if "main_force_control" in day.columns:
            day["score"] += (day["main_force_control"].fillna(0) > 0).astype(int) * 0.3

    day["trend_up"] = (day.get("ma_dist_20", pd.Series(0, index=day.index)) > 0).astype(int)
    day["mom_bad"] = (
        (day.get("ret_20d", pd.Series(0, index=day.index)) < 0) &
        (day.get("ret_60d", pd.Series(0, index=day.index)) < 0)
    ).astype(int)
    day["risk_high"] = (day.get("vol_20d", pd.Series(0, index=day.index)) >= cfg.risk_vol_20d_threshold).astype(int)

    if cfg.use_tradeability_filter:
        day = filter_tradeability(day, cfg)
    else:
        day["tradeable"] = True

    day["liquidity_ok"] = day.apply(lambda r: check_liquidity(r, cfg), axis=1)
    day["eligible"] = day.apply(lambda r: check_eligible(r, cfg), axis=1)

    day = day.sort_values("score", ascending=False).reset_index(drop=True)
    day["rank"] = np.arange(1, len(day) + 1)

    day["action"] = "HOLD"

    if cfg.use_tdx_protection:
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

    reduce_mask = (
        (day["action"] != "WITHDRAW") &
        (day["risk_high"] == 1) &
        (day["trend_up"] == 1) &
        (day["mom_bad"] == 0)
    )
    day.loc[reduce_mask, "action"] = "REDUCE"

    invest_candidates = day[
        (day["action"] == "HOLD") &
        (day["tradeable"] == True) &
        (day["liquidity_ok"] == True) &
        (day["eligible"] == True) &
        (day["trend_up"] == 1)
    ].copy()

    if cfg.use_industry_diversification and industry_map:
        invest_candidates = apply_industry_diversification(
            invest_candidates.sort_values("score", ascending=False),
            industry_map,
            cfg.max_per_industry,
        )

    if cfg.use_market_regime and not market_can_open:
        invest_syms = []
    else:
        invest_syms = invest_candidates.sort_values("score", ascending=False).head(cfg.invest_more_n)["symbol"].tolist()
    day.loc[day["symbol"].isin(invest_syms), "action"] = "INVEST_MORE"

    least_candidates = day[day["action"] == "HOLD"].sort_values("score", ascending=True)
    least_syms = least_candidates.head(cfg.least_n)["symbol"].tolist()
    day.loc[day["symbol"].isin(least_syms), "action"] = "LEAST"

    selected = day[day["action"] == "INVEST_MORE"]
    weights = compute_weights(selected, cfg)
    day["target_weight"] = day["symbol"].map(weights).fillna(0.0)

    day["industry"] = day["symbol"].astype(str).map(
        lambda s: industry_map.get(s, industry_map.get(s.split(".")[0], "其他"))
    )

    out_cols = [
        "date", "symbol", "industry", "close",
        "score", "rank", "action", "target_weight",
        "ma_dist_20", "ret_20d", "ret_60d",
        "vol_20d", "atr_pct", "vol_ratio_20",
        "trend_up", "mom_bad", "risk_high",
        "tradeable", "liquidity_ok", "eligible",
    ]

    tdx_cols = [
        "high30_breakout", "main_force_strong", "main_force_control",
        "has_limit_up_30d", "tdx_score",
    ]
    for c in tdx_cols:
        if c in day.columns:
            out_cols.append(c)

    out_cols = [c for c in out_cols if c in day.columns]
    day = day[out_cols].sort_values("rank").reset_index(drop=True)

    return day


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


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent

    print("加载特征...")
    feats = load_features(base_dir)
    print(f"日期范围: {feats['date'].min()} -> {feats['date'].max()}")

    print("\n=== V3.1 统一版配置 ===")
    cfg = SignalConfig()

    industry_map = build_industry_map(base_dir / "config.yaml")
    if not industry_map:
        industry_map = build_industry_map(base_dir / "config_v31.yaml")

    ranking = compute_daily_ranking(feats, cfg=cfg, industry_map=industry_map)
    print(ranking[["symbol", "action", "rank", "score", "target_weight"]].head(10))
