from __future__ import annotations

import argparse
import itertools
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class BacktestConfigV3:
    # core portfolio parameters
    top_k: int = 10
    invest_more_n: int = 10

    # entry conditions
    pullback_window_start: int = 3
    pullback_window_end: int = 10
    pullback_min_pct: float = 0.05
    pullback_max_pct: float = 0.25
    tdx_min_score: float = 1.5

    # risk controls
    stop_loss_pct: float = 0.08
    trailing_stop_pct: float = 0.10
    max_hold_days: int = 15

    # position sizing / costs
    initial_capital: float = 100000.0
    max_total_position: float = 0.80
    max_single_weight: float = 0.15
    cost_bps: float = 15.0

    # switches
    use_market_regime: bool = True
    use_tradeability_filter: bool = True

    # strategy enhancements
    use_industry_diversification: bool = True
    max_per_industry: int = 2

    use_market_cap_filter: bool = True
    min_float_mkt_cap: float = 8e9
    max_float_mkt_cap: float = 8e11

    use_liquidity_filter: bool = True
    min_amount_20d: float = 6e7
    min_turnover_20d: float = 0.6

    use_correlation_control: bool = True
    corr_lookback_days: int = 60
    max_pairwise_corr: float = 0.75

    # entry mode: "custom" / "strict" / "normal" / "loose"
    entry_mode: str = "normal"

    # TDX protection: relax stop-loss for strong main-force stocks
    use_tdx_protection: bool = True
    tdx_protection_threshold: float = 2.0

    # ── 方案1: 指数风控开关 ──
    use_index_filter: bool = True
    index_ma_period: int = 60          # 指数均线周期
    index_ma_short: int = 20           # 短期均线（判断拐头）
    index_crash_days: int = 3          # 近N天跌幅检测
    index_crash_threshold: float = -0.03  # 跌幅超此值暂停开仓
    index_pause_days: int = 5          # 暂停开仓天数

    # ── 方案2A: 普通路径 2-of-3 ──
    normal_min_conditions: int = 2     # 附加条件至少满足几个

    # ── 方案2B: 信号强度分级仓位 ──
    use_signal_tiered_sizing: bool = True
    tier_strong_multiplier: float = 1.2   # 强信号仓位倍数
    tier_normal_multiplier: float = 1.0   # 普通信号
    tier_weak_multiplier: float = 0.5     # 弱信号

    # ── 方案3: 退出规则升级 ──
    use_atr_stop: bool = True
    atr_stop_multiplier: float = 1.5   # 止损 = 入场价 - N * ATR

    use_failure_stop: bool = True
    failure_stop_days: int = 2         # 入场后N天内跌回突破位
    failure_stop_gain: float = 0.03    # N天内涨幅不足此值视为失败

    # 优化时间止损
    time_stop_min_gain: float = 0.0    # 持仓到期时，至少要有此涨幅才不被砍

    # ── 方案4: 盈利加仓 ──
    use_profit_pyramiding: bool = True
    pyramid_trigger_pct: float = 0.05  # 浮盈超过5%触发加仓
    pyramid_add_ratio: float = 0.5     # 加仓量 = 原仓位 * 0.5
    pyramid_max_adds: int = 1          # 最多加仓次数


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    top_k: int
    invest_more_n: int
    pullback_min_pct: float
    pullback_max_pct: float
    stop_loss_pct: float
    trailing_stop_pct: float
    annual_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    win_rate_pct: float


def _norm_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    if s.endswith(".SH") or s.endswith(".SZ"):
        return s
    if s.startswith(("688", "60", "601", "603", "605")):
        return f"{s}.SH"
    return f"{s}.SZ"


def _safe_zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sd


def _annual_return(nav: pd.Series, periods_per_year: int = 252) -> float:
    n = len(nav)
    if n < 2:
        return 0.0
    start = float(nav.iloc[0])
    end = float(nav.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    return (end / start) ** (periods_per_year / n) - 1.0


def _max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def _sharpe(r: pd.Series, periods_per_year: int = 252) -> float:
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd < 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def _win_rate(r: pd.Series) -> float:
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def load_and_prepare_features(base_dir: Path, start_date: str) -> pd.DataFrame:
    feats_path = base_dir / "data" / "features" / "features_daily.parquet"
    clean_path = base_dir / "data" / "clean" / "market_daily_all.parquet"

    if not feats_path.exists():
        raise FileNotFoundError(f"missing features file: {feats_path}")
    if not clean_path.exists():
        raise FileNotFoundError(f"missing market file: {clean_path}")

    feats = pd.read_parquet(feats_path)
    clean = pd.read_parquet(clean_path)

    feats["date"] = pd.to_datetime(feats["date"])
    clean["date"] = pd.to_datetime(clean["date"])

    feats["symbol"] = feats["symbol"].astype(str).str.upper()
    clean["symbol"] = clean["symbol"].astype(str).str.upper()

    feats = feats[feats["date"] >= pd.to_datetime(start_date)].copy()
    clean = clean[clean["date"] >= pd.to_datetime(start_date)].copy()

    clean_cols = [c for c in ["date", "symbol", "amount", "turnover", "volume"] if c in clean.columns]
    clean = clean[clean_cols].copy()

    merged = feats.merge(clean, on=["date", "symbol"], how="left")
    for col in ["amount", "turnover", "volume"]:
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)

    merged["amount_20d"] = merged.groupby("symbol")["amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    merged["turnover_20d"] = merged.groupby("symbol")["turnover"].transform(lambda s: s.rolling(20, min_periods=5).mean())

    turnover_daily = merged["turnover"].replace(0, np.nan)
    merged["float_mkt_cap"] = merged["amount"] / (turnover_daily / 100.0)
    merged["float_mkt_cap_20d"] = merged.groupby("symbol")["float_mkt_cap"].transform(
        lambda s: s.rolling(20, min_periods=5).median()
    )

    if "atr_14" not in merged.columns:
        merged["atr_14"] = np.nan
    merged["atr_pct"] = merged["atr_14"] / merged["close"].replace(0, np.nan)
    merged["atr_pct"] = merged["atr_pct"].fillna(0.01).clip(lower=0.003, upper=0.25)

    # ── 月线级别指标（用户自定义通达信公式）──
    # 主力控盘公式: GU1=(C*2+H+L)/4; 起爆=EMA(EMA(C,9),9); 主力控盘=(GU1-REF(起爆,1))/REF(起爆,1)
    merged["_gu1"] = (merged["close"] * 2 + merged["high"] + merged["low"]) / 4.0
    merged["_ema9"] = merged.groupby("symbol")["close"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    merged["_ema9_9"] = merged.groupby("symbol")["_ema9"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    merged["_qibao_prev"] = merged.groupby("symbol")["_ema9_9"].shift(1)
    merged["main_force_pct"] = (merged["_gu1"] - merged["_qibao_prev"]) / merged["_qibao_prev"].replace(0, np.nan)
    merged["main_force_pct"] = merged["main_force_pct"].fillna(0)

    # 高30公式: X1=(C+L+H)/1.5; X2=EMA(X1,3); 高30=HHV(X2,30); 条件:高30>前高30
    merged["_x1"] = (merged["close"] + merged["low"] + merged["high"]) / 1.5
    merged["_x2"] = merged.groupby("symbol")["_x1"].transform(lambda s: s.ewm(span=3, adjust=False).mean())
    merged["_high30"] = merged.groupby("symbol")["_x2"].transform(lambda s: s.rolling(30, min_periods=10).max())
    merged["_high30_prev"] = merged.groupby("symbol")["_high30"].shift(1)
    merged["high30_new_high"] = (merged["_high30"] > merged["_high30_prev"]).astype(int)

    # 30天内有涨停: 涨幅>9.5%的天数
    merged["_is_limit_up"] = (merged["close"] / merged.groupby("symbol")["close"].shift(1) > 1.095).astype(int)
    merged["has_limit_up_30d_calc"] = merged.groupby("symbol")["_is_limit_up"].transform(
        lambda s: s.rolling(30, min_periods=1).sum()
    )
    merged["has_limit_up_30d_calc"] = (merged["has_limit_up_30d_calc"] >= 1).astype(int)

    # 清理临时列
    drop_cols = [c for c in merged.columns if c.startswith("_")]
    merged.drop(columns=drop_cols, inplace=True, errors="ignore")

    return merged


def build_industry_map_from_config(base_dir: Path, symbols: pd.Series) -> dict[str, str]:
    mapping: dict[str, str] = {}

    # 尝试多个配置文件
    cfg_path = None
    for name in ["config_v31.yaml", "config_optimized.yaml", "config.yaml"]:
        p = base_dir / name
        if p.exists():
            cfg_path = p
            break

    if cfg_path is not None:
        text = cfg_path.read_text(encoding="utf-8", errors="ignore")
        current_group = "GROUP_0"
        group_id = 0

        for line in text.splitlines():
            if "==========" in line:
                group_id += 1
                slug = re.sub(r"[^A-Za-z0-9_]+", "_", line).strip("_")
                if not slug:
                    slug = f"GROUP_{group_id}"
                current_group = slug[:48]

            for code in re.findall(r"(?<!\d)(\d{6})(?!\d)", line):
                mapping[_norm_symbol(code)] = current_group

    def fallback_group(sym: str) -> str:
        code = sym.split(".")[0]
        if code.startswith("688"):
            return "STAR"
        if code.startswith("300"):
            return "CHINEXT"
        if code.startswith(("600", "601", "603", "605")):
            return "SSE_MAIN"
        if code.startswith(("000", "001", "002", "003")):
            return "SZ_MAIN"
        return "OTHER"

    full_map: dict[str, str] = {}
    for sym in symbols.astype(str).unique().tolist():
        s = sym.upper()
        full_map[s] = mapping.get(s, fallback_group(s))

    return full_map


def precompute_daily_universe(df: pd.DataFrame) -> pd.DataFrame:
    day_list: list[pd.DataFrame] = []

    required_fill_zero = [
        "ma_dist_20",
        "ret_20d",
        "ret_60d",
        "vol_20d",
        "vol_ratio_20",
        "tdx_score",
        "high30_breakout",
        "main_force_strong",
        "volume_breakout",
        "price_above_ma5",
        "days_since_limit_up",
        "pullback_pct",
        "amount_20d",
        "turnover_20d",
        "float_mkt_cap_20d",
        "amount",
        "turnover",
        "main_force_pct",
        "high30_new_high",
        "has_limit_up_30d_calc",
    ]
    for col in required_fill_zero:
        if col not in df.columns:
            df[col] = 0.0

    bool_like_cols = ["one_line_board", "limit_up_flag", "near_limit_up", "has_limit_up_30d", "monthly_bullish"]
    for col in bool_like_cols:
        if col not in df.columns:
            df[col] = 0

    for _, g in df.groupby("date", sort=True):
        x = g.copy()
        x["atr_pct"] = x["atr_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.01).clip(0.003, 0.25)

        z_ma = _safe_zscore(x["ma_dist_20"])
        z_r20 = _safe_zscore(x["ret_20d"])
        z_r60 = _safe_zscore(x["ret_60d"])
        z_vol = _safe_zscore(x["vol_20d"])
        z_atr = _safe_zscore(x["atr_pct"])
        z_vr = _safe_zscore(x["vol_ratio_20"])

        score = (
            2.0 * z_ma
            + 1.0 * z_r20
            + 0.5 * z_r60
            - 1.0 * z_vol
            - 0.5 * z_atr
            + 0.3 * z_vr
        )
        score = (
            score
            + x["high30_breakout"].fillna(0).astype(float) * 1.0
            + x["main_force_strong"].fillna(0).astype(float) * 1.5
            + x["has_limit_up_30d"].fillna(0).astype(float) * 0.5
        )
        score = score + x["monthly_bullish"].fillna(0).astype(float) * 0.8

        x["score"] = score
        x["trend_up"] = (x["ma_dist_20"] > 0).astype(int)

        tradeable = pd.Series(True, index=x.index)
        tradeable &= x["one_line_board"].fillna(0) != 1
        tradeable &= x["limit_up_flag"].fillna(0) != 1
        tradeable &= x["near_limit_up"].fillna(0) != 1
        tradeable &= x["close"] > 0
        x["tradeable"] = tradeable.astype(int)

        x = x.sort_values("score", ascending=False).reset_index(drop=True)
        x["rank"] = np.arange(1, len(x) + 1)
        day_list.append(x)

    return pd.concat(day_list, ignore_index=True)


def compute_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["trend_up"] = (tmp["ma_dist_20"] > 0).astype(int)

    mkt = tmp.groupby("date", as_index=False).agg(
        market_ret_20d=("ret_20d", "mean"),
        market_trend_pct=("trend_up", "mean"),
    )

    def label(row: pd.Series) -> str:
        if row["market_trend_pct"] > 0.6 and row["market_ret_20d"] > 0.03:
            return "BULL"
        if row["market_trend_pct"] < 0.4 and row["market_ret_20d"] < -0.03:
            return "BEAR"
        return "NEUTRAL"

    mkt["regime"] = mkt.apply(label, axis=1)
    mkt["regime_pos_cap"] = mkt["regime"].map({"BULL": 0.80, "NEUTRAL": 0.55, "BEAR": 0.30})
    return mkt


def compute_index_filter(base_dir: Path, start_date: str, cfg: BacktestConfigV3) -> dict:
    """
    方案1: 指数风控开关
    返回 {date: True/False}，True=允许开仓，False=暂停开仓
    """
    if not cfg.use_index_filter:
        return {}

    # 尝试加载沪深300指数
    idx_path = base_dir / "data" / "index" / "hs300_daily.parquet"
    if not idx_path.exists():
        # 如果没有本地指数数据，尝试用BaoStock获取
        try:
            import baostock as bs
            lg = bs.login()
            try:
                rs = bs.query_history_k_data_plus("sh.000300", "date,close",
                    start_date, "", "d", "3")
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    print("  ⚠ 无法获取指数数据，跳过指数风控")
                    return {}
                idx = pd.DataFrame(rows, columns=rs.fields)
                idx["date"] = pd.to_datetime(idx["date"])
                idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
            finally:
                bs.logout()
        except Exception as e:
            print(f"  ⚠ 获取指数数据失败: {e}，跳过指数风控")
            return {}
    else:
        idx = pd.read_parquet(idx_path)
        idx["date"] = pd.to_datetime(idx["date"])

    idx = idx.sort_values("date").reset_index(drop=True)

    # 计算均线
    idx["ma_long"] = idx["close"].rolling(cfg.index_ma_period, min_periods=20).mean()
    idx["ma_short"] = idx["close"].rolling(cfg.index_ma_short, min_periods=10).mean()
    idx["ma_short_prev"] = idx["ma_short"].shift(1)

    # 指数趋势: 收盘>MA60 且 MA20上拐
    idx["trend_ok"] = (idx["close"] > idx["ma_long"]) & (idx["ma_short"] > idx["ma_short_prev"])

    # 近N天跌幅检测
    idx["ret_n"] = idx["close"].pct_change(cfg.index_crash_days)
    idx["crash"] = idx["ret_n"] < cfg.index_crash_threshold

    # 暴跌后暂停N天
    idx["pause_until"] = pd.NaT
    pause_end = pd.NaT
    pause_flags = []
    for i, row in idx.iterrows():
        if row["crash"]:
            pause_end = row["date"] + pd.Timedelta(days=cfg.index_pause_days * 1.5)  # 交易日≈1.5倍自然日
        is_paused = pd.notna(pause_end) and row["date"] <= pause_end
        pause_flags.append(is_paused)
    idx["is_paused"] = pause_flags

    # 最终：趋势OK 且 不在暂停期
    idx["allow_open"] = idx["trend_ok"].fillna(False) & ~idx["is_paused"]

    result = dict(zip(idx["date"], idx["allow_open"]))
    allow_days = sum(1 for v in result.values() if v)
    total_days = len(result)
    print(f"  ✓ 指数风控: {allow_days}/{total_days} 天允许开仓 ({allow_days/max(total_days,1)*100:.0f}%)")
    return result


def _candidate_entry_mask(day: pd.DataFrame, cfg: BacktestConfigV3) -> pd.Series:
    """
    入场模式：
    - custom:  用户自定义必要条件（月线级别5条件全部满足）
    - strict:  必须 (涨停回调全套 OR TDX高分) AND trend_up
    - normal:  多路径灵活入场
    - loose:   仅需 trend_up
    """
    mode = cfg.entry_mode.lower()

    if mode == "custom":
        # ── 用户自定义：5个月线级别必要条件全部AND ──
        # 1. 30天内有涨停
        cond_limit_up = day["has_limit_up_30d_calc"].fillna(0) >= 1
        # 2. 主力控盘 > 0.5% (GU1偏离起爆线)
        cond_main_force = day["main_force_pct"].fillna(0) > 0.005
        # 3. 高30创新高
        cond_high30 = day["high30_new_high"].fillna(0) == 1
        # 4. 回调 10%-30%
        cond_pullback = (day["pullback_pct"].fillna(0) >= cfg.pullback_min_pct) & (
            day["pullback_pct"].fillna(0) <= cfg.pullback_max_pct
        )
        # 5. TDX评分 >= 1.5
        cond_tdx = day["tdx_score"].fillna(0) >= cfg.tdx_min_score

        return cond_limit_up & cond_main_force & cond_high30 & cond_pullback & cond_tdx

    # ── 基础条件计算 ──
    in_window = (day["days_since_limit_up"] >= cfg.pullback_window_start) & (
        day["days_since_limit_up"] <= cfg.pullback_window_end
    )
    pullback_ok = (day["pullback_pct"] >= cfg.pullback_min_pct) & (
        day["pullback_pct"] <= cfg.pullback_max_pct
    )
    breakout_ok = (day["volume_breakout"].fillna(0) == 1) & (
        day["price_above_ma5"].fillna(0) == 1
    )
    tdx_ok = (day["tdx_score"].fillna(0) >= cfg.tdx_min_score) | (
        (day["high30_breakout"].fillna(0) == 1) & (day["main_force_strong"].fillna(0) == 1)
    )
    trend_up = day["trend_up"] == 1

    # ── 涨停回调完整条件 ──
    limit_up_pullback = in_window & pullback_ok

    if mode == "strict":
        core_signal = (limit_up_pullback & breakout_ok) | tdx_ok
        return core_signal & trend_up

    elif mode == "loose":
        return trend_up

    else:  # "normal" (默认) - 方案2A: 2-of-3 + 信号强度分级
        # 附加条件计数: breakout_ok, tdx_ok, trend_up
        extra_count = (
            breakout_ok.astype(int)
            + tdx_ok.astype(int)
            + trend_up.astype(int)
        )

        # 路径1: 涨停回调 + 附加条件≥2 (2-of-3)
        path_pullback = limit_up_pullback & (extra_count >= cfg.normal_min_conditions)

        # 路径2: TDX高分 + trend_up (强信号)
        path_tdx = tdx_ok & trend_up

        # 路径3: 放量突破 + trend_up + 主力控盘 (需要更多验证)
        path_breakout = breakout_ok & trend_up & (day["main_force_strong"].fillna(0) == 1)

        mask = path_pullback | path_tdx | path_breakout

        # ── 方案2B: 信号强度分级（存到 DataFrame 供仓位分配用）──
        signal_strength = pd.Series(0.0, index=day.index)
        # 强信号: 涨停回调 + TDX + 趋势 全满足
        strong = limit_up_pullback & tdx_ok & trend_up
        # 普通信号: 路径2或路径1
        normal_sig = (path_tdx | path_pullback) & ~strong
        # 弱信号: 仅路径3
        weak = path_breakout & ~strong & ~normal_sig

        signal_strength = signal_strength.where(~strong, 2.0)    # strong=2
        signal_strength = signal_strength.where(~normal_sig, 1.0)  # normal=1
        signal_strength = signal_strength.where(~weak, 0.5)        # weak=0.5
        day["_signal_strength"] = signal_strength

        return mask


def _apply_enhancement_filters(day: pd.DataFrame, cfg: BacktestConfigV3) -> pd.DataFrame:
    out = day.copy()

    if cfg.use_tradeability_filter:
        out = out[out["tradeable"] == 1]

    if cfg.use_market_cap_filter:
        out = out[
            (out["float_mkt_cap_20d"] >= cfg.min_float_mkt_cap) & (out["float_mkt_cap_20d"] <= cfg.max_float_mkt_cap)
        ]

    if cfg.use_liquidity_filter:
        out = out[(out["amount_20d"] >= cfg.min_amount_20d) & (out["turnover_20d"] >= cfg.min_turnover_20d)]

    return out


def _corr_guard_select(
    day_candidates: pd.DataFrame,
    top_k: int,
    cfg: BacktestConfigV3,
    ret_window: pd.DataFrame,
) -> list[str]:
    if day_candidates.empty or top_k <= 0:
        return []

    selected: list[str] = []
    industry_counter: dict[str, int] = {}

    for _, row in day_candidates.iterrows():
        sym = str(row["symbol"])
        industry = str(row.get("industry", "OTHER"))

        if cfg.use_industry_diversification and industry_counter.get(industry, 0) >= cfg.max_per_industry:
            continue

        if cfg.use_correlation_control and len(selected) > 0 and len(ret_window) >= 15:
            if sym not in ret_window.columns:
                continue
            s = ret_window[sym]
            p = ret_window[selected].mean(axis=1)
            corr = s.corr(p)
            if np.isfinite(corr) and abs(float(corr)) > cfg.max_pairwise_corr:
                continue

        selected.append(sym)
        industry_counter[industry] = industry_counter.get(industry, 0) + 1
        if len(selected) >= top_k:
            break

    return selected


def run_backtest_v3(
    daily_universe: pd.DataFrame,
    price_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    cfg: BacktestConfigV3,
    regime_df: pd.DataFrame | None = None,
    date_start: pd.Timestamp | None = None,
    date_end: pd.Timestamp | None = None,
    index_filter: dict | None = None,
    allowed_symbols_by_date: dict[pd.Timestamp, set[str]] | None = None,
) -> dict[str, Any]:
    ddf = daily_universe.copy()
    ddf["date"] = pd.to_datetime(ddf["date"])

    if date_start is not None:
        ddf = ddf[ddf["date"] >= date_start]
    if date_end is not None:
        ddf = ddf[ddf["date"] <= date_end]

    if ddf.empty:
        empty_eq = pd.DataFrame(columns=["date", "equity", "daily_return", "n_holdings"])
        metrics = {"annual_return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe": 0.0, "win_rate_pct": 0.0}
        return {"metrics": metrics, "equity_curve": empty_eq, "daily_targets": {}}

    trading_days = sorted(ddf["date"].unique().tolist())
    if len(trading_days) < 20:
        empty_eq = pd.DataFrame(columns=["date", "equity", "daily_return", "n_holdings"])
        metrics = {"annual_return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe": 0.0, "win_rate_pct": 0.0}
        return {"metrics": metrics, "equity_curve": empty_eq, "daily_targets": {}}

    regime_pos: dict[pd.Timestamp, float] = {}
    if regime_df is not None and not regime_df.empty:
        rr = regime_df.copy()
        rr["date"] = pd.to_datetime(rr["date"])
        regime_pos = dict(zip(rr["date"], rr["regime_pos_cap"]))

    by_date = {d: g for d, g in ddf.groupby("date", sort=True)}
    idx_filter = index_filter or {}

    equity = cfg.initial_capital
    positions: dict[str, dict[str, float]] = {}
    rec: list[dict[str, Any]] = []
    daily_targets: dict[pd.Timestamp, list[str]] = {}
    trade_log: list[dict[str, Any]] = []  # 交易记录

    for i in range(1, len(trading_days)):
        prev_date = trading_days[i - 1]
        date = trading_days[i]
        prev_day = by_date.get(prev_date)
        if prev_day is None or prev_day.empty:
            continue

        if allowed_symbols_by_date:
            allowed = allowed_symbols_by_date.get(prev_date)
            if allowed is not None:
                prev_day = prev_day[prev_day["symbol"].isin(allowed)]

        # ── 方案1: 指数风控开关 ──
        index_allow = idx_filter.get(prev_date, True) if idx_filter else True

        if index_allow and not prev_day.empty:
            mask = _candidate_entry_mask(prev_day, cfg)
            cands = _apply_enhancement_filters(prev_day[mask].copy(), cfg).sort_values(
                "score", ascending=False
            ).head(cfg.invest_more_n)

            if cfg.use_correlation_control:
                ret_window = returns_df.loc[
                    (returns_df.index < prev_date)
                    & (returns_df.index >= prev_date - pd.Timedelta(days=cfg.corr_lookback_days * 2))
                ]
                if len(ret_window) > cfg.corr_lookback_days:
                    ret_window = ret_window.tail(cfg.corr_lookback_days)
            else:
                ret_window = returns_df.iloc[:0]

            target_symbols = _corr_guard_select(cands, cfg.top_k, cfg, ret_window)

            # 获取信号强度（方案2B）
            signal_strengths: dict[str, float] = {}
            if "_signal_strength" in prev_day.columns:
                for _, row in cands.iterrows():
                    signal_strengths[str(row["symbol"])] = float(row.get("_signal_strength", 1.0))
        else:
            # 指数不允许开仓或当日可选池为空 → 空目标（只做风控卖出，不新开仓）
            target_symbols = []
            signal_strengths = {}

        daily_targets[prev_date] = target_symbols

        # ── 当日持仓收益（先记已有仓位收益，再执行收盘调仓）──
        daily_ret_row = returns_df.loc[date] if date in returns_df.index else pd.Series(dtype=float)
        gross_ret = 0.0
        total_weight = 0.0
        for sym, pos in positions.items():
            gross_ret += pos["weight"] * float(daily_ret_row.get(sym, 0.0))
            total_weight += pos["weight"]

        # ── 退出逻辑（方案3升级）──
        to_sell: list[str] = []
        for sym, pos in positions.items():
            if sym not in price_df.columns or date not in price_df.index:
                to_sell.append(sym)
                continue

            px = price_df.at[date, sym]
            if not np.isfinite(px) or px <= 0:
                to_sell.append(sym)
                continue

            pos["hold_days"] += 1
            if px > pos["peak_price"]:
                pos["peak_price"] = float(px)

            pnl_pct = (float(px) - pos["entry_price"]) / pos["entry_price"]

            # (1) 硬止损（含TDX保护）
            effective_stop = cfg.stop_loss_pct
            if cfg.use_tdx_protection and pos.get("tdx_score", 0) >= cfg.tdx_protection_threshold:
                effective_stop = max(cfg.stop_loss_pct, 0.08)
            if pnl_pct <= -effective_stop:
                to_sell.append(sym)
                continue

            # (2) ATR止损（方案3）
            if cfg.use_atr_stop and pos.get("atr_val", 0) > 0:
                atr_stop_price = pos["entry_price"] - cfg.atr_stop_multiplier * pos["atr_val"]
                if float(px) <= atr_stop_price:
                    to_sell.append(sym)
                    continue

            # (3) 失败形态止损（方案3）: 入场后N天内涨幅不足，证伪出局
            if cfg.use_failure_stop and pos["hold_days"] <= cfg.failure_stop_days:
                if pos["hold_days"] == cfg.failure_stop_days and pnl_pct < cfg.failure_stop_gain:
                    to_sell.append(sym)
                    continue

            # (4) 移动止盈
            drawdown = (pos["peak_price"] - float(px)) / max(pos["peak_price"], 1e-12)
            # 动态移动止盈: 赚得越多，止盈线越紧
            effective_trail = cfg.trailing_stop_pct
            if pnl_pct > 0.10:  # 浮盈>10%时收紧到6%
                effective_trail = min(cfg.trailing_stop_pct, 0.06)
            elif pnl_pct > 0.05:  # 浮盈>5%时收紧到8%
                effective_trail = min(cfg.trailing_stop_pct, 0.08)
            if drawdown >= effective_trail:
                to_sell.append(sym)
                continue

            # (5) 时间止损（优化）: 到期且无明确盈利则清
            if pos["hold_days"] >= cfg.max_hold_days and pnl_pct < cfg.time_stop_min_gain:
                to_sell.append(sym)
                continue

            # ── 方案4: 盈利加仓 ──
            if cfg.use_profit_pyramiding and pnl_pct >= cfg.pyramid_trigger_pct:
                adds = pos.get("pyramid_adds", 0)
                if adds < cfg.pyramid_max_adds:
                    add_w = pos["weight"] * cfg.pyramid_add_ratio
                    pos["weight"] = pos["weight"] + add_w
                    pos["pyramid_adds"] = adds + 1

        for sym in to_sell:
            pos = positions.pop(sym, None)
            if pos is not None:
                exit_px = float(price_df.at[date, sym]) if (sym in price_df.columns and date in price_df.index) else pos["entry_price"]
                pnl = (exit_px - pos["entry_price"]) / pos["entry_price"] * 100
                trade_log.append({
                    "symbol": sym,
                    "entry_date": str(pos.get("entry_date", "")),
                    "exit_date": str(date)[:10],
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_px, 2),
                    "hold_days": int(pos["hold_days"]),
                    "pnl_pct": round(pnl, 2),
                    "weight": round(pos["weight"] * 100, 1),
                })

        # 指数允许时才保留在目标里的持仓，否则逐步清仓
        if index_allow:
            for sym in list(positions.keys()):
                if sym not in target_symbols:
                    pos = positions.pop(sym)
                    exit_px = float(price_df.at[date, sym]) if (sym in price_df.columns and date in price_df.index) else pos["entry_price"]
                    pnl = (exit_px - pos["entry_price"]) / pos["entry_price"] * 100
                    trade_log.append({
                        "symbol": sym,
                        "entry_date": str(pos.get("entry_date", "")),
                        "exit_date": str(date)[:10],
                        "entry_price": round(pos["entry_price"], 2),
                        "exit_price": round(exit_px, 2),
                        "hold_days": int(pos["hold_days"]),
                        "pnl_pct": round(pnl, 2),
                        "weight": round(pos["weight"] * 100, 1),
                    })

        # ── 仓位分配（方案2B: 信号分级）──
        max_pos_today = cfg.max_total_position
        if cfg.use_market_regime:
            max_pos_today = min(max_pos_today, float(regime_pos.get(prev_date, cfg.max_total_position)))

        # 指数风控减半仓位
        if not index_allow:
            max_pos_today *= 0.3  # 熊市仅保留30%已有仓位

        n_targets = max(1, len(target_symbols))
        base_w = min(max_pos_today / n_targets, cfg.max_single_weight)

        for sym in target_symbols:
            if sym in positions:
                continue
            if sym not in price_df.columns or date not in price_df.index:
                continue
            px = price_df.at[date, sym]
            if not np.isfinite(px) or px <= 0:
                continue

            # 信号分级仓位
            if cfg.use_signal_tiered_sizing:
                strength = signal_strengths.get(sym, 1.0)
                if strength >= 2.0:
                    w = base_w * cfg.tier_strong_multiplier
                elif strength >= 1.0:
                    w = base_w * cfg.tier_normal_multiplier
                else:
                    w = base_w * cfg.tier_weak_multiplier
                w = min(w, cfg.max_single_weight)
            else:
                w = base_w

            # 获取ATR用于ATR止损
            atr_val = 0.0
            if cfg.use_atr_stop and sym in prev_day["symbol"].values:
                atr_row = prev_day.loc[prev_day["symbol"] == sym, "atr_pct"]
                if not atr_row.empty:
                    atr_val = float(atr_row.iloc[0]) * float(px)

            tdx_val = 0.0
            if sym in prev_day["symbol"].values:
                tdx_row = prev_day.loc[prev_day["symbol"] == sym, "tdx_score"]
                if not tdx_row.empty:
                    tdx_val = float(tdx_row.iloc[0])

            positions[sym] = {
                "entry_price": float(px),
                "peak_price": float(px),
                "hold_days": 0.0,
                "weight": float(w),
                "tdx_score": tdx_val,
                "atr_val": atr_val,
                "pyramid_adds": 0,
                "entry_date": str(date)[:10],
            }

        # 重新平衡权重
        active_targets = [s for s in target_symbols if s in positions]
        if len(active_targets) > 0:
            total_w = sum(positions[s]["weight"] for s in active_targets)
            if total_w > max_pos_today:
                scale = max_pos_today / total_w
                for s in active_targets:
                    positions[s]["weight"] *= scale

        turnover = 0.10 if len(target_symbols) > 0 else 0.0
        cost = turnover * (cfg.cost_bps / 10000.0)
        net_ret = gross_ret - cost
        equity *= 1.0 + net_ret

        rec.append(
            {
                "date": date,
                "equity": equity,
                "daily_return": net_ret,
                "gross_return": gross_ret,
                "turnover": turnover,
                "n_holdings": len(positions),
                "total_position": total_weight,
            }
        )

    eq = pd.DataFrame(rec)
    if eq.empty:
        metrics = {"annual_return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe": 0.0, "win_rate_pct": 0.0}
        return {"metrics": metrics, "equity_curve": eq, "daily_targets": daily_targets, "trade_log": trade_log}

    nav = eq["equity"] / cfg.initial_capital
    r = eq["daily_return"]
    metrics = {
        "annual_return_pct": _annual_return(nav) * 100.0,
        "max_drawdown_pct": _max_drawdown(nav) * 100.0,
        "sharpe": _sharpe(r),
        "win_rate_pct": _win_rate(r) * 100.0,
    }
    return {"metrics": metrics, "equity_curve": eq, "daily_targets": daily_targets, "trade_log": trade_log}


def evaluate_params_cv(
    daily_universe: pd.DataFrame,
    price_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    param_grid: dict[str, list[Any]],
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(daily_universe["date"].unique()))
    if len(dates) < 80:
        raise ValueError("not enough dates for time-series CV")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    combos = list(
        itertools.product(
            param_grid["top_k"],
            param_grid["invest_more_n"],
            param_grid["pullback_min_pct"],
            param_grid["pullback_max_pct"],
            param_grid["stop_loss_pct"],
            param_grid["trailing_stop_pct"],
        )
    )

    fold_rows: list[dict[str, Any]] = []

    for fold_id, (train_idx, test_idx) in enumerate(tscv.split(dates), start=1):
        test_dates = [dates[i] for i in test_idx]
        train_dates = [dates[i] for i in train_idx]
        train_start, train_end = train_dates[0], train_dates[-1]
        test_start, test_end = test_dates[0], test_dates[-1]

        print(
            f"[CV] fold {fold_id}/{n_splits} | "
            f"train {train_start.date()}->{train_end.date()} | "
            f"test {test_start.date()}->{test_end.date()}"
        )

        for i, combo in enumerate(combos, start=1):
            top_k, invest_more_n, pmin, pmax, stop_loss, trailing = combo
            if pmin >= pmax or invest_more_n < top_k:
                continue

            cfg = BacktestConfigV3(
                top_k=int(top_k),
                invest_more_n=int(invest_more_n),
                pullback_min_pct=float(pmin),
                pullback_max_pct=float(pmax),
                stop_loss_pct=float(stop_loss),
                trailing_stop_pct=float(trailing),
            )

            res = run_backtest_v3(
                daily_universe=daily_universe,
                price_df=price_df,
                returns_df=returns_df,
                cfg=cfg,
                regime_df=regime_df,
                date_start=test_start,
                date_end=test_end,
            )
            m = res["metrics"]

            fold_rows.append(
                asdict(
                    FoldResult(
                        fold_id=fold_id,
                        train_start=train_start.strftime("%Y-%m-%d"),
                        train_end=train_end.strftime("%Y-%m-%d"),
                        test_start=test_start.strftime("%Y-%m-%d"),
                        test_end=test_end.strftime("%Y-%m-%d"),
                        top_k=int(top_k),
                        invest_more_n=int(invest_more_n),
                        pullback_min_pct=float(pmin),
                        pullback_max_pct=float(pmax),
                        stop_loss_pct=float(stop_loss),
                        trailing_stop_pct=float(trailing),
                        annual_return_pct=float(m["annual_return_pct"]),
                        max_drawdown_pct=float(m["max_drawdown_pct"]),
                        sharpe=float(m["sharpe"]),
                        win_rate_pct=float(m["win_rate_pct"]),
                    )
                )
            )

            if i % 256 == 0:
                print(f"  fold {fold_id}: {i}/{len(combos)} combos done")

    folds_df = pd.DataFrame(fold_rows)

    grp_cols = ["top_k", "invest_more_n", "pullback_min_pct", "pullback_max_pct", "stop_loss_pct", "trailing_stop_pct"]
    agg = folds_df.groupby(grp_cols, as_index=False).agg(
        annual_return_pct=("annual_return_pct", "mean"),
        max_drawdown_pct=("max_drawdown_pct", "mean"),
        sharpe=("sharpe", "mean"),
        win_rate_pct=("win_rate_pct", "mean"),
        stability_sharpe_std=("sharpe", "std"),
    )

    dd_penalty = np.maximum(np.abs(agg["max_drawdown_pct"]) - 15.0, 0.0)
    wr_bonus = np.maximum(agg["win_rate_pct"] - 55.0, 0.0)
    sharpe_bonus = np.maximum(agg["sharpe"] - 1.5, 0.0)

    agg["objective"] = (
        agg["sharpe"] * 2.2
        + agg["annual_return_pct"] / 25.0
        + wr_bonus / 20.0
        + sharpe_bonus * 0.5
        - dd_penalty / 8.0
        - agg["stability_sharpe_std"].fillna(0.0) * 0.7
    )

    agg = agg.sort_values("objective", ascending=False).reset_index(drop=True)
    return folds_df, agg


def sensitivity_analysis(agg_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = ["top_k", "invest_more_n", "pullback_min_pct", "pullback_max_pct", "stop_loss_pct", "trailing_stop_pct"]

    rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for p in params:
        g = agg_df.groupby(p, as_index=False).agg(
            mean_sharpe=("sharpe", "mean"),
            mean_max_dd=("max_drawdown_pct", "mean"),
            mean_win_rate=("win_rate_pct", "mean"),
            std_sharpe=("sharpe", "std"),
        )

        rows.append(
            {
                "parameter": p,
                "impact_on_sharpe": float(g["mean_sharpe"].max() - g["mean_sharpe"].min()),
                "stability_score": float(g["std_sharpe"].mean()),
            }
        )

        for _, r in g.iterrows():
            profile_rows.append(
                {
                    "parameter": p,
                    "value": r[p],
                    "mean_sharpe": r["mean_sharpe"],
                    "mean_max_dd": r["mean_max_dd"],
                    "mean_win_rate": r["mean_win_rate"],
                    "std_sharpe": r["std_sharpe"],
                }
            )

    importance_df = pd.DataFrame(rows).sort_values("impact_on_sharpe", ascending=False)
    profile_df = pd.DataFrame(profile_rows)
    return importance_df, profile_df


def plot_sensitivity(profile_df: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    params = ["top_k", "invest_more_n", "pullback_min_pct", "pullback_max_pct", "stop_loss_pct", "trailing_stop_pct"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=140)
    axes = axes.flatten()

    for ax, p in zip(axes, params):
        g = profile_df[profile_df["parameter"] == p].copy().sort_values("value")
        x = g["value"].astype(str)
        ax.plot(x, g["mean_sharpe"], marker="o", label="Mean Sharpe")
        ax_t = ax.twinx()
        ax_t.plot(x, g["mean_max_dd"], marker="s", linestyle="--", color="#d62728", label="Mean MaxDD")

        ax.set_title(p)
        ax.set_xlabel("value")
        ax.set_ylabel("Sharpe")
        ax_t.set_ylabel("MaxDD(%)")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_t.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")

    fig.suptitle("Parameter Sensitivity (Sharpe vs Max Drawdown)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def run_benchmark_hs300(base_dir: Path, start: pd.Timestamp, end: pd.Timestamp, initial_capital: float) -> pd.DataFrame:
    idx_path = base_dir / "data" / "index" / "hs300_daily.parquet"
    if not idx_path.exists():
        return pd.DataFrame(columns=["date", "equity", "daily_return"])

    df = pd.read_parquet(idx_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "equity", "daily_return"])

    df = df.sort_values("date").reset_index(drop=True)
    df["daily_return"] = df["close"].pct_change(fill_method=None).fillna(0.0)
    df["equity"] = initial_capital * (1.0 + df["daily_return"]).cumprod()
    return df[["date", "equity", "daily_return"]]


def run_benchmark_buy_hold(price_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, initial_capital: float) -> pd.DataFrame:
    sub = price_df[(price_df.index >= start) & (price_df.index <= end)]
    if sub.empty:
        return pd.DataFrame(columns=["date", "equity", "daily_return"])

    rets = sub.pct_change(fill_method=None).fillna(0.0)
    ew = rets.mean(axis=1)
    out = pd.DataFrame({"date": ew.index, "daily_return": ew.values})
    out["equity"] = initial_capital * (1.0 + out["daily_return"]).cumprod()
    return out[["date", "equity", "daily_return"]]


def summarize_metrics(equity_df: pd.DataFrame, initial_capital: float) -> dict[str, float]:
    if equity_df.empty:
        return {
            "annual_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "win_rate_pct": 0.0,
            "total_return_pct": 0.0,
        }

    nav = equity_df["equity"] / initial_capital
    r = equity_df["daily_return"]
    return {
        "annual_return_pct": _annual_return(nav) * 100.0,
        "max_drawdown_pct": _max_drawdown(nav) * 100.0,
        "sharpe": _sharpe(r),
        "win_rate_pct": _win_rate(r) * 100.0,
        "total_return_pct": float((nav.iloc[-1] - 1.0) * 100.0),
    }


def build_dynamic_watchlist_by_date(
    daily_universe: pd.DataFrame,
    max_symbols: int = 200,
    rebalance_freq: str = "M",
) -> dict[pd.Timestamp, set[str]]:
    """基于历史截面的条件，按月/周动态构建可交易股票池（避免用当下固定池回测全历史）。"""
    ddf = daily_universe.copy()
    ddf["date"] = pd.to_datetime(ddf["date"])
    ddf = ddf.sort_values(["date", "symbol"]).reset_index(drop=True)

    dates = sorted(ddf["date"].unique().tolist())
    if not dates:
        return {}

    dates_df = pd.DataFrame({"date": pd.to_datetime(dates)})
    dates_df["period"] = dates_df["date"].dt.to_period(rebalance_freq)
    rebalance_dates = set(dates_df.groupby("period")["date"].max().tolist())

    result: dict[pd.Timestamp, set[str]] = {}
    current_universe: set[str] | None = None

    for d in dates:
        day = ddf[ddf["date"] == d].copy()
        if day.empty:
            continue

        if d in rebalance_dates or current_universe is None:
            # 与自动选股逻辑保持一致：涨停活跃 + 主力控盘为正 + 高30创新高
            strict_mask = (
                (day["has_limit_up_30d_calc"].fillna(0) >= 1)
                & (day["main_force_pct"].fillna(0) > 0.005)
                & (day["high30_new_high"].fillna(0) == 1)
            )
            cands = day[strict_mask].copy()

            # 候选不足时，放宽到去掉创新高条件，防止池子过空
            if len(cands) < max_symbols // 3:
                loose_mask = (
                    (day["has_limit_up_30d_calc"].fillna(0) >= 1)
                    & (day["main_force_pct"].fillna(0) > 0)
                )
                cands = day[loose_mask].copy()

            if cands.empty:
                cands = day.copy()

            cands = cands.sort_values(
                ["tdx_score", "amount_20d", "turnover_20d"],
                ascending=[False, False, False],
            )
            current_universe = set(cands.head(max_symbols)["symbol"].astype(str).tolist())

        result[pd.to_datetime(d)] = current_universe or set()

    return result


def to_html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>(empty)</p>"
    return df.to_html(index=False, border=0, classes="table")


def save_reports(
    base_dir: Path,
    cfg_best: BacktestConfigV3,
    folds_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    strategy_eq: pd.DataFrame,
    strategy_metrics: dict[str, float],
    hs300_metrics: dict[str, float],
    buy_hold_metrics: dict[str, float],
    sensitivity_png: Path,
    compare_html_path: Path,
    compare_md_path: Path,
    tuning_md_path: Path,
    strategy_doc_path: Path,
    optimized_cfg_path: Path,
) -> None:
    optimized_cfg = {
        "strategy": {
            "top_k": cfg_best.top_k,
            "invest_more_n": cfg_best.invest_more_n,
            "pullback_min_pct": cfg_best.pullback_min_pct,
            "pullback_max_pct": cfg_best.pullback_max_pct,
            "use_limit_up_pullback": True,
            "use_tdx_indicators": True,
            "use_tradeability_filter": cfg_best.use_tradeability_filter,
            "use_volatility_sizing": True,
            "max_single_weight": cfg_best.max_single_weight,
            "max_total_position": cfg_best.max_total_position,
            "use_market_regime": cfg_best.use_market_regime,
            "industry_diversification": {
                "enabled": cfg_best.use_industry_diversification,
                "max_per_industry": cfg_best.max_per_industry,
            },
            "market_cap_filter": {
                "enabled": cfg_best.use_market_cap_filter,
                "min_float_mkt_cap": cfg_best.min_float_mkt_cap,
                "max_float_mkt_cap": cfg_best.max_float_mkt_cap,
            },
            "liquidity_filter": {
                "enabled": cfg_best.use_liquidity_filter,
                "min_amount_20d": cfg_best.min_amount_20d,
                "min_turnover_20d": cfg_best.min_turnover_20d,
            },
            "correlation_control": {
                "enabled": cfg_best.use_correlation_control,
                "corr_lookback_days": cfg_best.corr_lookback_days,
                "max_pairwise_corr": cfg_best.max_pairwise_corr,
            },
        },
        "risk_control": {
            "stop_loss_pct": cfg_best.stop_loss_pct,
            "trailing_stop_pct": cfg_best.trailing_stop_pct,
            "max_hold_days": cfg_best.max_hold_days,
            "use_stop_loss": True,
            "use_trailing_stop": True,
            "use_time_stop": True,
        },
        "backtest": {
            "initial_capital": cfg_best.initial_capital,
            "cost_bps": cfg_best.cost_bps,
            "start_date": "2023-01-01",
        },
    }
    with optimized_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(optimized_cfg, f, sort_keys=False, allow_unicode=True)

    best10 = agg_df.head(10).copy().round(4)
    fold_head = folds_df.head(40).copy().round(4)
    imp = importance_df.copy().round(4)

    impact_most = imp.iloc[0]["parameter"] if not imp.empty else "N/A"
    stable_most = imp.sort_values("stability_score", ascending=True).iloc[0]["parameter"] if not imp.empty else "N/A"

    tuning_md = f"""# 参数调优报告

## 优化目标
- 夏普比率 > 1.5
- 最大回撤 < 15%
- 胜率 > 55%

## 方法
- 参数搜索: 全网格 (4^6 = 4096 组)
- 时序验证: TimeSeriesSplit(n_splits=5)
- 回测区间: 2023-01-01 至最新
- 目标函数: Sharpe + 收益/胜率奖励 - 回撤/不稳定惩罚

## 最优参数
```yaml
{yaml.safe_dump(asdict(cfg_best), sort_keys=False, allow_unicode=True)}
```

## CV汇总 Top10
{best10.to_markdown(index=False)}

## Fold级结果（前40行）
{fold_head.to_markdown(index=False)}

## 敏感性结论
- 影响最大的参数: `{impact_most}`
- 最稳定的参数: `{stable_most}`

### 参数影响度表
{imp.to_markdown(index=False)}

## 目标达成检查（最优参数全样本）
- 夏普: {strategy_metrics['sharpe']:.2f}
- 最大回撤: {strategy_metrics['max_drawdown_pct']:.2f}%
- 胜率: {strategy_metrics['win_rate_pct']:.2f}%
"""
    tuning_md_path.write_text(tuning_md, encoding="utf-8")

    cmp_df = pd.DataFrame(
        [
            {"strategy": "Optimized V3", **strategy_metrics},
            {"strategy": "HS300", **hs300_metrics},
            {"strategy": "Buy&Hold EqualWeight", **buy_hold_metrics},
        ]
    ).round(4)

    compare_md = f"""# 回测对比报告

## 区间
- 2023-01-01 至数据最新交易日

## 结果对比
{cmp_df.to_markdown(index=False)}

## 结论
- 优化策略夏普: {strategy_metrics['sharpe']:.2f}
- 优化策略最大回撤: {strategy_metrics['max_drawdown_pct']:.2f}%
- 优化策略胜率: {strategy_metrics['win_rate_pct']:.2f}%

## 敏感性图
- `{sensitivity_png}`
"""
    compare_md_path.write_text(compare_md, encoding="utf-8")

    html = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>Backtest Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .table {{ border-collapse: collapse; width: 100%; }}
    .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
    .table th:first-child, .table td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>回测对比报告</h1>
  <h2>指标对比</h2>
  {to_html_table(cmp_df)}
  <h2>Top10 参数组合</h2>
  {to_html_table(best10)}
  <h2>敏感性分析图</h2>
  <p><img src="{sensitivity_png.name}" style="max-width: 100%;" /></p>
</body>
</html>
"""
    compare_html_path.write_text(html, encoding="utf-8")

    doc = f"""# 策略说明文档（优化版 v3）

## 策略框架
1. 入场信号: 涨停回调 + TDX + 趋势（严格同时满足）
2. 风控: 硬止损 + 移动止盈 + 时间止损
3. 仓位: 波动反比 + 单票上限 + 总仓上限 + 市场状态仓位上限

## 新增增强项
1. 行业分散控制
- 优先读取 `config.yaml` 里 watchlist 分组映射
- 无映射时回退到交易板块分组
- 控制单行业最大持仓数量

2. 市值过滤
- 使用 `amount / (turnover/100)` 估算流通市值
- 用20日中位数平滑，过滤极小/极大市值

3. 流动性过滤
- 20日成交额均值阈值
- 20日换手率均值阈值

4. 相关性控制
- 候选股与已选组合收益序列相关性约束
- 限制高相关标的同时入选

## 优化流程
1. TimeSeriesSplit 5折时序交叉验证
2. 4096组网格参数全搜索
3. 以 Sharpe/回撤/胜率 和 稳定性综合打分
4. 输出最优参数并全样本回测

## 最优参数（摘要）
- top_k: {cfg_best.top_k}
- invest_more_n: {cfg_best.invest_more_n}
- pullback_min_pct: {cfg_best.pullback_min_pct}
- pullback_max_pct: {cfg_best.pullback_max_pct}
- stop_loss_pct: {cfg_best.stop_loss_pct}
- trailing_stop_pct: {cfg_best.trailing_stop_pct}

## 指标结果（全样本）
- 年化收益: {strategy_metrics['annual_return_pct']:.2f}%
- 最大回撤: {strategy_metrics['max_drawdown_pct']:.2f}%
- 夏普比率: {strategy_metrics['sharpe']:.2f}
- 胜率: {strategy_metrics['win_rate_pct']:.2f}%
"""
    strategy_doc_path.write_text(doc, encoding="utf-8")

    eq_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_equity.csv"
    eq_out.parent.mkdir(parents=True, exist_ok=True)
    strategy_eq.to_csv(eq_out, index=False, encoding="utf-8-sig")

    stats_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_stats.json"
    stats_out.write_text(json.dumps(strategy_metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def run_optimization_pipeline(base_dir: Path, start_date: str) -> dict[str, Any]:
    print("Loading and preparing data...")
    feats = load_and_prepare_features(base_dir, start_date=start_date)

    industry_map = build_industry_map_from_config(base_dir, feats["symbol"])
    feats["industry"] = feats["symbol"].map(industry_map).fillna("OTHER")

    print("Precomputing daily universe...")
    daily = precompute_daily_universe(feats)

    price_df = feats.pivot_table(index="date", columns="symbol", values="close").sort_index()
    returns_df = price_df.pct_change(fill_method=None).fillna(0.0)
    regime_df = compute_market_regime(feats)

    param_grid = {
        "top_k": [5, 10, 15, 20],
        "invest_more_n": [3, 5, 10, 15],
        "pullback_min_pct": [0.03, 0.05, 0.07, 0.10],
        "pullback_max_pct": [0.20, 0.25, 0.30, 0.35],
        "stop_loss_pct": [0.06, 0.08, 0.10, 0.12],
        "trailing_stop_pct": [0.08, 0.10, 0.12, 0.15],
    }

    print("Running grid search + TimeSeriesSplit CV...")
    folds_df, agg_df = evaluate_params_cv(
        daily_universe=daily,
        price_df=price_df,
        returns_df=returns_df,
        regime_df=regime_df,
        param_grid=param_grid,
        n_splits=5,
    )
    if agg_df.empty:
        raise RuntimeError("No valid parameter combination found")

    best = agg_df.iloc[0]
    best_cfg = BacktestConfigV3(
        top_k=int(best["top_k"]),
        invest_more_n=int(best["invest_more_n"]),
        pullback_min_pct=float(best["pullback_min_pct"]),
        pullback_max_pct=float(best["pullback_max_pct"]),
        stop_loss_pct=float(best["stop_loss_pct"]),
        trailing_stop_pct=float(best["trailing_stop_pct"]),
    )

    print("Running final full-period backtest with best params...")
    full_res = run_backtest_v3(
        daily_universe=daily,
        price_df=price_df,
        returns_df=returns_df,
        cfg=best_cfg,
        regime_df=regime_df,
        date_start=pd.to_datetime(start_date),
        date_end=daily["date"].max(),
    )

    strategy_eq = full_res["equity_curve"]
    strategy_metrics = summarize_metrics(strategy_eq[["date", "equity", "daily_return"]], best_cfg.initial_capital)

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(daily["date"].max())

    hs300_eq = run_benchmark_hs300(base_dir, start=start, end=end, initial_capital=best_cfg.initial_capital)
    hs300_metrics = summarize_metrics(hs300_eq, best_cfg.initial_capital)

    buy_hold_eq = run_benchmark_buy_hold(price_df, start=start, end=end, initial_capital=best_cfg.initial_capital)
    buy_hold_metrics = summarize_metrics(buy_hold_eq, best_cfg.initial_capital)

    print("Running sensitivity analysis...")
    importance_df, profile_df = sensitivity_analysis(agg_df)

    out_dir = base_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_png = out_dir / "sensitivity_analysis.png"
    plot_sensitivity(profile_df, sensitivity_png)

    tuning_md_path = out_dir / "parameter_tuning_report.md"
    compare_md_path = out_dir / "backtest_comparison_report.md"
    compare_html_path = out_dir / "backtest_comparison_report.html"
    strategy_doc_path = out_dir / "strategy_v3_optimized_doc.md"
    optimized_cfg_path = base_dir / "config_optimized.yaml"

    save_reports(
        base_dir=base_dir,
        cfg_best=best_cfg,
        folds_df=folds_df,
        agg_df=agg_df,
        importance_df=importance_df,
        strategy_eq=strategy_eq,
        strategy_metrics=strategy_metrics,
        hs300_metrics=hs300_metrics,
        buy_hold_metrics=buy_hold_metrics,
        sensitivity_png=sensitivity_png,
        compare_html_path=compare_html_path,
        compare_md_path=compare_md_path,
        tuning_md_path=tuning_md_path,
        strategy_doc_path=strategy_doc_path,
        optimized_cfg_path=optimized_cfg_path,
    )

    cv_detail_path = out_dir / "cv_fold_results.csv"
    cv_summary_path = out_dir / "cv_param_summary.csv"
    folds_df.to_csv(cv_detail_path, index=False, encoding="utf-8-sig")
    agg_df.to_csv(cv_summary_path, index=False, encoding="utf-8-sig")

    return {
        "best_cfg": best_cfg,
        "strategy_metrics": strategy_metrics,
        "hs300_metrics": hs300_metrics,
        "buy_hold_metrics": buy_hold_metrics,
        "files": {
            "optimized_code": str(base_dir / "run_backtest_strategy_v3.py"),
            "optimized_config": str(optimized_cfg_path),
            "tuning_report": str(tuning_md_path),
            "sensitivity_png": str(sensitivity_png),
            "comparison_md": str(compare_md_path),
            "comparison_html": str(compare_html_path),
            "strategy_doc": str(strategy_doc_path),
            "cv_detail": str(cv_detail_path),
            "cv_summary": str(cv_summary_path),
        },
    }


def _load_cfg_from_yaml(base_dir: Path) -> BacktestConfigV3:
    """从 config_v31.yaml / config.yaml 读取参数构建 BacktestConfigV3"""
    for name in ["config_v31.yaml", "config_optimized.yaml", "config.yaml"]:
        p = base_dir / name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            print(f"  ✓ 配置: {name}")
            break
    else:
        print("  ⚠ 未找到配置文件，使用默认参数")
        return BacktestConfigV3()

    s = raw.get("strategy", {})
    r = raw.get("risk_control", {})
    b = raw.get("backtest", {})
    ind = s.get("industry_diversification", {})
    mc = s.get("market_cap_filter", {})
    liq = s.get("liquidity_filter", {})
    corr = s.get("correlation_control", {})

    return BacktestConfigV3(
        top_k=int(s.get("top_k", 15)),
        invest_more_n=int(s.get("invest_more_n", 15)),
        pullback_window_start=int(s.get("pullback_window_start", 3)),
        pullback_window_end=int(s.get("pullback_window_end", 10)),
        pullback_min_pct=float(s.get("pullback_min_pct", 0.10)),
        pullback_max_pct=float(s.get("pullback_max_pct", 0.30)),
        tdx_min_score=float(s.get("tdx_min_score", 1.5)),
        stop_loss_pct=float(r.get("stop_loss_pct", 0.06)),
        trailing_stop_pct=float(r.get("trailing_stop_pct", 0.08)),
        max_hold_days=int(r.get("max_hold_days", 15)),
        initial_capital=float(b.get("initial_capital", 100000.0)),
        max_total_position=float(s.get("max_total_position", 0.80)),
        max_single_weight=float(s.get("max_single_weight", 0.15)),
        cost_bps=float(b.get("cost_bps", 15.0)),
        use_market_regime=s.get("use_market_regime", True),
        use_tradeability_filter=s.get("use_tradeability_filter", True),
        use_industry_diversification=ind.get("enabled", True),
        max_per_industry=int(ind.get("max_per_industry", 2)),
        use_market_cap_filter=mc.get("enabled", True),
        min_float_mkt_cap=float(mc.get("min_float_mkt_cap", 8e9)),
        max_float_mkt_cap=float(mc.get("max_float_mkt_cap", 8e11)),
        use_liquidity_filter=liq.get("enabled", True),
        min_amount_20d=float(liq.get("min_amount_20d", 6e7)),
        min_turnover_20d=float(liq.get("min_turnover_20d", 0.6)),
        use_correlation_control=corr.get("enabled", True),
        corr_lookback_days=int(corr.get("corr_lookback_days", 60)),
        max_pairwise_corr=float(corr.get("max_pairwise_corr", 0.75)),
        entry_mode=str(s.get("entry_mode", "normal")),
        use_tdx_protection=bool(r.get("use_tdx_protection", True)),
        tdx_protection_threshold=float(r.get("tdx_protection_threshold", 2.0)),
        # 新增参数（使用默认值即可，配置文件中可选覆盖）
        use_index_filter=bool(r.get("use_index_filter", True)),
        index_ma_period=int(r.get("index_ma_period", 60)),
        index_ma_short=int(r.get("index_ma_short", 20)),
        normal_min_conditions=int(s.get("normal_min_conditions", 2)),
        use_signal_tiered_sizing=bool(s.get("use_signal_tiered_sizing", True)),
        use_atr_stop=bool(r.get("use_atr_stop", True)),
        atr_stop_multiplier=float(r.get("atr_stop_multiplier", 1.5)),
        use_failure_stop=bool(r.get("use_failure_stop", True)),
        failure_stop_days=int(r.get("failure_stop_days", 2)),
        failure_stop_gain=float(r.get("failure_stop_gain", 0.03)),
        use_profit_pyramiding=bool(s.get("use_profit_pyramiding", True)),
        pyramid_trigger_pct=float(s.get("pyramid_trigger_pct", 0.05)),
    )


def run_quick_backtest(
    base_dir: Path,
    start_date: str,
    watchlist_file: str = None,
    use_dynamic_watchlist: bool = False,
    dynamic_top_n: int = 200,
) -> None:
    """快速回测：读取配置文件参数，直接跑一次全区间回测并输出结果"""
    print("=" * 60)
    print("📊 策略 V3 回测")
    print("=" * 60)

    print("\n[1/6] 加载配置...")
    cfg = _load_cfg_from_yaml(base_dir)
    print(f"  入场模式: {cfg.entry_mode} (附加条件≥{cfg.normal_min_conditions})")
    print(f"  止损: {cfg.stop_loss_pct*100:.0f}% | 止盈: {cfg.trailing_stop_pct*100:.0f}% | 时间: {cfg.max_hold_days}天")
    print(f"  ATR止损: {'开' if cfg.use_atr_stop else '关'} ({cfg.atr_stop_multiplier}x)")
    print(f"  失败形态止损: {'开' if cfg.use_failure_stop else '关'} ({cfg.failure_stop_days}天内涨幅<{cfg.failure_stop_gain*100:.0f}%)")
    print(f"  指数风控: {'开' if cfg.use_index_filter else '关'} (MA{cfg.index_ma_period}+MA{cfg.index_ma_short})")
    print(f"  信号分级仓位: {'开' if cfg.use_signal_tiered_sizing else '关'}")
    print(f"  盈利加仓: {'开' if cfg.use_profit_pyramiding else '关'} (>{cfg.pyramid_trigger_pct*100:.0f}%加{cfg.pyramid_add_ratio*100:.0f}%)")
    print(f"  TDX保护: {'开' if cfg.use_tdx_protection else '关'}")
    if use_dynamic_watchlist:
        print(f"  动态股票池: 开 (按月重平衡, Top {dynamic_top_n})")

    print("\n[2/6] 加载数据...")
    feats = load_and_prepare_features(base_dir, start_date=start_date)

    # ── 股票池过滤 ──
    if watchlist_file:
        wl_path = base_dir / watchlist_file
        if wl_path.exists():
            import re as _re
            wl_text = wl_path.read_text(encoding="utf-8")

            # 尝试两种格式：
            # 1) yaml列表格式: - "600519" 或 - 600519
            # 2) watchlist_cache.yaml格式: ['601198', '600015', ...]
            raw_codes = _re.findall(r'(\d{6})', wl_text)

            # 也尝试直接用yaml解析（watchlist_cache.yaml格式）
            if not raw_codes:
                try:
                    wl_data = yaml.safe_load(wl_text) or {}
                    wl_list = wl_data.get("watchlist", [])
                    raw_codes = [str(c).split(".")[0] for c in wl_list if str(c).strip()]
                except Exception:
                    pass

            # 去重
            raw_codes = list(set(raw_codes))

            # 转换为带后缀格式
            wl_symbols = set()
            for code in raw_codes:
                code = code.zfill(6)
                if code.startswith(("6", "5")):
                    wl_symbols.add(f"{code}.SH")
                else:
                    wl_symbols.add(f"{code}.SZ")

            # 过滤掉指数代码（399xxx等）
            wl_symbols = {s for s in wl_symbols if not s.startswith("399")}

            before = feats["symbol"].nunique()
            feats = feats[feats["symbol"].isin(wl_symbols)].copy()
            after = feats["symbol"].nunique()
            print(f"  🎯 回测股票池: {watchlist_file} ({after}/{before} 只，池子{len(wl_symbols)}只)")
        else:
            print(f"  ⚠️ 股票池文件不存在: {wl_path}，使用全部股票")
    industry_map = build_industry_map_from_config(base_dir, feats["symbol"])
    feats["industry"] = feats["symbol"].map(industry_map).fillna("OTHER")
    print(f"  特征: {len(feats)} 行, {feats['symbol'].nunique()} 只, {feats['date'].min().date()} → {feats['date'].max().date()}")

    print("\n[3/6] 预计算...")
    daily = precompute_daily_universe(feats)
    dynamic_watchlist_map: dict[pd.Timestamp, set[str]] | None = None
    if use_dynamic_watchlist:
        dynamic_watchlist_map = build_dynamic_watchlist_by_date(
            daily_universe=daily,
            max_symbols=dynamic_top_n,
            rebalance_freq="M",
        )
        if dynamic_watchlist_map:
            sample_date = sorted(dynamic_watchlist_map.keys())[0]
            sample_n = len(dynamic_watchlist_map[sample_date])
            print(f"  🎯 动态池样例: {str(sample_date)[:10]} 有 {sample_n} 只")
    price_df = feats.pivot_table(index="date", columns="symbol", values="close").sort_index()
    returns_df = price_df.pct_change(fill_method=None).fillna(0.0)
    regime_df = compute_market_regime(feats)

    print("\n[4/6] 指数风控...")
    index_filter = compute_index_filter(base_dir, start_date, cfg)

    print("\n[5/6] 运行回测...")
    # 诊断：检查入场条件能筛出多少股票
    if cfg.entry_mode == "custom":
        sample_dates = sorted(daily["date"].unique())
        check_dates = sample_dates[::60][:5]  # 每60天取一个样本
        print(f"  [诊断] custom模式各条件命中率（抽样{len(check_dates)}天）:")
        for cd in check_dates:
            dd = daily[daily["date"] == cd]
            if dd.empty:
                continue
            n = len(dd)
            c1 = (dd["has_limit_up_30d_calc"].fillna(0) >= 1).sum()
            c2 = (dd["main_force_pct"].fillna(0) > 0.005).sum()
            c3 = (dd["high30_new_high"].fillna(0) == 1).sum()
            c4 = ((dd["pullback_pct"].fillna(0) >= cfg.pullback_min_pct) & (dd["pullback_pct"].fillna(0) <= cfg.pullback_max_pct)).sum()
            c5 = (dd["tdx_score"].fillna(0) >= cfg.tdx_min_score).sum()
            mask = _candidate_entry_mask(dd, cfg)
            call = mask.sum()
            print(f"    {str(cd)[:10]}: 总{n}只 | 涨停30d={c1} | 主力>{0.5}%={c2} | 高30新高={c3} | 回调OK={c4} | TDX≥{cfg.tdx_min_score}={c5} | 全部通过={call}")

    result = run_backtest_v3(
        daily_universe=daily,
        price_df=price_df,
        returns_df=returns_df,
        cfg=cfg,
        regime_df=regime_df,
        date_start=pd.to_datetime(start_date),
        date_end=daily["date"].max(),
        index_filter=index_filter,
        allowed_symbols_by_date=dynamic_watchlist_map,
    )

    eq = result["equity_curve"]
    m = result["metrics"]
    trades = result.get("trade_log", [])

    print(f"\n  年化收益: {m['annual_return_pct']:+.2f}%")
    print(f"  最大回撤: {m['max_drawdown_pct']:.2f}%")
    print(f"  夏普比率: {m['sharpe']:.2f}")
    print(f"  胜    率: {m['win_rate_pct']:.1f}%")

    if trades:
        tdf = pd.DataFrame(trades)
        n_trades = len(tdf)
        n_symbols = tdf["symbol"].nunique()
        wins = (tdf["pnl_pct"] > 0).sum()
        avg_pnl = tdf["pnl_pct"].mean()
        print(f"  交易次数: {n_trades} | 涉及股票: {n_symbols}只 | 交易胜率: {wins/n_trades*100:.1f}% | 平均盈亏: {avg_pnl:+.2f}%")

    if not eq.empty:
        has_trades = (eq["n_holdings"] > 0).sum()
        print(f"  有持仓天数: {has_trades} / {len(eq)}")

    print("\n[6/6] 保存结果...")
    eq_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_equity.csv"
    eq_out.parent.mkdir(parents=True, exist_ok=True)
    eq.to_csv(eq_out, index=False, encoding="utf-8-sig")

    # 保存交易记录
    if trades:
        trades_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_trades.csv"
        pd.DataFrame(trades).to_csv(trades_out, index=False, encoding="utf-8-sig")
        print(f"  ✓ 交易记录: {trades_out} ({len(trades)}条)")

    stats_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_stats.json"
    stats_out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"✅ 回测完成!")
    print(f"  📂 {eq_out}")
    print(f"  📅 {start_date} → {eq['date'].max() if not eq.empty else 'N/A'}")
    print(f"  📈 年化 {m['annual_return_pct']:+.2f}% | 回撤 {m['max_drawdown_pct']:.2f}% | 夏普 {m['sharpe']:.2f}")
    print(f"{'=' * 60}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Strategy V3 backtest")
    ap.add_argument("--base-dir", default=".", help="project root")
    ap.add_argument("--start-date", default=None, help="backtest start date (default: from config)")
    ap.add_argument("--entry-mode", default=None, choices=["custom", "strict", "normal", "loose"],
                     help="override entry mode")
    ap.add_argument("--optimize", action="store_true", help="run full grid-search optimization")
    ap.add_argument("--watchlist", default=None,
                     help="指定回测股票池文件 (如 backtest_watchlist.yaml)，不指定则用全部股票")
    ap.add_argument("--dynamic-watchlist", action="store_true",
                    help="启用动态股票池（按月重平衡），避免用当下静态池回测全历史")
    ap.add_argument("--dynamic-top-n", type=int, default=200,
                    help="动态股票池每期保留数量（默认200）")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()

    # 确定起始日期：命令行 > 配置文件 > 默认值
    if args.start_date:
        start_date = args.start_date
    else:
        for name in ["config_v31.yaml", "config_optimized.yaml", "config.yaml"]:
            p = base_dir / name
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                sd = raw.get("backtest", {}).get("start_date") or raw.get("market_data", {}).get("start_date")
                if sd:
                    start_date = str(sd)
                    break
        else:
            start_date = "2020-01-01"

    if args.optimize:
        result = run_optimization_pipeline(base_dir, start_date=start_date)
        best_cfg: BacktestConfigV3 = result["best_cfg"]
        m = result["strategy_metrics"]
        print("\n=== Best Parameters ===")
        print(yaml.safe_dump(asdict(best_cfg), sort_keys=False, allow_unicode=True))
        print("=== Final Metrics (Optimized Strategy) ===")
        print(f"Annual Return: {m['annual_return_pct']:.2f}%")
        print(f"Max Drawdown: {m['max_drawdown_pct']:.2f}%")
        print(f"Sharpe: {m['sharpe']:.2f}")
        print(f"Win Rate: {m['win_rate_pct']:.2f}%")
        print("\n=== Output Files ===")
        for k, v in result["files"].items():
            print(f"{k}: {v}")
    else:
        wl_file = args.watchlist
        if wl_file is None:
            print("  📋 未指定 --watchlist，默认使用全股票池（更接近真实历史回测）")
        run_quick_backtest(
            base_dir,
            start_date=start_date,
            watchlist_file=wl_file,
            use_dynamic_watchlist=bool(args.dynamic_watchlist),
            dynamic_top_n=int(args.dynamic_top_n),
        )


if __name__ == "__main__":
    main()
