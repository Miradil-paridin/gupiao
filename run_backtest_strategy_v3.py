from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit


def _enable_windows_utf8_console() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except Exception:
        pass


# Avoid Windows console encoding failures from legacy code-page defaults.
_enable_windows_utf8_console()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


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
    rebalance_band: float = 0.0
    use_dynamic_rebalance_band: bool = False
    dynamic_rebalance_band_sensitivity: float = 0.8
    dynamic_rebalance_band_signal_ref: float = 1.0
    dynamic_rebalance_band_cost_ref_bps: float = 20.0
    dynamic_rebalance_band_min: float = 0.0
    dynamic_rebalance_band_max: float = 0.05
    use_volatility_sizing: bool = False
    volatility_lookback_days: int = 20
    volatility_target_annual: float = 0.22
    volatility_floor_annual: float = 0.08
    volatility_pos_mult_min: float = 0.65
    volatility_pos_mult_max: float = 1.15
    use_liquidity_state_sizing: bool = False
    liquidity_lookback_days: int = 60
    liquidity_pos_mult_min: float = 0.85
    liquidity_pos_mult_max: float = 1.05
    liquidity_pos_mult_sensitivity: float = 0.50
    use_light_risk_budget: bool = False
    risk_budget_mode: str = "inverse_atr"
    risk_budget_atr_floor: float = 0.005
    risk_budget_power: float = 1.0
    risk_budget_min_multiplier: float = 0.75
    risk_budget_max_multiplier: float = 1.35
    risk_budget_blend: float = 0.50
    cost_bps: float = 15.0
    use_cn_fee_schedule: bool = False
    commission_bps_buy: float = 15.0
    commission_bps_sell: float = 15.0
    stamp_duty_bps_sell: float = 5.0
    exchange_fee_bps: float = 0.341
    regulatory_fee_bps: float = 0.200
    transfer_fee_bps: float = 0.100
    execution_price_mode: str = "close"  # close | next_open | parallel
    use_execution_realism: bool = False
    max_participation_rate: float = 0.10
    execution_slippage_bps: float = 5.0
    execution_impact_bps: float = 20.0
    execution_impact_exponent: float = 0.7
    enforce_lot_rounding: bool = True
    lot_size: int = 100
    use_price_limit_constraints: bool = True
    main_board_limit_pct: float = 0.10
    st_board_limit_pct: float = 0.05
    chinext_board_limit_pct: float = 0.20
    star_board_limit_pct: float = 0.20
    bse_board_limit_pct: float = 0.30
    chinext_new_limit_free_days: int = 5
    star_new_limit_free_days: int = 5

    # product KPI targets (for pass/fail checks)
    target_annual_return_min_pct: float = 26.0
    target_annual_return_max_pct: float = 33.0
    target_max_drawdown_limit_pct: float = 18.0
    target_sharpe_min: float = 1.2

    # switches
    use_market_regime: bool = True
    use_tradeability_filter: bool = True

    # strategy enhancements
    use_industry_diversification: bool = True
    max_per_industry: int = 2
    max_sector_weight: float = 0.35  # 单板块最大仓位占比 35%

    use_market_cap_filter: bool = True
    min_float_mkt_cap: float = 8e9
    max_float_mkt_cap: float = 8e11

    use_liquidity_filter: bool = True
    min_amount_20d: float = 6e7
    min_turnover_20d: float = 0.6
    use_institution_holding_filter: bool = False
    institution_filter_mode: str = "data"  # data | proxy
    institution_data_col: str = "inst_holding_ratio"
    institution_holding_min_pct: float = 5.0
    institution_holding_quantile: float = 0.6
    institution_proxy_quantile: float = 0.6

    use_correlation_control: bool = True
    corr_lookback_days: int = 60
    max_pairwise_corr: float = 0.75

    # entry mode: "custom" / "strict" / "normal" / "loose"
    entry_mode: str = "normal"

    # TDX protection: relax stop-loss for strong main-force stocks
    use_tdx_protection: bool = True
    tdx_protection_threshold: float = 2.0

    # 閳光偓閳光偓 閺傝1: 閹稿洦鏆熸搴㈠付瀵偓閸?閳光偓閳光偓
    use_index_filter: bool = True
    index_filter_hard_gate: bool = True
    index_filter_block_position_cap: float = 0.35
    index_ma_period: int = 60
    index_ma_short: int = 20
    index_crash_days: int = 3
    index_crash_threshold: float = -0.03
    index_pause_days: int = 5
    use_momentum_crash_protection: bool = False
    momentum_crash_lookback_days: int = 8
    momentum_crash_drop_threshold: float = -0.08
    momentum_rebound_lookback_days: int = 3
    momentum_rebound_threshold: float = 0.03
    momentum_crash_protection_days: int = 5
    momentum_crash_position_cap: float = 0.45
    use_rank_exit: bool = True
    rank_exit_rebalance_freq: str = "W"  # D/W/M
    rank_exit_min_hold_days: int = 5
    rank_exit_rank_buffer: int = 0
    rank_exit_only_when_trend_down: bool = False

    # 閳光偓閳光偓 閺傝2A: 閺呪偓姘崇熅瀵?2-of-3 閳光偓閳光偓
    normal_min_conditions: int = 2
    use_signal_tiered_sizing: bool = True
    tier_strong_multiplier: float = 1.2
    tier_normal_multiplier: float = 1.0
    tier_weak_multiplier: float = 0.5
    use_dual_layer_entry: bool = True
    use_regime_entry_gate: bool = True
    regime_gate_bull_min_strength: float = 0.5
    regime_gate_neutral_min_strength: float = 1.0
    regime_gate_bear_min_strength: float = 2.0
    dual_entry_strong_threshold: float = 2.0
    dual_entry_normal_threshold: float = 1.0
    weak_entry_mode: str = "observe"  # observe | micro
    weak_micro_max_new_positions: int = 1
    weak_micro_weight_multiplier: float = 0.35

    # alpha enhancement: low-correlation factors
    use_alpha_enhancement: bool = False
    alpha_industry_rs_weight: float = 0.35
    alpha_flow_persistence_weight: float = 0.45
    alpha_quality_weight: float = 0.30
    alpha_short_reversal_weight: float = 0.0
    alpha_turnover_reversal_weight: float = 0.0
    alpha_value_proxy_weight: float = 0.0
    use_news_sentiment_factor: bool = False
    news_sentiment_weight: float = 0.10
    news_sentiment_min_items: int = 3
    news_sentiment_lag_days: int = 1

    # cross-sectional score preprocessing (robust scaling / neutralization)
    use_robust_score_norm: bool = True
    score_winsor_quantile: float = 0.02
    score_neutralize_industry: bool = True
    score_neutralize_size: bool = True
    score_size_col: str = "float_mkt_cap_20d"
    score_neutralize_beta: bool = False
    score_beta_col: str = "beta_60d"

    # weak-signal de-risk: do not force full position on low-conviction days
    use_weak_signal_de_risk: bool = True
    weak_signal_threshold: float = 1.0
    weak_signal_cap_multiplier: float = 0.75

    # dynamic max_total_position by market regime
    use_dynamic_regime_position: bool = True
    regime_bull_pos_cap: float = 0.95
    regime_neutral_pos_cap: float = 0.70
    regime_bear_pos_cap: float = 0.35
    block_new_in_bear_regime: bool = False

    # 閳光偓閳光偓 閺傝3: 闁偓閸戦缚閸掓瑥宕岀痪?閳光偓閳光偓
    use_atr_stop: bool = True
    atr_stop_multiplier: float = 1.5   # 濮濄垺宕?= 閸忋儱婧€娴?- N * ATR

    use_failure_stop: bool = True
    failure_stop_days: int = 2
    failure_stop_gain: float = 0.03
    failure_stop_require_negative_pnl: bool = False
    failure_stop_negative_pnl_threshold: float = 0.0
    failure_stop_weak_signal_only: bool = False
    failure_stop_max_signal_strength: float = 1.0
    failure_stop_skip_if_still_target: bool = False
    failure_stop_skip_if_trend_up: bool = False
    use_portfolio_drawdown_brake: bool = True
    drawdown_brake_threshold: float = 0.12
    drawdown_brake_pause_days: int = 8
    drawdown_brake_position_cap: float = 0.35
    use_adaptive_drawdown_mode: bool = True
    adaptive_drawdown_trigger: float = 0.04
    adaptive_drawdown_position_cap_multiplier: float = 0.80
    adaptive_drawdown_gate_boost: float = 0.50
    adaptive_drawdown_top_k_multiplier: float = 0.70
    adaptive_drawdown_invest_more_multiplier: float = 0.80
    adaptive_drawdown_stop_loss_multiplier: float = 0.90
    adaptive_drawdown_trailing_multiplier: float = 0.85
    adaptive_drawdown_max_hold_multiplier: float = 0.80

    # 娴兼ê瀵查弮鍫曟？濮濄垺宕?
    time_stop_min_gain: float = 0.0    # 閹镐椒绮ㄩ崚鐗堟埂閺冭绱濋懛鍐茬毌鐟曚焦婀佸銈嗗畾楠炲懏澧犳稉宥堥惍?

    # 閳光偓閳光偓 閺傝4: 閻╁牆鍩勯崝鐘辩波 閳光偓閳光偓
    # MA 止损
    use_ma_stop: bool = True
    ma_stop_days: int = 2
    ma_trailing_priority: str = "trailing"  # trailing | ma

    # 止盈 (take profit)
    use_take_profit: bool = True
    take_profit_atr_multiplier: float = 3.0
    take_profit_fixed_pct: float = 0.20
    use_staged_take_profit: bool = False
    staged_tp_levels: tuple = ((0.08, 0.3), (0.15, 0.3), (0.25, 0.4))

    use_profit_pyramiding: bool = True
    pyramid_trigger_pct: float = 0.05  # 濞存畅鐡掑懓绻?%鐟欙箑褰傞崝鐘辩波
    pyramid_add_ratio: float = 0.5     # 閸旂姳绮ㄩ柌?= 閸樼喍绮ㄦ担?* 0.5
    pyramid_max_adds: int = 1          # 閺堚偓婢舵艾濮炴禒鎾撮弫?


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
    s = pd.to_numeric(x, errors="coerce")
    if int(s.notna().sum()) < 3:
        return pd.Series(0.0, index=s.index)
    mu = s.mean()
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def _winsorize_series(x: pd.Series, quantile: float = 0.02) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    q = float(np.clip(quantile, 0.0, 0.20))
    if q <= 0:
        return s
    if int(s.notna().sum()) < 3:
        return s
    lo = s.quantile(q)
    hi = s.quantile(1.0 - q)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return s
    return s.clip(lower=float(lo), upper=float(hi))


def _robust_zscore(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    if int(s.notna().sum()) < 3:
        return pd.Series(0.0, index=s.index)
    med = float(s.median())
    mad = float((s - med).abs().median())
    if not np.isfinite(mad) or mad < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - med) / (1.4826 * mad)


def _winsorized_zscore(x: pd.Series, quantile: float = 0.02, robust: bool = True) -> pd.Series:
    s = _winsorize_series(x, quantile=quantile)
    return _robust_zscore(s) if robust else _safe_zscore(s)


def _residualize_cross_section(y: pd.Series, exposures: pd.DataFrame) -> pd.Series:
    out = pd.to_numeric(y, errors="coerce").copy()
    if exposures is None or exposures.empty:
        return out.fillna(0.0)

    xx = exposures.copy()
    for c in xx.columns:
        xx[c] = pd.to_numeric(xx[c], errors="coerce")
    xx = xx.replace([np.inf, -np.inf], np.nan)
    xx = xx.loc[:, xx.notna().any(axis=0)]
    if xx.empty:
        return out.fillna(0.0)

    # drop near-constant exposures to reduce rank issues
    keep_cols = []
    for c in xx.columns:
        col = xx[c]
        if float(col.std(ddof=1)) > 1e-12:
            keep_cols.append(c)
    xx = xx[keep_cols] if keep_cols else xx.iloc[:, :0]
    if xx.empty:
        return out.fillna(0.0)

    valid = out.notna() & xx.notna().all(axis=1)
    if int(valid.sum()) < max(8, xx.shape[1] + 2):
        return (out - out.mean()).fillna(0.0)

    yv = out.loc[valid].astype(float).values
    xv = xx.loc[valid].astype(float).values
    x_design = np.column_stack([np.ones(len(yv)), xv])
    try:
        beta, *_ = np.linalg.lstsq(x_design, yv, rcond=None)
        fitted = x_design @ beta
        resid = yv - fitted
        out2 = pd.Series(0.0, index=out.index, dtype=float)
        out2.loc[valid] = resid
        return out2.fillna(0.0)
    except Exception:
        return (out - out.mean()).fillna(0.0)


def _compute_alpha_factors(day: pd.DataFrame) -> pd.DataFrame:
    """
    娴ｅ海娴夐崗?alpha 閸ョ姴鐡欓敍鍫熋幋娼伴敍澶涚窗
    1) 鐞涘奔绗熼惄绋垮鍝勬€?
    2) 鐠у嫰鍣鹃幐浣虹敾閹?    3) 鐠愩劑鍣洪崶鐘茬摍閿涘牓闂勨晞鐨熼弫鏉戞倵閻ㄥ嫯绉奸崝鑳窛闁插骏绱?
    4) 閻埂閸欏秷娴嗛敍鍦玊R閿?    5) 閹广垺澧滈崣宥堟祮娴溿倓绨伴敍鍫ョ彯閹广垺澧滄稉瀣畱閸欏秷娴嗗瑙勨偓褝绱?
    6) 娴犲嘲鈧棿鍞悶鍡礄E/P娴兼ê鍘涢敍宀€宸辨径杈ㄦ闁偓閸栨牔璐熺拹銊╁櫤閸╃儤婀伴棃鍞悶鍡礆
    """
    x = day.copy()
    idx = x.index

    if "industry" not in x.columns:
        x["industry"] = "OTHER"

    ret1 = pd.to_numeric(x.get("ret_1d", pd.Series(0.0, index=idx)), errors="coerce").fillna(0.0)
    ret5 = pd.to_numeric(x.get("ret_5d", pd.Series(0.0, index=idx)), errors="coerce").fillna(0.0)
    ret20 = pd.to_numeric(x.get("ret_20d", pd.Series(0.0, index=idx)), errors="coerce").fillna(0.0)
    ret60 = pd.to_numeric(x.get("ret_60d", pd.Series(0.0, index=idx)), errors="coerce").fillna(0.0)
    ma_dist_20 = pd.to_numeric(x.get("ma_dist_20", pd.Series(0.0, index=idx)), errors="coerce").fillna(0.0)
    vol20 = pd.to_numeric(x.get("vol_20d", pd.Series(0.0, index=idx)), errors="coerce").fillna(0.0)
    atr_pct = pd.to_numeric(x.get("atr_pct", pd.Series(0.01, index=idx)), errors="coerce").fillna(0.01)
    turnover_20d = pd.to_numeric(x.get("turnover_20d", pd.Series(np.nan, index=idx)), errors="coerce")
    min_cov = max(10, int(len(x) * 0.1))
    if turnover_20d.notna().sum() < min_cov:
        turnover_20d = pd.to_numeric(x.get("turnover", pd.Series(np.nan, index=idx)), errors="coerce")
    if turnover_20d.notna().sum() < min_cov:
        turnover_20d = pd.to_numeric(x.get("vol_ratio_20", pd.Series(1.0, index=idx)), errors="coerce")
    turnover_20d = turnover_20d.fillna(turnover_20d.median() if turnover_20d.notna().any() else 1.0)

    # 1) 鐞涘奔绗熼惄绋垮鍝勬€ラ敍姘虫稉姘繁鎼?+ 娑撳亗閻╃鐞涘奔绗熺搾鍛?
    ind_ret20 = x.groupby("industry")["ret_20d"].transform("mean") if "ret_20d" in x.columns else pd.Series(0.0, index=idx)
    ind_ret20 = pd.to_numeric(ind_ret20, errors="coerce").fillna(0.0)
    alpha_industry_rs = 0.7 * _safe_zscore(ind_ret20) + 0.3 * _safe_zscore(ret20 - ind_ret20)

    # flow persistence proxy
    main_force_ctrl = pd.to_numeric(x.get("main_force_control", pd.Series(0.0, index=idx)), errors="coerce").fillna(0.0).clip(-3.0, 3.0)
    main_force_strong = (pd.to_numeric(x.get("main_force_strong", pd.Series(0.0, index=idx)), errors="coerce").fillna(0.0) > 0).astype(float)
    recent_limit_up = (pd.to_numeric(x.get("has_limit_up_30d", pd.Series(0.0, index=idx)), errors="coerce").fillna(0.0) > 0).astype(float)
    flow_raw = _safe_zscore(main_force_ctrl) + 0.7 * main_force_strong + 0.3 * recent_limit_up
    alpha_flow_persistence = _safe_zscore(flow_raw)

    # 3) 鐠愩劑鍣洪崶鐘茬摍閿涙矮鑵戦張鐔告暪閻?+ 鐡掑濞嶆稉鈧懛瀛樷偓?+ 娴ｅ孩灏濈粙鍐蹭淮
    quality_raw = (
        0.45 * _safe_zscore(ret60)
        + 0.35 * _safe_zscore(ma_dist_20)
        + 0.20 * (-_safe_zscore(vol20) - 0.5 * _safe_zscore(atr_pct))
    )
    alpha_quality = _safe_zscore(quality_raw)

    # short-term reversal (STR)
    str_raw = -0.65 * _safe_zscore(ret5) - 0.35 * _safe_zscore(ret1)
    alpha_short_reversal = _safe_zscore(str_raw)

    # turnover reversal interaction
    turnover_reversal_raw = _safe_zscore(turnover_20d) * _safe_zscore(-ret5)
    alpha_turnover_reversal = _safe_zscore(turnover_reversal_raw)

    # 6) 娴犲嘲鈧棿鍞悶鍡窗娴兼ê鍘?E/P閿涙稓宸辨径杈ㄦ闁偓閸栨牔璐?ROE/CFO/YOY 缂佸嫬鎮?
    ep = pd.to_numeric(x.get("earnings_yield", pd.Series(np.nan, index=idx)), errors="coerce")
    if ep.notna().sum() < min_cov:
        pe = pd.to_numeric(x.get("pe_ttm", pd.Series(np.nan, index=idx)), errors="coerce")
        ep = (1.0 / pe.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if ep.notna().sum() >= min_cov:
        alpha_value_proxy = _safe_zscore(ep.fillna(ep.median()))
    else:
        roe = pd.to_numeric(x.get("roe_latest", pd.Series(np.nan, index=idx)), errors="coerce")
        yoy = pd.to_numeric(x.get("yoyni", pd.Series(np.nan, index=idx)), errors="coerce")
        cfo = pd.to_numeric(x.get("cfo_to_np", pd.Series(np.nan, index=idx)), errors="coerce")
        if roe.notna().sum() >= min_cov:
            value_raw = (
                0.50 * _safe_zscore(roe.fillna(roe.median()))
                + 0.30 * _safe_zscore(cfo.fillna(cfo.median() if cfo.notna().any() else 0.0))
                + 0.20 * _safe_zscore(yoy.fillna(yoy.median() if yoy.notna().any() else 0.0).clip(-100, 500))
            )
            alpha_value_proxy = _safe_zscore(value_raw)
        else:
            alpha_value_proxy = pd.Series(0.0, index=idx)

    return pd.DataFrame(
        {
            "alpha_industry_rs": alpha_industry_rs,
            "alpha_flow_persistence": alpha_flow_persistence,
            "alpha_quality": alpha_quality,
            "alpha_short_reversal": alpha_short_reversal,
            "alpha_turnover_reversal": alpha_turnover_reversal,
            "alpha_value_proxy": alpha_value_proxy,
        },
        index=idx,
    )


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


def evaluate_kpi_targets(metrics: dict[str, float], cfg: BacktestConfigV3) -> dict[str, Any]:
    annual = float(metrics.get("annual_return_pct", 0.0))
    max_dd = abs(float(metrics.get("max_drawdown_pct", 0.0)))
    sharpe = float(metrics.get("sharpe", 0.0))

    kpi = {
        "annual_range_ok": bool(cfg.target_annual_return_min_pct <= annual <= cfg.target_annual_return_max_pct),
        "max_drawdown_ok": bool(max_dd <= cfg.target_max_drawdown_limit_pct),
        "sharpe_ok": bool(sharpe >= cfg.target_sharpe_min),
    }
    kpi["all_ok"] = bool(kpi["annual_range_ok"] and kpi["max_drawdown_ok"] and kpi["sharpe_ok"])
    return kpi


def _pct_rank(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    if s.notna().sum() <= 1:
        return pd.Series(0.5, index=x.index, dtype=float)
    return s.rank(pct=True).fillna(0.5)


def _compute_rebalance_days(trading_days: list[pd.Timestamp], freq: str) -> set[pd.Timestamp]:
    if not trading_days:
        return set()

    f = str(freq or "D").strip().upper()
    if f in {"D", "DAILY"}:
        return set(pd.to_datetime(trading_days).tolist())

    days_df = pd.DataFrame({"date": pd.to_datetime(trading_days)})
    if f in {"W", "WEEKLY"}:
        days_df["period"] = days_df["date"].dt.to_period("W-FRI")
    elif f in {"M", "MONTHLY"}:
        days_df["period"] = days_df["date"].dt.to_period("M")
    else:
        return set(pd.to_datetime(trading_days).tolist())

    return set(pd.to_datetime(days_df.groupby("period")["date"].max().tolist()))


def _infer_board(sym: str) -> str:
    s = str(sym).upper().strip()
    code = s.split(".")[0]
    suffix = s.split(".")[1] if "." in s else ""
    if suffix == "BJ" or code.startswith(("4", "8")):
        return "BSE"
    if code.startswith("688"):
        return "STAR"
    if code.startswith(("300", "301")):
        return "CHINEXT"
    return "MAIN"


def _infer_is_st(row: pd.Series | None) -> bool:
    if row is None:
        return False
    for name_col in ["name", "code_name", "security_name", "stock_name"]:
        v = str(row.get(name_col, "")).upper()
        if "ST" in v:
            return True
    for flag_col in ["is_st", "st_flag", "risk_warning", "is_risk_warning"]:
        fv = pd.to_numeric(pd.Series([row.get(flag_col)]), errors="coerce").iloc[0]
        if pd.notna(fv) and float(fv) >= 1.0:
            return True
    return False


def _infer_days_since_listing(row: pd.Series | None) -> float | None:
    if row is None:
        return None
    for col in ["days_since_listing", "list_days", "listed_days"]:
        v = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        if pd.notna(v):
            return float(v)
    return None


def _infer_limit_pct(sym: str, row: pd.Series | None, cfg: BacktestConfigV3) -> float:
    board = _infer_board(sym)
    is_st = _infer_is_st(row)
    dsl = _infer_days_since_listing(row)
    if dsl is not None:
        if board == "STAR" and dsl < float(cfg.star_new_limit_free_days):
            return 9.99
        if board == "CHINEXT" and dsl < float(cfg.chinext_new_limit_free_days):
            return 9.99
    if is_st:
        return float(cfg.st_board_limit_pct)
    if board == "BSE":
        return float(cfg.bse_board_limit_pct)
    if board == "STAR":
        return float(cfg.star_board_limit_pct)
    if board == "CHINEXT":
        return float(cfg.chinext_board_limit_pct)
    return float(cfg.main_board_limit_pct)


def _limit_bounds(sym: str, row: pd.Series | None, cfg: BacktestConfigV3) -> tuple[float, float]:
    if row is not None:
        up = pd.to_numeric(pd.Series([row.get("limit_up_threshold")]), errors="coerce").iloc[0]
        dn = pd.to_numeric(pd.Series([row.get("limit_down_threshold")]), errors="coerce").iloc[0]
        if pd.notna(up) and pd.notna(dn) and float(up) > 0 and float(dn) > 0:
            return float(up), float(dn)
        prev_close = pd.to_numeric(pd.Series([row.get("prev_close")]), errors="coerce").iloc[0]
        if pd.notna(prev_close) and float(prev_close) > 0:
            lim = _infer_limit_pct(sym, row, cfg)
            return float(prev_close) * (1.0 + lim), float(prev_close) * (1.0 - lim)
    return float("nan"), float("nan")


def _is_trade_blocked_by_limit(
    side: str,
    sym: str,
    px: float,
    row: pd.Series | None,
    cfg: BacktestConfigV3,
) -> bool:
    if not cfg.use_execution_realism or not cfg.use_price_limit_constraints:
        return False
    if not np.isfinite(px) or px <= 0:
        return True

    side_u = str(side).upper()
    up, dn = _limit_bounds(sym, row, cfg)
    if row is not None:
        lu = pd.to_numeric(pd.Series([row.get("limit_up_flag")]), errors="coerce").fillna(0.0).iloc[0]
        ld = pd.to_numeric(pd.Series([row.get("limit_down_flag")]), errors="coerce").fillna(0.0).iloc[0]
        if side_u == "BUY" and float(lu) >= 1.0:
            return True
        if side_u == "SELL" and float(ld) >= 1.0:
            return True

    if side_u == "BUY" and np.isfinite(up):
        return bool(px >= up * 0.9995)
    if side_u == "SELL" and np.isfinite(dn):
        return bool(px <= dn * 1.0005)
    return False


def _apply_execution_order_constraints(
    desired_w: float,
    px: float,
    row: pd.Series | None,
    equity: float,
    cfg: BacktestConfigV3,
    side: str = "BUY",
) -> float:
    w = float(max(desired_w, 0.0))
    if w <= 0 or not np.isfinite(px) or px <= 0:
        return 0.0

    side_u = str(side).upper().strip()
    if cfg.use_execution_realism:
        if row is not None:
            amount = pd.to_numeric(pd.Series([row.get("amount")]), errors="coerce").iloc[0]
            if pd.notna(amount) and float(amount) > 0 and cfg.max_participation_rate > 0:
                cap_w = float(cfg.max_participation_rate) * float(amount) / max(float(equity), 1e-12)
                w = min(w, max(cap_w, 0.0))

        if cfg.enforce_lot_rounding and int(cfg.lot_size) > 0:
            step_w = int(cfg.lot_size) * float(px) / max(float(equity), 1e-12)
            if np.isfinite(step_w) and step_w > 0:
                rounded = np.floor(w / step_w) * step_w
                if side_u == "SELL" and rounded <= 0.0 < w:
                    # In a weight-based simulator, avoid immortal dust positions on sell.
                    rounded = w
                w = rounded
    return float(max(w, 0.0))


def _apply_execution_buy_constraints(
    desired_w: float,
    px: float,
    row: pd.Series | None,
    equity: float,
    cfg: BacktestConfigV3,
) -> float:
    """Backward-compatible buy-side wrapper."""
    return _apply_execution_order_constraints(
        desired_w=desired_w,
        px=px,
        row=row,
        equity=equity,
        cfg=cfg,
        side="BUY",
    )


def _execution_extra_cost_breakdown_from_turnover(
    prev_weights: dict[str, float],
    new_weights: dict[str, float],
    day_df: pd.DataFrame | None,
    equity: float,
    cfg: BacktestConfigV3,
) -> dict[str, float]:
    if not cfg.use_execution_realism:
        return {"slippage": 0.0, "impact": 0.0, "total": 0.0}

    amount_map: dict[str, float] = {}
    if day_df is not None and not day_df.empty and "symbol" in day_df.columns:
        for _, r in day_df.iterrows():
            s = str(r.get("symbol", ""))
            amt = pd.to_numeric(pd.Series([r.get("amount")]), errors="coerce").iloc[0]
            if s and pd.notna(amt) and float(amt) > 0:
                amount_map[s] = float(amt)

    slippage_total = 0.0
    impact_total = 0.0
    all_syms = set(prev_weights) | set(new_weights)
    for s in all_syms:
        dw = abs(float(new_weights.get(s, 0.0)) - float(prev_weights.get(s, 0.0)))
        if dw <= 0:
            continue
        slip = dw * (float(cfg.execution_slippage_bps) / 10000.0)
        amount = amount_map.get(s, np.nan)
        if np.isfinite(amount) and amount > 0:
            participation = (dw * float(equity)) / amount
            participation = max(float(participation), 0.0)
            impact_bps = float(cfg.execution_impact_bps) * (participation ** float(cfg.execution_impact_exponent))
        else:
            impact_bps = float(cfg.execution_impact_bps)
        impact = dw * (impact_bps / 10000.0)
        slippage_total += float(slip)
        impact_total += float(impact)
    total = float(max(slippage_total + impact_total, 0.0))
    return {
        "slippage": float(max(slippage_total, 0.0)),
        "impact": float(max(impact_total, 0.0)),
        "total": total,
    }


def _execution_extra_cost_from_turnover(
    prev_weights: dict[str, float],
    new_weights: dict[str, float],
    day_df: pd.DataFrame | None,
    equity: float,
    cfg: BacktestConfigV3,
) -> float:
    """Backward-compatible wrapper returning total execution extra cost."""
    info = _execution_extra_cost_breakdown_from_turnover(
        prev_weights=prev_weights,
        new_weights=new_weights,
        day_df=day_df,
        equity=equity,
        cfg=cfg,
    )
    return float(info.get("total", 0.0))


def _turnover_buy_sell(prev_weights: dict[str, float], new_weights: dict[str, float]) -> tuple[float, float]:
    buy_turnover = 0.0
    sell_turnover = 0.0
    all_syms = set(prev_weights) | set(new_weights)
    for s in all_syms:
        delta = float(new_weights.get(s, 0.0)) - float(prev_weights.get(s, 0.0))
        if delta > 0:
            buy_turnover += delta
        elif delta < 0:
            sell_turnover += -delta
    return float(max(buy_turnover, 0.0)), float(max(sell_turnover, 0.0))


def _explicit_cost_breakdown_from_turnover(
    buy_turnover: float,
    sell_turnover: float,
    cfg: BacktestConfigV3,
) -> dict[str, float]:
    bt = float(max(buy_turnover, 0.0))
    st = float(max(sell_turnover, 0.0))
    if not cfg.use_cn_fee_schedule:
        commission = float((bt + st) * (float(cfg.cost_bps) / 10000.0))
        return {
            "commission": commission,
            "stamp_duty": 0.0,
            "exchange_fee": 0.0,
            "regulatory_fee": 0.0,
            "transfer_fee": 0.0,
            "total": commission,
        }

    buy_comm = bt * (float(cfg.commission_bps_buy) / 10000.0)
    sell_comm = st * (float(cfg.commission_bps_sell) / 10000.0)
    exchange = (bt + st) * (float(cfg.exchange_fee_bps) / 10000.0)
    regulatory = (bt + st) * (float(cfg.regulatory_fee_bps) / 10000.0)
    transfer = (bt + st) * (float(cfg.transfer_fee_bps) / 10000.0)
    stamp = st * (float(cfg.stamp_duty_bps_sell) / 10000.0)
    total = buy_comm + sell_comm + exchange + regulatory + transfer + stamp
    return {
        "commission": float(max(buy_comm + sell_comm, 0.0)),
        "stamp_duty": float(max(stamp, 0.0)),
        "exchange_fee": float(max(exchange, 0.0)),
        "regulatory_fee": float(max(regulatory, 0.0)),
        "transfer_fee": float(max(transfer, 0.0)),
        "total": float(max(total, 0.0)),
    }


def _dynamic_rebalance_band(
    base_band: float,
    avg_signal_strength: float,
    cfg: BacktestConfigV3,
) -> float:
    band = float(max(0.0, base_band))
    if band <= 0 or (not bool(cfg.use_dynamic_rebalance_band)):
        return band

    signal_ref = float(max(cfg.dynamic_rebalance_band_signal_ref, 0.25))
    signal = float(max(avg_signal_strength, 0.25))
    signal_term = signal_ref / signal

    static_cost_bps = float(cfg.cost_bps)
    if cfg.use_cn_fee_schedule:
        # Cost baseline for one roundtrip turnover unit (buy+sell weights).
        static_cost_bps = float(
            0.5
            * (
                cfg.commission_bps_buy
                + cfg.commission_bps_sell
                + 2.0 * (cfg.exchange_fee_bps + cfg.regulatory_fee_bps + cfg.transfer_fee_bps)
                + cfg.stamp_duty_bps_sell
            )
        )
    exec_proxy_bps = 0.0
    if cfg.use_execution_realism:
        exec_proxy_bps = float(cfg.execution_slippage_bps) + float(cfg.execution_impact_bps) * float(
            max(0.0, min(1.0, cfg.max_participation_rate))
        )

    cost_proxy_bps = float(max(static_cost_bps + exec_proxy_bps, 1e-6))
    cost_ref_bps = float(max(cfg.dynamic_rebalance_band_cost_ref_bps, 1e-6))
    cost_term = cost_proxy_bps / cost_ref_bps

    sens = float(max(cfg.dynamic_rebalance_band_sensitivity, 0.0))
    multiplier = 1.0 + sens * (0.6 * (signal_term - 1.0) + 0.4 * (cost_term - 1.0))
    dyn = float(max(0.0, band * multiplier))

    lo = float(max(0.0, cfg.dynamic_rebalance_band_min))
    hi = float(max(lo, cfg.dynamic_rebalance_band_max))
    return float(min(hi, max(lo, dyn)))


def _apply_light_risk_budget(
    positions: dict[str, dict[str, float]],
    prev_day_symbol_idx: pd.DataFrame,
    cfg: BacktestConfigV3,
) -> int:
    """
    Lightweight risk-budget overlay:
    scale weights by inverse ATR risk proxy and keep existing portfolio constraints downstream.
    """
    if (not bool(cfg.use_light_risk_budget)) or len(positions) <= 1:
        return 0

    mode = str(cfg.risk_budget_mode).strip().lower()
    if mode not in {"inverse_atr", "atr_inverse"}:
        return 0

    atr_floor = float(max(1e-6, cfg.risk_budget_atr_floor))
    power = float(max(0.0, cfg.risk_budget_power))
    blend = float(np.clip(cfg.risk_budget_blend, 0.0, 1.0))
    min_mul = float(max(0.0, cfg.risk_budget_min_multiplier))
    max_mul = float(max(min_mul, cfg.risk_budget_max_multiplier))

    symbols = list(positions.keys())
    atr_map: dict[str, float] = {}
    valid_atr: list[float] = []
    for sym in symbols:
        atr_pct = float("nan")
        if (not prev_day_symbol_idx.empty) and (sym in prev_day_symbol_idx.index):
            row = prev_day_symbol_idx.loc[sym]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            atr_pct = float(pd.to_numeric(row.get("atr_pct", np.nan), errors="coerce"))
        if (not np.isfinite(atr_pct)) or atr_pct <= 0:
            pos = positions.get(sym, {})
            atr_val = float(pd.to_numeric(pos.get("atr_val", np.nan), errors="coerce"))
            entry_px = float(pd.to_numeric(pos.get("entry_price", np.nan), errors="coerce"))
            if np.isfinite(atr_val) and atr_val > 0 and np.isfinite(entry_px) and entry_px > 0:
                atr_pct = atr_val / entry_px
        if np.isfinite(atr_pct) and atr_pct > 0:
            atr_map[sym] = float(atr_pct)
            valid_atr.append(float(atr_pct))
        else:
            atr_map[sym] = float("nan")

    if len(valid_atr) < 2:
        return 0
    fallback_atr = float(np.median(valid_atr))
    if (not np.isfinite(fallback_atr)) or fallback_atr <= 0:
        fallback_atr = atr_floor

    raw_scales: dict[str, float] = {}
    for sym in symbols:
        atr_pct = float(atr_map.get(sym, fallback_atr))
        if (not np.isfinite(atr_pct)) or atr_pct <= 0:
            atr_pct = fallback_atr
        atr_pct = float(max(atr_floor, atr_pct))
        if power <= 0:
            raw_scale = 1.0
        else:
            raw_scale = float(1.0 / (atr_pct ** power))
        raw_scales[sym] = raw_scale

    avg_scale = float(np.mean(list(raw_scales.values()))) if raw_scales else 1.0
    if (not np.isfinite(avg_scale)) or avg_scale <= 0:
        return 0

    changed = 0
    for sym in symbols:
        pos = positions.get(sym)
        if pos is None:
            continue
        old_w = float(pos.get("weight", 0.0))
        base_scale = float(raw_scales.get(sym, avg_scale)) / avg_scale
        capped_scale = float(min(max_mul, max(min_mul, base_scale)))
        scale = float((1.0 - blend) + blend * capped_scale)
        new_w = float(min(cfg.max_single_weight, max(0.0, old_w * scale)))
        pos["weight"] = new_w
        if abs(new_w - old_w) > 1e-10:
            changed += 1
    return changed


def _apply_rebalance_band(
    prev_weights: dict[str, float],
    positions: dict[str, dict[str, float]],
    rebalance_band: float,
    locked_symbols: set[str] | None = None,
    max_total_position: float | None = None,
) -> int:
    band = max(0.0, float(rebalance_band))
    if band <= 0:
        return 0

    locked = locked_symbols or set()
    adjusted = 0

    # Suppress micro rebalances for non-forced symbols.
    all_syms = set(prev_weights) | set(positions)
    for sym in all_syms:
        if sym in locked:
            continue
        prev_w = float(prev_weights.get(sym, 0.0))
        new_w = float(positions.get(sym, {}).get("weight", 0.0))
        if prev_w <= 0.0 and new_w <= 0.0:
            continue
        if abs(new_w - prev_w) >= band:
            continue
        if prev_w <= 0.0:
            if sym in positions:
                positions.pop(sym, None)
                adjusted += 1
        else:
            if sym in positions:
                positions[sym]["weight"] = float(prev_w)
                adjusted += 1

    # Remove tiny dust positions after banding.
    tiny_eps = min(1e-6, band * 0.1 if band > 0 else 1e-6)
    for sym in list(positions.keys()):
        if sym in locked:
            continue
        if float(positions[sym].get("weight", 0.0)) <= tiny_eps:
            positions.pop(sym, None)
            adjusted += 1

    # Respect total-position cap after snapping.
    if max_total_position is not None:
        cap = float(max_total_position)
        total_w = float(sum(float(p.get("weight", 0.0)) for p in positions.values()))
        if cap > 0 and total_w > cap:
            scale = cap / total_w
            for p in positions.values():
                p["weight"] = float(p.get("weight", 0.0)) * scale

    return adjusted


def load_symbol_style_factors(base_dir: Path) -> pd.DataFrame:
    """
    鐠囪褰囬幐澶庡亗缁併劎娣惔锔炬畱妞嬪孩鐗?閸╃儤婀伴棃鍞悶鍡樻殶閹圭礄閸欌偓澶涚礆閵?    瑜版挸澧犳导妯哄帥閺夈儲绨? data/watchlist_report.csv
    """
    p = base_dir / "data" / "watchlist_report.csv"
    if not p.exists():
        return pd.DataFrame(columns=["symbol", "roe_latest", "yoyni", "cfo_to_np", "pe_ttm", "earnings_yield"])

    try:
        raw = pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=["symbol", "roe_latest", "yoyni", "cfo_to_np", "pe_ttm", "earnings_yield"])

    if "symbol" not in raw.columns:
        return pd.DataFrame(columns=["symbol", "roe_latest", "yoyni", "cfo_to_np", "pe_ttm", "earnings_yield"])

    out = pd.DataFrame()
    out["symbol"] = raw["symbol"].astype(str).map(_norm_symbol)
    for col in ["roe_latest", "yoyni", "cfo_to_np", "pe_ttm", "earnings_yield"]:
        if col in raw.columns:
            out[col] = pd.to_numeric(raw[col], errors="coerce")
        else:
            out[col] = np.nan

    out = out.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)
    return out


def load_institution_holding_factors(base_dir: Path) -> pd.DataFrame:
    candidates = [
        base_dir / "data" / "factors" / "institution_holding_daily.parquet",
        base_dir / "data" / "factors" / "institution_holding_daily.csv",
        base_dir / "data" / "institution_holding.parquet",
        base_dir / "data" / "institution_holding.csv",
    ]
    raw = pd.DataFrame()
    for p in candidates:
        if not p.exists():
            continue
        try:
            if p.suffix.lower() == ".parquet":
                raw = pd.read_parquet(p)
            else:
                raw = pd.read_csv(p)
        except Exception:
            continue
        if not raw.empty:
            break
    if raw.empty or "symbol" not in raw.columns:
        return pd.DataFrame(columns=["symbol", "inst_holding_ratio"])

    out = raw.copy()
    out["symbol"] = out["symbol"].astype(str).map(_norm_symbol)

    ratio_candidates = [
        "inst_holding_ratio",
        "institution_holding_ratio",
        "institution_ratio",
        "fund_holding_ratio",
        "fund_ratio",
        "inst_ratio",
    ]
    ratio_col = next((c for c in ratio_candidates if c in out.columns), None)
    if ratio_col is None:
        return pd.DataFrame(columns=["symbol", "inst_holding_ratio"])

    out["inst_holding_ratio"] = pd.to_numeric(out[ratio_col], errors="coerce")
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"])
        out = out[["date", "symbol", "inst_holding_ratio"]].drop_duplicates(subset=["date", "symbol"], keep="last")
        return out.reset_index(drop=True)

    out = out[["symbol", "inst_holding_ratio"]].drop_duplicates(subset=["symbol"], keep="last")
    return out.reset_index(drop=True)


def load_news_sentiment_factors(base_dir: Path, start_date: str) -> pd.DataFrame:
    news_root = base_dir / "data" / "news"
    cols = ["date", "symbol", "news_sentiment_score", "news_item_count"]
    if not news_root.exists():
        return pd.DataFrame(columns=cols)

    start_dt = pd.to_datetime(start_date, errors="coerce")
    rows: list[dict[str, Any]] = []

    for day_dir in sorted(news_root.iterdir(), key=lambda p: p.name):
        if not day_dir.is_dir():
            continue
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", day_dir.name):
            continue
        day_dt = pd.to_datetime(day_dir.name, errors="coerce")
        if pd.isna(day_dt):
            continue
        if pd.notna(start_dt) and day_dt < start_dt:
            continue

        manifest_path = day_dir / "manifest.json"
        used_manifest = False
        if manifest_path.exists():
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                as_of = pd.to_datetime(raw.get("as_of", day_dir.name), errors="coerce")
                if pd.isna(as_of):
                    as_of = day_dt
                symbols = raw.get("symbols", [])
                if isinstance(symbols, list):
                    for rec in symbols:
                        if not isinstance(rec, dict):
                            continue
                        sym_raw = rec.get("symbol") or rec.get("code6")
                        if not sym_raw:
                            continue
                        rows.append(
                            {
                                "date": as_of,
                                "symbol": _norm_symbol(str(sym_raw)),
                                "news_sentiment_score": rec.get("sentiment_score", 0.0),
                                "news_item_count": rec.get("item_count", 0),
                            }
                        )
                    used_manifest = True
            except Exception:
                used_manifest = False

        if used_manifest:
            continue

        for p in day_dir.glob("*.json"):
            if p.name in {"manifest.json", "qc_report.json"} or p.name.startswith("_"):
                continue
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            sym_raw = raw.get("symbol") or p.stem
            if not sym_raw:
                continue
            item_count = raw.get("item_count")
            if item_count is None and isinstance(raw.get("items"), list):
                item_count = len(raw["items"])
            rows.append(
                {
                    "date": day_dt,
                    "symbol": _norm_symbol(str(sym_raw)),
                    "news_sentiment_score": raw.get("sentiment_score", 0.0),
                    "news_item_count": item_count if item_count is not None else 0,
                }
            )

    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str).map(_norm_symbol)
    out["news_sentiment_score"] = pd.to_numeric(out["news_sentiment_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    out["news_item_count"] = pd.to_numeric(out["news_item_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = out.dropna(subset=["date"])
    out = out[out["symbol"].astype(str).str.len() > 0]
    if pd.notna(start_dt):
        out = out[out["date"] >= start_dt]
    out = out.sort_values(["date", "symbol"]).drop_duplicates(subset=["date", "symbol"], keep="last").reset_index(drop=True)
    return out[cols]


def load_and_prepare_features(base_dir: Path, start_date: str, news_sentiment_lag_days: int = 1) -> pd.DataFrame:
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

    style_df = load_symbol_style_factors(base_dir)
    if not style_df.empty:
        merged = merged.merge(style_df, on="symbol", how="left")
    else:
        for col in ["roe_latest", "yoyni", "cfo_to_np", "pe_ttm", "earnings_yield"]:
            merged[col] = np.nan

    inst_df = load_institution_holding_factors(base_dir)
    if not inst_df.empty:
        if "date" in inst_df.columns:
            left = merged.drop(columns=["inst_holding_ratio"], errors="ignore").sort_values(["symbol", "date"])
            right = inst_df.sort_values(["symbol", "date"])
            right_map: dict[str, pd.DataFrame] = {
                str(sym): g[["date", "inst_holding_ratio"]].sort_values("date")
                for sym, g in right.groupby("symbol", sort=False)
            }
            parts: list[pd.DataFrame] = []
            for sym, g in left.groupby("symbol", sort=False):
                rs = right_map.get(str(sym))
                if rs is None or rs.empty:
                    gg = g.copy()
                    gg["inst_holding_ratio"] = np.nan
                else:
                    gg = pd.merge_asof(
                        g.sort_values("date"),
                        rs,
                        on="date",
                        direction="backward",
                    )
                parts.append(gg)
            merged = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
        else:
            merged = merged.merge(inst_df, on="symbol", how="left")
    if "inst_holding_ratio" not in merged.columns:
        merged["inst_holding_ratio"] = np.nan

    news_df = load_news_sentiment_factors(base_dir, start_date=start_date)
    if not news_df.empty:
        merged = merged.merge(news_df, on=["date", "symbol"], how="left")
    if "news_sentiment_score" not in merged.columns:
        merged["news_sentiment_score"] = np.nan
    if "news_item_count" not in merged.columns:
        merged["news_item_count"] = np.nan
    merged["news_sentiment_score"] = pd.to_numeric(merged["news_sentiment_score"], errors="coerce").fillna(0.0)
    merged["news_item_count"] = pd.to_numeric(merged["news_item_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    lag_days = max(0, int(news_sentiment_lag_days))
    if lag_days > 0:
        merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
        merged["news_sentiment_score"] = merged.groupby("symbol")["news_sentiment_score"].shift(lag_days).fillna(0.0)
        merged["news_item_count"] = merged.groupby("symbol")["news_item_count"].shift(lag_days).fillna(0.0)

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

    # 閳光偓閳光偓 閺堝牏鍤庣痪褍鍩嗛幐鍥ㄧ垼閿涘牏鏁ら幋鐤殰鐎规矮绠熼柅姘虫彧娣団€冲彆瀵骏绱氶埞鈧埞鈧?
    # 娑撹濮忛幒褏娲忛崗绱? GU1=(C*2+H+L)/4; 鐠ч鍨?EMA(EMA(C,9),9); 娑撹濮忛幒褏娲?(GU1-REF(鐠ч鍨?1))/REF(鐠ч鍨?1)
    merged["_gu1"] = (merged["close"] * 2 + merged["high"] + merged["low"]) / 4.0
    merged["_ema9"] = merged.groupby("symbol")["close"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    merged["_ema9_9"] = merged.groupby("symbol")["_ema9"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    merged["_qibao_prev"] = merged.groupby("symbol")["_ema9_9"].shift(1)
    merged["main_force_pct"] = (merged["_gu1"] - merged["_qibao_prev"]) / merged["_qibao_prev"].replace(0, np.nan)
    merged["main_force_pct"] = merged["main_force_pct"].fillna(0)

    # 妤?0閸忕础: X1=(C+L+H)/1.5; X2=EMA(X1,3); 妤?0=HHV(X2,30); 閺夆€叉:妤?0>閸撳秹鐝?0
    merged["_x1"] = (merged["close"] + merged["low"] + merged["high"]) / 1.5
    merged["_x2"] = merged.groupby("symbol")["_x1"].transform(lambda s: s.ewm(span=3, adjust=False).mean())
    merged["_high30"] = merged.groupby("symbol")["_x2"].transform(lambda s: s.rolling(30, min_periods=10).max())
    merged["_high30_prev"] = merged.groupby("symbol")["_high30"].shift(1)
    merged["high30_new_high"] = (merged["_high30"] > merged["_high30_prev"]).astype(int)

    # 30婢垛晛鍞撮張澶嬪畾閸? 濞戙劌绠?9.5%閻ㄥ嫬銇夐弫?
    merged["_is_limit_up"] = (merged["close"] / merged.groupby("symbol")["close"].shift(1) > 1.095).astype(int)
    merged["has_limit_up_30d_calc"] = merged.groupby("symbol")["_is_limit_up"].transform(
        lambda s: s.rolling(30, min_periods=1).sum()
    )
    merged["has_limit_up_30d_calc"] = (merged["has_limit_up_30d_calc"] >= 1).astype(int)

    # 濞撳懐鎮婃稉瀛樻閸?
    drop_cols = [c for c in merged.columns if c.startswith("_")]
    merged.drop(columns=drop_cols, inplace=True, errors="ignore")

    return merged


def build_industry_map_from_config(base_dir: Path, symbols: pd.Series) -> dict[str, str]:
    mapping: dict[str, str] = {}

    # 鐏忔繆鐦径姘嚋闁板秶鐤嗛弬鍥︽
    cfg_path = None
    p = base_dir / "config.yaml"
    if p.exists():
        cfg_path = p

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


def precompute_daily_universe(df: pd.DataFrame, cfg: BacktestConfigV3 | None = None) -> pd.DataFrame:
    day_list: list[pd.DataFrame] = []
    use_alpha = bool(cfg.use_alpha_enhancement) if cfg is not None else False
    robust_norm = bool(cfg.use_robust_score_norm) if cfg is not None else False
    winsor_q = float(cfg.score_winsor_quantile) if cfg is not None else 0.0
    neutral_ind = bool(cfg.score_neutralize_industry) if cfg is not None else False
    neutral_size = bool(cfg.score_neutralize_size) if cfg is not None else False
    neutral_beta = bool(cfg.score_neutralize_beta) if cfg is not None else False
    size_col = str(cfg.score_size_col) if cfg is not None else "float_mkt_cap_20d"
    beta_col = str(cfg.score_beta_col) if cfg is not None else "beta_60d"
    alpha_ind_w = float(cfg.alpha_industry_rs_weight) if cfg is not None else 0.35
    alpha_flow_w = float(cfg.alpha_flow_persistence_weight) if cfg is not None else 0.45
    alpha_quality_w = float(cfg.alpha_quality_weight) if cfg is not None else 0.30
    alpha_str_w = float(cfg.alpha_short_reversal_weight) if cfg is not None else 0.0
    alpha_trev_w = float(cfg.alpha_turnover_reversal_weight) if cfg is not None else 0.0
    alpha_value_w = float(cfg.alpha_value_proxy_weight) if cfg is not None else 0.0
    use_news_sentiment = bool(cfg.use_news_sentiment_factor) if cfg is not None else False
    news_sentiment_weight = float(cfg.news_sentiment_weight) if cfg is not None else 0.10
    news_sentiment_min_items = int(cfg.news_sentiment_min_items) if cfg is not None else 3

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
        "news_sentiment_score",
        "news_item_count",
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

        z_ma = _winsorized_zscore(x["ma_dist_20"], quantile=winsor_q, robust=robust_norm)
        z_r20 = _winsorized_zscore(x["ret_20d"], quantile=winsor_q, robust=robust_norm)
        z_r60 = _winsorized_zscore(x["ret_60d"], quantile=winsor_q, robust=robust_norm)
        z_vol = _winsorized_zscore(x["vol_20d"], quantile=winsor_q, robust=robust_norm)
        z_atr = _winsorized_zscore(x["atr_pct"], quantile=winsor_q, robust=robust_norm)
        z_vr = _winsorized_zscore(x["vol_ratio_20"], quantile=winsor_q, robust=robust_norm)

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

        if use_alpha:
            alpha_df = _compute_alpha_factors(x)
            x["alpha_industry_rs"] = alpha_df["alpha_industry_rs"]
            x["alpha_flow_persistence"] = alpha_df["alpha_flow_persistence"]
            x["alpha_quality"] = alpha_df["alpha_quality"]
            x["alpha_short_reversal"] = alpha_df["alpha_short_reversal"]
            x["alpha_turnover_reversal"] = alpha_df["alpha_turnover_reversal"]
            x["alpha_value_proxy"] = alpha_df["alpha_value_proxy"]
            score = (
                score
                + alpha_ind_w * x["alpha_industry_rs"]
                + alpha_flow_w * x["alpha_flow_persistence"]
                + alpha_quality_w * x["alpha_quality"]
                + alpha_str_w * x["alpha_short_reversal"]
                + alpha_trev_w * x["alpha_turnover_reversal"]
                + alpha_value_w * x["alpha_value_proxy"]
            )
        else:
            x["alpha_industry_rs"] = 0.0
            x["alpha_flow_persistence"] = 0.0
            x["alpha_quality"] = 0.0
            x["alpha_short_reversal"] = 0.0
            x["alpha_turnover_reversal"] = 0.0
            x["alpha_value_proxy"] = 0.0

        news_raw = pd.to_numeric(x.get("news_sentiment_score", 0.0), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
        news_items = pd.to_numeric(x.get("news_item_count", 0.0), errors="coerce").fillna(0.0)
        if news_sentiment_min_items > 0:
            news_raw = news_raw.where(news_items >= float(news_sentiment_min_items), 0.0)
        if use_news_sentiment:
            valid_mask = news_raw.abs() > 1e-12
            if int(valid_mask.sum()) >= 3:
                news_alpha = _safe_zscore(news_raw.where(valid_mask, np.nan)).clip(lower=-3.0, upper=3.0).fillna(0.0)
            else:
                news_alpha = pd.Series(0.0, index=x.index, dtype=float)
            score = score + float(news_sentiment_weight) * news_alpha
        else:
            news_alpha = pd.Series(0.0, index=x.index, dtype=float)
        x["alpha_news_sentiment"] = news_alpha
        x["news_sentiment_score_effective"] = news_raw

        x["score_raw"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
        score_adj = x["score_raw"].copy()
        if neutral_ind or neutral_size or neutral_beta:
            exposures_parts: list[pd.DataFrame] = []

            if neutral_ind and "industry" in x.columns:
                ind = x["industry"].astype(str).fillna("OTHER")
                dummies = pd.get_dummies(ind, prefix="ind", drop_first=True)
                if not dummies.empty:
                    exposures_parts.append(dummies)

            if neutral_size and size_col in x.columns:
                sz = pd.to_numeric(x[size_col], errors="coerce")
                sz = np.log(sz.clip(lower=1.0))
                sz = _winsorized_zscore(sz, quantile=winsor_q, robust=robust_norm)
                exposures_parts.append(pd.DataFrame({"size": sz}, index=x.index))

            if neutral_beta and beta_col in x.columns:
                bt = pd.to_numeric(x[beta_col], errors="coerce")
                bt = _winsorized_zscore(bt, quantile=winsor_q, robust=robust_norm)
                exposures_parts.append(pd.DataFrame({"beta": bt}, index=x.index))

            if exposures_parts:
                ex = pd.concat(exposures_parts, axis=1)
                score_adj = _residualize_cross_section(score_adj, ex)

        x["score_neutralized"] = pd.to_numeric(score_adj, errors="coerce").fillna(0.0)
        preprocess_enabled = bool(robust_norm or winsor_q > 0.0 or neutral_ind or neutral_size or neutral_beta)
        if preprocess_enabled:
            x["score"] = _winsorized_zscore(
                x["score_neutralized"],
                quantile=winsor_q,
                robust=robust_norm,
            ).fillna(0.0)
        else:
            x["score"] = x["score_neutralized"].copy()
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


def compute_market_regime(df: pd.DataFrame, cfg: BacktestConfigV3 | None = None) -> pd.DataFrame:
    tmp = df.copy()
    tmp["trend_up"] = (tmp["ma_dist_20"] > 0).astype(int)
    tmp["ret_1d"] = pd.to_numeric(tmp.get("ret_1d", 0.0), errors="coerce").fillna(0.0)
    tmp["ret_20d"] = pd.to_numeric(tmp.get("ret_20d", 0.0), errors="coerce").fillna(0.0)
    tmp["amount_20d"] = pd.to_numeric(tmp.get("amount_20d", np.nan), errors="coerce")

    mkt = tmp.groupby("date", as_index=False).agg(
        market_ret_1d=("ret_1d", "mean"),
        market_ret_20d=("ret_20d", "mean"),
        market_trend_pct=("trend_up", "mean"),
        market_amount_20d=("amount_20d", "median"),
    )
    mkt = mkt.sort_values("date").reset_index(drop=True)

    def label(row: pd.Series) -> str:
        if row["market_trend_pct"] > 0.6 and row["market_ret_20d"] > 0.03:
            return "BULL"
        if row["market_trend_pct"] < 0.4 and row["market_ret_20d"] < -0.03:
            return "BEAR"
        return "NEUTRAL"

    mkt["regime"] = mkt.apply(label, axis=1)
    if cfg is None:
        cap_map = {"BULL": 0.80, "NEUTRAL": 0.55, "BEAR": 0.30}
    else:
        cap_map = {
            "BULL": float(cfg.regime_bull_pos_cap),
            "NEUTRAL": float(cfg.regime_neutral_pos_cap),
            "BEAR": float(cfg.regime_bear_pos_cap),
        }
    mkt["regime_pos_cap"] = mkt["regime"].map(cap_map)

    use_vol_sizing = bool(cfg.use_volatility_sizing) if cfg is not None else False
    vol_lb = max(5, int(cfg.volatility_lookback_days)) if cfg is not None else 20
    vol_target = float(cfg.volatility_target_annual) if cfg is not None else 0.22
    vol_floor = float(cfg.volatility_floor_annual) if cfg is not None else 0.08
    vol_min = float(cfg.volatility_pos_mult_min) if cfg is not None else 0.65
    vol_max = float(cfg.volatility_pos_mult_max) if cfg is not None else 1.15
    mkt["market_vol_annual"] = (
        pd.to_numeric(mkt["market_ret_1d"], errors="coerce")
        .fillna(0.0)
        .rolling(vol_lb, min_periods=max(5, vol_lb // 2))
        .std(ddof=0)
        * np.sqrt(252.0)
    )
    if use_vol_sizing:
        vol_denom = pd.to_numeric(mkt["market_vol_annual"], errors="coerce").clip(lower=max(vol_floor, 1e-6))
        vol_mult = (vol_target / vol_denom).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        mkt["volatility_pos_mult"] = vol_mult.clip(lower=min(vol_min, vol_max), upper=max(vol_min, vol_max))
    else:
        mkt["volatility_pos_mult"] = 1.0

    use_liq_sizing = bool(cfg.use_liquidity_state_sizing) if cfg is not None else False
    liq_lb = max(20, int(cfg.liquidity_lookback_days)) if cfg is not None else 60
    liq_min = float(cfg.liquidity_pos_mult_min) if cfg is not None else 0.85
    liq_max = float(cfg.liquidity_pos_mult_max) if cfg is not None else 1.05
    liq_sens = float(cfg.liquidity_pos_mult_sensitivity) if cfg is not None else 0.50
    mkt["market_liq_proxy"] = pd.to_numeric(mkt["market_amount_20d"], errors="coerce")
    liq_anchor = mkt["market_liq_proxy"].rolling(liq_lb, min_periods=max(10, liq_lb // 3)).median()
    liq_ratio = (mkt["market_liq_proxy"] / liq_anchor).replace([np.inf, -np.inf], np.nan)
    if use_liq_sizing:
        liq_pow = np.power(liq_ratio.clip(lower=1e-6).fillna(1.0), max(liq_sens, 0.0))
        mkt["liquidity_pos_mult"] = liq_pow.clip(lower=min(liq_min, liq_max), upper=max(liq_min, liq_max))
    else:
        mkt["liquidity_pos_mult"] = 1.0

    mkt["state_pos_mult"] = (
        pd.to_numeric(mkt["volatility_pos_mult"], errors="coerce").fillna(1.0)
        * pd.to_numeric(mkt["liquidity_pos_mult"], errors="coerce").fillna(1.0)
    ).clip(lower=0.25, upper=1.50)

    use_crash_protect = bool(cfg.use_momentum_crash_protection) if cfg is not None else False
    crash_cap = float(cfg.momentum_crash_position_cap) if cfg is not None else 0.45
    if use_crash_protect:
        lb = max(3, int(cfg.momentum_crash_lookback_days))
        rebound_lb = max(1, int(cfg.momentum_rebound_lookback_days))
        protect_days = max(1, int(cfg.momentum_crash_protection_days))
        drop_th = float(cfg.momentum_crash_drop_threshold)
        rebound_th = float(cfg.momentum_rebound_threshold)

        mkt["crash_lb_ret"] = (
            (1.0 + mkt["market_ret_1d"].fillna(0.0))
            .rolling(lb, min_periods=lb)
            .apply(np.prod, raw=True)
            - 1.0
        )
        mkt["rebound_lb_ret"] = (
            (1.0 + mkt["market_ret_1d"].fillna(0.0))
            .rolling(rebound_lb, min_periods=rebound_lb)
            .apply(np.prod, raw=True)
            - 1.0
        )
        trigger = (mkt["crash_lb_ret"] <= drop_th) & (mkt["rebound_lb_ret"] >= rebound_th)
        trigger = trigger.fillna(False).astype(bool)

        active_flags: list[bool] = []
        protect_left = 0
        for is_trigger in trigger.tolist():
            if bool(is_trigger):
                protect_left = protect_days
            active = protect_left > 0
            active_flags.append(active)
            if protect_left > 0:
                protect_left -= 1

        mkt["momentum_crash_trigger"] = trigger.astype(int)
        mkt["momentum_crash_active"] = active_flags
        mkt["momentum_crash_pos_cap"] = np.where(mkt["momentum_crash_active"], crash_cap, 1.0)
    else:
        mkt["crash_lb_ret"] = np.nan
        mkt["rebound_lb_ret"] = np.nan
        mkt["momentum_crash_trigger"] = 0
        mkt["momentum_crash_active"] = False
        mkt["momentum_crash_pos_cap"] = 1.0
    return mkt


def compute_index_filter(base_dir: Path, start_date: str, cfg: BacktestConfigV3) -> dict:
    """
    閺傝1: 閹稿洦鏆熸搴㈠付瀵偓閸?
    鏉╂柨娲?{date: True/False}閿涘rue=閸忎浇瀵偓娴犳搫绱滷alse=閺嗗倸浠犲鈧禒?
    """
    if not cfg.use_index_filter:
        return {}

    # 鐏忔繆鐦崝鐘烘祰濞岀箒300閹稿洦鏆?
    idx_path = base_dir / "data" / "index" / "hs300_daily.parquet"
    if not idx_path.exists():
        # 婵″倹鐏夊▽鈩冩箒閺堟勾閹稿洦鏆熼弫鐗堝祦閿涘苯鐨剧拠鏇犳暏BaoStock閼惧嘲褰?
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
                    print("  [Index Filter] failed to load index data, skip index filter")
                    return {}
                idx = pd.DataFrame(rows, columns=rs.fields)
                idx["date"] = pd.to_datetime(idx["date"])
                idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
            finally:
                bs.logout()
        except Exception as e:
            print(f"  [Index Filter] failed to load index data: {e}; skip index filter")
            return {}
    else:
        idx = pd.read_parquet(idx_path)
        idx["date"] = pd.to_datetime(idx["date"])

    idx = idx.sort_values("date").reset_index(drop=True)

    # 鐠侊紕鐣婚崸鍥╁殠
    idx["ma_long"] = idx["close"].rolling(cfg.index_ma_period, min_periods=20).mean()
    idx["ma_short"] = idx["close"].rolling(cfg.index_ma_short, min_periods=10).mean()
    idx["ma_short_prev"] = idx["ma_short"].shift(1)

    # 閹稿洦鏆熺搾瀣◢: 閺€鍓佹磸>MA60 娑?MA20娑撳﹥瀚?
    idx["trend_ok"] = (idx["close"] > idx["ma_long"]) & (idx["ma_short"] > idx["ma_short_prev"])

    # 鏉╂厸婢垛晞绌奸獮鍛ù?
    idx["ret_n"] = idx["close"].pct_change(cfg.index_crash_days)
    idx["crash"] = idx["ret_n"] < cfg.index_crash_threshold

    # 閺嗙绌奸崥搴㈡畯閸嬫罚婢?
    idx["pause_until"] = pd.NaT
    pause_end = pd.NaT
    pause_flags = []
    for i, row in idx.iterrows():
        if row["crash"]:
            pause_end = row["date"] + pd.Timedelta(days=cfg.index_pause_days * 1.5)  # 娴溿倖妲楅弮銉㈠1.5閸婂秷鍤滈悞鑸垫）
        is_paused = pd.notna(pause_end) and row["date"] <= pause_end
        pause_flags.append(is_paused)
    idx["is_paused"] = pause_flags

    # 閺堚偓缂佸牞绱扮搾瀣◢OK 娑?娑撳秴婀弳鍌氫粻閺?
    idx["allow_open"] = idx["trend_ok"].fillna(False) & ~idx["is_paused"]

    result = dict(zip(idx["date"], idx["allow_open"]))
    allow_days = sum(1 for v in result.values() if v)
    total_days = len(result)
    print(f"  📈 指数过滤可开仓日: {allow_days}/{total_days} ({allow_days/max(total_days,1)*100:.0f}%)")
    return result


def _candidate_entry_mask(day: pd.DataFrame, cfg: BacktestConfigV3) -> tuple[pd.Series, pd.Series]:
    """
    閸忋儱婧€濡€崇础閿?
    - custom:  閻劍鍩涢懛鐣炬稊澶婄箑鐟曚焦娼禒璁圭礄閺堝牏鍤庣痪褍鍩?閺夆€叉閸忋劑鍎村陇鍐婚敍?
    - strict:  韫囧懘銆?(濞戙劌浠犻崶鐐剁殶閸忋劌 OR TDX妤傛ê鍨? AND trend_up
    - normal:  婢舵俺鐭惧鍕紥濞茶鍙嗛崷?
    - loose:   娴犲懘娓?trend_up
    """
    day = day.copy()
    mode = cfg.entry_mode.lower()

    if mode == "custom":
        # 閳光偓閳光偓 閻劍鍩涢懛鐣炬稊澶涚窗5娑撴箑缁捐法楠囬崚绻€鐟曚焦娼禒璺哄弿闁婲D 閳光偓閳光偓
        # 1. 30婢垛晛鍞撮張澶嬪畾閸?
        cond_limit_up = day["has_limit_up_30d_calc"].fillna(0) >= 1
        # 2. 娑撹濮忛幒褏娲?> 0.5% (GU1閸嬪繒鐠ч鍨庣痪?
        cond_main_force = day["main_force_pct"].fillna(0) > 0.005
        # 3. 妤?0閸掓稒鏌婃?
        cond_high30 = day["high30_new_high"].fillna(0) == 1
        # 4. 閸ョ偠鐨?10%-30%
        cond_pullback = (day["pullback_pct"].fillna(0) >= cfg.pullback_min_pct) & (
            day["pullback_pct"].fillna(0) <= cfg.pullback_max_pct
        )
        # 5. TDX鐠囧嫬鍨?>= 1.5
        cond_tdx = day["tdx_score"].fillna(0) >= cfg.tdx_min_score

        mask = cond_limit_up & cond_main_force & cond_high30 & cond_pullback & cond_tdx
        strength = pd.Series(0.0, index=day.index)
        strength = strength.where(~mask, 1.0)
        return mask, strength

    # 閳光偓閳光偓 閸╄櫣閺夆€叉鐠侊紕鐣?閳光偓閳光偓
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

    # 閳光偓閳光偓 濞戙劌浠犻崶鐐剁殶鐎瑰本鏆ｉ弶鈥叉 閳光偓閳光偓
    limit_up_pullback = in_window & pullback_ok

    if mode == "strict":
        core_signal = (limit_up_pullback & breakout_ok) | tdx_ok
        mask = core_signal & trend_up
        strong = limit_up_pullback & tdx_ok & trend_up
        strength = pd.Series(0.0, index=day.index)
        strength = strength.where(~mask, 1.0)
        strength = strength.where(~strong, 2.0)
        return mask, strength

    elif mode == "loose":
        mask = trend_up
        strength = pd.Series(0.0, index=day.index)
        strength = strength.where(~mask, 1.0)
        return mask, strength

    else:  # "normal" (姒涙) - 閺傝2A: 2-of-3 + 娣団€冲娇瀵搫瀹抽崚鍡欓獓
        # 闂勫嫬濮為弶鈥叉鐠佲剝鏆? breakout_ok, tdx_ok, trend_up
        extra_count = (
            breakout_ok.astype(int)
            + tdx_ok.astype(int)
            + trend_up.astype(int)
        )

        # 鐠虹窞1: 濞戙劌浠犻崶鐐剁殶 + 闂勫嫬濮為弶鈥叉閳? (2-of-3)
        path_pullback = limit_up_pullback & (extra_count >= cfg.normal_min_conditions)

        # 鐠虹窞2: TDX妤傛ê鍨?+ trend_up (瀵桨淇婇崣?
        path_tdx = tdx_ok & trend_up

        # 鐠虹窞3: 閺€楣冨櫤缁愪胶鐗?+ trend_up + 娑撹濮忛幒褏娲?(闂団偓鐟曚焦娲挎径姘剁崣鐠?
        path_breakout = breakout_ok & trend_up & (day["main_force_strong"].fillna(0) == 1)

        mask = path_pullback | path_tdx | path_breakout

        # 閳光偓閳光偓 閺傝2B: 娣団€冲娇瀵搫瀹抽崚鍡欓獓閿涘牆鐡ㄩ崚?DataFrame 娓氭稐绮ㄦ担宥呭瀻闁板秶鏁ら敍澶嗘敘閳光偓
        signal_strength = pd.Series(0.0, index=day.index)
        # 瀵桨淇婇崣? 濞戙劌浠犻崶鐐剁殶 + TDX + 鐡掑濞?閸忋劍寮х搾?
        strong = limit_up_pullback & tdx_ok & trend_up
        # 閺呪偓姘繆閸? 鐠虹窞2閹存牞鐭惧?
        normal_sig = (path_tdx | path_pullback) & ~strong
        # 瀵彉淇婇崣? 娴犲懓鐭惧?
        weak = path_breakout & ~strong & ~normal_sig

        signal_strength = signal_strength.where(~strong, 2.0)    # strong=2
        signal_strength = signal_strength.where(~normal_sig, 1.0)  # normal=1
        signal_strength = signal_strength.where(~weak, 0.5)        # weak=0.5
        return mask, signal_strength


def _regime_entry_strength_floor(regime_tag: str, cfg: BacktestConfigV3) -> float:
    regime = str(regime_tag).upper()
    if regime == "BULL":
        return float(cfg.regime_gate_bull_min_strength)
    if regime == "BEAR":
        return float(cfg.regime_gate_bear_min_strength)
    return float(cfg.regime_gate_neutral_min_strength)


def _build_entry_open_lists(
    target_symbols: list[str],
    signal_strengths: dict[str, float],
    regime_tag: str,
    cfg: BacktestConfigV3,
    gate_boost: float = 0.0,
) -> tuple[list[str], list[str], float]:
    if not target_symbols:
        return [], [], 0.0

    gated_symbols = list(target_symbols)
    gate_floor = 0.0
    if cfg.use_regime_entry_gate:
        gate_floor = _regime_entry_strength_floor(regime_tag, cfg) + max(0.0, float(gate_boost))
        gated_symbols = [s for s in gated_symbols if float(signal_strengths.get(s, 1.0)) >= gate_floor]

    if not cfg.use_dual_layer_entry:
        open_symbols = gated_symbols
        open_set = set(open_symbols)
        observed = [s for s in target_symbols if s not in open_set]
        return open_symbols, observed, gate_floor

    strong: list[str] = []
    normal: list[str] = []
    weak: list[str] = []
    for sym in gated_symbols:
        strength = float(signal_strengths.get(sym, 1.0))
        if strength >= float(cfg.dual_entry_strong_threshold):
            strong.append(sym)
        elif strength >= float(cfg.dual_entry_normal_threshold):
            normal.append(sym)
        else:
            weak.append(sym)

    open_symbols = strong + normal
    weak_mode = str(cfg.weak_entry_mode).strip().lower()
    if weak_mode not in {"observe", "micro"}:
        weak_mode = "observe"
    if weak_mode == "micro" and weak:
        max_new = int(cfg.weak_micro_max_new_positions)
        weak_take = weak if max_new <= 0 else weak[:max_new]
        open_symbols.extend(weak_take)

    open_set = set(open_symbols)
    observed_symbols = [s for s in target_symbols if s not in open_set]
    return open_symbols, observed_symbols, gate_floor


def _compute_institution_proxy_score(day: pd.DataFrame) -> pd.Series:
    idx = day.index
    roe = pd.to_numeric(day.get("roe_latest", pd.Series(np.nan, index=idx)), errors="coerce")
    cfo = pd.to_numeric(day.get("cfo_to_np", pd.Series(np.nan, index=idx)), errors="coerce")
    yoy = pd.to_numeric(day.get("yoyni", pd.Series(np.nan, index=idx)), errors="coerce").clip(-100, 500)
    vol20 = pd.to_numeric(day.get("vol_20d", pd.Series(np.nan, index=idx)), errors="coerce")
    amount20 = pd.to_numeric(day.get("amount_20d", pd.Series(np.nan, index=idx)), errors="coerce")

    proxy = (
        0.35 * _pct_rank(roe)
        + 0.20 * _pct_rank(cfo)
        + 0.15 * _pct_rank(yoy)
        + 0.15 * (1.0 - _pct_rank(vol20))
        + 0.15 * _pct_rank(amount20)
    )
    return proxy.fillna(0.5)


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

    if cfg.use_institution_holding_filter and not out.empty:
        mode = str(cfg.institution_filter_mode).strip().lower()
        min_keep = max(5, int(np.ceil(len(out) * 0.05)))
        if mode == "proxy":
            proxy = _compute_institution_proxy_score(out)
            q = float(np.clip(cfg.institution_proxy_quantile, 0.0, 0.99))
            thr = float(proxy.quantile(q))
            keep = proxy >= thr
            if int(keep.sum()) >= min_keep:
                out = out.loc[keep].copy()
        else:
            data_col = str(cfg.institution_data_col).strip()
            if not data_col:
                data_col = "inst_holding_ratio"
            if data_col in out.columns:
                ratio = pd.to_numeric(out[data_col], errors="coerce")
                q = float(np.clip(cfg.institution_holding_quantile, 0.0, 0.99))
                mask_q = ratio >= float(ratio.quantile(q))
                min_pct = float(cfg.institution_holding_min_pct)
                if min_pct > 0:
                    mask = mask_q & (ratio >= min_pct)
                else:
                    mask = mask_q
                if int(mask.sum()) >= min_keep:
                    out = out.loc[mask].copy()

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

    trading_days = [pd.to_datetime(d) for d in sorted(ddf["date"].unique().tolist())]
    if len(trading_days) < 20:
        empty_eq = pd.DataFrame(columns=["date", "equity", "daily_return", "n_holdings"])
        metrics = {"annual_return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe": 0.0, "win_rate_pct": 0.0}
        return {"metrics": metrics, "equity_curve": empty_eq, "daily_targets": {}}
    rank_rebalance_days = _compute_rebalance_days(trading_days, cfg.rank_exit_rebalance_freq)

    regime_pos: dict[pd.Timestamp, float] = {}
    regime_name: dict[pd.Timestamp, str] = {}
    crash_protect_active_map: dict[pd.Timestamp, bool] = {}
    crash_protect_cap_map: dict[pd.Timestamp, float] = {}
    volatility_pos_mult_map: dict[pd.Timestamp, float] = {}
    liquidity_pos_mult_map: dict[pd.Timestamp, float] = {}
    state_pos_mult_map: dict[pd.Timestamp, float] = {}
    market_vol_annual_map: dict[pd.Timestamp, float] = {}
    market_liq_proxy_map: dict[pd.Timestamp, float] = {}
    if regime_df is not None and not regime_df.empty:
        rr = regime_df.copy()
        rr["date"] = pd.to_datetime(rr["date"])
        regime_pos = dict(zip(rr["date"], rr["regime_pos_cap"]))
        if "regime" in rr.columns:
            regime_name = dict(zip(rr["date"], rr["regime"]))
        if "volatility_pos_mult" in rr.columns:
            volatility_pos_mult_map = {
                pd.to_datetime(d): float(v)
                for d, v in zip(rr["date"], rr["volatility_pos_mult"])
            }
        if "liquidity_pos_mult" in rr.columns:
            liquidity_pos_mult_map = {
                pd.to_datetime(d): float(v)
                for d, v in zip(rr["date"], rr["liquidity_pos_mult"])
            }
        if "state_pos_mult" in rr.columns:
            state_pos_mult_map = {
                pd.to_datetime(d): float(v)
                for d, v in zip(rr["date"], rr["state_pos_mult"])
            }
        if "market_vol_annual" in rr.columns:
            market_vol_annual_map = {
                pd.to_datetime(d): float(v) if np.isfinite(v) else float("nan")
                for d, v in zip(rr["date"], rr["market_vol_annual"])
            }
        if "market_liq_proxy" in rr.columns:
            market_liq_proxy_map = {
                pd.to_datetime(d): float(v) if np.isfinite(v) else float("nan")
                for d, v in zip(rr["date"], rr["market_liq_proxy"])
            }
        if "momentum_crash_active" in rr.columns:
            crash_protect_active_map = {
                pd.to_datetime(d): bool(v)
                for d, v in zip(rr["date"], rr["momentum_crash_active"])
            }
        if "momentum_crash_pos_cap" in rr.columns:
            crash_protect_cap_map = {
                pd.to_datetime(d): float(v)
                for d, v in zip(rr["date"], rr["momentum_crash_pos_cap"])
            }

    by_date = {d: g for d, g in ddf.groupby("date", sort=True)}
    idx_filter = index_filter or {}

    equity = cfg.initial_capital
    equity_peak = cfg.initial_capital
    brake_days_left = 0
    dd_above_threshold_prev = False
    positions: dict[str, dict[str, float]] = {}
    rec: list[dict[str, Any]] = []
    daily_targets: dict[pd.Timestamp, list[str]] = {}
    trade_log: list[dict[str, Any]] = []  # 娴溿倖妲楃拋鏉跨秿

    for i in range(1, len(trading_days)):
        prev_date = trading_days[i - 1]
        date = trading_days[i]
        prev_day = by_date.get(prev_date)
        day = by_date.get(date)
        if prev_day is None or prev_day.empty:
            continue
        prev_day = prev_day.copy()
        if "symbol" in prev_day.columns:
            prev_day_symbol_idx = prev_day.set_index("symbol", drop=False)
        else:
            prev_day_symbol_idx = pd.DataFrame()
        if day is not None and not day.empty and "symbol" in day.columns:
            day_symbol_idx = day.set_index("symbol", drop=False)
        else:
            day_symbol_idx = pd.DataFrame()

        if allowed_symbols_by_date:
            allowed = allowed_symbols_by_date.get(prev_date)
            if allowed is not None:
                prev_day = prev_day[prev_day["symbol"].isin(allowed)].copy()

        # 閳光偓閳光偓 缂佸嫬鎮庨崶鐐存寵閸掔婧呴敍鍫濆帥娴滃骸绱戞禒鎾冲灲閺傜礆閳光偓閳光偓
        portfolio_dd = 1.0 - equity / max(equity_peak, 1e-12)
        dd_now = portfolio_dd >= cfg.drawdown_brake_threshold
        if (
            cfg.use_portfolio_drawdown_brake
            and dd_now
            and (not dd_above_threshold_prev)
            and brake_days_left <= 0
        ):
            brake_days_left = max(1, int(cfg.drawdown_brake_pause_days))
        dd_above_threshold_prev = dd_now

        brake_active = brake_days_left > 0
        if brake_active:
            brake_days_left -= 1

        # 閳光偓閳光偓 閺傝1: 閹稿洦鏆熸搴㈠付瀵偓閸?閳光偓閳光偓
        regime_tag = str(regime_name.get(prev_date, "UNKNOWN"))
        index_allow = idx_filter.get(prev_date, True) if idx_filter else True
        index_blocked = not bool(index_allow)
        index_hard_block = bool(cfg.use_index_filter and cfg.index_filter_hard_gate and index_blocked)
        can_open = (not brake_active) and (not index_hard_block)
        bear_block_active = bool(cfg.block_new_in_bear_regime and regime_tag == "BEAR")
        if bear_block_active:
            can_open = False
        adaptive_dd_active = bool(
            cfg.use_adaptive_drawdown_mode
            and (portfolio_dd >= float(cfg.adaptive_drawdown_trigger))
            and (str(regime_tag).upper() != "BULL")
        )
        momentum_crash_active = bool(
            cfg.use_momentum_crash_protection and crash_protect_active_map.get(prev_date, False)
        )
        momentum_crash_cap = float(
            crash_protect_cap_map.get(prev_date, float(cfg.momentum_crash_position_cap))
        )
        vol_pos_mult = float(volatility_pos_mult_map.get(prev_date, 1.0))
        liq_pos_mult = float(liquidity_pos_mult_map.get(prev_date, 1.0))
        state_pos_mult = float(state_pos_mult_map.get(prev_date, vol_pos_mult * liq_pos_mult))
        market_vol_annual = float(market_vol_annual_map.get(prev_date, float("nan")))
        market_liq_proxy = float(market_liq_proxy_map.get(prev_date, float("nan")))
        entry_top_k = int(cfg.top_k)
        entry_invest_more_n = int(cfg.invest_more_n)
        adaptive_gate_boost = 0.0
        if adaptive_dd_active:
            entry_top_k = max(1, int(np.ceil(cfg.top_k * float(cfg.adaptive_drawdown_top_k_multiplier))))
            entry_invest_more_n = max(
                entry_top_k,
                int(np.ceil(cfg.invest_more_n * float(cfg.adaptive_drawdown_invest_more_multiplier))),
            )
            adaptive_gate_boost = max(0.0, float(cfg.adaptive_drawdown_gate_boost))

        if not prev_day.empty:
            mask, signal_strength_series = _candidate_entry_mask(prev_day, cfg)
            prev_day.loc[:, "_signal_strength"] = signal_strength_series
            cands = _apply_enhancement_filters(prev_day[mask].copy(), cfg).sort_values(
                "score", ascending=False
            ).head(entry_invest_more_n)

            if cfg.use_correlation_control:
                ret_window = returns_df.loc[
                    (returns_df.index < prev_date)
                    & (returns_df.index >= prev_date - pd.Timedelta(days=cfg.corr_lookback_days * 2))
                ]
                if len(ret_window) > cfg.corr_lookback_days:
                    ret_window = ret_window.tail(cfg.corr_lookback_days)
            else:
                ret_window = returns_df.iloc[:0]

            target_symbols = _corr_guard_select(cands, entry_top_k, cfg, ret_window)

            signal_strengths: dict[str, float] = {}
            for _, row in cands.iterrows():
                signal_strengths[str(row["symbol"])] = float(row.get("_signal_strength", 1.0))
        else:
            # 瑜版挻妫╅崣鈧鐫滄稉铏光敄
            target_symbols = []
            signal_strengths = {}

        open_symbols, observed_symbols, gate_floor = _build_entry_open_lists(
            target_symbols=target_symbols,
            signal_strengths=signal_strengths,
            regime_tag=regime_tag,
            cfg=cfg,
            gate_boost=adaptive_gate_boost,
        )
        daily_targets[prev_date] = target_symbols
        target_set = set(target_symbols)
        rank_by_symbol: dict[str, float] = {}
        if "rank" in prev_day.columns:
            for _sym, _rk in zip(prev_day["symbol"], pd.to_numeric(prev_day["rank"], errors="coerce")):
                if np.isfinite(_rk):
                    rank_by_symbol[str(_sym)] = float(_rk)

        # 閳光偓閳光偓 瑜版挻妫╅幐浣风波閺€鍓佹抄閿涘牆鍘涚拋鏉垮嚒閺堝绮ㄦ担宥嗘暪閻╁绱濋崘宥嗗⒔鐞涘本鏁归惄妯跨殶娴犳搫绱氶埞鈧埞鈧?
        prev_weights = {sym: float(pos.get("weight", 0.0)) for sym, pos in positions.items()}

        daily_ret_row = returns_df.loc[date] if date in returns_df.index else pd.Series(dtype=float)
        gross_ret = 0.0
        for sym, pos in positions.items():
            gross_ret += pos["weight"] * float(daily_ret_row.get(sym, 0.0))

        # 閳光偓閳光偓 闁偓閸戞椽鈧槒绶敍鍫熸煙濡?閸楀洨楠囬敍澶嗘敘閳光偓
        risk_state = (
            f"regime={regime_tag};"
            f"index={'OPEN' if index_allow else 'BLOCK'};"
            f"index_mode={'HARD' if cfg.index_filter_hard_gate else 'SOFT'};"
            f"dd_brake={'ON' if brake_active else 'OFF'};"
            f"adaptive_dd={'ON' if adaptive_dd_active else 'OFF'};"
            f"mom_crash={'ON' if momentum_crash_active else 'OFF'};"
            f"bear_block={'ON' if bear_block_active else 'OFF'};"
            f"entry_open={len(open_symbols)}/{len(target_symbols)}"
        )
        if cfg.use_index_filter and (not index_allow) and (not cfg.index_filter_hard_gate):
            risk_state = f"{risk_state};index_soft_cap={cfg.index_filter_block_position_cap:.2f}"
        if cfg.use_regime_entry_gate:
            risk_state = f"{risk_state};gate_floor={gate_floor:.2f}"
        if cfg.use_volatility_sizing:
            risk_state = f"{risk_state};vol_pos_mult={vol_pos_mult:.3f}"
            if np.isfinite(market_vol_annual):
                risk_state = f"{risk_state};mkt_vol={market_vol_annual:.2%}"
            else:
                risk_state = f"{risk_state};mkt_vol=NA"
        if cfg.use_liquidity_state_sizing:
            risk_state = f"{risk_state};liq_pos_mult={liq_pos_mult:.3f}"
            if np.isfinite(market_liq_proxy):
                risk_state = f"{risk_state};mkt_liq={market_liq_proxy:.0f}"
            else:
                risk_state = f"{risk_state};mkt_liq=NA"
        if cfg.use_light_risk_budget:
            risk_state = f"{risk_state};risk_budget={cfg.risk_budget_mode}"
        if adaptive_dd_active:
            risk_state = (
                f"{risk_state};pre_brake=dd>={cfg.adaptive_drawdown_trigger:.2f};"
                f"topk={entry_top_k};invest_n={entry_invest_more_n}"
            )
        adaptive_stop_mult = float(cfg.adaptive_drawdown_stop_loss_multiplier) if adaptive_dd_active else 1.0
        adaptive_trail_mult = float(cfg.adaptive_drawdown_trailing_multiplier) if adaptive_dd_active else 1.0
        adaptive_hold_mult = float(cfg.adaptive_drawdown_max_hold_multiplier) if adaptive_dd_active else 1.0
        max_hold_days_now = max(3, int(round(float(cfg.max_hold_days) * adaptive_hold_mult)))
        to_sell: list[tuple[str, str]] = []
        for sym, pos in positions.items():
            if sym not in price_df.columns or date not in price_df.index:
                to_sell.append((sym, "missing_price"))
                continue

            px = price_df.at[date, sym]
            if not np.isfinite(px) or px <= 0:
                to_sell.append((sym, "invalid_price"))
                continue

            pos["hold_days"] += 1
            if px > pos["peak_price"]:
                pos["peak_price"] = float(px)

            pnl_pct = (float(px) - pos["entry_price"]) / pos["entry_price"]

            # (1) 绾幑鐕傜礄閸氱帄DX娣囨繃濮㈤敍?
            effective_stop = float(cfg.stop_loss_pct) * adaptive_stop_mult
            if cfg.use_tdx_protection and pos.get("tdx_score", 0) >= cfg.tdx_protection_threshold:
                effective_stop = max(cfg.stop_loss_pct, 0.08)
            if pnl_pct <= -effective_stop:
                to_sell.append((sym, "hard_stop_loss"))
                continue

            # (2) ATR濮濄垺宕敍鍫熸煙濡?閿?
            if cfg.use_atr_stop and pos.get("atr_val", 0) > 0:
                atr_stop_price = pos["entry_price"] - cfg.atr_stop_multiplier * pos["atr_val"]
                if float(px) <= atr_stop_price:
                    to_sell.append((sym, "atr_stop"))
                    continue

            # (3) 婢惰精瑙﹁ぐ銏♀偓浣归幑鐕傜礄閺傝3閿? 閸忋儱婧€閸氬动婢垛晛鍞村☉銊ョ畽娑撳秷鍐婚敍宀冪槈娴煎毉鐏炩偓
            if cfg.use_failure_stop and pos["hold_days"] <= cfg.failure_stop_days:
                if pos["hold_days"] == cfg.failure_stop_days and pnl_pct < cfg.failure_stop_gain:
                    entry_strength = float(pos.get("entry_signal_strength", 1.0))
                    if cfg.failure_stop_weak_signal_only and entry_strength > float(cfg.failure_stop_max_signal_strength):
                        pass
                    elif cfg.failure_stop_require_negative_pnl and pnl_pct >= float(cfg.failure_stop_negative_pnl_threshold):
                        pass
                    elif cfg.failure_stop_skip_if_still_target and sym in target_set:
                        pass
                    else:
                        trend_up_now = False
                        if cfg.failure_stop_skip_if_trend_up:
                            prev_row = (
                                prev_day_symbol_idx.loc[sym]
                                if (not prev_day_symbol_idx.empty and sym in prev_day_symbol_idx.index)
                                else None
                            )
                            if isinstance(prev_row, pd.DataFrame):
                                prev_row = prev_row.iloc[0]
                            if prev_row is not None:
                                _trend_v = pd.to_numeric(prev_row.get("trend_up", 0), errors="coerce")
                                trend_up_now = bool(np.isfinite(_trend_v) and float(_trend_v) > 0.0)
                        if cfg.failure_stop_skip_if_trend_up and trend_up_now:
                            pass
                        else:
                            to_sell.append((sym, "failure_stop"))
                            continue

            # (4/4.5) trailing / MA stop conflict-resolved by priority.
            trailing_hit = False
            ma_hit = False

            drawdown = (pos["peak_price"] - float(px)) / max(pos["peak_price"], 1e-12)
            effective_trail = float(cfg.trailing_stop_pct) * adaptive_trail_mult
            if pnl_pct > 0.10:
                effective_trail = min(cfg.trailing_stop_pct, 0.06)
            elif pnl_pct > 0.05:
                effective_trail = min(cfg.trailing_stop_pct, 0.08)
            if drawdown >= effective_trail:
                trailing_hit = True

            if cfg.use_ma_stop and pos["hold_days"] >= 3 and sym in price_df.columns:
                ma_slice = price_df.loc[:date, sym].dropna().tail(20)
                if len(ma_slice) >= 10:
                    ma20 = ma_slice.mean()
                    if float(px) < ma20:
                        pos["ma_below_count"] = pos.get("ma_below_count", 0) + 1
                        ma_hit = bool(pos["ma_below_count"] >= cfg.ma_stop_days)
                    else:
                        pos["ma_below_count"] = 0

            if trailing_hit or ma_hit:
                pri = str(cfg.ma_trailing_priority).strip().lower()
                if pri not in {"trailing", "ma"}:
                    pri = "trailing"
                if trailing_hit and ma_hit:
                    reason = "ma_stop" if pri == "ma" else "trailing_stop"
                elif trailing_hit:
                    reason = "trailing_stop"
                else:
                    reason = "ma_stop"
                to_sell.append((sym, reason))
                continue

            # (5) 閺冨爼妫垮銏″疮閿涘牅绱崠鏍电礆: 閸掔増婀℃稉鏃€妫ら弰搴ｂ€橀惄鍫濆焺閸掓瑦绔?
            if pos["hold_days"] >= max_hold_days_now and pnl_pct < cfg.time_stop_min_gain:
                to_sell.append((sym, "time_stop"))
                continue

            # 閳光偓閳光偓 閺傝4: 閻╁牆鍩勯崝鐘辩波 閳光偓閳光偓

            # (6) ATR 止盈 / 阶梯止盈 (regime-aware)
            if cfg.use_take_profit:
                atr_val = pos.get("atr_val", 0)
                # 根据市场状态动态调整止盈阈值
                # BULL: 放宽止盈让赢家跑更远; BEAR: 收紧止盈锁定利润
                regime_tp_mult = 1.0
                if regime_tag == "BULL":
                    regime_tp_mult = 1.5
                elif regime_tag == "BEAR":
                    regime_tp_mult = 0.7
                dynamic_tp_pct = cfg.take_profit_fixed_pct * regime_tp_mult
                dynamic_atr_mult = cfg.take_profit_atr_multiplier * regime_tp_mult

                tp_hit = False
                if atr_val > 0:
                    tp_price = pos["entry_price"] + dynamic_atr_mult * atr_val
                    if float(px) >= tp_price:
                        tp_hit = True
                if pnl_pct >= dynamic_tp_pct:
                    tp_hit = True
                if tp_hit:
                    if cfg.use_staged_take_profit:
                        staged_taken = pos.get("staged_tp_taken", 0)
                        levels = cfg.staged_tp_levels
                        if staged_taken < len(levels):
                            threshold_pct, close_ratio = levels[staged_taken]
                            if pnl_pct >= threshold_pct * regime_tp_mult:
                                pos["weight"] = pos["weight"] * (1.0 - close_ratio)
                                pos["staged_tp_taken"] = staged_taken + 1
                                if pos["weight"] < 0.005:
                                    to_sell.append((sym, "take_profit_staged_final"))
                                    continue
                        else:
                            to_sell.append((sym, "take_profit_staged_final"))
                            continue
                    else:
                        to_sell.append((sym, "take_profit"))
                        continue


            if cfg.use_profit_pyramiding and pnl_pct >= cfg.pyramid_trigger_pct:
                adds = pos.get("pyramid_adds", 0)
                if adds < cfg.pyramid_max_adds:
                    add_w = pos["weight"] * cfg.pyramid_add_ratio
                    pos["weight"] = pos["weight"] + add_w
                    pos["pyramid_adds"] = adds + 1

        blocked_buy_orders = 0
        blocked_sell_orders = 0
        partial_buy_orders = 0
        partial_sell_orders = 0
        forced_exit_symbols: set[str] = set()
        for sym, exit_reason in to_sell:
            pos = positions.get(sym)
            if pos is None:
                continue

            if sym in price_df.columns and date in price_df.index:
                exit_px = float(price_df.at[date, sym])
            else:
                exit_px = float(pos["entry_price"])

            if exit_reason in {"missing_price", "invalid_price"}:
                pos = positions.pop(sym, None)
                if pos is None:
                    continue
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
                    "entry_signal_strength": round(float(pos.get("entry_signal_strength", 1.0)), 2),
                    "entry_reason": str(pos.get("entry_reason", "")),
                    "exit_reason": exit_reason,
                    "risk_state": risk_state,
                })
                forced_exit_symbols.add(sym)
                continue

            row = day_symbol_idx.loc[sym] if (not day_symbol_idx.empty and sym in day_symbol_idx.index) else None
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if _is_trade_blocked_by_limit("SELL", sym, exit_px, row, cfg):
                blocked_sell_orders += 1
                continue

            desired_w = float(pos.get("weight", 0.0))
            sell_w = float(desired_w)
            sell_w = _apply_execution_order_constraints(
                desired_w=sell_w,
                px=exit_px,
                row=row,
                equity=equity,
                cfg=cfg,
                side="SELL",
            )
            if sell_w <= 0:
                blocked_sell_orders += 1
                continue

            pnl = (exit_px - pos["entry_price"]) / pos["entry_price"] * 100
            if sell_w + 1e-12 < desired_w:
                partial_sell_orders += 1
                positions[sym]["weight"] = max(0.0, desired_w - sell_w)
                trade_log.append({
                    "symbol": sym,
                    "entry_date": str(pos.get("entry_date", "")),
                    "exit_date": str(date)[:10],
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_px, 2),
                    "hold_days": int(pos["hold_days"]),
                    "pnl_pct": round(pnl, 2),
                    "weight": round(sell_w * 100, 1),
                    "entry_signal_strength": round(float(pos.get("entry_signal_strength", 1.0)), 2),
                    "entry_reason": str(pos.get("entry_reason", "")),
                    "exit_reason": f"{exit_reason}_partial",
                    "risk_state": risk_state,
                })
                forced_exit_symbols.add(sym)
            else:
                pos = positions.pop(sym, None)
                if pos is None:
                    continue
                trade_log.append({
                    "symbol": sym,
                    "entry_date": str(pos.get("entry_date", "")),
                    "exit_date": str(date)[:10],
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_px, 2),
                    "hold_days": int(pos["hold_days"]),
                    "pnl_pct": round(pnl, 2),
                    "weight": round(pos["weight"] * 100, 1),
                    "entry_signal_strength": round(float(pos.get("entry_signal_strength", 1.0)), 2),
                    "entry_reason": str(pos.get("entry_reason", "")),
                    "exit_reason": exit_reason,
                    "risk_state": risk_state,
                })
                forced_exit_symbols.add(sym)

        # Rank-exit rebalance with a minimum holding period to avoid excessive turnover.
        if cfg.use_rank_exit and date in rank_rebalance_days and (index_allow or brake_active):
            rank_exit_cut = int(max(1, cfg.top_k) + max(0, int(cfg.rank_exit_rank_buffer)))
            for sym in list(positions.keys()):
                if sym in target_set:
                    continue
                sym_rank = rank_by_symbol.get(sym, float("inf"))
                if np.isfinite(sym_rank) and sym_rank <= float(rank_exit_cut):
                    continue
                pos = positions.get(sym)
                if pos is None:
                    continue
                if float(pos.get("hold_days", 0.0)) < cfg.rank_exit_min_hold_days:
                    continue
                if cfg.rank_exit_only_when_trend_down:
                    prev_row = (
                        prev_day_symbol_idx.loc[sym]
                        if (not prev_day_symbol_idx.empty and sym in prev_day_symbol_idx.index)
                        else None
                    )
                    if isinstance(prev_row, pd.DataFrame):
                        prev_row = prev_row.iloc[0]
                    trend_up_now = False
                    if prev_row is not None:
                        _trend_v = pd.to_numeric(prev_row.get("trend_up", 0), errors="coerce")
                        trend_up_now = bool(np.isfinite(_trend_v) and float(_trend_v) > 0.0)
                    if trend_up_now:
                        continue
                exit_px = float(price_df.at[date, sym]) if (sym in price_df.columns and date in price_df.index) else float(pos["entry_price"])
                row = day_symbol_idx.loc[sym] if (not day_symbol_idx.empty and sym in day_symbol_idx.index) else None
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                if _is_trade_blocked_by_limit("SELL", sym, exit_px, row, cfg):
                    blocked_sell_orders += 1
                    continue
                desired_w = float(pos.get("weight", 0.0))
                sell_w = _apply_execution_order_constraints(
                    desired_w=desired_w,
                    px=exit_px,
                    row=row,
                    equity=equity,
                    cfg=cfg,
                    side="SELL",
                )
                if sell_w <= 0:
                    blocked_sell_orders += 1
                    continue
                pnl = (exit_px - pos["entry_price"]) / pos["entry_price"] * 100
                if sell_w + 1e-12 < desired_w:
                    partial_sell_orders += 1
                    positions[sym]["weight"] = max(0.0, desired_w - sell_w)
                    trade_log.append({
                        "symbol": sym,
                        "entry_date": str(pos.get("entry_date", "")),
                        "exit_date": str(date)[:10],
                        "entry_price": round(pos["entry_price"], 2),
                        "exit_price": round(exit_px, 2),
                        "hold_days": int(pos["hold_days"]),
                        "pnl_pct": round(pnl, 2),
                        "weight": round(sell_w * 100, 1),
                        "entry_signal_strength": round(float(pos.get("entry_signal_strength", 1.0)), 2),
                        "entry_reason": str(pos.get("entry_reason", "")),
                        "exit_reason": "rank_exit_partial",
                        "risk_state": risk_state,
                    })
                    forced_exit_symbols.add(sym)
                else:
                    pos = positions.pop(sym)
                    trade_log.append({
                        "symbol": sym,
                        "entry_date": str(pos.get("entry_date", "")),
                        "exit_date": str(date)[:10],
                        "entry_price": round(pos["entry_price"], 2),
                        "exit_price": round(exit_px, 2),
                        "hold_days": int(pos["hold_days"]),
                        "pnl_pct": round(pnl, 2),
                        "weight": round(pos["weight"] * 100, 1),
                        "entry_signal_strength": round(float(pos.get("entry_signal_strength", 1.0)), 2),
                        "entry_reason": str(pos.get("entry_reason", "")),
                        "exit_reason": "rank_exit",
                        "risk_state": risk_state,
                    })
                    forced_exit_symbols.add(sym)

        # 閳光偓閳光偓 娴犳挷缍呴崚鍡涘帳閿涘牊鏌熷?B: 娣団€冲娇閸掑棛楠囬敍澶嗘敘閳光偓
        max_pos_today = cfg.max_total_position
        if cfg.use_volatility_sizing:
            max_pos_today *= float(vol_pos_mult)
        if cfg.use_liquidity_state_sizing:
            max_pos_today *= float(liq_pos_mult)
        if cfg.use_market_regime:
            regime_cap = float(regime_pos.get(prev_date, cfg.max_total_position))
            if cfg.use_dynamic_regime_position:
                max_pos_today = min(max_pos_today, regime_cap)
        if cfg.use_index_filter and index_blocked and (not cfg.index_filter_hard_gate):
            max_pos_today = min(max_pos_today, float(cfg.index_filter_block_position_cap))
        if adaptive_dd_active:
            max_pos_today *= float(cfg.adaptive_drawdown_position_cap_multiplier)
        if momentum_crash_active:
            max_pos_today = min(max_pos_today, float(momentum_crash_cap))

        # 閹稿洦鏆熸搴㈠付閿涙矮绮庨弳鍌氫粻閺傛澘绱戞禒鎿勭礉娑撳秴宸遍崚璺哄櫤娴犳搫绱欓崙蹇庣波閻㈠崬娲栭幘銈呭煘鏉烇箒袝閸欐埊绱?

        # During drawdown-brake pause days, force a tighter portfolio cap.
        if brake_active:
            max_pos_today = min(max_pos_today, float(cfg.drawdown_brake_position_cap))

        signal_vals = [float(signal_strengths.get(s, 1.0)) for s in open_symbols]
        avg_signal_strength = float(np.mean(signal_vals)) if signal_vals else 0.0
        if cfg.use_weak_signal_de_risk and signal_vals and avg_signal_strength < cfg.weak_signal_threshold:
            max_pos_today *= float(cfg.weak_signal_cap_multiplier)
        max_pos_today = min(max(0.0, float(max_pos_today)), 1.0)

        n_targets = max(1, len(open_symbols))
        base_w = min(max_pos_today / n_targets, cfg.max_single_weight)

        for sym in open_symbols:
            if not can_open:
                break
            if sym in positions:
                continue
            if sym not in price_df.columns or date not in price_df.index:
                continue
            px = price_df.at[date, sym]
            if not np.isfinite(px) or px <= 0:
                continue
            row = day_symbol_idx.loc[sym] if (not day_symbol_idx.empty and sym in day_symbol_idx.index) else None
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if _is_trade_blocked_by_limit("BUY", sym, float(px), row, cfg):
                blocked_buy_orders += 1
                continue

            # 娣団€冲娇閸掑棛楠囨禒鎾茬秴
            strength = float(signal_strengths.get(sym, 1.0))
            if cfg.use_signal_tiered_sizing:
                if strength >= 2.0:
                    w = base_w * cfg.tier_strong_multiplier
                elif strength >= 1.0:
                    w = base_w * cfg.tier_normal_multiplier
                else:
                    w = base_w * cfg.tier_weak_multiplier
                w = min(w, cfg.max_single_weight)
            else:
                w = base_w

            if (
                str(cfg.weak_entry_mode).strip().lower() == "micro"
                and strength < float(cfg.dual_entry_normal_threshold)
            ):
                w = w * float(cfg.weak_micro_weight_multiplier)
            desired_w = float(w)
            w = _apply_execution_buy_constraints(
                desired_w=desired_w,
                px=float(px),
                row=row,
                equity=equity,
                cfg=cfg,
            )
            if w <= 0:
                blocked_buy_orders += 1
                continue
            if w + 1e-12 < desired_w:
                partial_buy_orders += 1

            # 閼惧嘲褰嘇TR閻劋绨珹TR濮濄垺宕?
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

            strength_label = "strong" if strength >= 2.0 else ("normal" if strength >= 1.0 else "weak")
            score_val = np.nan
            if sym in prev_day["symbol"].values:
                score_row = prev_day.loc[prev_day["symbol"] == sym, "score"]
                if not score_row.empty:
                    score_val = float(score_row.iloc[0])
            if np.isfinite(score_val):
                entry_reason = f"{strength_label};score={score_val:.2f};tdx={tdx_val:.2f}"
            else:
                entry_reason = f"{strength_label};tdx={tdx_val:.2f}"

            # 获取行业信息用于板块集中度控制
            pos_industry = "OTHER"
            if sym in prev_day["symbol"].values and "industry" in prev_day.columns:
                ind_row = prev_day.loc[prev_day["symbol"] == sym, "industry"]
                if not ind_row.empty:
                    pos_industry = str(ind_row.iloc[0])

            positions[sym] = {
                "entry_price": float(px),
                "peak_price": float(px),
                "hold_days": 0.0,
                "weight": float(w),
                "tdx_score": tdx_val,
                "atr_val": atr_val,
                "pyramid_adds": 0,
                "entry_date": str(date)[:10],
                "entry_signal_strength": strength,
                "entry_reason": entry_reason,
                "industry": pos_industry,
            }

        # 闁插秵鏌婇獮瀹犮€€閺夊啴鍣搁敍鍫濆弿閹镐椒绮ㄩ敍澶涚礉绾箽閸︺劑闂勨晝濮搁幀浣风瑓閸欏厴閸戝繋绮?
        risk_budget_adjusted = 0
        if positions:
            risk_budget_adjusted = _apply_light_risk_budget(
                positions=positions,
                prev_day_symbol_idx=prev_day_symbol_idx,
                cfg=cfg,
            )
        if positions:
            total_w_all = sum(float(p.get("weight", 0.0)) for p in positions.values())
            if total_w_all > max_pos_today > 0:
                scale = max_pos_today / total_w_all
                for s in list(positions.keys()):
                    positions[s]["weight"] *= scale

        # 板块集中度控制：限制单板块总仓位
        if cfg.use_industry_diversification and cfg.max_sector_weight < 1.0 and positions:
            sector_weights: dict[str, float] = {}
            for sym_s, pos_s in positions.items():
                ind_s = str(pos_s.get("industry", "OTHER"))
                sector_weights[ind_s] = sector_weights.get(ind_s, 0.0) + float(pos_s.get("weight", 0.0))
            for ind_s, sw in sector_weights.items():
                if sw > cfg.max_sector_weight:
                    scale_down = cfg.max_sector_weight / sw
                    for sym_s, pos_s in positions.items():
                        if str(pos_s.get("industry", "OTHER")) == ind_s:
                            pos_s["weight"] *= scale_down

        band_used = _dynamic_rebalance_band(
            base_band=float(cfg.rebalance_band),
            avg_signal_strength=float(avg_signal_strength),
            cfg=cfg,
        )
        rebalance_band_adjusted = _apply_rebalance_band(
            prev_weights=prev_weights,
            positions=positions,
            rebalance_band=float(band_used),
            locked_symbols=forced_exit_symbols,
            max_total_position=float(max_pos_today),
        )

        # 閻喎鐤勭拫鍐х波閹广垺澧滈敍姘唨娴滃孩妫╅崘鍛扮殶娴犳挸澧犻崥搴㈡綀闁插秴妯婄拋锛勭暬
        new_weights = {sym: float(pos.get("weight", 0.0)) for sym, pos in positions.items()}
        all_syms = set(prev_weights) | set(new_weights)
        turnover = float(sum(abs(new_weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in all_syms))
        buy_turnover, sell_turnover = _turnover_buy_sell(prev_weights, new_weights)
        explicit_cost = _explicit_cost_breakdown_from_turnover(
            buy_turnover=buy_turnover,
            sell_turnover=sell_turnover,
            cfg=cfg,
        )

        exec_cost = _execution_extra_cost_breakdown_from_turnover(
            prev_weights=prev_weights,
            new_weights=new_weights,
            day_df=day,
            equity=equity,
            cfg=cfg,
        )
        extra_exec_cost = float(exec_cost.get("total", 0.0))
        cost = float(explicit_cost.get("total", 0.0)) + extra_exec_cost
        net_ret = gross_ret - cost
        equity *= 1.0 + net_ret
        if equity > equity_peak:
            equity_peak = equity

        total_weight = float(sum(float(pos.get("weight", 0.0)) for pos in positions.values()))

        rec.append(
            {
                "date": date,
                "equity": equity,
                "daily_return": net_ret,
                "gross_return": gross_ret,
                "turnover": turnover,
                "buy_turnover": float(buy_turnover),
                "sell_turnover": float(sell_turnover),
                "explicit_cost": float(explicit_cost.get("total", 0.0)),
                "commission_cost": float(explicit_cost.get("commission", 0.0)),
                "stamp_duty_cost": float(explicit_cost.get("stamp_duty", 0.0)),
                "exchange_fee_cost": float(explicit_cost.get("exchange_fee", 0.0)),
                "regulatory_fee_cost": float(explicit_cost.get("regulatory_fee", 0.0)),
                "transfer_fee_cost": float(explicit_cost.get("transfer_fee", 0.0)),
                "slippage_cost": float(exec_cost.get("slippage", 0.0)),
                "impact_cost": float(exec_cost.get("impact", 0.0)),
                "total_cost": float(cost),
                "extra_exec_cost": float(extra_exec_cost),
                "n_holdings": len(positions),
                "total_position": total_weight,
                "entry_allowed": bool(can_open),
                "dd_brake_active": bool(brake_active),
                "momentum_crash_protect_active": bool(momentum_crash_active),
                "momentum_crash_position_cap": float(momentum_crash_cap if momentum_crash_active else 1.0),
                "portfolio_drawdown": float(1.0 - equity / max(equity_peak, 1e-12)),
                "avg_signal_strength": float(avg_signal_strength),
                "blocked_buy_orders": int(blocked_buy_orders),
                "blocked_sell_orders": int(blocked_sell_orders),
                "partial_buy_orders": int(partial_buy_orders),
                "partial_sell_orders": int(partial_sell_orders),
                "rebalance_band_used": float(band_used),
                "rebalance_band_adjusted": int(rebalance_band_adjusted),
                "risk_budget_adjusted": int(risk_budget_adjusted),
                "volatility_pos_mult": float(vol_pos_mult),
                "liquidity_pos_mult": float(liq_pos_mult),
                "state_pos_mult": float(state_pos_mult),
                "market_vol_annual": float(market_vol_annual) if np.isfinite(market_vol_annual) else np.nan,
                "market_liq_proxy": float(market_liq_proxy) if np.isfinite(market_liq_proxy) else np.nan,
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


def run_walk_forward_oos(
    daily_universe: pd.DataFrame,
    price_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    cfg: BacktestConfigV3,
    regime_df: pd.DataFrame | None,
    index_filter: dict | None,
    allowed_symbols_by_date: dict[pd.Timestamp, set[str]] | None,
    train_days: int = 504,
    test_days: int = 126,
    step_days: int = 126,
) -> pd.DataFrame:
    """Rolling out-of-sample evaluation with fixed parameters."""
    dates = sorted(pd.to_datetime(daily_universe["date"].unique().tolist()))
    if len(dates) < (train_days + test_days + 20):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    fold_id = 0
    max_start = len(dates) - train_days - test_days
    for start_idx in range(0, max_start + 1, max(1, step_days)):
        fold_id += 1
        train_start = dates[start_idx]
        train_end = dates[start_idx + train_days - 1]
        test_start = dates[start_idx + train_days]
        test_end = dates[start_idx + train_days + test_days - 1]

        res = run_backtest_v3(
            daily_universe=daily_universe,
            price_df=price_df,
            returns_df=returns_df,
            cfg=cfg,
            regime_df=regime_df,
            date_start=test_start,
            date_end=test_end,
            index_filter=index_filter,
            allowed_symbols_by_date=allowed_symbols_by_date,
        )
        m = res["metrics"]
        kpi = evaluate_kpi_targets(m, cfg)
        rows.append(
            {
                "fold_id": fold_id,
                "train_start": str(train_start)[:10],
                "train_end": str(train_end)[:10],
                "test_start": str(test_start)[:10],
                "test_end": str(test_end)[:10],
                "annual_return_pct": float(m.get("annual_return_pct", 0.0)),
                "max_drawdown_pct": float(m.get("max_drawdown_pct", 0.0)),
                "sharpe": float(m.get("sharpe", 0.0)),
                "win_rate_pct": float(m.get("win_rate_pct", 0.0)),
                "kpi_all_ok": bool(kpi["all_ok"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_walk_forward_stability(
    wf_df: pd.DataFrame,
    cfg: BacktestConfigV3,
    sharpe_floor: float = 1.0,
) -> dict[str, Any]:
    """Summarize out-of-sample stability with per-fold pass ratios."""
    if wf_df.empty:
        return {}

    total = int(len(wf_df))
    sharpe = pd.to_numeric(wf_df["sharpe"], errors="coerce").fillna(0.0)
    annual = pd.to_numeric(wf_df["annual_return_pct"], errors="coerce").fillna(0.0)
    max_dd = pd.to_numeric(wf_df["max_drawdown_pct"], errors="coerce").fillna(0.0)
    kpi_all_ok = wf_df["kpi_all_ok"].astype(bool) if "kpi_all_ok" in wf_df.columns else pd.Series(False, index=wf_df.index)

    sharpe_ok = sharpe >= float(sharpe_floor)
    annual_positive = annual > 0.0
    drawdown_ok = max_dd.abs() <= float(cfg.target_max_drawdown_limit_pct)

    def _count_and_ratio(mask: pd.Series) -> tuple[int, float]:
        cnt = int(mask.sum())
        ratio = float(cnt / total) if total > 0 else 0.0
        return cnt, ratio

    sharpe_ok_count, sharpe_ok_ratio = _count_and_ratio(sharpe_ok)
    annual_pos_count, annual_pos_ratio = _count_and_ratio(annual_positive)
    drawdown_ok_count, drawdown_ok_ratio = _count_and_ratio(drawdown_ok)
    kpi_all_ok_count, kpi_all_ok_ratio = _count_and_ratio(kpi_all_ok)

    return {
        "fold_count": total,
        "mean_annual_return_pct": float(annual.mean()),
        "mean_max_drawdown_pct": float(max_dd.mean()),
        "mean_sharpe": float(sharpe.mean()),
        "std_sharpe": float(sharpe.std(ddof=1)) if total > 1 else 0.0,
        "sharpe_floor": float(sharpe_floor),
        "sharpe_ok_count": sharpe_ok_count,
        "sharpe_ok_ratio": sharpe_ok_ratio,
        "annual_positive_count": annual_pos_count,
        "annual_positive_ratio": annual_pos_ratio,
        "drawdown_ok_count": drawdown_ok_count,
        "drawdown_ok_ratio": drawdown_ok_ratio,
        "kpi_all_ok_count": kpi_all_ok_count,
        "kpi_all_ok_ratio": kpi_all_ok_ratio,
    }


def summarize_execution_realism_layer(
    eq: pd.DataFrame,
    trades: list[dict[str, Any]],
    cfg: BacktestConfigV3,
    parallel_results: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if eq is None or eq.empty:
        return {
            "status": "no_data",
            "realism_enabled": bool(cfg.use_execution_realism),
            "days": 0,
        }

    df = eq.copy()
    for col in [
        "turnover",
        "buy_turnover",
        "sell_turnover",
        "volatility_pos_mult",
        "liquidity_pos_mult",
        "state_pos_mult",
        "market_vol_annual",
        "market_liq_proxy",
        "explicit_cost",
        "commission_cost",
        "stamp_duty_cost",
        "exchange_fee_cost",
        "regulatory_fee_cost",
        "transfer_fee_cost",
        "slippage_cost",
        "impact_cost",
        "total_cost",
        "extra_exec_cost",
        "blocked_buy_orders",
        "blocked_sell_orders",
        "partial_buy_orders",
        "partial_sell_orders",
    ]:
        if col not in df.columns:
            df[col] = 0.0

    days = int(len(df))
    blocked_buy = int(pd.to_numeric(df["blocked_buy_orders"], errors="coerce").fillna(0).sum())
    blocked_sell = int(pd.to_numeric(df["blocked_sell_orders"], errors="coerce").fillna(0).sum())
    partial_buy = int(pd.to_numeric(df["partial_buy_orders"], errors="coerce").fillna(0).sum())
    partial_sell = int(pd.to_numeric(df["partial_sell_orders"], errors="coerce").fillna(0).sum())

    blocked_days = int(
        (
            pd.to_numeric(df["blocked_buy_orders"], errors="coerce").fillna(0)
            + pd.to_numeric(df["blocked_sell_orders"], errors="coerce").fillna(0)
        ).gt(0).sum()
    )
    partial_days = int(
        (
            pd.to_numeric(df["partial_buy_orders"], errors="coerce").fillna(0)
            + pd.to_numeric(df["partial_sell_orders"], errors="coerce").fillna(0)
        ).gt(0).sum()
    )

    avg_turnover = float(pd.to_numeric(df["turnover"], errors="coerce").fillna(0).mean())
    total_turnover = float(pd.to_numeric(df["turnover"], errors="coerce").fillna(0).sum())
    avg_buy_turnover = float(pd.to_numeric(df["buy_turnover"], errors="coerce").fillna(0).mean())
    avg_sell_turnover = float(pd.to_numeric(df["sell_turnover"], errors="coerce").fillna(0).mean())
    avg_vol_pos_mult = float(pd.to_numeric(df["volatility_pos_mult"], errors="coerce").fillna(1.0).mean())
    avg_liq_pos_mult = float(pd.to_numeric(df["liquidity_pos_mult"], errors="coerce").fillna(1.0).mean())
    avg_state_pos_mult = float(pd.to_numeric(df["state_pos_mult"], errors="coerce").fillna(1.0).mean())
    avg_market_vol_annual = float(pd.to_numeric(df["market_vol_annual"], errors="coerce").dropna().mean()) if pd.to_numeric(df["market_vol_annual"], errors="coerce").notna().any() else float("nan")
    avg_market_liq_proxy = float(pd.to_numeric(df["market_liq_proxy"], errors="coerce").dropna().mean()) if pd.to_numeric(df["market_liq_proxy"], errors="coerce").notna().any() else float("nan")
    avg_explicit_cost_bps = float(pd.to_numeric(df["explicit_cost"], errors="coerce").fillna(0).mean() * 10000.0)
    avg_commission_cost_bps = float(pd.to_numeric(df["commission_cost"], errors="coerce").fillna(0).mean() * 10000.0)
    avg_stamp_cost_bps = float(pd.to_numeric(df["stamp_duty_cost"], errors="coerce").fillna(0).mean() * 10000.0)
    avg_exchange_cost_bps = float(pd.to_numeric(df["exchange_fee_cost"], errors="coerce").fillna(0).mean() * 10000.0)
    avg_regulatory_cost_bps = float(pd.to_numeric(df["regulatory_fee_cost"], errors="coerce").fillna(0).mean() * 10000.0)
    avg_transfer_cost_bps = float(pd.to_numeric(df["transfer_fee_cost"], errors="coerce").fillna(0).mean() * 10000.0)
    avg_slippage_cost_bps = float(pd.to_numeric(df["slippage_cost"], errors="coerce").fillna(0).mean() * 10000.0)
    avg_impact_cost_bps = float(pd.to_numeric(df["impact_cost"], errors="coerce").fillna(0).mean() * 10000.0)
    avg_total_cost_bps = float(pd.to_numeric(df["total_cost"], errors="coerce").fillna(0).mean() * 10000.0)
    avg_extra_cost_bps = float(pd.to_numeric(df["extra_exec_cost"], errors="coerce").fillna(0).mean() * 10000.0)
    total_explicit_cost_pct = float(pd.to_numeric(df["explicit_cost"], errors="coerce").fillna(0).sum() * 100.0)
    total_extra_cost_pct = float(pd.to_numeric(df["extra_exec_cost"], errors="coerce").fillna(0).sum() * 100.0)
    total_cost_pct = float(pd.to_numeric(df["total_cost"], errors="coerce").fillna(0).sum() * 100.0)
    realized_extra_bps_per_turnover = (
        float(total_extra_cost_pct * 100.0 / max(total_turnover, 1e-12)) if total_turnover > 0 else 0.0
    )

    trade_count = int(len(trades))
    partial_trade_logs = int(
        sum(1 for t in trades if "_partial" in str(t.get("exit_reason", "")).lower())
    )

    mode_sensitivity: dict[str, Any] = {}
    if parallel_results and {"close", "next_open"}.issubset(set(parallel_results.keys())):
        m_close = parallel_results["close"].get("metrics", {})
        m_open = parallel_results["next_open"].get("metrics", {})
        annual_gap = float(m_open.get("annual_return_pct", 0.0)) - float(m_close.get("annual_return_pct", 0.0))
        sharpe_gap = float(m_open.get("sharpe", 0.0)) - float(m_close.get("sharpe", 0.0))
        dd_gap = float(m_open.get("max_drawdown_pct", 0.0)) - float(m_close.get("max_drawdown_pct", 0.0))
        mode_sensitivity = {
            "annual_gap_pctpt_next_open_minus_close": annual_gap,
            "sharpe_gap_next_open_minus_close": sharpe_gap,
            "max_drawdown_gap_pctpt_next_open_minus_close": dd_gap,
            "abs_annual_gap_pctpt": abs(annual_gap),
            "abs_sharpe_gap": abs(sharpe_gap),
        }

    if not cfg.use_execution_realism:
        status = "baseline_only"
    else:
        blocked_day_ratio = blocked_days / max(days, 1)
        if blocked_day_ratio > 0.40 or avg_extra_cost_bps > 40.0:
            status = "warning"
        else:
            status = "ok"

    return {
        "status": status,
        "realism_enabled": bool(cfg.use_execution_realism),
        "days": days,
        "blocked_buy_orders_total": blocked_buy,
        "blocked_sell_orders_total": blocked_sell,
        "partial_buy_orders_total": partial_buy,
        "partial_sell_orders_total": partial_sell,
        "blocked_days": blocked_days,
        "partial_days": partial_days,
        "blocked_day_ratio": float(blocked_days / max(days, 1)),
        "partial_day_ratio": float(partial_days / max(days, 1)),
        "avg_turnover": avg_turnover,
        "avg_buy_turnover": avg_buy_turnover,
        "avg_sell_turnover": avg_sell_turnover,
        "avg_volatility_pos_mult": avg_vol_pos_mult,
        "avg_liquidity_pos_mult": avg_liq_pos_mult,
        "avg_state_pos_mult": avg_state_pos_mult,
        "avg_market_vol_annual": avg_market_vol_annual,
        "avg_market_liq_proxy": avg_market_liq_proxy,
        "total_turnover": total_turnover,
        "avg_explicit_cost_bps": avg_explicit_cost_bps,
        "avg_commission_cost_bps": avg_commission_cost_bps,
        "avg_stamp_duty_cost_bps": avg_stamp_cost_bps,
        "avg_exchange_fee_cost_bps": avg_exchange_cost_bps,
        "avg_regulatory_fee_cost_bps": avg_regulatory_cost_bps,
        "avg_transfer_fee_cost_bps": avg_transfer_cost_bps,
        "avg_slippage_cost_bps": avg_slippage_cost_bps,
        "avg_impact_cost_bps": avg_impact_cost_bps,
        "avg_total_cost_bps": avg_total_cost_bps,
        "total_explicit_cost_pct": total_explicit_cost_pct,
        "avg_extra_exec_cost_bps": avg_extra_cost_bps,
        "total_extra_exec_cost_pct": total_extra_cost_pct,
        "total_cost_pct": total_cost_pct,
        "extra_cost_bps_per_turnover_unit": realized_extra_bps_per_turnover,
        "trade_count": trade_count,
        "partial_trade_log_count": partial_trade_logs,
        "mode_sensitivity": mode_sensitivity,
    }


def _extract_active_windows(
    dates: pd.Series,
    active_flags: pd.Series,
    trigger_flags: pd.Series | None = None,
    crash_lb_ret: pd.Series | None = None,
    rebound_lb_ret: pd.Series | None = None,
) -> list[dict[str, Any]]:
    if dates is None or active_flags is None or len(dates) == 0:
        return []

    d = pd.to_datetime(dates)
    active = pd.Series(active_flags).fillna(False).astype(bool).reset_index(drop=True)
    trigger = (
        pd.Series(trigger_flags).fillna(False).astype(bool).reset_index(drop=True)
        if trigger_flags is not None
        else pd.Series(False, index=active.index)
    )
    crash = (
        pd.to_numeric(pd.Series(crash_lb_ret), errors="coerce").reset_index(drop=True)
        if crash_lb_ret is not None
        else pd.Series(np.nan, index=active.index)
    )
    rebound = (
        pd.to_numeric(pd.Series(rebound_lb_ret), errors="coerce").reset_index(drop=True)
        if rebound_lb_ret is not None
        else pd.Series(np.nan, index=active.index)
    )

    windows: list[dict[str, Any]] = []
    n = int(len(active))
    i = 0
    wid = 0
    while i < n:
        if not bool(active.iloc[i]):
            i += 1
            continue
        s = i
        while i + 1 < n and bool(active.iloc[i + 1]):
            i += 1
        e = i
        seg_trigger_idx = trigger.iloc[s : e + 1][trigger.iloc[s : e + 1]].index.tolist()
        t_idx = int(seg_trigger_idx[0]) if seg_trigger_idx else int(s)
        windows.append(
            {
                "window_id": int(wid),
                "start_date": str(d.iloc[s].date()),
                "end_date": str(d.iloc[e].date()),
                "days": int(e - s + 1),
                "trigger_date": str(d.iloc[t_idx].date()),
                "crash_lb_ret_at_trigger": float(crash.iloc[t_idx]) if np.isfinite(crash.iloc[t_idx]) else None,
                "rebound_lb_ret_at_trigger": float(rebound.iloc[t_idx]) if np.isfinite(rebound.iloc[t_idx]) else None,
            }
        )
        wid += 1
        i += 1

    for k in range(len(windows)):
        if k + 1 < len(windows):
            end_dt = pd.to_datetime(windows[k]["end_date"])
            next_start_dt = pd.to_datetime(windows[k + 1]["start_date"])
            gap = int(max(0, (next_start_dt - end_dt).days - 1))
            windows[k]["recovery_gap_days_to_next_window"] = int(gap)
        else:
            windows[k]["recovery_gap_days_to_next_window"] = None
    return windows


def summarize_momentum_crash_layer(
    cfg: BacktestConfigV3,
    regime_df: pd.DataFrame | None,
    eq: pd.DataFrame | None,
) -> dict[str, Any]:
    enabled = bool(cfg.use_momentum_crash_protection)
    base = {
        "enabled": enabled,
        "params": {
            "crash_lookback_days": int(cfg.momentum_crash_lookback_days),
            "crash_drop_threshold": float(cfg.momentum_crash_drop_threshold),
            "rebound_lookback_days": int(cfg.momentum_rebound_lookback_days),
            "rebound_threshold": float(cfg.momentum_rebound_threshold),
            "protection_days": int(cfg.momentum_crash_protection_days),
            "position_cap": float(cfg.momentum_crash_position_cap),
        },
    }
    if not enabled:
        return {
            **base,
            "status": "disabled",
            "window_count": 0,
            "trigger_count": 0,
            "active_days": 0,
            "active_day_ratio": 0.0,
            "windows": [],
            "recovery_conditions": ["崩盘保护未启用。"],
        }
    if regime_df is None or regime_df.empty:
        return {
            **base,
            "status": "no_data",
            "window_count": 0,
            "trigger_count": 0,
            "active_days": 0,
            "active_day_ratio": 0.0,
            "windows": [],
            "recovery_conditions": ["缺少市场状态数据，无法评估崩盘保护触发。"],
        }

    rr = regime_df.copy()
    rr["date"] = pd.to_datetime(rr["date"])
    rr = rr.sort_values("date").reset_index(drop=True)
    active = pd.to_numeric(rr.get("momentum_crash_active", False), errors="coerce").fillna(0).astype(bool)
    trigger = pd.to_numeric(rr.get("momentum_crash_trigger", 0), errors="coerce").fillna(0).astype(int).gt(0)
    crash_lb = pd.to_numeric(rr.get("crash_lb_ret", np.nan), errors="coerce")
    rebound_lb = pd.to_numeric(rr.get("rebound_lb_ret", np.nan), errors="coerce")

    windows = _extract_active_windows(
        dates=rr["date"],
        active_flags=active,
        trigger_flags=trigger,
        crash_lb_ret=crash_lb,
        rebound_lb_ret=rebound_lb,
    )
    total_days = max(1, int(len(rr)))
    active_days = int(active.sum())
    trigger_count = int(trigger.sum())
    active_ratio = float(active_days / total_days)

    active_ret_mean = 0.0
    inactive_ret_mean = 0.0
    active_ret_sharpe = 0.0
    inactive_ret_sharpe = 0.0
    if eq is not None and (not eq.empty) and ("date" in eq.columns) and ("daily_return" in eq.columns):
        ee = eq[["date", "daily_return"]].copy()
        ee["date"] = pd.to_datetime(ee["date"])
        ee["daily_return"] = pd.to_numeric(ee["daily_return"], errors="coerce").fillna(0.0)
        mm = rr[["date"]].copy()
        mm["active"] = active.values
        merged = ee.merge(mm, on="date", how="inner")
        if not merged.empty:
            act_r = pd.to_numeric(merged.loc[merged["active"], "daily_return"], errors="coerce").fillna(0.0)
            inact_r = pd.to_numeric(merged.loc[~merged["active"], "daily_return"], errors="coerce").fillna(0.0)
            active_ret_mean = float(act_r.mean()) if len(act_r) > 0 else 0.0
            inactive_ret_mean = float(inact_r.mean()) if len(inact_r) > 0 else 0.0
            active_ret_sharpe = _sharpe(act_r) if len(act_r) > 1 else 0.0
            inactive_ret_sharpe = _sharpe(inact_r) if len(inact_r) > 1 else 0.0

    gap_vals = [
        int(w.get("recovery_gap_days_to_next_window"))
        for w in windows
        if w.get("recovery_gap_days_to_next_window") is not None
    ]
    recovery_p50 = int(np.median(gap_vals)) if gap_vals else None
    recovery_p75 = int(np.percentile(gap_vals, 75)) if gap_vals else None

    if trigger_count == 0:
        status = "idle"
    elif active_ratio > 0.35:
        status = "warning"
    elif trigger_count >= 15:
        status = "watch"
    else:
        status = "ok"

    hold_n = max(2, min(6, int(cfg.momentum_crash_protection_days)))
    recovery_conditions = [
        f"连续 {hold_n} 天无新触发。",
        f"crash_lb_ret 回升至 {float(cfg.momentum_crash_drop_threshold) * 0.5:+.2%} 以上。",
        f"rebound_lb_ret 回落到 {float(cfg.momentum_rebound_threshold):+.2%} 以下。",
    ]
    if recovery_p50 is not None:
        recovery_conditions.append(f"历史中位恢复间隔约 {int(recovery_p50)} 天（P75={int(recovery_p75 or recovery_p50)} 天）。")

    return {
        **base,
        "status": status,
        "window_count": int(len(windows)),
        "trigger_count": trigger_count,
        "active_days": active_days,
        "active_day_ratio": active_ratio,
        "active_return_mean": active_ret_mean,
        "inactive_return_mean": inactive_ret_mean,
        "active_return_sharpe": active_ret_sharpe,
        "inactive_return_sharpe": inactive_ret_sharpe,
        "recovery_gap_p50_days": recovery_p50,
        "recovery_gap_p75_days": recovery_p75,
        "windows": windows,
        "recovery_conditions": recovery_conditions,
    }


def summarize_stability_layer(
    eq: pd.DataFrame,
    wf_stability_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if eq is None or eq.empty:
        return {"status": "no_data", "month_count": 0}

    df = eq.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    nav = pd.to_numeric(df["equity"], errors="coerce")
    date_index = pd.DatetimeIndex(df["date"])
    month_nav = pd.Series(nav.values, index=date_index).resample("ME").last().dropna()
    month_ret = month_nav.pct_change(fill_method=None).dropna()

    if month_ret.empty:
        month_count = 0
        pos_ratio = 0.0
        mean_m = 0.0
        std_m = 0.0
        worst_m = 0.0
        best_m = 0.0
    else:
        month_count = int(len(month_ret))
        pos_ratio = float((month_ret > 0).mean())
        mean_m = float(month_ret.mean() * 100.0)
        std_m = float(month_ret.std(ddof=1) * 100.0) if month_count > 1 else 0.0
        worst_m = float(month_ret.min() * 100.0)
        best_m = float(month_ret.max() * 100.0)

    status = "ok"
    wf = wf_stability_summary or {}
    if wf:
        sharpe_ok_ratio = float(wf.get("sharpe_ok_ratio", 0.0))
        annual_pos_ratio = float(wf.get("annual_positive_ratio", 0.0))
        drawdown_ok_ratio = float(wf.get("drawdown_ok_ratio", 0.0))
        if sharpe_ok_ratio < 0.50 or annual_pos_ratio < 0.60 or drawdown_ok_ratio < 0.80:
            status = "warning"
    else:
        if month_count >= 3 and (pos_ratio < 0.45 or worst_m < -15.0):
            status = "warning"

    return {
        "status": status,
        "month_count": month_count,
        "positive_month_ratio": pos_ratio,
        "mean_monthly_return_pct": mean_m,
        "std_monthly_return_pct": std_m,
        "worst_month_return_pct": worst_m,
        "best_month_return_pct": best_m,
        "walk_forward": wf,
    }


def build_three_layer_evaluation(
    cfg: BacktestConfigV3,
    metrics: dict[str, float],
    kpi_status: dict[str, Any],
    eq: pd.DataFrame,
    trades: list[dict[str, Any]],
    wf_stability_summary: dict[str, Any] | None = None,
    parallel_results: dict[str, dict[str, Any]] | None = None,
    momentum_crash_layer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layer1_status = "ok" if bool(kpi_status.get("all_ok", False)) else "warning"
    layer1 = {
        "status": layer1_status,
        "metrics": {
            "annual_return_pct": float(metrics.get("annual_return_pct", 0.0)),
            "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
            "sharpe": float(metrics.get("sharpe", 0.0)),
            "win_rate_pct": float(metrics.get("win_rate_pct", 0.0)),
        },
        "kpi_status": kpi_status,
        "kpi_targets": {
            "annual_return_min_pct": float(cfg.target_annual_return_min_pct),
            "annual_return_max_pct": float(cfg.target_annual_return_max_pct),
            "max_drawdown_limit_pct": float(cfg.target_max_drawdown_limit_pct),
            "sharpe_min": float(cfg.target_sharpe_min),
        },
    }
    layer2 = summarize_execution_realism_layer(eq, trades, cfg, parallel_results=parallel_results)
    layer2b = momentum_crash_layer or {"status": "no_data", "enabled": bool(cfg.use_momentum_crash_protection)}
    layer3 = summarize_stability_layer(eq, wf_stability_summary=wf_stability_summary)

    statuses = [
        str(layer1.get("status", "")),
        str(layer2.get("status", "")),
        str(layer2b.get("status", "")),
        str(layer3.get("status", "")),
    ]
    if "warning" in statuses:
        overall = "warning"
    elif "watch" in statuses:
        overall = "warning"
    elif "no_data" in statuses:
        overall = "partial"
    else:
        overall = "ok"

    period_start = None
    period_end = None
    if eq is not None and not eq.empty and "date" in eq.columns:
        period_start = str(pd.to_datetime(eq["date"]).min())[:10]
        period_end = str(pd.to_datetime(eq["date"]).max())[:10]

    return {
        "overall_status": overall,
        "period_start": period_start,
        "period_end": period_end,
        "layer_1_backtest": layer1,
        "layer_2_execution": layer2,
        "layer_2b_momentum_crash": layer2b,
        "layer_3_stability": layer3,
    }


def render_three_layer_report_md(three_layer: dict[str, Any]) -> str:
    l1 = three_layer.get("layer_1_backtest", {})
    l2 = three_layer.get("layer_2_execution", {})
    l2b = three_layer.get("layer_2b_momentum_crash", {})
    l3 = three_layer.get("layer_3_stability", {})
    m1 = l1.get("metrics", {})
    kpi = l1.get("kpi_status", {})
    wf = l3.get("walk_forward", {}) if isinstance(l3.get("walk_forward", {}), dict) else {}

    lines = [
        "# Three-Layer Evaluation",
        "",
        f"- Overall status: **{three_layer.get('overall_status', 'unknown')}**",
        f"- Period: {three_layer.get('period_start', 'N/A')} -> {three_layer.get('period_end', 'N/A')}",
        "",
        "## Layer 1 - Backtest",
        f"- Status: {l1.get('status', 'unknown')}",
        f"- Annual return: {float(m1.get('annual_return_pct', 0.0)):+.2f}%",
        f"- Max drawdown: {float(m1.get('max_drawdown_pct', 0.0)):.2f}%",
        f"- Sharpe: {float(m1.get('sharpe', 0.0)):.2f}",
        f"- Win rate: {float(m1.get('win_rate_pct', 0.0)):.2f}%",
        f"- KPI all ok: {bool(kpi.get('all_ok', False))}",
        "",
        "## Layer 2 - Execution",
        f"- Status: {l2.get('status', 'unknown')}",
        f"- Realism enabled: {bool(l2.get('realism_enabled', False))}",
        f"- Blocked orders (buy/sell): {int(l2.get('blocked_buy_orders_total', 0))}/{int(l2.get('blocked_sell_orders_total', 0))}",
        f"- Partial orders (buy/sell): {int(l2.get('partial_buy_orders_total', 0))}/{int(l2.get('partial_sell_orders_total', 0))}",
        f"- Avg turnover: {float(l2.get('avg_turnover', 0.0)):.4f}",
        f"- Avg position multipliers (vol/liquidity/combined): "
        f"{float(l2.get('avg_volatility_pos_mult', 1.0)):.3f}/"
        f"{float(l2.get('avg_liquidity_pos_mult', 1.0)):.3f}/"
        f"{float(l2.get('avg_state_pos_mult', 1.0)):.3f}",
        f"- Avg explicit cost: {float(l2.get('avg_explicit_cost_bps', 0.0)):.2f} bps",
        f"- Avg total trading cost: {float(l2.get('avg_total_cost_bps', 0.0)):.2f} bps",
        f"- Cost split (commission/stamp/slippage/impact): "
        f"{float(l2.get('avg_commission_cost_bps', 0.0)):.2f}/"
        f"{float(l2.get('avg_stamp_duty_cost_bps', 0.0)):.2f}/"
        f"{float(l2.get('avg_slippage_cost_bps', 0.0)):.2f}/"
        f"{float(l2.get('avg_impact_cost_bps', 0.0)):.2f} bps",
        f"- Avg extra execution cost: {float(l2.get('avg_extra_exec_cost_bps', 0.0)):.2f} bps",
        f"- Total extra execution cost: {float(l2.get('total_extra_exec_cost_pct', 0.0)):.3f}%",
        f"- Total explicit/extra/total cost: "
        f"{float(l2.get('total_explicit_cost_pct', 0.0)):.3f}%/"
        f"{float(l2.get('total_extra_exec_cost_pct', 0.0)):.3f}%/"
        f"{float(l2.get('total_cost_pct', 0.0)):.3f}%",
        "",
        "## Layer 2B - Momentum Crash Protection",
        f"- Status: {l2b.get('status', 'unknown')}",
        f"- Enabled: {bool(l2b.get('enabled', False))}",
        f"- Trigger count: {int(l2b.get('trigger_count', 0))}",
        f"- Active days / ratio: {int(l2b.get('active_days', 0))} / {float(l2b.get('active_day_ratio', 0.0)) * 100.0:.1f}%",
        f"- Window count: {int(l2b.get('window_count', 0))}",
        f"- Active return mean: {float(l2b.get('active_return_mean', 0.0)) * 100.0:+.3f}%",
        f"- Inactive return mean: {float(l2b.get('inactive_return_mean', 0.0)) * 100.0:+.3f}%",
        "",
        "## Layer 3 - Stability",
        f"- Status: {l3.get('status', 'unknown')}",
        f"- Positive month ratio: {float(l3.get('positive_month_ratio', 0.0)) * 100.0:.1f}%",
        f"- Mean monthly return: {float(l3.get('mean_monthly_return_pct', 0.0)):+.2f}%",
        f"- Worst month: {float(l3.get('worst_month_return_pct', 0.0)):.2f}%",
        f"- Best month: {float(l3.get('best_month_return_pct', 0.0)):.2f}%",
    ]
    if wf:
        lines.extend(
            [
                "",
                "### Walk-Forward",
                f"- Fold count: {int(wf.get('fold_count', 0))}",
                f"- Sharpe OK ratio: {float(wf.get('sharpe_ok_ratio', 0.0)) * 100.0:.1f}%",
                f"- Annual positive ratio: {float(wf.get('annual_positive_ratio', 0.0)) * 100.0:.1f}%",
                f"- Drawdown OK ratio: {float(wf.get('drawdown_ok_ratio', 0.0)) * 100.0:.1f}%",
            ]
        )
    return "\n".join(lines)


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
    """Build monthly (or periodic) dynamic watchlist from daily-universe ranking."""
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
            cands = day.copy()
            if "tradeable" in cands.columns:
                cands = cands[cands["tradeable"].fillna(1).astype(int) == 1].copy()
            if cands.empty:
                cands = day.copy()

            # 娑撳骸鍙嗛崷娲偓鏄忕帆鐎靛綊缍堥敍姘喘閸忓牅绻氶悾娆愬畾閸嬫粍妞跨捄?+ 娑撹濮忛幒褏娲忔稉鐑?
            strict_mask = (
                (cands.get("has_limit_up_30d_calc", pd.Series(0.0, index=cands.index)).fillna(0) >= 1)
                & (cands.get("main_force_pct", pd.Series(0.0, index=cands.index)).fillna(0) > 0)
            )
            strict_cands = cands[strict_mask].copy()
            if len(strict_cands) >= max_symbols // 2:
                cands = strict_cands

            s_tdx = _pct_rank(cands.get("tdx_score", pd.Series(0.0, index=cands.index)))
            s_main_force = _pct_rank(cands.get("main_force_pct", pd.Series(0.0, index=cands.index)))
            s_ma = _pct_rank(cands.get("ma_dist_20", pd.Series(0.0, index=cands.index)))
            s_ret20 = _pct_rank(cands.get("ret_20d", pd.Series(0.0, index=cands.index)))
            s_amount = _pct_rank(cands.get("amount_20d", pd.Series(0.0, index=cands.index)))
            s_turn = _pct_rank(cands.get("turnover_20d", pd.Series(0.0, index=cands.index)))
            s_low_vol = 1.0 - _pct_rank(cands.get("vol_20d", pd.Series(0.0, index=cands.index)))
            s_low_atr = 1.0 - _pct_rank(cands.get("atr_pct", pd.Series(0.01, index=cands.index)))
            s_breakout = cands.get("high30_new_high", pd.Series(0.0, index=cands.index)).fillna(0).astype(float)
            s_limit = cands.get("has_limit_up_30d_calc", pd.Series(0.0, index=cands.index)).fillna(0).astype(float)
            s_month = cands.get("monthly_bullish", pd.Series(0.0, index=cands.index)).fillna(0).astype(float)

            cands["dyn_pool_score"] = (
                0.28 * s_tdx
                + 0.16 * s_main_force
                + 0.12 * s_breakout
                + 0.10 * s_ma
                + 0.08 * s_ret20
                + 0.10 * s_amount
                + 0.08 * s_turn
                + 0.05 * s_low_vol
                + 0.03 * s_low_atr
                + 0.04 * s_limit
                + 0.03 * s_month
            )

            if "vol_20d" in cands.columns:
                cands.loc[cands["vol_20d"].fillna(0) > 90, "dyn_pool_score"] -= 0.20
            if "extreme_days_30d" in cands.columns:
                cands.loc[cands["extreme_days_30d"].fillna(0) > 4, "dyn_pool_score"] -= 0.20

            cands = cands.sort_values(
                ["dyn_pool_score", "tdx_score", "amount_20d", "turnover_20d"],
                ascending=[False, False, False, False],
            )
            current_universe = set(cands.head(max_symbols)["symbol"].astype(str).str.upper().tolist())

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

    tuning_md = "\n".join(
        [
            "# Parameter Tuning Report",
            "",
            "## Targets",
            "- Sharpe > 1.5",
            "- Max drawdown < 15%",
            "- Win rate > 55%",
            "",
            "## Method",
            "- Grid search: full grid (4^6 = 4096 combos)",
            "- Time-series CV: TimeSeriesSplit(n_splits=5)",
            "- Backtest window: 2023-01-01 to latest trading day",
            "- Objective: Sharpe + return/win-rate bonus - drawdown/instability penalty",
            "",
            "## Best Parameters",
            "```yaml",
            yaml.safe_dump(asdict(cfg_best), sort_keys=False, allow_unicode=True).rstrip(),
            "```",
            "",
            "## CV Summary Top10",
            best10.to_markdown(index=False),
            "",
            "## Fold Details (Top 40 rows)",
            fold_head.to_markdown(index=False),
            "",
            "## Sensitivity Notes",
            f"- Largest-impact parameter: `{impact_most}`",
            f"- Most-stable parameter: `{stable_most}`",
            "",
            "### Parameter Impact Table",
            imp.to_markdown(index=False),
            "",
            "## Target Check (Best Params on Full Sample)",
            f"- Sharpe: {strategy_metrics['sharpe']:.2f}",
            f"- Max drawdown: {strategy_metrics['max_drawdown_pct']:.2f}%",
            f"- Win rate: {strategy_metrics['win_rate_pct']:.2f}%",
        ]
    )
    tuning_md_path.write_text(tuning_md, encoding="utf-8")

    cmp_df = pd.DataFrame(
        [
            {"strategy": "Optimized V3", **strategy_metrics},
            {"strategy": "HS300", **hs300_metrics},
            {"strategy": "Buy&Hold EqualWeight", **buy_hold_metrics},
        ]
    ).round(4)

    compare_md = "\n".join(
        [
            "# Backtest Comparison Report",
            "",
            "## Window",
            "- 2023-01-01 to latest trading day",
            "",
            "## Result Comparison",
            cmp_df.to_markdown(index=False),
            "",
            "## Key Metrics",
            f"- Optimized strategy Sharpe: {strategy_metrics['sharpe']:.2f}",
            f"- Optimized strategy Max drawdown: {strategy_metrics['max_drawdown_pct']:.2f}%",
            f"- Optimized strategy Win rate: {strategy_metrics['win_rate_pct']:.2f}%",
            "",
            "## Sensitivity Figure",
            f"- `{sensitivity_png}`",
        ]
    )
    compare_md_path.write_text(compare_md, encoding="utf-8")

    html = "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh">',
            "<head>",
            '  <meta charset="utf-8" />',
            "  <title>Backtest Comparison</title>",
            "  <style>",
            "    body { font-family: Arial, sans-serif; margin: 24px; }",
            "    h1, h2 { margin: 0 0 12px 0; }",
            "    .table { border-collapse: collapse; width: 100%; }",
            "    .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: right; }",
            "    .table th:first-child, .table td:first-child { text-align: left; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Backtest Comparison Report</h1>",
            "  <h2>Metric Comparison</h2>",
            f"  {to_html_table(cmp_df)}",
            "  <h2>Top10 Parameter Combos</h2>",
            f"  {to_html_table(best10)}",
            "  <h2>Sensitivity Figure</h2>",
            f'  <p><img src="{sensitivity_png.name}" style="max-width: 100%;" /></p>',
            "</body>",
            "</html>",
        ]
    )
    compare_html_path.write_text(html, encoding="utf-8")

    doc = "\n".join(
        [
            "# Strategy Notes (Optimized v3)",
            "",
            "## Framework",
            "1. Entry: limit-up pullback + TDX + trend checks",
            "2. Risk: hard stop + trailing stop + time stop",
            "3. Positioning: volatility-based sizing + single-name/portfolio caps",
            "",
            "## Enhancements",
            "1. Industry diversification",
            "2. Liquidity filters",
            "3. Market-cap filters",
            "4. Dynamic watchlist support",
            "",
            "## Optimization Setup",
            "1. TimeSeriesSplit with 5 folds",
            "2. 4096 parameter combinations",
            "3. Composite objective with stability penalty",
            "",
            "## Best Params",
            f"- top_k: {cfg_best.top_k}",
            f"- invest_more_n: {cfg_best.invest_more_n}",
            f"- pullback_min_pct: {cfg_best.pullback_min_pct}",
            f"- pullback_max_pct: {cfg_best.pullback_max_pct}",
            f"- stop_loss_pct: {cfg_best.stop_loss_pct}",
            f"- trailing_stop_pct: {cfg_best.trailing_stop_pct}",
            "",
            "## Full-Sample Metrics",
            f"- annual_return_pct: {strategy_metrics['annual_return_pct']:.2f}%",
            f"- max_drawdown_pct: {strategy_metrics['max_drawdown_pct']:.2f}%",
            f"- sharpe: {strategy_metrics['sharpe']:.2f}",
            f"- win_rate_pct: {strategy_metrics['win_rate_pct']:.2f}%",
        ]
    )
    strategy_doc_path.write_text(doc, encoding="utf-8")

    eq_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_equity.csv"
    eq_out.parent.mkdir(parents=True, exist_ok=True)
    strategy_eq.to_csv(eq_out, index=False, encoding="utf-8-sig")

    stats_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_stats.json"
    stats_out.write_text(json.dumps(strategy_metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def run_optimization_pipeline(base_dir: Path, start_date: str) -> dict[str, Any]:
    print("Loading and preparing data...")
    feats = load_and_prepare_features(
        base_dir,
        start_date=start_date,
        news_sentiment_lag_days=int(cfg.news_sentiment_lag_days),
    )

    industry_map = build_industry_map_from_config(base_dir, feats["symbol"])
    feats["industry"] = feats["symbol"].map(industry_map).fillna("OTHER")

    print("Precomputing daily universe...")
    daily = precompute_daily_universe(feats)

    price_df = feats.pivot_table(index="date", columns="symbol", values="close").sort_index()
    returns_df = price_df.pct_change(fill_method=None).fillna(0.0)
    regime_df = compute_market_regime(feats)

    param_grid = {
        "top_k": [5, 10, 15, 20],
        "invest_more_n": [5, 10, 15, 20, 25],
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


def _resolve_config_file(base_dir: Path, config_path: str | None = None) -> Path | None:
    if config_path:
        p = Path(config_path)
        if not p.is_absolute():
            p = base_dir / p
        if not p.exists():
            raise FileNotFoundError(f"指定的配置文件不存在: {p}")
        return p

    candidates: list[Path] = []
    env_cfg = str(os.getenv("PIPELINE_CONFIG", "")).strip()
    if env_cfg:
        env_path = Path(env_cfg)
        if not env_path.is_absolute():
            env_path = base_dir / env_path
        candidates.append(env_path)
    candidates.extend([base_dir / "config.yaml", base_dir / "config_v31.yaml"])

    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            return p
    return None


def _load_cfg_from_yaml(base_dir: Path, config_path: str | None = None) -> BacktestConfigV3:
    """从配置文件加载配置构建 BacktestConfigV3。"""
    p = _resolve_config_file(base_dir, config_path=config_path)
    if p is None:
        print("  ⚠ 未找到配置文件（config.yaml / config_v31.yaml），使用默认配置")
        return BacktestConfigV3()
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    print(f"  📂 配置: {p.name}")


    s = raw.get("strategy", {})
    r = raw.get("risk_control", {})
    b = raw.get("backtest", {})
    ind = s.get("industry_diversification", {})
    mc = s.get("market_cap_filter", {})
    liq = s.get("liquidity_filter", {})
    inst_filter = s.get("institution_holding_filter", {})
    corr = s.get("correlation_control", {})
    dyn_pos = s.get("dynamic_position_by_regime", {})
    bear_block = s.get("bear_entry_block", {})
    weak = s.get("weak_signal_de_risk", {})
    alpha = s.get("alpha_enhancement", {})
    news_sent = s.get("news_sentiment_factor", {}) if isinstance(s.get("news_sentiment_factor", {}), dict) else {}
    factor_pre = s.get("factor_preprocess", {})
    factor_neu = factor_pre.get("neutralize", {}) if isinstance(factor_pre.get("neutralize", {}), dict) else {}
    dyn_rb = s.get("dynamic_rebalance_band", {}) if isinstance(s.get("dynamic_rebalance_band", {}), dict) else {}
    vol_sizing = s.get("volatility_sizing", {}) if isinstance(s.get("volatility_sizing", {}), dict) else {}
    liq_state = s.get("liquidity_state_sizing", {}) if isinstance(s.get("liquidity_state_sizing", {}), dict) else {}
    risk_budget = s.get("risk_budget", {}) if isinstance(s.get("risk_budget", {}), dict) else {}
    layered = s.get("layered_entry", {})
    rank_refine = s.get("rank_exit_refine", {}) if isinstance(s.get("rank_exit_refine", {}), dict) else {}
    dd_brake = r.get("portfolio_drawdown_brake", {})
    dd_adapt = r.get("adaptive_drawdown_mode", {})
    mom_crash = r.get("momentum_crash_protection", {})
    failure_refine = r.get("failure_stop_refine", {}) if isinstance(r.get("failure_stop_refine", {}), dict) else {}
    exit_conf = r.get("exit_conflict_policy", {}) if isinstance(r.get("exit_conflict_policy", {}), dict) else {}
    kpi = b.get("kpi_targets", {})
    exe_root = raw.get("execution", {})
    exe = b.get("execution", {}) if isinstance(b.get("execution", {}), dict) else {}
    if isinstance(exe_root, dict) and exe_root:
        exe = exe_root
    lot_cfg = exe.get("lot_rounding", {}) if isinstance(exe.get("lot_rounding", {}), dict) else {}
    limit_cfg = exe.get("price_limit", {}) if isinstance(exe.get("price_limit", {}), dict) else {}
    fees = b.get("fees", {}) if isinstance(b.get("fees", {}), dict) else {}

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
        rebalance_band=float(s.get("rebalance_band", b.get("rebalance_band", 0.0))),
        use_dynamic_rebalance_band=bool(
            dyn_rb.get("enabled", s.get("use_dynamic_rebalance_band", False))
        ),
        dynamic_rebalance_band_sensitivity=float(
            dyn_rb.get("sensitivity", s.get("dynamic_rebalance_band_sensitivity", 0.8))
        ),
        dynamic_rebalance_band_signal_ref=float(
            dyn_rb.get("signal_ref", s.get("dynamic_rebalance_band_signal_ref", 1.0))
        ),
        dynamic_rebalance_band_cost_ref_bps=float(
            dyn_rb.get("cost_ref_bps", s.get("dynamic_rebalance_band_cost_ref_bps", 20.0))
        ),
        dynamic_rebalance_band_min=float(
            dyn_rb.get("min_band", s.get("dynamic_rebalance_band_min", 0.0))
        ),
        dynamic_rebalance_band_max=float(
            dyn_rb.get("max_band", s.get("dynamic_rebalance_band_max", 0.05))
        ),
        use_volatility_sizing=bool(
            vol_sizing.get("enabled", s.get("use_volatility_sizing", False))
        ),
        volatility_lookback_days=int(
            vol_sizing.get("lookback_days", s.get("volatility_lookback_days", 20))
        ),
        volatility_target_annual=float(
            vol_sizing.get("target_annual", s.get("volatility_target_annual", 0.22))
        ),
        volatility_floor_annual=float(
            vol_sizing.get("floor_annual", s.get("volatility_floor_annual", 0.08))
        ),
        volatility_pos_mult_min=float(
            vol_sizing.get("min_multiplier", s.get("volatility_pos_mult_min", 0.65))
        ),
        volatility_pos_mult_max=float(
            vol_sizing.get("max_multiplier", s.get("volatility_pos_mult_max", 1.15))
        ),
        use_liquidity_state_sizing=bool(
            liq_state.get("enabled", s.get("use_liquidity_state_sizing", False))
        ),
        liquidity_lookback_days=int(
            liq_state.get("lookback_days", s.get("liquidity_lookback_days", 60))
        ),
        liquidity_pos_mult_min=float(
            liq_state.get("min_multiplier", s.get("liquidity_pos_mult_min", 0.85))
        ),
        liquidity_pos_mult_max=float(
            liq_state.get("max_multiplier", s.get("liquidity_pos_mult_max", 1.05))
        ),
        liquidity_pos_mult_sensitivity=float(
            liq_state.get("sensitivity", s.get("liquidity_pos_mult_sensitivity", 0.50))
        ),
        use_light_risk_budget=bool(
            risk_budget.get("enabled", s.get("use_light_risk_budget", False))
        ),
        risk_budget_mode=str(
            risk_budget.get("mode", s.get("risk_budget_mode", "inverse_atr"))
        ).strip().lower(),
        risk_budget_atr_floor=float(
            risk_budget.get("atr_floor", s.get("risk_budget_atr_floor", 0.005))
        ),
        risk_budget_power=float(
            risk_budget.get("power", s.get("risk_budget_power", 1.0))
        ),
        risk_budget_min_multiplier=float(
            risk_budget.get("min_multiplier", s.get("risk_budget_min_multiplier", 0.75))
        ),
        risk_budget_max_multiplier=float(
            risk_budget.get("max_multiplier", s.get("risk_budget_max_multiplier", 1.35))
        ),
        risk_budget_blend=float(
            risk_budget.get("blend", s.get("risk_budget_blend", 0.50))
        ),
        cost_bps=float(b.get("cost_bps", 15.0)),
        use_cn_fee_schedule=bool(fees.get("enabled", b.get("use_cn_fee_schedule", False))),
        commission_bps_buy=float(fees.get("commission_bps_buy", b.get("commission_bps_buy", b.get("cost_bps", 15.0)))),
        commission_bps_sell=float(fees.get("commission_bps_sell", b.get("commission_bps_sell", b.get("cost_bps", 15.0)))),
        stamp_duty_bps_sell=float(fees.get("stamp_duty_bps_sell", b.get("stamp_duty_bps_sell", 5.0))),
        exchange_fee_bps=float(fees.get("exchange_fee_bps", b.get("exchange_fee_bps", 0.341))),
        regulatory_fee_bps=float(fees.get("regulatory_fee_bps", b.get("regulatory_fee_bps", 0.200))),
        transfer_fee_bps=float(fees.get("transfer_fee_bps", b.get("transfer_fee_bps", 0.100))),
        execution_price_mode=str(exe.get("price_mode", b.get("execution_price_mode", s.get("execution_price_mode", "close")))),
        use_execution_realism=bool(exe.get("realism_enabled", b.get("use_execution_realism", s.get("use_execution_realism", False)))),
        max_participation_rate=float(
            exe.get("max_participation_rate", b.get("max_participation_rate", s.get("max_participation_rate", 0.10)))
        ),
        execution_slippage_bps=float(
            exe.get("slippage_bps", b.get("execution_slippage_bps", s.get("execution_slippage_bps", 5.0)))
        ),
        execution_impact_bps=float(
            exe.get("impact_bps", b.get("execution_impact_bps", s.get("execution_impact_bps", 20.0)))
        ),
        execution_impact_exponent=float(
            exe.get("impact_exponent", b.get("execution_impact_exponent", s.get("execution_impact_exponent", 0.7)))
        ),
        enforce_lot_rounding=bool(
            lot_cfg.get("enabled", exe.get("enforce_lot_rounding", b.get("enforce_lot_rounding", True)))
        ),
        lot_size=int(lot_cfg.get("lot_size", exe.get("lot_size", b.get("lot_size", 100)))),
        use_price_limit_constraints=bool(
            limit_cfg.get("enabled", exe.get("use_price_limit_constraints", b.get("use_price_limit_constraints", True)))
        ),
        main_board_limit_pct=float(
            limit_cfg.get("main_board_pct", exe.get("main_board_limit_pct", b.get("main_board_limit_pct", 0.10)))
        ),
        st_board_limit_pct=float(
            limit_cfg.get("st_pct", exe.get("st_board_limit_pct", b.get("st_board_limit_pct", 0.05)))
        ),
        chinext_board_limit_pct=float(
            limit_cfg.get(
                "chinext_pct",
                exe.get("chinext_board_limit_pct", b.get("chinext_board_limit_pct", 0.20)),
            )
        ),
        star_board_limit_pct=float(
            limit_cfg.get("star_pct", exe.get("star_board_limit_pct", b.get("star_board_limit_pct", 0.20)))
        ),
        bse_board_limit_pct=float(
            limit_cfg.get("bse_pct", exe.get("bse_board_limit_pct", b.get("bse_board_limit_pct", 0.30)))
        ),
        chinext_new_limit_free_days=int(
            limit_cfg.get(
                "chinext_new_limit_free_days",
                exe.get("chinext_new_limit_free_days", b.get("chinext_new_limit_free_days", 5)),
            )
        ),
        star_new_limit_free_days=int(
            limit_cfg.get(
                "star_new_limit_free_days",
                exe.get("star_new_limit_free_days", b.get("star_new_limit_free_days", 5)),
            )
        ),
        target_annual_return_min_pct=float(kpi.get("annual_return_min_pct", b.get("target_annual_return_min_pct", 26.0))),
        target_annual_return_max_pct=float(kpi.get("annual_return_max_pct", b.get("target_annual_return_max_pct", 33.0))),
        target_max_drawdown_limit_pct=float(kpi.get("max_drawdown_limit_pct", b.get("target_max_drawdown_limit_pct", 18.0))),
        target_sharpe_min=float(kpi.get("sharpe_min", b.get("target_sharpe_min", 1.2))),
        use_market_regime=s.get("use_market_regime", True),
        use_tradeability_filter=s.get("use_tradeability_filter", True),
        use_industry_diversification=ind.get("enabled", True),
        max_per_industry=int(ind.get("max_per_industry", 2)),
        max_sector_weight=float(ind.get("max_sector_weight", 0.35)),
        use_market_cap_filter=mc.get("enabled", True),
        min_float_mkt_cap=float(mc.get("min_float_mkt_cap", 8e9)),
        max_float_mkt_cap=float(mc.get("max_float_mkt_cap", 8e11)),
        use_liquidity_filter=liq.get("enabled", True),
        min_amount_20d=float(liq.get("min_amount_20d", 6e7)),
        min_turnover_20d=float(liq.get("min_turnover_20d", 0.6)),
        use_institution_holding_filter=bool(
            inst_filter.get("enabled", s.get("use_institution_holding_filter", False))
        ),
        institution_filter_mode=str(inst_filter.get("mode", s.get("institution_filter_mode", "data"))),
        institution_data_col=str(inst_filter.get("data_col", s.get("institution_data_col", "inst_holding_ratio"))),
        institution_holding_min_pct=float(
            inst_filter.get("min_pct", s.get("institution_holding_min_pct", 5.0))
        ),
        institution_holding_quantile=float(
            inst_filter.get("quantile", s.get("institution_holding_quantile", 0.6))
        ),
        institution_proxy_quantile=float(
            inst_filter.get("proxy_quantile", s.get("institution_proxy_quantile", 0.6))
        ),
        use_correlation_control=corr.get("enabled", True),
        corr_lookback_days=int(corr.get("corr_lookback_days", 60)),
        max_pairwise_corr=float(corr.get("max_pairwise_corr", 0.75)),
        entry_mode=str(s.get("entry_mode", "normal")),
        use_tdx_protection=bool(r.get("use_tdx_protection", True)),
        tdx_protection_threshold=float(r.get("tdx_protection_threshold", 2.0)),
        # 閺傛澘閸欏倹鏆熼敍鍫滃▏閻劑绮拋銈呪偓鐓庡祮閸欑礉闁板秶鐤嗛弬鍥︽娑撳讲闁閻╂牭绱?
        use_index_filter=bool(r.get("use_index_filter", True)),
        index_filter_hard_gate=bool(r.get("index_filter_hard_gate", True)),
        index_filter_block_position_cap=float(r.get("index_filter_block_position_cap", 0.35)),
        index_ma_period=int(r.get("index_ma_period", 60)),
        index_ma_short=int(r.get("index_ma_short", 20)),
        index_crash_days=int(r.get("index_crash_days", 3)),
        index_crash_threshold=float(r.get("index_crash_threshold", -0.03)),
        index_pause_days=int(r.get("index_pause_days", 5)),
        use_momentum_crash_protection=bool(
            mom_crash.get("enabled", r.get("use_momentum_crash_protection", False))
        ),
        momentum_crash_lookback_days=int(
            mom_crash.get("crash_lookback_days", r.get("momentum_crash_lookback_days", 8))
        ),
        momentum_crash_drop_threshold=float(
            mom_crash.get("crash_drop_threshold", r.get("momentum_crash_drop_threshold", -0.08))
        ),
        momentum_rebound_lookback_days=int(
            mom_crash.get("rebound_lookback_days", r.get("momentum_rebound_lookback_days", 3))
        ),
        momentum_rebound_threshold=float(
            mom_crash.get("rebound_threshold", r.get("momentum_rebound_threshold", 0.03))
        ),
        momentum_crash_protection_days=int(
            mom_crash.get("protection_days", r.get("momentum_crash_protection_days", 5))
        ),
        momentum_crash_position_cap=float(
            mom_crash.get("position_cap", r.get("momentum_crash_position_cap", 0.45))
        ),
        normal_min_conditions=int(s.get("normal_min_conditions", 2)),
        use_signal_tiered_sizing=bool(s.get("use_signal_tiered_sizing", True)),
        tier_strong_multiplier=float(s.get("tier_strong_multiplier", 1.2)),
        tier_normal_multiplier=float(s.get("tier_normal_multiplier", 1.0)),
        tier_weak_multiplier=float(s.get("tier_weak_multiplier", 0.5)),
        use_dual_layer_entry=bool(layered.get("enabled", s.get("use_dual_layer_entry", True))),
        use_regime_entry_gate=bool(layered.get("regime_gate_enabled", s.get("use_regime_entry_gate", True))),
        regime_gate_bull_min_strength=float(layered.get("bull_min_strength", s.get("regime_gate_bull_min_strength", 0.5))),
        regime_gate_neutral_min_strength=float(layered.get("neutral_min_strength", s.get("regime_gate_neutral_min_strength", 1.0))),
        regime_gate_bear_min_strength=float(layered.get("bear_min_strength", s.get("regime_gate_bear_min_strength", 2.0))),
        dual_entry_strong_threshold=float(layered.get("strong_threshold", s.get("dual_entry_strong_threshold", 2.0))),
        dual_entry_normal_threshold=float(layered.get("normal_threshold", s.get("dual_entry_normal_threshold", 1.0))),
        weak_entry_mode=str(layered.get("weak_mode", s.get("weak_entry_mode", "observe"))),
        weak_micro_max_new_positions=int(layered.get("weak_micro_max_new", s.get("weak_micro_max_new_positions", 1))),
        weak_micro_weight_multiplier=float(layered.get("weak_micro_weight_multiplier", s.get("weak_micro_weight_multiplier", 0.35))),
        use_alpha_enhancement=bool(alpha.get("enabled", s.get("use_alpha_enhancement", False))),
        alpha_industry_rs_weight=float(alpha.get("industry_rs_weight", s.get("alpha_industry_rs_weight", 0.35))),
        alpha_flow_persistence_weight=float(alpha.get("flow_persistence_weight", s.get("alpha_flow_persistence_weight", 0.45))),
        alpha_quality_weight=float(alpha.get("quality_weight", s.get("alpha_quality_weight", 0.30))),
        alpha_short_reversal_weight=float(alpha.get("short_reversal_weight", s.get("alpha_short_reversal_weight", 0.0))),
        alpha_turnover_reversal_weight=float(alpha.get("turnover_reversal_weight", s.get("alpha_turnover_reversal_weight", 0.0))),
        alpha_value_proxy_weight=float(alpha.get("value_proxy_weight", s.get("alpha_value_proxy_weight", 0.0))),
        use_news_sentiment_factor=bool(
            news_sent.get("enabled", s.get("use_news_sentiment_factor", False))
        ),
        news_sentiment_weight=float(
            news_sent.get("weight", s.get("news_sentiment_weight", 0.10))
        ),
        news_sentiment_min_items=int(
            news_sent.get("min_items", s.get("news_sentiment_min_items", 3))
        ),
        news_sentiment_lag_days=int(
            news_sent.get("lag_days", s.get("news_sentiment_lag_days", 1))
        ),
        use_robust_score_norm=bool(
            factor_pre.get("robust_zscore", s.get("use_robust_score_norm", True))
        ),
        score_winsor_quantile=float(
            factor_pre.get("winsorize_quantile", s.get("score_winsor_quantile", 0.02))
        ),
        score_neutralize_industry=bool(
            factor_neu.get("industry", factor_pre.get("neutralize_industry", s.get("score_neutralize_industry", True)))
        ),
        score_neutralize_size=bool(
            factor_neu.get("size", factor_pre.get("neutralize_size", s.get("score_neutralize_size", True)))
        ),
        score_size_col=str(
            factor_pre.get("size_col", s.get("score_size_col", "float_mkt_cap_20d"))
        ),
        score_neutralize_beta=bool(
            factor_neu.get("beta", factor_pre.get("neutralize_beta", s.get("score_neutralize_beta", False)))
        ),
        score_beta_col=str(
            factor_pre.get("beta_col", s.get("score_beta_col", "beta_60d"))
        ),
        use_weak_signal_de_risk=bool(weak.get("enabled", s.get("use_weak_signal_de_risk", True))),
        weak_signal_threshold=float(weak.get("threshold", s.get("weak_signal_threshold", 1.0))),
        weak_signal_cap_multiplier=float(weak.get("cap_multiplier", s.get("weak_signal_cap_multiplier", 0.75))),
        use_dynamic_regime_position=bool(dyn_pos.get("enabled", s.get("use_dynamic_regime_position", True))),
        regime_bull_pos_cap=float(dyn_pos.get("bull_cap", s.get("regime_bull_pos_cap", 0.95))),
        regime_neutral_pos_cap=float(dyn_pos.get("neutral_cap", s.get("regime_neutral_pos_cap", 0.70))),
        regime_bear_pos_cap=float(dyn_pos.get("bear_cap", s.get("regime_bear_pos_cap", 0.35))),
        block_new_in_bear_regime=bool(bear_block.get("enabled", s.get("block_new_in_bear_regime", False))),
        use_rank_exit=bool(s.get("use_rank_exit", True)),
        rank_exit_rebalance_freq=str(s.get("rank_exit_rebalance_freq", "W")),
        rank_exit_min_hold_days=int(s.get("rank_exit_min_hold_days", 5)),
        rank_exit_rank_buffer=int(rank_refine.get("rank_buffer", s.get("rank_exit_rank_buffer", 0))),
        rank_exit_only_when_trend_down=bool(
            rank_refine.get("only_when_trend_down", s.get("rank_exit_only_when_trend_down", False))
        ),
        use_atr_stop=bool(r.get("use_atr_stop", True)),
        atr_stop_multiplier=float(r.get("atr_stop_multiplier", 1.5)),
        use_failure_stop=bool(r.get("use_failure_stop", True)),
        failure_stop_days=int(r.get("failure_stop_days", 2)),
        failure_stop_gain=float(r.get("failure_stop_gain", 0.03)),
        failure_stop_require_negative_pnl=bool(
            failure_refine.get(
                "require_negative_pnl",
                r.get("failure_stop_require_negative_pnl", False),
            )
        ),
        failure_stop_negative_pnl_threshold=float(
            failure_refine.get(
                "negative_pnl_threshold",
                r.get("failure_stop_negative_pnl_threshold", 0.0),
            )
        ),
        failure_stop_weak_signal_only=bool(
            failure_refine.get(
                "weak_signal_only",
                r.get("failure_stop_weak_signal_only", False),
            )
        ),
        failure_stop_max_signal_strength=float(
            failure_refine.get(
                "max_signal_strength",
                r.get("failure_stop_max_signal_strength", 1.0),
            )
        ),
        failure_stop_skip_if_still_target=bool(
            failure_refine.get(
                "skip_if_still_target",
                r.get("failure_stop_skip_if_still_target", False),
            )
        ),
        failure_stop_skip_if_trend_up=bool(
            failure_refine.get(
                "skip_if_trend_up",
                r.get("failure_stop_skip_if_trend_up", False),
            )
        ),
        time_stop_min_gain=float(r.get("time_stop_min_gain", 0.0)),
        use_portfolio_drawdown_brake=bool(dd_brake.get("enabled", r.get("use_portfolio_drawdown_brake", True))),
        drawdown_brake_threshold=float(dd_brake.get("threshold", r.get("drawdown_brake_threshold", 0.12))),
        drawdown_brake_pause_days=int(dd_brake.get("pause_days", r.get("drawdown_brake_pause_days", 8))),
        drawdown_brake_position_cap=float(dd_brake.get("position_cap", r.get("drawdown_brake_position_cap", 0.35))),
        use_adaptive_drawdown_mode=bool(dd_adapt.get("enabled", r.get("use_adaptive_drawdown_mode", True))),
        adaptive_drawdown_trigger=float(dd_adapt.get("trigger", r.get("adaptive_drawdown_trigger", 0.04))),
        adaptive_drawdown_position_cap_multiplier=float(
            dd_adapt.get("position_cap_multiplier", r.get("adaptive_drawdown_position_cap_multiplier", 0.80))
        ),
        adaptive_drawdown_gate_boost=float(dd_adapt.get("gate_boost", r.get("adaptive_drawdown_gate_boost", 0.50))),
        adaptive_drawdown_top_k_multiplier=float(
            dd_adapt.get("top_k_multiplier", r.get("adaptive_drawdown_top_k_multiplier", 0.70))
        ),
        adaptive_drawdown_invest_more_multiplier=float(
            dd_adapt.get("invest_more_multiplier", r.get("adaptive_drawdown_invest_more_multiplier", 0.80))
        ),
        adaptive_drawdown_stop_loss_multiplier=float(
            dd_adapt.get("stop_loss_multiplier", r.get("adaptive_drawdown_stop_loss_multiplier", 0.90))
        ),
        adaptive_drawdown_trailing_multiplier=float(
            dd_adapt.get("trailing_multiplier", r.get("adaptive_drawdown_trailing_multiplier", 0.85))
        ),
        adaptive_drawdown_max_hold_multiplier=float(
            dd_adapt.get("max_hold_multiplier", r.get("adaptive_drawdown_max_hold_multiplier", 0.80))
        ),
        use_ma_stop=bool(r.get("use_ma_stop", True)),
        ma_stop_days=int(r.get("ma_stop_days", 2)),
        ma_trailing_priority=str(
            exit_conf.get("ma_vs_trailing", r.get("ma_trailing_priority", "trailing"))
        ).strip().lower(),
        use_take_profit=bool(r.get("use_take_profit", True)),
        take_profit_atr_multiplier=float(r.get("take_profit_atr_multiplier", 3.0)),
        take_profit_fixed_pct=float(r.get("take_profit_fixed_pct", 0.20)),
        use_staged_take_profit=bool(r.get("use_staged_take_profit", False)),
        use_profit_pyramiding=bool(s.get("use_profit_pyramiding", True)),
        pyramid_trigger_pct=float(s.get("pyramid_trigger_pct", 0.05)),
        pyramid_add_ratio=float(s.get("pyramid_add_ratio", 0.5)),
        pyramid_max_adds=int(s.get("pyramid_max_adds", 1)),
    )


def run_quick_backtest(
    base_dir: Path,
    start_date: str,
    watchlist_file: str = None,
    config_path: str | None = None,
    use_dynamic_watchlist: bool = False,
    dynamic_top_n: int = 300,
    run_walk_forward: bool = False,
    wf_train_days: int = 504,
    wf_test_days: int = 126,
    wf_step_days: int = 126,
    top_k_override: int | None = None,
    invest_more_n_override: int | None = None,
    entry_mode_override: str | None = None,
    min_float_mkt_cap_override: float | None = None,
    min_amount_20d_override: float | None = None,
    min_turnover_20d_override: float | None = None,
    use_institution_filter_override: bool | None = None,
    institution_filter_mode_override: str | None = None,
    institution_holding_min_pct_override: float | None = None,
    institution_holding_quantile_override: float | None = None,
    institution_proxy_quantile_override: float | None = None,
    use_alpha_enhancement_override: bool | None = None,
    alpha_industry_weight_override: float | None = None,
    alpha_flow_weight_override: float | None = None,
    alpha_quality_weight_override: float | None = None,
    alpha_short_reversal_weight_override: float | None = None,
    alpha_turnover_reversal_weight_override: float | None = None,
    alpha_value_proxy_weight_override: float | None = None,
    use_news_sentiment_factor_override: bool | None = None,
    news_sentiment_weight_override: float | None = None,
    use_index_filter_override: bool | None = None,
    use_dynamic_regime_position_override: bool | None = None,
    regime_bull_pos_cap_override: float | None = None,
    regime_neutral_pos_cap_override: float | None = None,
    regime_bear_pos_cap_override: float | None = None,
    use_volatility_sizing_override: bool | None = None,
    use_liquidity_state_sizing_override: bool | None = None,
    use_momentum_crash_protection_override: bool | None = None,
    momentum_crash_position_cap_override: float | None = None,
    block_new_in_bear_override: bool | None = None,
    regime_gate_bull_min_override: float | None = None,
    regime_gate_neutral_min_override: float | None = None,
    regime_gate_bear_min_override: float | None = None,
    use_weak_signal_de_risk_override: bool | None = None,
    weak_signal_threshold_override: float | None = None,
    weak_signal_cap_multiplier_override: float | None = None,
    use_drawdown_brake_override: bool | None = None,
    drawdown_brake_threshold_override: float | None = None,
    drawdown_brake_position_cap_override: float | None = None,
    use_adaptive_drawdown_mode_override: bool | None = None,
    adaptive_drawdown_trigger_override: float | None = None,
    adaptive_drawdown_position_cap_multiplier_override: float | None = None,
    adaptive_drawdown_gate_boost_override: float | None = None,
    adaptive_drawdown_top_k_multiplier_override: float | None = None,
    adaptive_drawdown_invest_more_multiplier_override: float | None = None,
    adaptive_drawdown_stop_loss_multiplier_override: float | None = None,
    adaptive_drawdown_trailing_multiplier_override: float | None = None,
    adaptive_drawdown_max_hold_multiplier_override: float | None = None,
    stop_loss_pct_override: float | None = None,
    trailing_stop_pct_override: float | None = None,
    max_hold_days_override: int | None = None,
    atr_stop_multiplier_override: float | None = None,
    failure_stop_days_override: int | None = None,
    failure_stop_gain_override: float | None = None,
    failure_stop_deconflict_override: bool | None = None,
    rank_exit_buffer_override: int | None = None,
    rank_exit_only_when_trend_down_override: bool | None = None,
    ma_trailing_priority_override: str | None = None,
    execution_mode_override: str | None = None,
    use_execution_realism_override: bool | None = None,
    max_participation_rate_override: float | None = None,
    execution_slippage_bps_override: float | None = None,
    execution_impact_bps_override: float | None = None,
    execution_impact_exponent_override: float | None = None,
    enforce_lot_rounding_override: bool | None = None,
    lot_size_override: int | None = None,
    use_price_limit_constraints_override: bool | None = None,
) -> None:
    """Run one full-period backtest from config with optional CLI overrides."""
    print("=" * 60)
    print("Quick Backtest: Strategy V3")
    print("=" * 60)

    print("\n[1/6] 加载策略配置...")
    cfg = _load_cfg_from_yaml(base_dir, config_path=config_path)
    # 閳光偓閳光偓 鏉炲鍣洪崣鍌涙殶鐟曞棛娲婇敍鍫㈡暏娴滃骸鎻╅柅鐑?B妤犲矁鐦夐敍灞肩瑝閺€閫涘瘜闁板秶鐤嗛敍澶嗘敘閳光偓
    if top_k_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "top_k": int(top_k_override)})
    if invest_more_n_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "invest_more_n": int(invest_more_n_override)})
    if entry_mode_override:
        cfg = BacktestConfigV3(**{**asdict(cfg), "entry_mode": str(entry_mode_override)})
    if min_float_mkt_cap_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "min_float_mkt_cap": float(min_float_mkt_cap_override)})
    if min_amount_20d_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "min_amount_20d": float(min_amount_20d_override)})
    if min_turnover_20d_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "min_turnover_20d": float(min_turnover_20d_override)})
    if use_institution_filter_override is not None:
        cfg = BacktestConfigV3(
            **{**asdict(cfg), "use_institution_holding_filter": bool(use_institution_filter_override)}
        )
    if institution_filter_mode_override:
        cfg = BacktestConfigV3(
            **{**asdict(cfg), "institution_filter_mode": str(institution_filter_mode_override).strip().lower()}
        )
    if institution_holding_min_pct_override is not None:
        cfg = BacktestConfigV3(
            **{**asdict(cfg), "institution_holding_min_pct": float(institution_holding_min_pct_override)}
        )
    if institution_holding_quantile_override is not None:
        cfg = BacktestConfigV3(
            **{**asdict(cfg), "institution_holding_quantile": float(institution_holding_quantile_override)}
        )
    if institution_proxy_quantile_override is not None:
        cfg = BacktestConfigV3(
            **{**asdict(cfg), "institution_proxy_quantile": float(institution_proxy_quantile_override)}
        )
    if use_alpha_enhancement_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "use_alpha_enhancement": bool(use_alpha_enhancement_override)})
    if alpha_industry_weight_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "alpha_industry_rs_weight": float(alpha_industry_weight_override)})
    if alpha_flow_weight_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "alpha_flow_persistence_weight": float(alpha_flow_weight_override)})
    if alpha_quality_weight_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "alpha_quality_weight": float(alpha_quality_weight_override)})
    if alpha_short_reversal_weight_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "alpha_short_reversal_weight": float(alpha_short_reversal_weight_override)})
    if alpha_turnover_reversal_weight_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "alpha_turnover_reversal_weight": float(alpha_turnover_reversal_weight_override)})
    if alpha_value_proxy_weight_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "alpha_value_proxy_weight": float(alpha_value_proxy_weight_override)})
    if use_news_sentiment_factor_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "use_news_sentiment_factor": bool(use_news_sentiment_factor_override)})
    if news_sentiment_weight_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "news_sentiment_weight": float(news_sentiment_weight_override)})
    if use_index_filter_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "use_index_filter": bool(use_index_filter_override)})
    if use_dynamic_regime_position_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "use_dynamic_regime_position": bool(use_dynamic_regime_position_override)})
    if regime_bull_pos_cap_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "regime_bull_pos_cap": float(regime_bull_pos_cap_override)})
    if regime_neutral_pos_cap_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "regime_neutral_pos_cap": float(regime_neutral_pos_cap_override)})
    if regime_bear_pos_cap_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "regime_bear_pos_cap": float(regime_bear_pos_cap_override)})
    if use_volatility_sizing_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "use_volatility_sizing": bool(use_volatility_sizing_override)})
    if use_liquidity_state_sizing_override is not None:
        cfg = BacktestConfigV3(
            **{**asdict(cfg), "use_liquidity_state_sizing": bool(use_liquidity_state_sizing_override)}
        )
    if use_momentum_crash_protection_override is not None:
        cfg = BacktestConfigV3(
            **{**asdict(cfg), "use_momentum_crash_protection": bool(use_momentum_crash_protection_override)}
        )
    if momentum_crash_position_cap_override is not None:
        cfg = BacktestConfigV3(
            **{**asdict(cfg), "momentum_crash_position_cap": float(momentum_crash_position_cap_override)}
        )
    if block_new_in_bear_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "block_new_in_bear_regime": bool(block_new_in_bear_override)})
    if regime_gate_bull_min_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "regime_gate_bull_min_strength": float(regime_gate_bull_min_override)})
    if regime_gate_neutral_min_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "regime_gate_neutral_min_strength": float(regime_gate_neutral_min_override)})
    if regime_gate_bear_min_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "regime_gate_bear_min_strength": float(regime_gate_bear_min_override)})
    if use_weak_signal_de_risk_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "use_weak_signal_de_risk": bool(use_weak_signal_de_risk_override)})
    if weak_signal_threshold_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "weak_signal_threshold": float(weak_signal_threshold_override)})
    if weak_signal_cap_multiplier_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "weak_signal_cap_multiplier": float(weak_signal_cap_multiplier_override)})
    if use_drawdown_brake_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "use_portfolio_drawdown_brake": bool(use_drawdown_brake_override)})
    if drawdown_brake_threshold_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "drawdown_brake_threshold": float(drawdown_brake_threshold_override)})
    if drawdown_brake_position_cap_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "drawdown_brake_position_cap": float(drawdown_brake_position_cap_override)})
    if use_adaptive_drawdown_mode_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "use_adaptive_drawdown_mode": bool(use_adaptive_drawdown_mode_override)})
    if adaptive_drawdown_trigger_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "adaptive_drawdown_trigger": float(adaptive_drawdown_trigger_override)})
    if adaptive_drawdown_position_cap_multiplier_override is not None:
        cfg = BacktestConfigV3(
            **{
                **asdict(cfg),
                "adaptive_drawdown_position_cap_multiplier": float(adaptive_drawdown_position_cap_multiplier_override),
            }
        )
    if adaptive_drawdown_gate_boost_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "adaptive_drawdown_gate_boost": float(adaptive_drawdown_gate_boost_override)})
    if adaptive_drawdown_top_k_multiplier_override is not None:
        cfg = BacktestConfigV3(
            **{**asdict(cfg), "adaptive_drawdown_top_k_multiplier": float(adaptive_drawdown_top_k_multiplier_override)}
        )
    if adaptive_drawdown_invest_more_multiplier_override is not None:
        cfg = BacktestConfigV3(
            **{
                **asdict(cfg),
                "adaptive_drawdown_invest_more_multiplier": float(adaptive_drawdown_invest_more_multiplier_override),
            }
        )
    if adaptive_drawdown_stop_loss_multiplier_override is not None:
        cfg = BacktestConfigV3(
            **{
                **asdict(cfg),
                "adaptive_drawdown_stop_loss_multiplier": float(adaptive_drawdown_stop_loss_multiplier_override),
            }
        )
    if adaptive_drawdown_trailing_multiplier_override is not None:
        cfg = BacktestConfigV3(
            **{
                **asdict(cfg),
                "adaptive_drawdown_trailing_multiplier": float(adaptive_drawdown_trailing_multiplier_override),
            }
        )
    if adaptive_drawdown_max_hold_multiplier_override is not None:
        cfg = BacktestConfigV3(
            **{
                **asdict(cfg),
                "adaptive_drawdown_max_hold_multiplier": float(adaptive_drawdown_max_hold_multiplier_override),
            }
        )
    if stop_loss_pct_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "stop_loss_pct": float(stop_loss_pct_override)})
    if trailing_stop_pct_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "trailing_stop_pct": float(trailing_stop_pct_override)})
    if max_hold_days_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "max_hold_days": int(max_hold_days_override)})
    if atr_stop_multiplier_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "atr_stop_multiplier": float(atr_stop_multiplier_override)})
    if failure_stop_days_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "failure_stop_days": int(failure_stop_days_override)})
    if failure_stop_gain_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "failure_stop_gain": float(failure_stop_gain_override)})
    if failure_stop_deconflict_override is not None:
        cfg = BacktestConfigV3(
            **{
                **asdict(cfg),
                "failure_stop_require_negative_pnl": bool(failure_stop_deconflict_override),
                "failure_stop_weak_signal_only": bool(failure_stop_deconflict_override),
                "failure_stop_skip_if_still_target": bool(failure_stop_deconflict_override),
                "failure_stop_skip_if_trend_up": bool(failure_stop_deconflict_override),
            }
        )
    if rank_exit_buffer_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "rank_exit_rank_buffer": int(rank_exit_buffer_override)})
    if rank_exit_only_when_trend_down_override is not None:
        cfg = BacktestConfigV3(
            **{**asdict(cfg), "rank_exit_only_when_trend_down": bool(rank_exit_only_when_trend_down_override)}
        )
    if ma_trailing_priority_override:
        pr = str(ma_trailing_priority_override).strip().lower()
        if pr not in {"trailing", "ma"}:
            pr = "trailing"
        cfg = BacktestConfigV3(**{**asdict(cfg), "ma_trailing_priority": pr})
    if execution_mode_override:
        cfg = BacktestConfigV3(**{**asdict(cfg), "execution_price_mode": str(execution_mode_override).strip().lower()})
    if use_execution_realism_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "use_execution_realism": bool(use_execution_realism_override)})
    if max_participation_rate_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "max_participation_rate": float(max_participation_rate_override)})
    if execution_slippage_bps_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "execution_slippage_bps": float(execution_slippage_bps_override)})
    if execution_impact_bps_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "execution_impact_bps": float(execution_impact_bps_override)})
    if execution_impact_exponent_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "execution_impact_exponent": float(execution_impact_exponent_override)})
    if enforce_lot_rounding_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "enforce_lot_rounding": bool(enforce_lot_rounding_override)})
    if lot_size_override is not None:
        cfg = BacktestConfigV3(**{**asdict(cfg), "lot_size": int(lot_size_override)})
    if use_price_limit_constraints_override is not None:
        cfg = BacktestConfigV3(
            **{**asdict(cfg), "use_price_limit_constraints": bool(use_price_limit_constraints_override)}
        )

    print(f"  Entry mode: {cfg.entry_mode} (min extra conditions >= {cfg.normal_min_conditions})")
    print(f"  Stops: SL {cfg.stop_loss_pct*100:.0f}% | TS {cfg.trailing_stop_pct*100:.0f}% | Max hold {cfg.max_hold_days}d")
    print(f"  Rebalance band: {cfg.rebalance_band:.4f}")
    print(
        f"  Dynamic rebalance band: {'on' if cfg.use_dynamic_rebalance_band else 'off'} "
        f"(sens {cfg.dynamic_rebalance_band_sensitivity:.2f}, signal_ref {cfg.dynamic_rebalance_band_signal_ref:.2f}, "
        f"cost_ref {cfg.dynamic_rebalance_band_cost_ref_bps:.1f}bps, "
        f"range {cfg.dynamic_rebalance_band_min:.4f}-{cfg.dynamic_rebalance_band_max:.4f})"
    )
    print(
        f"  Volatility sizing: {'on' if cfg.use_volatility_sizing else 'off'} "
        f"(target {cfg.volatility_target_annual:.0%}, lookback {cfg.volatility_lookback_days}d, "
        f"range x{cfg.volatility_pos_mult_min:.2f}-x{cfg.volatility_pos_mult_max:.2f})"
    )
    print(
        f"  Liquidity-state sizing: {'on' if cfg.use_liquidity_state_sizing else 'off'} "
        f"(lookback {cfg.liquidity_lookback_days}d, sens {cfg.liquidity_pos_mult_sensitivity:.2f}, "
        f"range x{cfg.liquidity_pos_mult_min:.2f}-x{cfg.liquidity_pos_mult_max:.2f})"
    )
    print(
        f"  Light risk budget: {'on' if cfg.use_light_risk_budget else 'off'} "
        f"(mode {cfg.risk_budget_mode}, blend {cfg.risk_budget_blend:.2f}, "
        f"clamp x{cfg.risk_budget_min_multiplier:.2f}-x{cfg.risk_budget_max_multiplier:.2f})"
    )
    print(f"  ATR stop: {'on' if cfg.use_atr_stop else 'off'} ({cfg.atr_stop_multiplier}x)")
    print(f"  MA stop: {'on' if cfg.use_ma_stop else 'off'} ({cfg.ma_stop_days}d below MA20)")
    print(f"  MA vs trailing priority: {cfg.ma_trailing_priority}")
    print(
        f"  Take profit: {'on' if cfg.use_take_profit else 'off'} "
        f"(ATR {cfg.take_profit_atr_multiplier}x | fixed {cfg.take_profit_fixed_pct*100:.0f}% | "
        f"staged {'on' if cfg.use_staged_take_profit else 'off'})"
    )
    print(
        f"  Failure stop: {'on' if cfg.use_failure_stop else 'off'} "
        f"({cfg.failure_stop_days}d gain<{cfg.failure_stop_gain*100:.0f}%; "
        f"neg_only={'on' if cfg.failure_stop_require_negative_pnl else 'off'}; "
        f"weak_only={'on' if cfg.failure_stop_weak_signal_only else 'off'}; "
        f"skip_target={'on' if cfg.failure_stop_skip_if_still_target else 'off'}; "
        f"skip_trend_up={'on' if cfg.failure_stop_skip_if_trend_up else 'off'})"
    )
    if cfg.use_index_filter:
        if cfg.index_filter_hard_gate:
            index_mode = "hard gate"
        else:
            index_mode = f"soft cap {cfg.index_filter_block_position_cap:.2f}"
        print(f"  Index filter: on ({index_mode}; MA{cfg.index_ma_period}+MA{cfg.index_ma_short})")
    else:
        print(f"  Index filter: off (MA{cfg.index_ma_period}+MA{cfg.index_ma_short})")
    print(
        f"  Momentum crash protection: {'on' if cfg.use_momentum_crash_protection else 'off'} "
        f"(drop {cfg.momentum_crash_drop_threshold:.1%}/{cfg.momentum_crash_lookback_days}d, "
        f"rebound {cfg.momentum_rebound_threshold:.1%}/{cfg.momentum_rebound_lookback_days}d, "
        f"hold {cfg.momentum_crash_protection_days}d, cap {cfg.momentum_crash_position_cap:.2f})"
    )
    print(
        f"  Rank exit: {'on' if cfg.use_rank_exit else 'off'} "
        f"({cfg.rank_exit_rebalance_freq}, min hold {cfg.rank_exit_min_hold_days}d, "
        f"rank_buffer={cfg.rank_exit_rank_buffer}, "
        f"trend_down_only={'on' if cfg.rank_exit_only_when_trend_down else 'off'})"
    )
    print(f"  Signal tier sizing: {'on' if cfg.use_signal_tiered_sizing else 'off'}")
    print(
        f"  Alpha enhancement: {'on' if cfg.use_alpha_enhancement else 'off'} "
        f"(industry {cfg.alpha_industry_rs_weight:.2f}/flow {cfg.alpha_flow_persistence_weight:.2f}/quality {cfg.alpha_quality_weight:.2f}/"
        f"STR {cfg.alpha_short_reversal_weight:.2f}/turnover_rev {cfg.alpha_turnover_reversal_weight:.2f}/value {cfg.alpha_value_proxy_weight:.2f})"
    )
    print(
        f"  News sentiment factor: {'on' if cfg.use_news_sentiment_factor else 'off'} "
        f"(weight {cfg.news_sentiment_weight:.2f}, min_items {cfg.news_sentiment_min_items}, lag {cfg.news_sentiment_lag_days}d)"
    )
    print(
        f"  Factor preprocess: robust={'on' if cfg.use_robust_score_norm else 'off'} "
        f"| winsor={cfg.score_winsor_quantile:.3f} "
        f"| neutral(ind/size/beta)="
        f"{'on' if cfg.score_neutralize_industry else 'off'}/"
        f"{'on' if cfg.score_neutralize_size else 'off'}/"
        f"{'on' if cfg.score_neutralize_beta else 'off'}"
    )
    print(
        f"  Weak-signal de-risk: {'on' if cfg.use_weak_signal_de_risk else 'off'} "
        f"(threshold {cfg.weak_signal_threshold:.2f}, cap x{cfg.weak_signal_cap_multiplier:.2f})"
    )
    print(
        f"  Institution filter: {'on' if cfg.use_institution_holding_filter else 'off'} "
        f"(mode={cfg.institution_filter_mode}; min={cfg.institution_holding_min_pct:.1f}%; "
        f"q={cfg.institution_holding_quantile:.2f}; proxy_q={cfg.institution_proxy_quantile:.2f})"
    )
    weak_mode = str(cfg.weak_entry_mode).strip().lower()
    if weak_mode not in {"observe", "micro"}:
        weak_mode = "observe"
    weak_desc = (
        "weak signal: observe only"
        if weak_mode == "observe"
        else f"weak signal: micro x{cfg.weak_micro_weight_multiplier:.2f}, max new {cfg.weak_micro_max_new_positions}"
    )
    print(
        f"  Dual-layer entry: {'on' if cfg.use_dual_layer_entry else 'off'} "
        f"(regime gate {'on' if cfg.use_regime_entry_gate else 'off'}; "
        f"B/N/B {cfg.regime_gate_bull_min_strength:.2f}/{cfg.regime_gate_neutral_min_strength:.2f}/{cfg.regime_gate_bear_min_strength:.2f}; "
        f"{weak_desc})"
    )
    print(
        f"  Dynamic regime position: {'on' if cfg.use_dynamic_regime_position else 'off'} "
        f"(B/N/B={cfg.regime_bull_pos_cap:.2f}/{cfg.regime_neutral_pos_cap:.2f}/{cfg.regime_bear_pos_cap:.2f})"
    )
    print(f"  Bear entry block: {'on' if cfg.block_new_in_bear_regime else 'off'}")
    print(
        f"  Drawdown brake: {'on' if cfg.use_portfolio_drawdown_brake else 'off'} "
        f"(threshold {cfg.drawdown_brake_threshold*100:.1f}%, pause {cfg.drawdown_brake_pause_days}d, "
        f"cap {cfg.drawdown_brake_position_cap:.2f})"
    )
    print(
        f"  Adaptive pre-brake: {'on' if cfg.use_adaptive_drawdown_mode else 'off'} "
        f"(trigger {cfg.adaptive_drawdown_trigger*100:.1f}%, pos x{cfg.adaptive_drawdown_position_cap_multiplier:.2f}, "
        f"gate +{cfg.adaptive_drawdown_gate_boost:.2f}, topk x{cfg.adaptive_drawdown_top_k_multiplier:.2f})"
    )
    print(
        f"  Profit pyramiding: {'on' if cfg.use_profit_pyramiding else 'off'} "
        f"(trigger >{cfg.pyramid_trigger_pct*100:.0f}%, add {cfg.pyramid_add_ratio*100:.0f}%)"
    )
    print(f"  TDX protection: {'on' if cfg.use_tdx_protection else 'off'}")
    print(
        f"  Execution: mode={cfg.execution_price_mode} | realism={'on' if cfg.use_execution_realism else 'off'} | "
        f"lot={cfg.lot_size} ({'on' if cfg.enforce_lot_rounding else 'off'}) | "
        f"price_limit={'on' if cfg.use_price_limit_constraints else 'off'}"
    )
    print(
        f"  Exec costs: slippage={cfg.execution_slippage_bps:.1f}bps | "
        f"impact={cfg.execution_impact_bps:.1f}bps ^ {cfg.execution_impact_exponent:.2f} | "
        f"max_participation={cfg.max_participation_rate:.2%}"
    )
    print(
        f"  Explicit fees: model={'CN' if cfg.use_cn_fee_schedule else 'legacy'} | "
        f"commission(b/s)={cfg.commission_bps_buy:.2f}/{cfg.commission_bps_sell:.2f}bps | "
        f"stamp={cfg.stamp_duty_bps_sell:.2f}bps"
    )
    print(
        f"  Price limits(fallback): main={cfg.main_board_limit_pct:.0%} | st={cfg.st_board_limit_pct:.0%} | "
        f"chinext={cfg.chinext_board_limit_pct:.0%} | star={cfg.star_board_limit_pct:.0%} | "
        f"bse={cfg.bse_board_limit_pct:.0%}"
    )
    print(f"  top_k/invest_more_n: {cfg.top_k}/{cfg.invest_more_n}")
    print(
        f"  Liquidity floor: amount20d>={cfg.min_amount_20d:.2e}, "
        f"turnover20d>={cfg.min_turnover_20d:.2f}"
    )
    print(f"  Market-cap floor: float_mkt_cap>={cfg.min_float_mkt_cap:.2e}")
    if use_dynamic_watchlist:
        print(f"  Dynamic watchlist: on (monthly rebalance Top {dynamic_top_n})")

    print("\n[2/6] 加载特征数据...")
    feats = load_and_prepare_features(base_dir, start_date=start_date)

    # 閳光偓閳光偓 閼诧紕銈ㄥЧ鐘虹箖濠?閳光偓閳光偓
    if watchlist_file:
        wl_path = base_dir / watchlist_file
        if wl_path.exists():
            import re as _re
            wl_text = wl_path.read_text(encoding="utf-8")

            # 鐏忔繆鐦稉銈囬弽鐓庣础閿?
            # 1) yaml閸掓銆冮弽鐓庣础: - "600519" 閹?- 600519
            # 2) watchlist_cache.yaml閺嶇厧绱? ['601198', '600015', ...]
            raw_codes = _re.findall(r'(\d{6})', wl_text)

            # 娑旂喎鐨剧拠鏇犳纯閹恒儳鏁aml鐟欙絾鐎介敍鍧礱tchlist_cache.yaml閺嶇厧绱￠敍?
            if not raw_codes:
                try:
                    wl_data = yaml.safe_load(wl_text) or {}
                    wl_list = wl_data.get("watchlist", [])
                    raw_codes = [str(c).split(".")[0] for c in wl_list if str(c).strip()]
                except Exception:
                    pass

            # 閸樺鍣?
            raw_codes = list(set(raw_codes))

            # 鏉炲床娑撳搫鐢崥搴ｇ磻閺嶇厧绱?
            wl_symbols = set()
            for code in raw_codes:
                code = code.zfill(6)
                if code.startswith(("6", "5")):
                    wl_symbols.add(f"{code}.SH")
                else:
                    wl_symbols.add(f"{code}.SZ")

            # 鏉╁洦鎶ら幒澶嬪瘹閺侀鍞惍渚婄礄399xxx缁涘绱?
            wl_symbols = {s for s in wl_symbols if not s.startswith("399")}

            before = feats["symbol"].nunique()
            feats = feats[feats["symbol"].isin(wl_symbols)].copy()
            after = feats["symbol"].nunique()
            print(
                f"  [Watchlist] using {watchlist_file} "
                f"({after}/{before} symbols kept, pool size {len(wl_symbols)})"
            )
        else:
            print(f"  [Watchlist] file not found: {wl_path}; fallback to full universe")
    industry_map = build_industry_map_from_config(base_dir, feats["symbol"])
    feats["industry"] = feats["symbol"].map(industry_map).fillna("OTHER")
    print(
        f"  Features: {len(feats)} rows | {feats['symbol'].nunique()} symbols | "
        f"{feats['date'].min().date()} -> {feats['date'].max().date()}"
    )

    print("\n[3/6] 预计算日度候选池...")
    daily = precompute_daily_universe(feats, cfg=cfg)
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
            print(f"  [Dynamic Watchlist] sample: {str(sample_date)[:10]} -> {sample_n} symbols")
    close_price_df = feats.pivot_table(index="date", columns="symbol", values="close").sort_index()
    if "open" in feats.columns:
        open_price_df = feats.pivot_table(index="date", columns="symbol", values="open").sort_index()
    else:
        open_price_df = close_price_df.copy()
        print("  [Execution] 'open' column missing, next_open mode will fallback to close.")
    close_returns_df = close_price_df.pct_change(fill_method=None).fillna(0.0)
    open_returns_df = open_price_df.pct_change(fill_method=None).fillna(0.0)
    regime_df = compute_market_regime(feats, cfg)

    print("\n[4/6] 计算指数过滤...")
    index_filter = compute_index_filter(base_dir, start_date, cfg)

    print("\n[5/6] 执行策略回测...")
    # 鐠囧﹥鏌囬敍姘弻銉ュ弳閸︾儤娼禒鎯板厴缁涙稑鍤径姘毌閼诧紕銈?
    if cfg.entry_mode == "custom":
        sample_dates = sorted(daily["date"].unique())
        check_dates = sample_dates[::60][:5]  # 濮?0婢垛晛褰囨稉鈧稉鐗遍張?
        print(f"  [Diagnostic] custom entry hit-rates (sample {len(check_dates)} days):")
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
            mask, _ = _candidate_entry_mask(dd, cfg)
            call = mask.sum()
            print(
                f"    {str(cd)[:10]}: total={n} | limit_up_30d={c1} | main_force>0.5%={c2} | "
                f"high30_new_high={c3} | pullback_ok={c4} | tdx>={cfg.tdx_min_score}={c5} | all_pass={call}"
            )

    def _run_one_mode(run_cfg: BacktestConfigV3, run_price_df: pd.DataFrame, run_returns_df: pd.DataFrame) -> dict[str, Any]:
        return run_backtest_v3(
            daily_universe=daily,
            price_df=run_price_df,
            returns_df=run_returns_df,
            cfg=run_cfg,
            regime_df=regime_df,
            date_start=pd.to_datetime(start_date),
            date_end=daily["date"].max(),
            index_filter=index_filter,
            allowed_symbols_by_date=dynamic_watchlist_map,
        )

    exec_mode = str(cfg.execution_price_mode).strip().lower()
    if exec_mode not in {"close", "next_open", "parallel"}:
        print(f"  [Execution] unknown mode '{cfg.execution_price_mode}', fallback to close.")
        exec_mode = "close"
        cfg = BacktestConfigV3(**{**asdict(cfg), "execution_price_mode": "close"})

    parallel_results: dict[str, dict[str, Any]] = {}
    if exec_mode == "parallel":
        cfg_close = BacktestConfigV3(**{**asdict(cfg), "execution_price_mode": "close"})
        cfg_open = BacktestConfigV3(**{**asdict(cfg), "execution_price_mode": "next_open"})
        parallel_results["close"] = _run_one_mode(cfg_close, close_price_df, close_returns_df)
        parallel_results["next_open"] = _run_one_mode(cfg_open, open_price_df, open_returns_df)
        result = parallel_results["close"]
        active_price_df = close_price_df
        active_returns_df = close_returns_df
    elif exec_mode == "next_open":
        result = _run_one_mode(cfg, open_price_df, open_returns_df)
        active_price_df = open_price_df
        active_returns_df = open_returns_df
    else:
        result = _run_one_mode(cfg, close_price_df, close_returns_df)
        active_price_df = close_price_df
        active_returns_df = close_returns_df

    eq = result["equity_curve"]
    m = result["metrics"]
    trades = result.get("trade_log", [])

    print(f"\n  Annual return: {m['annual_return_pct']:+.2f}%")
    print(f"  Max drawdown: {m['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe: {m['sharpe']:.2f}")
    print(f"  Win rate: {m['win_rate_pct']:.1f}%")
    if parallel_results:
        close_m = parallel_results["close"]["metrics"]
        open_m = parallel_results["next_open"]["metrics"]
        print(
            "  [Execution Compare] "
            f"close annual {close_m['annual_return_pct']:+.2f}% / sharpe {close_m['sharpe']:.2f} | "
            f"next_open annual {open_m['annual_return_pct']:+.2f}% / sharpe {open_m['sharpe']:.2f}"
        )
    kpi_status = evaluate_kpi_targets(m, cfg)
    annual_band = f"{cfg.target_annual_return_min_pct:.1f}%~{cfg.target_annual_return_max_pct:.1f}%"
    print(
        "  KPI status: "
        f"annual[{annual_band}]={'OK' if kpi_status['annual_range_ok'] else 'FAIL'} | "
        f"max_dd<={cfg.target_max_drawdown_limit_pct:.1f}%={'OK' if kpi_status['max_drawdown_ok'] else 'FAIL'} | "
        f"sharpe>={cfg.target_sharpe_min:.2f}={'OK' if kpi_status['sharpe_ok'] else 'FAIL'}"
    )

    trade_level_metrics = {}
    if trades:
        tdf = pd.DataFrame(trades)
        n_trades = len(tdf)
        n_symbols = tdf["symbol"].nunique()
        wins = (tdf["pnl_pct"] > 0).sum()
        avg_pnl = tdf["pnl_pct"].mean()

        winning = tdf[tdf["pnl_pct"] > 0]["pnl_pct"]
        losing = tdf[tdf["pnl_pct"] <= 0]["pnl_pct"]

        profit_factor = float(winning.sum() / abs(losing.sum())) if losing.sum() != 0 else float("inf")
        avg_win = float(winning.mean()) if len(winning) > 0 else 0.0
        avg_loss = float(losing.mean()) if len(losing) > 0 else 0.0
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        win_rate = len(winning) / n_trades if n_trades > 0 else 0.0
        expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)

        print(f"  交易总数: {n_trades} | 个股数: {n_symbols} | 胜率: {win_rate*100:.1f}% | 平均盈亏: {avg_pnl:+.2f}%")
        print(
            f"  Profit Factor: {profit_factor:.2f} | Payoff Ratio: {payoff_ratio:.2f} | "
            f"Avg Win: {avg_win:+.2f}% | Avg Loss: {avg_loss:+.2f}% | Expectancy: {expectancy:+.3f}%"
        )

        # 按退出原因分组统计
        if "exit_reason" in tdf.columns:
            exit_stats = tdf.groupby("exit_reason").agg(
                count=("pnl_pct", "size"),
                avg_pnl=("pnl_pct", "mean"),
                total_pnl=("pnl_pct", "sum"),
            )
            print("  Exit reason breakdown:")
            for reason, row in exit_stats.iterrows():
                print(f"    {reason}: {int(row['count'])} trades, avg {row['avg_pnl']:+.2f}%, total {row['total_pnl']:+.2f}%")
            trade_level_metrics["exit_reason_stats"] = {
                reason: {"count": int(row["count"]), "avg_pnl": round(float(row["avg_pnl"]), 4), "total_pnl": round(float(row["total_pnl"]), 4)}
                for reason, row in exit_stats.iterrows()
            }

        trade_level_metrics.update({
            "n_trades": n_trades,
            "n_symbols": n_symbols,
            "win_rate": round(win_rate, 4),
            "avg_pnl_pct": round(avg_pnl, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
            "payoff_ratio": round(payoff_ratio, 4) if payoff_ratio != float("inf") else "inf",
            "avg_win_pct": round(avg_win, 4),
            "avg_loss_pct": round(avg_loss, 4),
            "expectancy_pct": round(expectancy, 4),
        })


    if not eq.empty:
        has_trades = (eq["n_holdings"] > 0).sum()
        print(f"  有持仓交易日: {has_trades} / {len(eq)}")

    wf_df = pd.DataFrame()
    wf_stability_summary: dict[str, Any] = {}
    if run_walk_forward:
        print("\n  [Walk-Forward] 开始样本外稳定性评估...")
        wf_df = run_walk_forward_oos(
            daily_universe=daily,
            price_df=active_price_df,
            returns_df=active_returns_df,
            cfg=cfg,
            regime_df=regime_df,
            index_filter=index_filter,
            allowed_symbols_by_date=dynamic_watchlist_map,
            train_days=int(wf_train_days),
            test_days=int(wf_test_days),
            step_days=int(wf_step_days),
        )
        if wf_df.empty:
            print("  [Walk-Forward] No valid fold results.")
        else:
            wf_stability_summary = summarize_walk_forward_stability(wf_df, cfg=cfg, sharpe_floor=1.0)
            fold_count = int(wf_stability_summary.get("fold_count", len(wf_df)))
            mean_annual = float(wf_stability_summary.get("mean_annual_return_pct", 0.0))
            mean_dd = float(wf_stability_summary.get("mean_max_drawdown_pct", 0.0))
            mean_sharpe = float(wf_stability_summary.get("mean_sharpe", 0.0))
            kpi_ok_count = int(wf_stability_summary.get("kpi_all_ok_count", 0))
            sharpe_floor = float(wf_stability_summary.get("sharpe_floor", 1.0))
            sharpe_ok_count = int(wf_stability_summary.get("sharpe_ok_count", 0))
            sharpe_ok_ratio = float(wf_stability_summary.get("sharpe_ok_ratio", 0.0))
            annual_pos_count = int(wf_stability_summary.get("annual_positive_count", 0))
            annual_pos_ratio = float(wf_stability_summary.get("annual_positive_ratio", 0.0))
            drawdown_ok_count = int(wf_stability_summary.get("drawdown_ok_count", 0))
            drawdown_ok_ratio = float(wf_stability_summary.get("drawdown_ok_ratio", 0.0))
            print(
                "  [Walk-Forward] "
                f"{fold_count} folds | annual_mean {mean_annual:+.2f}% | "
                f"max_dd_mean {mean_dd:.2f}% | sharpe_mean {mean_sharpe:.2f} | "
                f"kpi_all_ok {kpi_ok_count}/{fold_count}"
            )
            print(
                "  [Walk-Forward Stability] "
                f"Sharpe>={sharpe_floor:.1f}: {sharpe_ok_count}/{fold_count} ({sharpe_ok_ratio * 100:.1f}%) | "
                f"annual>0: {annual_pos_count}/{fold_count} ({annual_pos_ratio * 100:.1f}%) | "
                f"drawdown_ok: {drawdown_ok_count}/{fold_count} ({drawdown_ok_ratio * 100:.1f}%)"
            )

    print("\n[6/6] 保存回测结果...")
    eq_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_equity.csv"
    eq_out.parent.mkdir(parents=True, exist_ok=True)
    eq.to_csv(eq_out, index=False, encoding="utf-8-sig")

    regime_snapshot_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_market_regime_snapshot.csv"
    if regime_df is not None and not regime_df.empty:
        regime_cols = [
            "date",
            "market_ret_1d",
            "market_ret_20d",
            "market_trend_pct",
            "regime",
            "regime_pos_cap",
            "market_vol_annual",
            "volatility_pos_mult",
            "market_liq_proxy",
            "liquidity_pos_mult",
            "state_pos_mult",
            "crash_lb_ret",
            "rebound_lb_ret",
            "momentum_crash_trigger",
            "momentum_crash_active",
            "momentum_crash_pos_cap",
        ]
        rr = regime_df.copy()
        keep_cols = [c for c in regime_cols if c in rr.columns]
        if keep_cols:
            rr[keep_cols].to_csv(regime_snapshot_out, index=False, encoding="utf-8-sig")

    # 娣囨繂鐡ㄦ禍銈嗘鐠佹澘缍?
    if trades:
        trades_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_trades.csv"
        pd.DataFrame(trades).to_csv(trades_out, index=False, encoding="utf-8-sig")
        print(f"  📄 交易明细输出: {trades_out} ({len(trades)}笔)")

    if parallel_results:
        compare_rows: list[dict[str, Any]] = []
        for mode_name, mode_result in parallel_results.items():
            mode_eq = mode_result.get("equity_curve", pd.DataFrame())
            mode_m = mode_result.get("metrics", {})
            mode_trades = mode_result.get("trade_log", [])
            mode_eq_out = base_dir / "data" / "backtests" / f"backtest_strategy_v3_equity_{mode_name}.csv"
            mode_eq.to_csv(mode_eq_out, index=False, encoding="utf-8-sig")
            if mode_trades:
                mode_trades_out = base_dir / "data" / "backtests" / f"backtest_strategy_v3_trades_{mode_name}.csv"
                pd.DataFrame(mode_trades).to_csv(mode_trades_out, index=False, encoding="utf-8-sig")
            compare_rows.append(
                {
                    "mode": mode_name,
                    "annual_return_pct": float(mode_m.get("annual_return_pct", 0.0)),
                    "max_drawdown_pct": float(mode_m.get("max_drawdown_pct", 0.0)),
                    "sharpe": float(mode_m.get("sharpe", 0.0)),
                    "win_rate_pct": float(mode_m.get("win_rate_pct", 0.0)),
                    "trades": int(len(mode_trades)),
                }
            )
        compare_df = pd.DataFrame(compare_rows)
        compare_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_execution_compare.csv"
        compare_df.to_csv(compare_out, index=False, encoding="utf-8-sig")
        print(f"  [Execution Compare] {compare_out}")

    momentum_crash_layer = summarize_momentum_crash_layer(cfg=cfg, regime_df=regime_df, eq=eq)
    crash_summary_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_momentum_crash_summary.json"
    crash_summary_out.write_text(json.dumps(momentum_crash_layer, ensure_ascii=False, indent=2), encoding="utf-8")
    crash_log_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_momentum_crash_log.csv"
    crash_windows = momentum_crash_layer.get("windows", [])
    if isinstance(crash_windows, list) and crash_windows:
        pd.DataFrame(crash_windows).to_csv(crash_log_out, index=False, encoding="utf-8-sig")
    elif crash_log_out.exists():
        crash_log_out.unlink()

    three_layer = build_three_layer_evaluation(
        cfg=cfg,
        metrics=m,
        kpi_status=kpi_status,
        eq=eq,
        trades=trades,
        wf_stability_summary=wf_stability_summary,
        parallel_results=parallel_results,
        momentum_crash_layer=momentum_crash_layer,
    )
    three_layer_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_three_layer_report.json"
    three_layer_out.write_text(json.dumps(three_layer, ensure_ascii=False, indent=2), encoding="utf-8")
    three_layer_md_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_three_layer_report.md"
    three_layer_md_out.write_text(render_three_layer_report_md(three_layer), encoding="utf-8")
    print(
        "  [Three-Layer] "
        f"{three_layer_out.name} / {three_layer_md_out.name} "
        f"(overall={three_layer.get('overall_status', 'unknown')})"
    )

    stats_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_stats.json"
    stats_payload = {**m, "kpi_status": kpi_status}
    if trade_level_metrics:
        stats_payload["trade_level_metrics"] = trade_level_metrics
    stats_payload["three_layer_evaluation"] = three_layer
    if parallel_results:
        stats_payload["execution_compare"] = {
            k: {
                "annual_return_pct": float(v.get("metrics", {}).get("annual_return_pct", 0.0)),
                "max_drawdown_pct": float(v.get("metrics", {}).get("max_drawdown_pct", 0.0)),
                "sharpe": float(v.get("metrics", {}).get("sharpe", 0.0)),
                "win_rate_pct": float(v.get("metrics", {}).get("win_rate_pct", 0.0)),
            }
            for k, v in parallel_results.items()
        }
    if wf_stability_summary:
        stats_payload["walk_forward_stability"] = wf_stability_summary
    stats_payload["momentum_crash_protection"] = momentum_crash_layer
    stats_out.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if run_walk_forward and not wf_df.empty:
        wf_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_walk_forward.csv"
        wf_df.to_csv(wf_out, index=False, encoding="utf-8-sig")
        print(f"  📄 Walk-Forward: {wf_out} ({len(wf_df)}行)")
        wf_summary_out = base_dir / "data" / "backtests" / "backtest_strategy_v3_walk_forward_summary.json"
        wf_summary_out.write_text(json.dumps(wf_stability_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✅ Walk-Forward Summary: {wf_summary_out}")

    print(f"\n{'=' * 60}")
    print("📌 回测结果汇总")
    print(f"  输出文件 {eq_out}")
    if regime_snapshot_out.exists():
        print(f"  市场状态快照 {regime_snapshot_out}")
    print(f"  崩盘保护汇总 {crash_summary_out}")
    print(f"  回测区间 {start_date} 至 {eq['date'].max() if not eq.empty else 'N/A'}")
    print(f"  指标 年化{m['annual_return_pct']:+.2f}% | 最大回撤{m['max_drawdown_pct']:.2f}% | Sharpe {m['sharpe']:.2f}")
    print(f"{'=' * 60}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Strategy V3 backtest")
    ap.add_argument("--base-dir", default=".", help="project root")
    ap.add_argument(
        "--config",
        default=None,
        help="path to config yaml (default: config.yaml, fallback: config_v31.yaml)",
    )
    ap.add_argument("--start-date", default=None, help="backtest start date (default: from config)")
    ap.add_argument("--entry-mode", default=None, choices=["custom", "strict", "normal", "loose"],
                     help="override entry mode")
    ap.add_argument("--optimize", action="store_true", help="run full grid-search optimization")
    ap.add_argument("--watchlist", default=None, help="watchlist yaml file for backtest universe")
    ap.add_argument(
        "--no-default-watchlist",
        action="store_true",
        help="do not auto-use backtest_watchlist.yaml/watchlist_cache.yaml when --watchlist is omitted",
    )
    ap.add_argument(
        "--dynamic-watchlist",
        action="store_true",
        help="use monthly rebalanced dynamic watchlist to avoid static-universe bias",
    )
    ap.add_argument("--dynamic-top-n", type=int, default=300, help="max symbols kept per dynamic watchlist rebalance")
    ap.add_argument("--walk-forward", action="store_true", help="run walk-forward out-of-sample evaluation")
    ap.add_argument("--wf-train-days", type=int, default=504, help="walk-forward train window (trading days)")
    ap.add_argument("--wf-test-days", type=int, default=126, help="walk-forward test window (trading days)")
    ap.add_argument("--wf-step-days", type=int, default=126, help="walk-forward rolling step (trading days)")
    ap.add_argument("--top-k", type=int, default=None, help="temporary override for top_k")
    ap.add_argument("--invest-more-n", type=int, default=None, help="temporary override for invest_more_n")
    ap.add_argument("--min-float-mkt-cap", type=float, default=None, help="temporary override for min float market cap")
    ap.add_argument("--min-amount-20d", type=float, default=None, help="temporary override for min 20d amount")
    ap.add_argument("--min-turnover-20d", type=float, default=None, help="temporary override for min 20d turnover")
    ap.add_argument("--enable-institution-filter", action="store_true", help="temporarily enable institution-holding filter")
    ap.add_argument("--disable-institution-filter", action="store_true", help="temporarily disable institution-holding filter")
    ap.add_argument(
        "--institution-filter-mode",
        choices=["data", "proxy"],
        default=None,
        help="institution filter mode",
    )
    ap.add_argument("--institution-min-pct", type=float, default=None, help="institution holding minimum percent")
    ap.add_argument("--institution-quantile", type=float, default=None, help="institution data filter quantile")
    ap.add_argument("--institution-proxy-quantile", type=float, default=None, help="institution proxy filter quantile")
    ap.add_argument("--enable-alpha-enhancement", action="store_true", help="temporarily enable alpha enhancement")
    ap.add_argument("--disable-alpha-enhancement", action="store_true", help="temporarily disable alpha enhancement")
    ap.add_argument("--alpha-industry-w", type=float, default=None, help="temporary override for alpha industry-rs weight")
    ap.add_argument("--alpha-flow-w", type=float, default=None, help="temporary override for alpha flow-persistence weight")
    ap.add_argument("--alpha-quality-w", type=float, default=None, help="temporary override for alpha quality weight")
    ap.add_argument("--alpha-str-w", type=float, default=None, help="temporary override for alpha short-reversal (STR) weight")
    ap.add_argument("--alpha-turnover-rev-w", type=float, default=None, help="temporary override for alpha turnover-reversal weight")
    ap.add_argument("--alpha-value-w", type=float, default=None, help="temporary override for alpha value-proxy weight")
    ap.add_argument("--enable-news-sentiment", action="store_true", help="temporarily enable news sentiment factor")
    ap.add_argument("--disable-news-sentiment", action="store_true", help="temporarily disable news sentiment factor")
    ap.add_argument("--news-sentiment-weight", type=float, default=None, help="temporary override for news sentiment factor weight")
    ap.add_argument("--enable-index-filter", action="store_true", help="temporarily enable index filter")
    ap.add_argument("--disable-index-filter", action="store_true", help="temporarily disable index filter")
    ap.add_argument("--enable-dynamic-regime-position", action="store_true", help="temporarily enable regime-based position caps")
    ap.add_argument("--disable-dynamic-regime-position", action="store_true", help="temporarily disable regime-based position caps")
    ap.add_argument("--regime-bull-cap", type=float, default=None, help="temporary override for bull regime position cap")
    ap.add_argument("--regime-neutral-cap", type=float, default=None, help="temporary override for neutral regime position cap")
    ap.add_argument("--regime-bear-cap", type=float, default=None, help="temporary override for bear regime position cap")
    ap.add_argument("--enable-volatility-sizing", action="store_true", help="temporarily enable volatility-based position sizing")
    ap.add_argument("--disable-volatility-sizing", action="store_true", help="temporarily disable volatility-based position sizing")
    ap.add_argument("--enable-liquidity-sizing", action="store_true", help="temporarily enable liquidity-state position sizing")
    ap.add_argument("--disable-liquidity-sizing", action="store_true", help="temporarily disable liquidity-state position sizing")
    ap.add_argument("--enable-momentum-crash-protect", action="store_true", help="temporarily enable momentum crash protection")
    ap.add_argument("--disable-momentum-crash-protect", action="store_true", help="temporarily disable momentum crash protection")
    ap.add_argument("--momentum-crash-cap", type=float, default=None, help="temporary override for momentum crash position cap")
    ap.add_argument("--enable-bear-entry-block", action="store_true", help="temporarily block new entries in BEAR regime")
    ap.add_argument("--disable-bear-entry-block", action="store_true", help="temporarily allow new entries in BEAR regime")
    ap.add_argument("--gate-bull-min", type=float, default=None, help="temporary override for bull regime min signal strength")
    ap.add_argument("--gate-neutral-min", type=float, default=None, help="temporary override for neutral regime min signal strength")
    ap.add_argument("--gate-bear-min", type=float, default=None, help="temporary override for bear regime min signal strength")
    ap.add_argument("--enable-weak-de-risk", action="store_true", help="temporarily enable weak-signal de-risk")
    ap.add_argument("--disable-weak-de-risk", action="store_true", help="temporarily disable weak-signal de-risk")
    ap.add_argument("--weak-signal-threshold", type=float, default=None, help="temporary override for weak-signal threshold")
    ap.add_argument("--weak-signal-cap-mul", type=float, default=None, help="temporary override for weak-signal cap multiplier")
    ap.add_argument("--enable-drawdown-brake", action="store_true", help="temporarily enable portfolio drawdown brake")
    ap.add_argument("--disable-drawdown-brake", action="store_true", help="temporarily disable portfolio drawdown brake")
    ap.add_argument("--drawdown-brake-threshold", type=float, default=None, help="temporary override for drawdown brake threshold (e.g. 0.10)")
    ap.add_argument("--drawdown-brake-position-cap", type=float, default=None, help="temporary override for drawdown brake position cap")
    ap.add_argument("--enable-adaptive-pre-brake", action="store_true", help="temporarily enable adaptive pre-brake mode")
    ap.add_argument("--disable-adaptive-pre-brake", action="store_true", help="temporarily disable adaptive pre-brake mode")
    ap.add_argument("--adaptive-dd-trigger", type=float, default=None, help="temporary override for adaptive pre-brake trigger (e.g. 0.04)")
    ap.add_argument("--adaptive-dd-pos-mul", type=float, default=None, help="temporary override for adaptive pre-brake position multiplier")
    ap.add_argument("--adaptive-dd-gate-boost", type=float, default=None, help="temporary override for adaptive pre-brake gate boost")
    ap.add_argument("--adaptive-dd-topk-mul", type=float, default=None, help="temporary override for adaptive pre-brake top_k multiplier")
    ap.add_argument("--adaptive-dd-invest-mul", type=float, default=None, help="temporary override for adaptive pre-brake invest_more_n multiplier")
    ap.add_argument("--adaptive-dd-stop-mul", type=float, default=None, help="temporary override for adaptive pre-brake stop-loss multiplier")
    ap.add_argument("--adaptive-dd-trail-mul", type=float, default=None, help="temporary override for adaptive pre-brake trailing-stop multiplier")
    ap.add_argument("--adaptive-dd-hold-mul", type=float, default=None, help="temporary override for adaptive pre-brake max-hold-days multiplier")
    ap.add_argument("--stop-loss-pct", type=float, default=None, help="temporary override for stop loss pct (e.g. 0.08)")
    ap.add_argument("--trailing-stop-pct", type=float, default=None, help="temporary override for trailing stop pct (e.g. 0.10)")
    ap.add_argument("--max-hold-days", type=int, default=None, help="temporary override for max hold days")
    ap.add_argument("--atr-stop-mult", type=float, default=None, help="temporary override for ATR stop multiplier")
    ap.add_argument("--failure-stop-days", type=int, default=None, help="temporary override for failure-stop days")
    ap.add_argument("--failure-stop-gain", type=float, default=None, help="temporary override for failure-stop min gain (e.g. 0.02)")
    ap.add_argument("--enable-failure-stop-deconflict", action="store_true", help="enable failure-stop deconflict guards")
    ap.add_argument("--disable-failure-stop-deconflict", action="store_true", help="disable failure-stop deconflict guards")
    ap.add_argument("--rank-exit-buffer", type=int, default=None, help="temporary override for rank-exit buffer around top_k")
    ap.add_argument("--rank-exit-trend-down-only", action="store_true", help="only run rank-exit when trend is down")
    ap.add_argument("--rank-exit-allow-trend-up", action="store_true", help="allow rank-exit even when trend is up")
    ap.add_argument(
        "--ma-trailing-priority",
        choices=["trailing", "ma"],
        default=None,
        help="conflict priority between ma_stop and trailing_stop",
    )
    ap.add_argument(
        "--execution-mode",
        default=None,
        choices=["close", "next_open", "parallel"],
        help="execution price mode: close | next_open | parallel",
    )
    ap.add_argument("--enable-execution-realism", action="store_true", help="enable execution realism constraints")
    ap.add_argument("--disable-execution-realism", action="store_true", help="disable execution realism constraints")
    ap.add_argument("--max-participation-rate", type=float, default=None, help="temporary override for max participation rate")
    ap.add_argument("--execution-slippage-bps", type=float, default=None, help="temporary override for execution slippage bps")
    ap.add_argument("--execution-impact-bps", type=float, default=None, help="temporary override for execution impact bps")
    ap.add_argument("--execution-impact-exp", type=float, default=None, help="temporary override for execution impact exponent")
    ap.add_argument("--enable-lot-rounding", action="store_true", help="enable lot-size rounding")
    ap.add_argument("--disable-lot-rounding", action="store_true", help="disable lot-size rounding")
    ap.add_argument("--lot-size", type=int, default=None, help="temporary override for lot size")
    ap.add_argument("--enable-price-limit", action="store_true", help="enable price-limit execution constraints")
    ap.add_argument("--disable-price-limit", action="store_true", help="disable price-limit execution constraints")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()

    # 绾暰鐠у嘲閺冦儲婀￠敍姘嚒娴犮倛 > 闁板秶鐤嗛弬鍥︽ > 姒涙閸?
    if args.start_date:
        start_date = args.start_date
    else:
        cfg_file = _resolve_config_file(base_dir, config_path=args.config)
        if cfg_file and cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            sd = raw.get("backtest", {}).get("start_date") or raw.get("market_data", {}).get("start_date")
            start_date = str(sd) if sd else "2020-01-01"
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
        if wl_file is None and not args.no_default_watchlist:
            for cand in ["backtest_watchlist.yaml", "watchlist_cache.yaml"]:
                if (base_dir / cand).exists():
                    wl_file = cand
                    print(f"  Auto watchlist: {cand} (use --no-default-watchlist to disable)")
                    break
        if wl_file is None:
            print("  Hint: use --watchlist to specify a stock pool file.")
        institution_filter_override = None
        if args.enable_institution_filter:
            institution_filter_override = True
        if args.disable_institution_filter:
            institution_filter_override = False
        alpha_override = None
        if args.enable_alpha_enhancement:
            alpha_override = True
        if args.disable_alpha_enhancement:
            alpha_override = False
        news_sentiment_override = None
        if args.enable_news_sentiment:
            news_sentiment_override = True
        if args.disable_news_sentiment:
            news_sentiment_override = False
        index_filter_override = None
        if args.enable_index_filter:
            index_filter_override = True
        if args.disable_index_filter:
            index_filter_override = False
        regime_pos_override = None
        if args.enable_dynamic_regime_position:
            regime_pos_override = True
        if args.disable_dynamic_regime_position:
            regime_pos_override = False
        vol_sizing_override = None
        if args.enable_volatility_sizing:
            vol_sizing_override = True
        if args.disable_volatility_sizing:
            vol_sizing_override = False
        liq_sizing_override = None
        if args.enable_liquidity_sizing:
            liq_sizing_override = True
        if args.disable_liquidity_sizing:
            liq_sizing_override = False
        momentum_crash_override = None
        if args.enable_momentum_crash_protect:
            momentum_crash_override = True
        if args.disable_momentum_crash_protect:
            momentum_crash_override = False
        bear_entry_block_override = None
        if args.enable_bear_entry_block:
            bear_entry_block_override = True
        if args.disable_bear_entry_block:
            bear_entry_block_override = False
        weak_de_risk_override = None
        if args.enable_weak_de_risk:
            weak_de_risk_override = True
        if args.disable_weak_de_risk:
            weak_de_risk_override = False
        drawdown_brake_override = None
        if args.enable_drawdown_brake:
            drawdown_brake_override = True
        if args.disable_drawdown_brake:
            drawdown_brake_override = False
        adaptive_pre_brake_override = None
        if args.enable_adaptive_pre_brake:
            adaptive_pre_brake_override = True
        if args.disable_adaptive_pre_brake:
            adaptive_pre_brake_override = False
        failure_stop_deconflict_override = None
        if args.enable_failure_stop_deconflict:
            failure_stop_deconflict_override = True
        if args.disable_failure_stop_deconflict:
            failure_stop_deconflict_override = False
        rank_exit_trend_only_override = None
        if args.rank_exit_trend_down_only:
            rank_exit_trend_only_override = True
        if args.rank_exit_allow_trend_up:
            rank_exit_trend_only_override = False
        execution_realism_override = None
        if args.enable_execution_realism:
            execution_realism_override = True
        if args.disable_execution_realism:
            execution_realism_override = False
        lot_rounding_override = None
        if args.enable_lot_rounding:
            lot_rounding_override = True
        if args.disable_lot_rounding:
            lot_rounding_override = False
        price_limit_override = None
        if args.enable_price_limit:
            price_limit_override = True
        if args.disable_price_limit:
            price_limit_override = False
        run_quick_backtest(
            base_dir,
            start_date=start_date,
            watchlist_file=wl_file,
            config_path=args.config,
            use_dynamic_watchlist=bool(args.dynamic_watchlist),
            dynamic_top_n=int(args.dynamic_top_n),
            run_walk_forward=bool(args.walk_forward),
            wf_train_days=int(args.wf_train_days),
            wf_test_days=int(args.wf_test_days),
            wf_step_days=int(args.wf_step_days),
            top_k_override=args.top_k,
            invest_more_n_override=args.invest_more_n,
            entry_mode_override=args.entry_mode,
            min_float_mkt_cap_override=args.min_float_mkt_cap,
            min_amount_20d_override=args.min_amount_20d,
            min_turnover_20d_override=args.min_turnover_20d,
            use_institution_filter_override=institution_filter_override,
            institution_filter_mode_override=args.institution_filter_mode,
            institution_holding_min_pct_override=args.institution_min_pct,
            institution_holding_quantile_override=args.institution_quantile,
            institution_proxy_quantile_override=args.institution_proxy_quantile,
            use_alpha_enhancement_override=alpha_override,
            alpha_industry_weight_override=args.alpha_industry_w,
            alpha_flow_weight_override=args.alpha_flow_w,
            alpha_quality_weight_override=args.alpha_quality_w,
            alpha_short_reversal_weight_override=args.alpha_str_w,
            alpha_turnover_reversal_weight_override=args.alpha_turnover_rev_w,
            alpha_value_proxy_weight_override=args.alpha_value_w,
            use_news_sentiment_factor_override=news_sentiment_override,
            news_sentiment_weight_override=args.news_sentiment_weight,
            use_index_filter_override=index_filter_override,
            use_dynamic_regime_position_override=regime_pos_override,
            regime_bull_pos_cap_override=args.regime_bull_cap,
            regime_neutral_pos_cap_override=args.regime_neutral_cap,
            regime_bear_pos_cap_override=args.regime_bear_cap,
            use_volatility_sizing_override=vol_sizing_override,
            use_liquidity_state_sizing_override=liq_sizing_override,
            use_momentum_crash_protection_override=momentum_crash_override,
            momentum_crash_position_cap_override=args.momentum_crash_cap,
            block_new_in_bear_override=bear_entry_block_override,
            regime_gate_bull_min_override=args.gate_bull_min,
            regime_gate_neutral_min_override=args.gate_neutral_min,
            regime_gate_bear_min_override=args.gate_bear_min,
            use_weak_signal_de_risk_override=weak_de_risk_override,
            weak_signal_threshold_override=args.weak_signal_threshold,
            weak_signal_cap_multiplier_override=args.weak_signal_cap_mul,
            use_drawdown_brake_override=drawdown_brake_override,
            drawdown_brake_threshold_override=args.drawdown_brake_threshold,
            drawdown_brake_position_cap_override=args.drawdown_brake_position_cap,
            use_adaptive_drawdown_mode_override=adaptive_pre_brake_override,
            adaptive_drawdown_trigger_override=args.adaptive_dd_trigger,
            adaptive_drawdown_position_cap_multiplier_override=args.adaptive_dd_pos_mul,
            adaptive_drawdown_gate_boost_override=args.adaptive_dd_gate_boost,
            adaptive_drawdown_top_k_multiplier_override=args.adaptive_dd_topk_mul,
            adaptive_drawdown_invest_more_multiplier_override=args.adaptive_dd_invest_mul,
            adaptive_drawdown_stop_loss_multiplier_override=args.adaptive_dd_stop_mul,
            adaptive_drawdown_trailing_multiplier_override=args.adaptive_dd_trail_mul,
            adaptive_drawdown_max_hold_multiplier_override=args.adaptive_dd_hold_mul,
            stop_loss_pct_override=args.stop_loss_pct,
            trailing_stop_pct_override=args.trailing_stop_pct,
            max_hold_days_override=args.max_hold_days,
            atr_stop_multiplier_override=args.atr_stop_mult,
            failure_stop_days_override=args.failure_stop_days,
            failure_stop_gain_override=args.failure_stop_gain,
            failure_stop_deconflict_override=failure_stop_deconflict_override,
            rank_exit_buffer_override=args.rank_exit_buffer,
            rank_exit_only_when_trend_down_override=rank_exit_trend_only_override,
            ma_trailing_priority_override=args.ma_trailing_priority,
            execution_mode_override=args.execution_mode,
            use_execution_realism_override=execution_realism_override,
            max_participation_rate_override=args.max_participation_rate,
            execution_slippage_bps_override=args.execution_slippage_bps,
            execution_impact_bps_override=args.execution_impact_bps,
            execution_impact_exponent_override=args.execution_impact_exp,
            enforce_lot_rounding_override=lot_rounding_override,
            lot_size_override=args.lot_size,
            use_price_limit_constraints_override=price_limit_override,
        )


if __name__ == "__main__":
    main()
