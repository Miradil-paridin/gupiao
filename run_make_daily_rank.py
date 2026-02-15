"""
每日排名信号生成 V3.2（与回测脚本入场逻辑一致）

核心修复：
1. 入场条件与 run_backtest_strategy_v3.py 的 _candidate_entry_mask 完全一致
2. normal模式: 3条路径(涨停回调+2of3 / TDX+趋势 / 放量+趋势+主力)
3. 从config读取所有V3.2参数(tdx_min_score/pullback/流动性等)
4. 行业映射支持大股票池(不依赖config注释)
5. 诊断输出：告诉你哪个环节过滤掉了股票

使用方法：
    python run_make_daily_rank.py
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd
import yaml
import re


@dataclass
class SignalConfigV32:
    """信号配置 V3.2 - 与回测脚本参数一致"""
    invest_more_n: int = 20
    least_n: int = 3
    withdraw_score_threshold: float = -0.5
    risk_vol_20d_threshold: float = 0.55
    use_tradeability_filter: bool = True
    use_limit_up_entry: bool = True
    pullback_window_start: int = 2
    pullback_window_end: int = 15
    pullback_min_pct: float = 0.05
    pullback_max_pct: float = 0.35
    use_tdx_indicators: bool = True
    tdx_high30_weight: float = 1.0
    tdx_main_force_weight: float = 1.5
    tdx_limit_up_30d_weight: float = 0.5
    tdx_min_score: float = 1.0
    use_volatility_sizing: bool = True
    max_single_weight: float = 0.12
    entry_mode: str = "normal"
    normal_min_conditions: int = 2
    use_market_regime: bool = False
    use_industry_diversification: bool = True
    max_per_industry: int = 3
    use_liquidity_filter: bool = True
    min_amount_20d: float = 5e7
    min_turnover_20d: float = 0.4
    use_correlation_control: bool = True
    max_pairwise_corr: float = 0.80
    stop_loss_pct: float = 0.08
    trailing_stop_pct: float = 0.10
    use_tdx_protection: bool = True
    tdx_protection_threshold: float = 2.0
    use_signal_tiered_sizing: bool = True
    tier_strong_multiplier: float = 1.2
    tier_normal_multiplier: float = 1.0
    tier_weak_multiplier: float = 0.5


def _safe_float(value, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_config(base_dir: Path) -> dict:
    for cfg_name in ["config_v31.yaml", "config_optimized.yaml", "config.yaml"]:
        cfg_path = base_dir / cfg_name
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                print(f"  📂 使用配置: {cfg_name}")
                return yaml.safe_load(f) or {}
    return {}


def build_industry_map(base_dir: Path, symbols: pd.Series) -> Dict[str, str]:
    mapping = {}
    for cfg_name in ["config_v31.yaml", "config.yaml"]:
        cfg_path = base_dir / cfg_name
        if not cfg_path.exists():
            continue
        content = cfg_path.read_text(encoding="utf-8", errors="ignore")
        current_industry = "其他"
        for line in content.split('\n'):
            if "==========" in line:
                match = re.search(r'=+\s*([^=]+)\s*=+', line)
                if match:
                    raw = match.group(1).strip()
                    if "消费" in raw: current_industry = "消费"
                    elif "医药" in raw: current_industry = "医药"
                    elif "科技" in raw: current_industry = "科技"
                    elif "金融" in raw: current_industry = "金融"
                    elif "能源" in raw: current_industry = "新能源"
                    else: current_industry = "其他"
            code_match = re.search(r'"(\d{6})"', line)
            if code_match:
                code = code_match.group(1)
                mapping[code] = current_industry
                if code.startswith(('6', '5')):
                    mapping[f"{code}.SH"] = current_industry
                else:
                    mapping[f"{code}.SZ"] = current_industry
        if mapping:
            break
    industry_parquet = base_dir / "data" / "industry_map.parquet"
    if industry_parquet.exists():
        try:
            idf = pd.read_parquet(industry_parquet)
            for _, row in idf.iterrows():
                sym = str(row.get("symbol", ""))
                ind = str(row.get("industry", "其他"))
                if sym and sym not in mapping:
                    mapping[sym] = ind
        except Exception:
            pass

    # ── 未映射的股票：用BaoStock查行业 ──
    unmapped = [str(s) for s in symbols.unique() if str(s) not in mapping and str(s).split(".")[0] not in mapping]
    if unmapped:
        try:
            import baostock as bs
            bs.login()
            for sym in unmapped:
                code6 = sym.split(".")[0]
                bs_code = f"sh.{code6}" if code6.startswith("6") else f"sz.{code6}"
                rs = bs.query_stock_industry(code=bs_code)
                while rs.next():
                    row_data = rs.get_row_data()
                    if len(row_data) >= 4 and row_data[3]:
                        ind_name = row_data[3]  # 行业名称
                        mapping[sym] = ind_name
                        mapping[code6] = ind_name
                        break
            bs.logout()
        except Exception as e:
            pass  # BaoStock不可用就用下面的回退方案

    # ── 回退：按代码前缀粗分（比全部"其他"好）──
    # A股代码段对应的大致板块
    prefix_industry = {
        "000": "深主板A", "001": "深主板A", "002": "中小板", "003": "中小板",
        "300": "创业板", "301": "创业板",
        "600": "沪主板A", "601": "沪主板B", "603": "沪主板C", "605": "沪主板D",
        "688": "科创板",
    }
    for sym in symbols.unique():
        s = str(sym).split(".")[0]
        if sym not in mapping and s not in mapping:
            prefix3 = s[:3]
            ind = prefix_industry.get(prefix3, "其他")
            mapping[str(sym)] = ind
            mapping[s] = ind

    # 缓存行业映射（下次更快）
    try:
        cache_path = base_dir / "data" / "industry_map.parquet"
        records = [{"symbol": k, "industry": v} for k, v in mapping.items() if "." in k]
        if records:
            pd.DataFrame(records).to_parquet(cache_path, index=False)
    except Exception:
        pass

    return mapping


def _zscore(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    mu = x.mean(); sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sd


def load_features(base_dir: Path) -> pd.DataFrame:
    p = base_dir / "data" / "features" / "features_daily.parquet"
    if not p.exists():
        raise FileNotFoundError("features_daily.parquet not found")
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def check_tradeable(row, cfg):
    if not cfg.use_tradeability_filter: return True
    if row.get("one_line_board", 0) == 1: return False
    if row.get("limit_up_flag", 0) == 1: return False
    if row.get("near_limit_up", 0) == 1: return False
    return True


def check_liquidity(row, cfg):
    if not cfg.use_liquidity_filter: return True

    # 成交额检查 — 兼容多种列名
    amount_20d = _safe_float(row.get("amount_20d", 0))
    if amount_20d == 0:
        amount_20d = _safe_float(row.get("amount", 0))
    if amount_20d == 0:
        # 没有成交额列 → 用 volume * close 估算
        # volume 单位是"手"(100股)，所以要 ×100
        vol = _safe_float(row.get("volume", 0))
        close = _safe_float(row.get("close", 0))
        if vol > 0 and close > 0:
            amount_20d = vol * 100 * close  # 手 → 股 → 元
    # 如果完全没有成交额数据，跳过此检查
    if amount_20d == 0:
        pass  # 不过滤
    elif amount_20d < _safe_float(cfg.min_amount_20d):
        return False

    # 换手率检查 — 兼容多种列名
    turnover_20d = _safe_float(row.get("turnover_20d", 0))
    if turnover_20d == 0:
        turnover_20d = _safe_float(row.get("turnover", 0))
    if turnover_20d == 0:
        turnover_20d = _safe_float(row.get("turn", 0))
    # 如果完全没有换手率数据，跳过此检查（不因缺数据而误杀）
    if turnover_20d == 0:
        pass  # 不过滤
    elif turnover_20d < _safe_float(cfg.min_turnover_20d):
        return False

    return True


def check_eligible(row, cfg):
    """入场条件 - 与 run_backtest_strategy_v3.py 的 _candidate_entry_mask 一致"""
    mode = cfg.entry_mode.lower()
    days_since = _safe_float(row.get("days_since_limit_up", np.nan))
    pullback = _safe_float(row.get("pullback_pct", np.nan))
    volume_breakout = int(_safe_float(row.get("volume_breakout", 0)))
    price_above_ma5 = int(_safe_float(row.get("price_above_ma5", 0)))
    trend_up = _safe_float(row.get("ma_dist_20", 0)) > 0
    tdx_score = _safe_float(row.get("tdx_score", 0))
    high30 = int(_safe_float(row.get("high30_breakout", 0)))
    main_force = int(_safe_float(row.get("main_force_strong", 0)))

    in_window = (cfg.pullback_window_start <= days_since <= cfg.pullback_window_end) if np.isfinite(days_since) else False
    pullback_ok = (cfg.pullback_min_pct <= pullback <= cfg.pullback_max_pct) if np.isfinite(pullback) else False
    limit_up_pullback = in_window and pullback_ok
    breakout_ok = (volume_breakout == 1) and (price_above_ma5 == 1)
    tdx_ok = (tdx_score >= cfg.tdx_min_score) or (high30 == 1 and main_force == 1)

    if mode == "strict":
        return ((limit_up_pullback and breakout_ok) or tdx_ok) and trend_up
    elif mode == "loose":
        return trend_up
    else:  # normal - 3条路径
        extra_count = int(breakout_ok) + int(tdx_ok) + int(trend_up)
        path_pullback = limit_up_pullback and (extra_count >= cfg.normal_min_conditions)
        path_tdx = tdx_ok and trend_up
        path_breakout = breakout_ok and trend_up and (main_force == 1)
        return path_pullback or path_tdx or path_breakout


def get_signal_strength(row, cfg):
    days_since = _safe_float(row.get("days_since_limit_up", np.nan))
    pullback = _safe_float(row.get("pullback_pct", np.nan))
    trend_up = _safe_float(row.get("ma_dist_20", 0)) > 0
    tdx_score = _safe_float(row.get("tdx_score", 0))
    high30 = int(_safe_float(row.get("high30_breakout", 0)))
    main_force = int(_safe_float(row.get("main_force_strong", 0)))
    in_window = (cfg.pullback_window_start <= days_since <= cfg.pullback_window_end) if np.isfinite(days_since) else False
    pullback_ok = (cfg.pullback_min_pct <= pullback <= cfg.pullback_max_pct) if np.isfinite(pullback) else False
    limit_up_pullback = in_window and pullback_ok
    tdx_ok = (tdx_score >= cfg.tdx_min_score) or (high30 == 1 and main_force == 1)
    if limit_up_pullback and tdx_ok and trend_up: return 2.0
    if (tdx_ok and trend_up) or (limit_up_pullback and trend_up): return 1.0
    return 0.5


def get_entry_path(row, cfg) -> str:
    """判断入场路径，用于建议买入时机"""
    days_since = _safe_float(row.get("days_since_limit_up", np.nan))
    pullback = _safe_float(row.get("pullback_pct", np.nan))
    volume_breakout = int(_safe_float(row.get("volume_breakout", 0)))
    price_above_ma5 = int(_safe_float(row.get("price_above_ma5", 0)))
    trend_up = _safe_float(row.get("ma_dist_20", 0)) > 0
    tdx_score = _safe_float(row.get("tdx_score", 0))
    high30 = int(_safe_float(row.get("high30_breakout", 0)))
    main_force = int(_safe_float(row.get("main_force_strong", 0)))

    in_window = (cfg.pullback_window_start <= days_since <= cfg.pullback_window_end) if np.isfinite(days_since) else False
    pullback_ok = (cfg.pullback_min_pct <= pullback <= cfg.pullback_max_pct) if np.isfinite(pullback) else False
    limit_up_pullback = in_window and pullback_ok
    breakout_ok = (volume_breakout == 1) and (price_above_ma5 == 1)
    tdx_ok = (tdx_score >= cfg.tdx_min_score) or (high30 == 1 and main_force == 1)

    # 强信号：涨停回调+TDX全满
    if limit_up_pullback and tdx_ok and trend_up:
        return "涨停回调+主力"

    # 路径1：涨停回调
    extra_count = int(breakout_ok) + int(tdx_ok) + int(trend_up)
    if limit_up_pullback and extra_count >= cfg.normal_min_conditions:
        return "涨停回调"

    # 路径2：TDX+趋势
    if tdx_ok and trend_up:
        return "主力控盘"

    # 路径3：放量突破+趋势+主力
    if breakout_ok and trend_up and main_force == 1:
        return "放量突破"

    return "其他"


def get_timing_advice(entry_path: str, signal_strength: float, vol_20d: float) -> str:
    """根据入场路径和信号强度给出买入时机建议"""
    if entry_path == "涨停回调+主力":
        return "开盘确认放量后买入"
    elif entry_path == "涨停回调":
        return "开盘观察30分钟确认"
    elif entry_path == "放量突破":
        return "盘中放量突破时买入"
    elif entry_path == "主力控盘":
        if vol_20d > 0.5:
            return "尾盘买入(波动大)"
        else:
            return "盘中低点分批买入"
    else:
        return "尾盘择机买入"


def load_portfolio(base_dir: Path) -> dict:
    """加载当前持仓（如果有）"""
    port_path = base_dir / "data" / "portfolio.yaml"
    if port_path.exists():
        with open(port_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("holdings", {})
    return {}


def save_portfolio_advice(base_dir: Path, holdings: dict, new_buys: list, sells: list, keeps: list):
    """保存持仓建议到文件"""
    port_path = base_dir / "data" / "portfolio_advice.yaml"
    advice = {
        "current_holdings": holdings,
        "new_buys": new_buys,
        "sells": sells,
        "keeps": keeps,
        "updated": str(dt.date.today()),
    }
    port_path.parent.mkdir(parents=True, exist_ok=True)
    with open(port_path, "w", encoding="utf-8") as f:
        yaml.dump(advice, f, allow_unicode=True, default_flow_style=False)


def apply_industry_diversification(candidates, industry_map, max_per_industry):
    if candidates.empty: return candidates
    candidates = candidates.copy()
    candidates["industry"] = candidates["symbol"].astype(str).map(
        lambda s: industry_map.get(s, industry_map.get(s.split(".")[0], "其他")))
    result = []; counts = {}
    for _, row in candidates.iterrows():
        ind = row["industry"]; c = counts.get(ind, 0)
        if c < max_per_industry: result.append(row); counts[ind] = c + 1
    return pd.DataFrame(result)


def compute_weights(selected, cfg):
    if selected.empty: return {}
    selected = selected.copy(); n = len(selected)
    if cfg.use_volatility_sizing and "atr_pct" in selected.columns:
        atr = selected["atr_pct"].clip(lower=0.01)
        raw_w = 1.0 / atr; raw_w = raw_w / raw_w.sum()
    else:
        raw_w = pd.Series(1.0 / n, index=selected.index)
    if cfg.use_signal_tiered_sizing and "_signal_strength" in selected.columns:
        for idx in selected.index:
            s = _safe_float(selected.at[idx, "_signal_strength"], 1.0)
            if s >= 2.0: raw_w[idx] *= cfg.tier_strong_multiplier
            elif s >= 1.0: raw_w[idx] *= cfg.tier_normal_multiplier
            else: raw_w[idx] *= cfg.tier_weak_multiplier
    weights = raw_w.clip(upper=cfg.max_single_weight)
    if weights.sum() > 1.0: weights = weights / weights.sum()
    return dict(zip(selected["symbol"], weights))


def compute_daily_ranking_v32(feats, as_of=None, cfg=None, industry_map=None, market_can_open=True):
    if cfg is None: cfg = SignalConfigV32()
    if industry_map is None: industry_map = {}
    d = feats["date"].max() if as_of is None else pd.to_datetime(as_of).date()
    day = feats[feats["date"] == d].copy()
    if day.empty: raise ValueError(f"No rows for date: {d}")

    numeric_cols = ["ma_dist_20","ret_20d","ret_60d","vol_20d","vol_ratio_20","atr_14","close",
        "amount_20d","amount","turnover_20d","turnover","tdx_score","days_since_limit_up",
        "pullback_pct","high30_breakout","main_force_strong","main_force_control",
        "has_limit_up_30d","volume_breakout","price_above_ma5"]
    for col in numeric_cols:
        if col in day.columns: day[col] = pd.to_numeric(day[col], errors="coerce")

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

    day["score"] = 2.0*z_ma + 1.0*z_r20 + 0.5*z_r60 - 1.0*z_vol - 0.5*z_atr + 0.3*z_vr

    if cfg.use_tdx_indicators:
        if "high30_breakout" in day.columns: day["score"] += day["high30_breakout"].fillna(0) * cfg.tdx_high30_weight
        if "main_force_strong" in day.columns: day["score"] += day["main_force_strong"].fillna(0) * cfg.tdx_main_force_weight
        if "has_limit_up_30d" in day.columns: day["score"] += day["has_limit_up_30d"].fillna(0) * cfg.tdx_limit_up_30d_weight
        if "main_force_control" in day.columns: day["score"] += (day["main_force_control"].fillna(0) > 0).astype(int) * 0.3

    day["trend_up"] = (day.get("ma_dist_20", pd.Series(0, index=day.index)) > 0).astype(int)
    day["mom_bad"] = ((day.get("ret_20d", pd.Series(0, index=day.index)) < 0) & (day.get("ret_60d", pd.Series(0, index=day.index)) < 0)).astype(int)
    day["risk_high"] = (day.get("vol_20d", pd.Series(0, index=day.index)) >= cfg.risk_vol_20d_threshold).astype(int)

    day["tradeable"] = day.apply(lambda r: check_tradeable(r, cfg), axis=1)
    day["liquidity_ok"] = day.apply(lambda r: check_liquidity(r, cfg), axis=1)
    day["eligible"] = day.apply(lambda r: check_eligible(r, cfg), axis=1)
    day["_signal_strength"] = day.apply(lambda r: get_signal_strength(r, cfg), axis=1)

    day = day.sort_values("score", ascending=False).reset_index(drop=True)
    day["rank"] = np.arange(1, len(day) + 1)
    day["action"] = "HOLD"

    if cfg.use_tdx_protection:
        tdx_s = day.get("tdx_score", pd.Series(0, index=day.index))
        wm = ((day["trend_up"]==0)&(day["mom_bad"]==1)&(tdx_s<cfg.tdx_protection_threshold))|(day["score"]<=cfg.withdraw_score_threshold)
    else:
        wm = ((day["trend_up"]==0)&(day["mom_bad"]==1))|(day["score"]<=cfg.withdraw_score_threshold)
    day.loc[wm, "action"] = "WITHDRAW"

    rm = (day["action"]!="WITHDRAW")&(day["risk_high"]==1)&(day["trend_up"]==1)&(day["mom_bad"]==0)
    day.loc[rm, "action"] = "REDUCE"

    # INVEST_MORE 候选 — HOLD 和 REDUCE 都可以入选
    # 回测脚本是先选股再管风控，信号脚本应一致
    # REDUCE的股票符合入场但波动高，给低仓位而不是完全排除
    invest_candidates = day[
        (day["action"].isin(["HOLD", "REDUCE"])) &
        (day["tradeable"]==True) &
        (day["liquidity_ok"]==True) &
        (day["eligible"]==True) &
        (day["trend_up"]==1)
    ].copy()

    # REDUCE的股票信号强度降一级（仓位会更小）
    if not invest_candidates.empty:
        reduce_mask = invest_candidates["action"] == "REDUCE"
        invest_candidates.loc[reduce_mask, "_signal_strength"] = \
            invest_candidates.loc[reduce_mask, "_signal_strength"].clip(upper=0.5)

    if cfg.use_industry_diversification and industry_map:
        invest_candidates = apply_industry_diversification(invest_candidates.sort_values("score", ascending=False), industry_map, cfg.max_per_industry)

    if cfg.use_market_regime and not market_can_open:
        invest_syms = []
    else:
        invest_syms = invest_candidates.sort_values("score", ascending=False).head(cfg.invest_more_n)["symbol"].tolist()
    day.loc[day["symbol"].isin(invest_syms), "action"] = "INVEST_MORE"

    least_syms = day[day["action"]=="HOLD"].sort_values("score").head(cfg.least_n)["symbol"].tolist()
    day.loc[day["symbol"].isin(least_syms), "action"] = "LEAST"

    selected = day[day["action"]=="INVEST_MORE"]
    weights = compute_weights(selected, cfg)
    day["target_weight"] = day["symbol"].map(weights).fillna(0.0)
    day["industry"] = day["symbol"].astype(str).map(lambda s: industry_map.get(s, industry_map.get(s.split(".")[0], "其他")))

    # 入场路径和买入时机
    day["entry_path"] = day.apply(lambda r: get_entry_path(r, cfg), axis=1)
    day["timing"] = day.apply(
        lambda r: get_timing_advice(r["entry_path"], _safe_float(r.get("_signal_strength", 1)), _safe_float(r.get("vol_20d", 0))),
        axis=1
    )

    out_cols = ["date","symbol","industry","close","score","rank","action","target_weight","_signal_strength",
        "entry_path","timing",
        "ma_dist_20","ret_20d","ret_60d","vol_20d","atr_pct","vol_ratio_20","trend_up","mom_bad","risk_high",
        "tradeable","liquidity_ok","eligible"]
    for c in ["high30_breakout","main_force_strong","main_force_control","has_limit_up_30d","tdx_score"]:
        if c in day.columns: out_cols.append(c)
    out_cols = [c for c in out_cols if c in day.columns]
    return day[out_cols].sort_values("rank").reset_index(drop=True)


def save_daily_ranking(base_dir, ranking):
    out_dir = base_dir / "data" / "signals"; out_dir.mkdir(parents=True, exist_ok=True)
    d = pd.to_datetime(ranking["date"].iloc[0]).date().isoformat()
    dated_path = out_dir / f"daily_rank_{d}.csv"
    latest_path = out_dir / "latest_daily_rank.csv"
    ranking.to_csv(dated_path, index=False, encoding="utf-8-sig")
    ranking.to_csv(latest_path, index=False, encoding="utf-8-sig")
    return dated_path, latest_path


def main():
    base_dir = Path(__file__).resolve().parent
    print("="*60); print("📊 每日排名信号生成 V3.2"); print("="*60)

    config = load_config(base_dir)
    strategy = config.get("strategy", {}); risk = config.get("risk_control", {})

    cfg = SignalConfigV32(
        invest_more_n=int(strategy.get("top_k", 20)),
        pullback_window_start=int(strategy.get("pullback_window_start", 2)),
        pullback_window_end=int(strategy.get("pullback_window_end", 15)),
        pullback_min_pct=float(strategy.get("pullback_min_pct", 0.05)),
        pullback_max_pct=float(strategy.get("pullback_max_pct", 0.35)),
        tdx_min_score=float(strategy.get("tdx_min_score", 1.0)),
        max_single_weight=float(strategy.get("max_single_weight", 0.12)),
        entry_mode=str(strategy.get("entry_mode", "normal")),
        normal_min_conditions=int(strategy.get("normal_min_conditions", 2)),
        use_industry_diversification=strategy.get("industry_diversification", {}).get("enabled", True),
        max_per_industry=int(strategy.get("industry_diversification", {}).get("max_per_industry", 3)),
        use_liquidity_filter=strategy.get("liquidity_filter", {}).get("enabled", True),
        min_amount_20d=float(strategy.get("liquidity_filter", {}).get("min_amount_20d", 5e7)),
        min_turnover_20d=float(strategy.get("liquidity_filter", {}).get("min_turnover_20d", 0.4)),
        use_signal_tiered_sizing=bool(strategy.get("use_signal_tiered_sizing", True)),
        stop_loss_pct=float(risk.get("stop_loss_pct", 0.08)),
        trailing_stop_pct=float(risk.get("trailing_stop_pct", 0.10)),
        use_tdx_protection=risk.get("use_tdx_protection", True),
        tdx_protection_threshold=float(risk.get("tdx_protection_threshold", 2.0)),
        use_market_regime=strategy.get("use_market_regime", False),
    )

    print(f"\n📋 配置:")
    print(f"   入场模式: {cfg.entry_mode} (附加条件≥{cfg.normal_min_conditions})")
    print(f"   选股数: {cfg.invest_more_n}")
    print(f"   回调: {cfg.pullback_min_pct*100:.0f}%-{cfg.pullback_max_pct*100:.0f}% | 窗口: {cfg.pullback_window_start}-{cfg.pullback_window_end}天")
    print(f"   TDX最低分: {cfg.tdx_min_score}")
    print(f"   行业分散: 每行业最多{cfg.max_per_industry}只")
    print(f"   流动性: 均额>{cfg.min_amount_20d/1e4:.0f}万, 换手>{cfg.min_turnover_20d}%")

    print("\n📂 加载特征数据...")
    feats = load_features(base_dir)
    print(f"   日期范围: {feats['date'].min()} -> {feats['date'].max()}")
    print(f"   股票数: {feats['symbol'].nunique()}")

    industry_map = build_industry_map(base_dir, feats["symbol"])
    print(f"   行业映射: {len(industry_map)} 条")

    latest_date = str(feats["date"].max())
    print(f"\n📈 计算 {latest_date} 的信号...")

    market_can_open = True
    ranking = compute_daily_ranking_v32(feats, as_of=latest_date, cfg=cfg, industry_map=industry_map, market_can_open=market_can_open)

    invest_more = ranking[ranking["action"] == "INVEST_MORE"]

    print(f"\n📊 诊断:")
    print(f"   总股票: {len(ranking)}")
    print(f"   可交易: {ranking['tradeable'].sum()}")
    print(f"   流动性OK: {ranking['liquidity_ok'].sum()}")
    print(f"   符合入场: {ranking['eligible'].sum()}")
    print(f"   最终选中: {len(invest_more)}")

    if ranking["liquidity_ok"].sum() == 0:
        print(f"\n   ⚠️ 流动性全不通过！")
        for col in ["amount_20d", "amount", "turnover_20d", "turnover", "turn", "volume", "close"]:
            if col in ranking.columns:
                vals = ranking[col].dropna()
                if not vals.empty:
                    print(f"      {col}: {vals.min():.2f} ~ {vals.max():.2f}")
            else:
                print(f"      {col}: ❌ 不存在")

    if ranking["eligible"].sum() == 0:
        print(f"\n   ⚠️ 入场条件全不满足！检查列:")
        for col in ["days_since_limit_up","pullback_pct","volume_breakout","price_above_ma5","tdx_score","high30_breakout","main_force_strong"]:
            exists = col in feats.columns
            print(f"      {col}: {'✅' if exists else '❌ 缺失'}")

    dated_path, latest_path = save_daily_ranking(base_dir, ranking)
    print(f"\n📁 已保存: {latest_path}")

    # ── 加载当前持仓（如果有）──
    holdings = load_portfolio(base_dir)
    held_symbols = set(holdings.keys()) if holdings else set()
    invest_symbols = set(invest_more["symbol"].tolist()) if not invest_more.empty else set()
    withdraw_symbols = set(ranking[ranking["action"]=="WITHDRAW"]["symbol"].tolist())

    new_buys = invest_symbols - held_symbols
    keeps = invest_symbols & held_symbols
    sells = (held_symbols & withdraw_symbols) | (held_symbols - invest_symbols - withdraw_symbols)
    # 持仓中被WITHDRAW的一定要卖，其他持仓中不在推荐里的可以考虑卖
    must_sell = held_symbols & withdraw_symbols
    may_sell = held_symbols - invest_symbols - withdraw_symbols

    print("\n" + "="*70)
    print("📋 今日信号")
    print("="*70)

    if not invest_more.empty:
        print(f"\n🎯 建议买入 ({len(invest_more)} 只):")
        print("-"*70)
        for _, row in invest_more.iterrows():
            sym = row["symbol"]
            ind = row.get("industry", "")
            score = row.get("score", 0)
            weight = row.get("target_weight", 0)
            strength = row.get("_signal_strength", 1)
            tdx = row.get("tdx_score", 0)
            path = row.get("entry_path", "")
            timing = row.get("timing", "")
            vol = row.get("vol_20d", 0)

            # 信号强度标签
            if strength >= 2.0:
                strength_label = "🔴强"
            elif strength >= 1.0:
                strength_label = "🟡普通"
            else:
                strength_label = "🟢弱"

            # 新买 vs 续持
            status = "📌续持" if sym in held_symbols else "🆕新买"

            print(f"  {status} {sym} [{ind}]")
            print(f"     评分: {score:.2f} | 仓位: {weight*100:.1f}% | 信号: {strength_label} | TDX: {tdx:.1f}")
            print(f"     路径: {path} | 波动: {vol*100:.1f}%")
            print(f"     ⏰ 时机: {timing}")
            print()

        total_weight = invest_more['target_weight'].sum()
        print(f"  📊 总仓位: {total_weight*100:.1f}% | 现金: {(1-total_weight)*100:.1f}%")
        print(f"  📊 持股数: {len(invest_more)} 只")
    else:
        print("\n⚠️ 今日无符合条件的股票，建议空仓观望")

    # ── 持仓变动 ──
    if holdings:
        print(f"\n{'='*70}")
        print(f"📦 持仓变动（当前持有 {len(holdings)} 只）")
        print("-"*70)
        if must_sell:
            print(f"  🔴 必须卖出（WITHDRAW）: {', '.join(must_sell)}")
        if may_sell:
            print(f"  🟡 建议卖出（不在推荐）: {', '.join(may_sell)}")
        if keeps:
            print(f"  🟢 继续持有: {', '.join(keeps)}")
        if new_buys:
            print(f"  🆕 新增买入: {', '.join(new_buys)}")
        print(f"\n  操作后持股: {len(invest_symbols)} 只")
        save_portfolio_advice(base_dir, holdings, list(new_buys), list(must_sell | may_sell), list(keeps))
    else:
        print(f"\n💡 提示: 创建 data/portfolio.yaml 可跟踪持仓变动")
        print(f"   格式:")
        print(f"   holdings:")
        print(f"     603968.SH: {{shares: 1000, cost: 50.0}}")
        print(f"     002283.SZ: {{shares: 500, cost: 30.0}}")

    # ── 卖出信号 ──
    withdraws = ranking[ranking["action"]=="WITHDRAW"]
    if not withdraws.empty:
        print(f"\n{'='*70}")
        print(f"⛔ 建议卖出 (WITHDRAW, {len(withdraws)} 只):")
        wcols = [c for c in ["symbol","industry","score","rank","tdx_score"] if c in withdraws.columns]
        top_withdraws = withdraws.head(10)
        print(top_withdraws[wcols].to_string(index=False))
        if len(withdraws) > 10:
            print(f"  ... 共 {len(withdraws)} 只")

    # ── 自动更新持仓（假设完全按建议操作）──
    if not invest_more.empty:
        new_holdings = {}
        for _, row in invest_more.iterrows():
            sym = str(row["symbol"])
            close = float(_safe_float(row.get("close", 0)))
            weight = float(_safe_float(row.get("target_weight", 0)))
            strength = float(_safe_float(row.get("_signal_strength", 1)))
            path = str(row.get("entry_path", ""))
            # 如果是续持，保留原来的成本和股数
            if sym in holdings:
                old = holdings[sym]
                new_holdings[sym] = {
                    "shares": int(old.get("shares", 0)),
                    "cost": float(old.get("cost", close)),
                    "weight": round(weight, 4),
                    "signal_strength": round(strength, 2),
                    "entry_path": path,
                }
            else:
                new_holdings[sym] = {
                    "shares": 0,
                    "cost": round(close, 2),
                    "weight": round(weight, 4),
                    "signal_strength": round(strength, 2),
                    "entry_path": path,
                }

        port_path = base_dir / "data" / "portfolio.yaml"
        port_data = {
            "updated": str(dt.date.today()),
            "total_positions": len(new_holdings),
            "total_weight": round(float(invest_more["target_weight"].sum()), 4),
            "holdings": new_holdings,
        }
        with open(port_path, "w", encoding="utf-8") as f:
            yaml.dump(port_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"\n📝 持仓已自动更新: {port_path} ({len(new_holdings)} 只)")
    else:
        # 空仓：清空持仓
        port_path = base_dir / "data" / "portfolio.yaml"
        port_data = {
            "updated": str(dt.date.today()),
            "total_positions": 0,
            "total_weight": 0,
            "holdings": {},
        }
        with open(port_path, "w", encoding="utf-8") as f:
            yaml.dump(port_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"\n📝 持仓已清空: {port_path}")

    print("="*70 + "\nDone ✅")


if __name__ == "__main__":
    main()