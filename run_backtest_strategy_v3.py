
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

    return merged


def build_industry_map_from_config(base_dir: Path, symbols: pd.Series) -> dict[str, str]:
    cfg_path = base_dir / "config.yaml"
    mapping: dict[str, str] = {}

    if cfg_path.exists():
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


def _candidate_entry_mask(day: pd.DataFrame, cfg: BacktestConfigV3) -> pd.Series:
    in_window = (day["days_since_limit_up"] >= cfg.pullback_window_start) & (
        day["days_since_limit_up"] <= cfg.pullback_window_end
    )
    pullback_ok = (day["pullback_pct"] >= cfg.pullback_min_pct) & (day["pullback_pct"] <= cfg.pullback_max_pct)
    breakout_ok = (day["volume_breakout"].fillna(0) == 1) & (day["price_above_ma5"].fillna(0) == 1)
    tdx_ok = (day["tdx_score"].fillna(0) >= cfg.tdx_min_score) | (
        (day["high30_breakout"].fillna(0) == 1) & (day["main_force_strong"].fillna(0) == 1)
    )
    return in_window & pullback_ok & breakout_ok & tdx_ok & (day["trend_up"] == 1)


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

    equity = cfg.initial_capital
    positions: dict[str, dict[str, float]] = {}
    rec: list[dict[str, Any]] = []
    daily_targets: dict[pd.Timestamp, list[str]] = {}

    for i in range(1, len(trading_days)):
        prev_date = trading_days[i - 1]
        date = trading_days[i]
        prev_day = by_date.get(prev_date)
        if prev_day is None or prev_day.empty:
            continue

        mask = _candidate_entry_mask(prev_day, cfg)
        cands = _apply_enhancement_filters(prev_day[mask].copy(), cfg).sort_values("score", ascending=False).head(
            cfg.invest_more_n
        )

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
        daily_targets[prev_date] = target_symbols

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
            if pnl_pct <= -cfg.stop_loss_pct:
                to_sell.append(sym)
                continue

            drawdown = (pos["peak_price"] - float(px)) / max(pos["peak_price"], 1e-12)
            if drawdown >= cfg.trailing_stop_pct:
                to_sell.append(sym)
                continue

            if pos["hold_days"] >= cfg.max_hold_days:
                to_sell.append(sym)
                continue

        for sym in to_sell:
            positions.pop(sym, None)

        for sym in list(positions.keys()):
            if sym not in target_symbols:
                positions.pop(sym, None)

        max_pos_today = cfg.max_total_position
        if cfg.use_market_regime:
            max_pos_today = min(max_pos_today, float(regime_pos.get(prev_date, cfg.max_total_position)))

        n_targets = max(1, len(target_symbols))
        equal_w = min(max_pos_today / n_targets, cfg.max_single_weight)

        for sym in target_symbols:
            if sym in positions:
                continue
            if sym not in price_df.columns or date not in price_df.index:
                continue
            px = price_df.at[date, sym]
            if not np.isfinite(px) or px <= 0:
                continue
            positions[sym] = {"entry_price": float(px), "peak_price": float(px), "hold_days": 0.0, "weight": float(equal_w)}

        active_targets = [s for s in target_symbols if s in positions]
        if len(active_targets) > 0:
            new_w = min(max_pos_today / len(active_targets), cfg.max_single_weight)
            for s in active_targets:
                positions[s]["weight"] = float(new_w)

        daily_ret_row = returns_df.loc[date] if date in returns_df.index else pd.Series(dtype=float)
        gross_ret = 0.0
        total_weight = 0.0
        for sym, pos in positions.items():
            gross_ret += pos["weight"] * float(daily_ret_row.get(sym, 0.0))
            total_weight += pos["weight"]

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
        return {"metrics": metrics, "equity_curve": eq, "daily_targets": daily_targets}

    nav = eq["equity"] / cfg.initial_capital
    r = eq["daily_return"]
    metrics = {
        "annual_return_pct": _annual_return(nav) * 100.0,
        "max_drawdown_pct": _max_drawdown(nav) * 100.0,
        "sharpe": _sharpe(r),
        "win_rate_pct": _win_rate(r) * 100.0,
    }
    return {"metrics": metrics, "equity_curve": eq, "daily_targets": daily_targets}


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep optimization for strategy v3")
    ap.add_argument("--base-dir", default=".", help="project root")
    ap.add_argument("--start-date", default="2023-01-01", help="backtest start date")
    ap.add_argument("--optimize", action="store_true", help="run full optimization pipeline")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()

    if not args.optimize:
        print("Use --optimize to run deep optimization pipeline.")
        return

    result = run_optimization_pipeline(base_dir, start_date=args.start_date)

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


if __name__ == "__main__":
    main()
