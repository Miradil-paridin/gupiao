from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd
import yaml

from run_backtest_strategy_v3 import (
    _apply_enhancement_filters,
    _build_entry_open_lists,
    _candidate_entry_mask,
    _corr_guard_select,
    _load_cfg_from_yaml,
    compute_index_filter,
    compute_market_regime,
    load_and_prepare_features,
    precompute_daily_universe,
)


@dataclass
class SignalRuntimeContext:
    base_dir: Path
    start_date: str
    watchlist_file: str | None = None
    config_path: str | None = None


def _resolve_config_path(base_dir: Path, config_arg: str = "") -> Path | None:
    explicit = str(config_arg or "").strip()
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = base_dir / p
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        return p

    env_cfg = str(os.getenv("PIPELINE_CONFIG", "")).strip()
    if env_cfg:
        p = Path(env_cfg)
        if not p.is_absolute():
            p = base_dir / p
        if p.exists():
            return p

    for name in ["config.yaml", "config_v31.yaml"]:
        p = base_dir / name
        if p.exists():
            return p
    return None


def _load_start_date_from_config(base_dir: Path, config_arg: str = "") -> str:
    p = _resolve_config_path(base_dir, config_arg=config_arg)
    if p and p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        sd = raw.get("backtest", {}).get("start_date") or raw.get("market_data", {}).get("start_date")
        if sd:
            return str(sd)
    return "2020-01-01"


def _entry_path_from_strength(strength: float) -> str:
    if strength >= 2.0:
        return "strong_signal"
    if strength >= 1.0:
        return "normal_signal"
    return "weak_signal"


def _timing_advice_from_strength(strength: float) -> str:
    if strength >= 2.0:
        return "open_confirm_then_buy"
    if strength >= 1.0:
        return "intraday_pullback_buy"
    return "watch_or_micro_position"


def _load_watchlist_symbols(base_dir: Path, watchlist_file: str) -> set[str]:
    wl_path = base_dir / watchlist_file
    if not wl_path.exists():
        raise FileNotFoundError(f"Watchlist file not found: {wl_path}")

    wl_text = wl_path.read_text(encoding="utf-8")
    raw_codes = re.findall(r"(\d{6})", wl_text)
    if not raw_codes:
        try:
            wl_data = yaml.safe_load(wl_text) or {}
            if isinstance(wl_data, dict):
                wl_list = wl_data.get("watchlist", []) or []
            elif isinstance(wl_data, list):
                wl_list = wl_data
            else:
                wl_list = []
            raw_codes = [str(c).split(".")[0] for c in wl_list if str(c).strip()]
        except Exception:
            raw_codes = []

    symbols: set[str] = set()
    for code in set(raw_codes):
        c = str(code).strip().zfill(6)
        if c.startswith(("6", "5")):
            symbols.add(f"{c}.SH")
        else:
            symbols.add(f"{c}.SZ")
    return symbols


def _compute_target_weights(
    open_symbols: list[str],
    signal_strengths: dict[str, float],
    cfg: Any,
    regime_pos_cap: float,
) -> dict[str, float]:
    if not open_symbols:
        return {}

    max_pos_today = float(cfg.max_total_position)
    if bool(cfg.use_market_regime) and bool(cfg.use_dynamic_regime_position):
        max_pos_today = min(max_pos_today, float(regime_pos_cap))

    signal_vals = [float(signal_strengths.get(s, 1.0)) for s in open_symbols]
    avg_signal = float(np.mean(signal_vals)) if signal_vals else 0.0
    if bool(cfg.use_weak_signal_de_risk) and signal_vals and avg_signal < float(cfg.weak_signal_threshold):
        max_pos_today *= float(cfg.weak_signal_cap_multiplier)

    base_w = min(max_pos_today / max(1, len(open_symbols)), float(cfg.max_single_weight))

    out: dict[str, float] = {}
    for sym in open_symbols:
        s = float(signal_strengths.get(sym, 1.0))
        w = base_w
        if bool(cfg.use_signal_tiered_sizing):
            if s >= 2.0:
                w *= float(cfg.tier_strong_multiplier)
            elif s >= 1.0:
                w *= float(cfg.tier_normal_multiplier)
            else:
                w *= float(cfg.tier_weak_multiplier)
            w = min(w, float(cfg.max_single_weight))
        if str(cfg.weak_entry_mode).strip().lower() == "micro" and s < float(cfg.dual_entry_normal_threshold):
            w *= float(cfg.weak_micro_weight_multiplier)
        out[sym] = float(w)

    total_w = float(sum(out.values()))
    if total_w > max_pos_today > 0:
        scale = max_pos_today / total_w
        out = {k: float(v * scale) for k, v in out.items()}
    return out


def build_daily_ranking(ctx: SignalRuntimeContext) -> pd.DataFrame:
    cfg = _load_cfg_from_yaml(ctx.base_dir, config_path=ctx.config_path)

    feats = load_and_prepare_features(
        ctx.base_dir,
        start_date=ctx.start_date,
        news_sentiment_lag_days=int(getattr(cfg, "news_sentiment_lag_days", 1)),
    )
    if feats.empty:
        raise ValueError("No feature data available.")
    if ctx.watchlist_file:
        symbols = _load_watchlist_symbols(ctx.base_dir, ctx.watchlist_file)
        if symbols:
            before = feats["symbol"].nunique()
            feats = feats[feats["symbol"].isin(symbols)].copy()
            after = feats["symbol"].nunique()
            print(f"[Watchlist] using {ctx.watchlist_file} ({after}/{before} symbols kept, pool size {len(symbols)})")

    daily = precompute_daily_universe(feats, cfg=cfg)
    if daily.empty:
        raise ValueError("Daily universe is empty.")

    price_df = feats.pivot_table(index="date", columns="symbol", values="close").sort_index()
    returns_df = price_df.pct_change(fill_method=None).fillna(0.0)

    latest_date = pd.to_datetime(daily["date"].max())
    day = daily[daily["date"] == latest_date].copy()
    if day.empty:
        raise ValueError(f"No universe rows for date {latest_date.date()}")

    regime_df = compute_market_regime(daily, cfg=cfg)
    regime_row = regime_df[regime_df["date"] == latest_date]
    regime_tag = str(regime_row["regime"].iloc[0]) if not regime_row.empty else "UNKNOWN"
    regime_pos_cap = float(regime_row["regime_pos_cap"].iloc[0]) if not regime_row.empty else float(cfg.max_total_position)

    index_filter_map = compute_index_filter(ctx.base_dir, ctx.start_date, cfg)
    market_can_open = bool(index_filter_map.get(latest_date, True)) if index_filter_map else True

    mask, signal_strength_series = _candidate_entry_mask(day, cfg)
    day.loc[:, "_signal_strength"] = signal_strength_series

    candidates = _apply_enhancement_filters(day[mask].copy(), cfg).sort_values("score", ascending=False).head(int(cfg.invest_more_n))

    if bool(cfg.use_correlation_control):
        ret_window = returns_df.loc[
            (returns_df.index < latest_date)
            & (returns_df.index >= latest_date - pd.Timedelta(days=int(cfg.corr_lookback_days) * 2))
        ]
        if len(ret_window) > int(cfg.corr_lookback_days):
            ret_window = ret_window.tail(int(cfg.corr_lookback_days))
    else:
        ret_window = returns_df.iloc[:0]

    target_symbols = _corr_guard_select(candidates, int(cfg.top_k), cfg, ret_window)
    signal_strengths = {str(r["symbol"]): float(r.get("_signal_strength", 1.0)) for _, r in candidates.iterrows()}

    open_symbols, observe_symbols, gate_floor = _build_entry_open_lists(
        target_symbols=target_symbols,
        signal_strengths=signal_strengths,
        regime_tag=regime_tag,
        cfg=cfg,
    )

    if (not market_can_open) and bool(getattr(cfg, "index_filter_hard_gate", True)):
        observe_symbols = list(target_symbols)
        open_symbols = []
    if (not market_can_open) and (not bool(getattr(cfg, "index_filter_hard_gate", True))):
        regime_pos_cap = min(regime_pos_cap, float(getattr(cfg, "index_filter_block_position_cap", regime_pos_cap)))

    out = day.sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    out["trend_up"] = (pd.to_numeric(out.get("ma_dist_20", 0), errors="coerce").fillna(0) > 0).astype(int)
    out["mom_bad"] = (
        (pd.to_numeric(out.get("ret_20d", 0), errors="coerce").fillna(0) < 0)
        & (pd.to_numeric(out.get("ret_60d", 0), errors="coerce").fillna(0) < 0)
    ).astype(int)
    out["risk_high"] = (pd.to_numeric(out.get("vol_20d", 0), errors="coerce").fillna(0) >= 0.55).astype(int)

    out["action"] = "HOLD"
    withdraw_mask = ((out["trend_up"] == 0) & (out["mom_bad"] == 1)) | (pd.to_numeric(out["score"], errors="coerce").fillna(0) <= -0.5)
    out.loc[withdraw_mask, "action"] = "WITHDRAW"

    reduce_mask = (out["action"] != "WITHDRAW") & (out["risk_high"] == 1) & (out["trend_up"] == 1) & (out["mom_bad"] == 0)
    out.loc[reduce_mask, "action"] = "REDUCE"

    out["entry_gate"] = "none"
    out.loc[out["symbol"].isin(target_symbols), "entry_gate"] = "candidate"
    out.loc[out["symbol"].isin(observe_symbols), "entry_gate"] = "observe"
    out.loc[out["symbol"].isin(open_symbols), "entry_gate"] = "open"
    out.loc[out["symbol"].isin(open_symbols), "action"] = "INVEST_MORE"

    least_pool = out[(out["action"] == "HOLD") & (~out["symbol"].isin(target_symbols))]
    least_symbols = least_pool.sort_values("score", ascending=True).head(int(getattr(cfg, "least_n", 3)))["symbol"].tolist()
    out.loc[out["symbol"].isin(least_symbols), "action"] = "LEAST"

    weights = _compute_target_weights(open_symbols, signal_strengths, cfg, regime_pos_cap)
    out["target_weight"] = out["symbol"].map(weights).fillna(0.0)

    out["_signal_strength"] = out["symbol"].map(signal_strengths).fillna(0.0)
    out["entry_path"] = out["_signal_strength"].map(_entry_path_from_strength)
    out["timing"] = out["_signal_strength"].map(_timing_advice_from_strength)
    out["market_regime"] = regime_tag
    out["regime_gate_floor"] = float(gate_floor)
    out["index_open"] = int(bool(market_can_open))
    if "vol_ratio_20" not in out.columns:
        out["vol_ratio_20"] = 0.0

    keep_cols = [
        "date", "symbol", "industry", "close", "score", "rank", "action", "entry_gate", "market_regime", "regime_gate_floor",
        "index_open", "target_weight", "_signal_strength", "entry_path", "timing", "tradeable", "eligible"
    ]
    for extra in [
        "liquidity_ok",
        "trend_up",
        "mom_bad",
        "risk_high",
        "tdx_score",
        "ma_dist_20",
        "ret_20d",
        "ret_60d",
        "vol_20d",
        "atr_pct",
        "vol_ratio_20",
        "news_sentiment_score",
        "news_item_count",
        "alpha_news_sentiment",
    ]:
        if extra in out.columns:
            keep_cols.append(extra)

    return out[[c for c in keep_cols if c in out.columns]].sort_values("rank").reset_index(drop=True)


def save_daily_ranking(base_dir: Path, ranking: pd.DataFrame) -> tuple[Path, Path]:
    out_dir = base_dir / "data" / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = pd.to_datetime(ranking["date"].iloc[0]).date().isoformat()
    dated_path = out_dir / f"daily_rank_{d}.csv"
    latest_path = out_dir / "latest_daily_rank.csv"
    ranking.to_csv(dated_path, index=False, encoding="utf-8-sig")
    ranking.to_csv(latest_path, index=False, encoding="utf-8-sig")
    return dated_path, latest_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily signal ranking aligned with Strategy V3")
    ap.add_argument("--config", default="", help="config path (default: config.yaml, fallback: config_v31.yaml)")
    ap.add_argument("--watchlist", default=None, help="watchlist file (yaml/txt) to filter signal universe")
    ap.add_argument(
        "--no-default-watchlist",
        action="store_true",
        help="do not auto-use backtest_watchlist.yaml/watchlist_cache.yaml when --watchlist is omitted",
    )
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    start_date = _load_start_date_from_config(base_dir, config_arg=args.config)
    watchlist_file = args.watchlist
    if watchlist_file is None and not args.no_default_watchlist:
        for cand in ["backtest_watchlist.yaml", "watchlist_cache.yaml"]:
            if (base_dir / cand).exists():
                watchlist_file = cand
                print(f"auto watchlist: {cand} (use --no-default-watchlist to disable)")
                break

    print("=" * 60)
    print("Daily Signal Ranking (Aligned with Backtest V3)")
    print("=" * 60)

    cfg_arg = str(args.config).strip() or None
    ctx = SignalRuntimeContext(
        base_dir=base_dir,
        start_date=start_date,
        watchlist_file=watchlist_file,
        config_path=cfg_arg,
    )
    ranking = build_daily_ranking(ctx)
    dated_path, latest_path = save_daily_ranking(base_dir, ranking)

    invest_more = ranking[ranking["action"] == "INVEST_MORE"]
    print(f"date: {ranking['date'].iloc[0]}")
    print(f"total: {len(ranking)} | invest_more: {len(invest_more)} | withdraw: {(ranking['action'] == 'WITHDRAW').sum()}")
    if "index_open" in ranking.columns:
        print(f"index_open: {int(ranking['index_open'].iloc[0])}")
    print(f"saved: {latest_path}")
    print(f"dated: {dated_path}")


if __name__ == "__main__":
    main()
