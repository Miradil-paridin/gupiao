from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import run_backtest_strategy_v3 as bt


def _resolve_config_path(base_dir: Path, config_path: str | None = None) -> Path | None:
    if config_path:
        p = Path(config_path)
        if not p.is_absolute():
            p = base_dir / p
        if not p.exists():
            raise FileNotFoundError(f"指定的配置文件不存在: {p}")
        return p

    env_cfg = str(os.getenv("PIPELINE_CONFIG", "")).strip()
    candidates: list[Path] = []
    if env_cfg:
        p = Path(env_cfg)
        if not p.is_absolute():
            p = base_dir / p
        candidates.append(p)
    candidates.extend([base_dir / "config.yaml", base_dir / "config_v31.yaml"])

    for p in candidates:
        if p.exists():
            return p
    return None


def _load_start_date(base_dir: Path, cli_start: str | None, config_path: str | None = None) -> str:
    if cli_start:
        return str(cli_start)
    p = _resolve_config_path(base_dir, config_path=config_path)
    if p and p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        sd = raw.get("backtest", {}).get("start_date") or raw.get("market_data", {}).get("start_date")
        if sd:
            return str(sd)
    return "2020-01-01"


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _clamp01(x: Any, default: float | None = None) -> float | None:
    try:
        v = float(x)
    except Exception:
        return default
    if not np.isfinite(v):
        return default
    return float(min(1.0, max(0.0, v)))


def _safe_annual_from_returns(r: pd.Series) -> float:
    rr = pd.to_numeric(r, errors="coerce").fillna(0.0)
    if len(rr) < 2:
        return 0.0
    nav = (1.0 + rr).cumprod()
    years = len(rr) / 252.0
    if years <= 0 or nav.iloc[-1] <= 0:
        return 0.0
    return float((nav.iloc[-1] ** (1.0 / years)) - 1.0)


def _max_drawdown_from_returns(r: pd.Series) -> float:
    rr = pd.to_numeric(r, errors="coerce").fillna(0.0)
    if rr.empty:
        return 0.0
    nav = (1.0 + rr).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def _sharpe_from_returns(r: pd.Series) -> float:
    rr = pd.to_numeric(r, errors="coerce").fillna(0.0)
    sd = float(rr.std(ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(rr.mean() / sd * np.sqrt(252.0))


def _compute_trade_stats(trades: list[dict[str, Any]], net_daily_returns: pd.Series) -> dict[str, float]:
    if trades:
        t = pd.DataFrame(trades).copy()
        pnl = pd.to_numeric(t.get("pnl_pct", 0.0), errors="coerce").fillna(0.0) / 100.0
        wt = pd.to_numeric(t.get("weight", 100.0), errors="coerce").fillna(100.0) / 100.0
        pnl_w = pnl * wt
    else:
        pnl_w = pd.to_numeric(net_daily_returns, errors="coerce").fillna(0.0)

    if pnl_w.empty:
        return {
            "trade_expectancy": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "trade_win_rate_pct": 0.0,
            "trade_count": 0.0,
        }

    wins = pnl_w[pnl_w > 0]
    losses = pnl_w[pnl_w < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-12 else 0.0

    return {
        "trade_expectancy": float(pnl_w.mean()),
        "profit_factor": profit_factor,
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "trade_win_rate_pct": float((pnl_w > 0).mean() * 100.0),
        "trade_count": float(len(pnl_w)),
    }


def _extended_metrics(result: dict[str, Any], initial_capital: float) -> dict[str, float]:
    eq = result.get("equity_curve", pd.DataFrame()).copy()
    trades = result.get("trade_log", []) or []
    if eq is None or eq.empty:
        return {
            "annual_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "win_rate_pct": 0.0,
            "daily_expectancy": 0.0,
            "trade_expectancy": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "avg_turnover": 0.0,
            "avg_extra_exec_cost_bps": 0.0,
            "gross_annual_return_pct": 0.0,
            "net_annual_return_pct": 0.0,
            "slippage_drag_pct": 0.0,
            "trade_count": 0.0,
            "trade_win_rate_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    net_ret = pd.to_numeric(eq.get("daily_return", 0.0), errors="coerce").fillna(0.0)
    gross_ret = pd.to_numeric(eq.get("gross_return", net_ret), errors="coerce").fillna(0.0)
    turnover = pd.to_numeric(eq.get("turnover", 0.0), errors="coerce").fillna(0.0)
    extra_cost = pd.to_numeric(eq.get("extra_exec_cost", 0.0), errors="coerce").fillna(0.0)

    nav = pd.to_numeric(eq.get("equity", initial_capital), errors="coerce").fillna(initial_capital)
    nav = nav / float(initial_capital)
    if len(nav) > 1:
        annual_return_pct = float((nav.iloc[-1] ** (252.0 / len(nav)) - 1.0) * 100.0)
    else:
        annual_return_pct = 0.0
    mdd_pct = float((nav / nav.cummax() - 1.0).min() * 100.0) if len(nav) > 0 else 0.0
    sharpe = _sharpe_from_returns(net_ret)
    calmar = float(annual_return_pct / abs(mdd_pct)) if mdd_pct < 0 else 0.0
    win_rate_pct = float((net_ret > 0).mean() * 100.0) if len(net_ret) else 0.0

    gross_ann = _safe_annual_from_returns(gross_ret) * 100.0
    net_ann = _safe_annual_from_returns(net_ret) * 100.0
    slip_drag = float(gross_ann - net_ann)

    trade_stats = _compute_trade_stats(trades, net_ret)
    daily_expectancy = float(net_ret.mean()) if len(net_ret) else 0.0
    return {
        "annual_return_pct": annual_return_pct,
        "max_drawdown_pct": mdd_pct,
        "sharpe": sharpe,
        "calmar": calmar,
        "win_rate_pct": win_rate_pct,
        "daily_expectancy": daily_expectancy,
        "trade_expectancy": float(trade_stats["trade_expectancy"]),
        "expectancy": daily_expectancy,
        "profit_factor": float(trade_stats["profit_factor"]),
        "avg_turnover": float(turnover.mean()),
        "avg_extra_exec_cost_bps": float(extra_cost.mean() * 10000.0),
        "gross_annual_return_pct": float(gross_ann),
        "net_annual_return_pct": float(net_ann),
        "slippage_drag_pct": slip_drag,
        "trade_count": float(trade_stats["trade_count"]),
        "trade_win_rate_pct": float(trade_stats["trade_win_rate_pct"]),
        "avg_win": float(trade_stats["avg_win"]),
        "avg_loss": float(trade_stats["avg_loss"]),
    }


def _run_once(
    daily_universe: pd.DataFrame,
    price_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    cfg: bt.BacktestConfigV3,
    regime_df: pd.DataFrame,
    index_filter: dict,
    start_date: str,
) -> dict[str, Any]:
    res = bt.run_backtest_v3(
        daily_universe=daily_universe,
        price_df=price_df,
        returns_df=returns_df,
        cfg=cfg,
        regime_df=regime_df,
        date_start=pd.to_datetime(start_date),
        date_end=daily_universe["date"].max(),
        index_filter=index_filter,
        allowed_symbols_by_date=None,
    )
    return res


def _load_watchlist_symbols(base_dir: Path, watchlist_file: str) -> set[str]:
    wl_path = base_dir / watchlist_file
    if not wl_path.exists():
        raise FileNotFoundError(f"watchlist file not found: {wl_path}")

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


def _rolling_window_metrics(eq: pd.DataFrame, window_days: int = 252, step_days: int = 21) -> pd.DataFrame:
    if eq is None or eq.empty:
        return pd.DataFrame()
    r = pd.to_numeric(eq.get("daily_return", 0.0), errors="coerce").fillna(0.0).reset_index(drop=True)
    d = pd.to_datetime(eq["date"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    if len(r) < max(window_days, 30):
        return pd.DataFrame()

    for i in range(0, len(r) - window_days + 1, max(1, step_days)):
        seg = r.iloc[i : i + window_days]
        rows.append(
            {
                "start_date": str(d.iloc[i])[:10],
                "end_date": str(d.iloc[i + window_days - 1])[:10],
                "annual_return_pct": _safe_annual_from_returns(seg) * 100.0,
                "max_drawdown_pct": _max_drawdown_from_returns(seg) * 100.0,
                "sharpe": _sharpe_from_returns(seg),
                "win_rate_pct": float((seg > 0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def _wf_summary(ctx: dict[str, Any], cfg: bt.BacktestConfigV3) -> tuple[pd.DataFrame, dict[str, Any]]:
    wf_df = bt.run_walk_forward_oos(
        daily_universe=ctx["daily"],
        price_df=ctx["price_df"],
        returns_df=ctx["returns_df"],
        cfg=cfg,
        regime_df=ctx["regime_df"],
        index_filter=ctx["index_filter"],
        allowed_symbols_by_date=None,
        train_days=504,
        test_days=126,
        step_days=126,
    )
    wf_summary = bt.summarize_walk_forward_stability(wf_df, cfg=cfg, sharpe_floor=1.0) if not wf_df.empty else {}
    return wf_df, wf_summary


def _make_cfg(base_cfg: bt.BacktestConfigV3, **kwargs: Any) -> bt.BacktestConfigV3:
    d = asdict(base_cfg)
    d.update(kwargs)
    return bt.BacktestConfigV3(**d)


def _grid_combo_count(param_grid: dict[str, list[Any]]) -> int:
    keys = [
        "top_k",
        "invest_more_n",
        "pullback_min_pct",
        "pullback_max_pct",
        "stop_loss_pct",
        "trailing_stop_pct",
    ]
    total = 1
    for k in keys:
        total *= max(1, len(param_grid.get(k, [])))
    return int(total)


def _drop_farthest(values: list[Any], center: float) -> list[Any]:
    vals = list(values)
    if len(vals) <= 1:
        return vals
    i = max(range(len(vals)), key=lambda j: (abs(float(vals[j]) - center), float(vals[j])))
    vals.pop(i)
    return sorted(vals)


def _build_overfit_param_grid(base_cfg: bt.BacktestConfigV3, max_combos: int) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {
        "top_k": sorted({max(3, int(base_cfg.top_k) - 2), int(base_cfg.top_k), int(base_cfg.top_k) + 2}),
        "invest_more_n": sorted(
            {
                max(int(base_cfg.top_k), int(base_cfg.invest_more_n) - 2),
                int(base_cfg.invest_more_n),
                int(base_cfg.invest_more_n) + 2,
            }
        ),
        "pullback_min_pct": [float(base_cfg.pullback_min_pct)],
        "pullback_max_pct": [float(base_cfg.pullback_max_pct)],
        "stop_loss_pct": sorted(
            {max(0.03, float(base_cfg.stop_loss_pct) - 0.01), float(base_cfg.stop_loss_pct), float(base_cfg.stop_loss_pct) + 0.01}
        ),
        "trailing_stop_pct": sorted(
            {
                max(0.04, float(base_cfg.trailing_stop_pct) - 0.01),
                float(base_cfg.trailing_stop_pct),
                float(base_cfg.trailing_stop_pct) + 0.01,
            }
        ),
    }

    cap = max(4, int(max_combos))
    squeeze_order = [
        ("top_k", float(base_cfg.top_k)),
        ("invest_more_n", float(base_cfg.invest_more_n)),
        ("stop_loss_pct", float(base_cfg.stop_loss_pct)),
        ("trailing_stop_pct", float(base_cfg.trailing_stop_pct)),
    ]
    for _ in range(64):
        if _grid_combo_count(grid) <= cap:
            break
        changed = False
        for key, center in squeeze_order:
            vals = list(grid.get(key, []))
            if len(vals) <= 1:
                continue
            grid[key] = _drop_farthest(vals, center)
            changed = True
            if _grid_combo_count(grid) <= cap:
                break
        if not changed:
            break
    return grid


def _estimate_pbo_from_folds(folds_df: pd.DataFrame) -> dict[str, Any]:
    if folds_df is None or folds_df.empty:
        return {"status": "no_data", "pbo": float("nan")}

    combo_cols = [
        "top_k",
        "invest_more_n",
        "pullback_min_pct",
        "pullback_max_pct",
        "stop_loss_pct",
        "trailing_stop_pct",
    ]
    required_cols = set(combo_cols + ["fold_id", "sharpe"])
    if not required_cols.issubset(folds_df.columns):
        return {"status": "missing_columns", "pbo": float("nan")}

    frame = folds_df.copy()
    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="coerce").astype("Int64")
    frame["sharpe"] = pd.to_numeric(frame["sharpe"], errors="coerce")
    frame = frame.dropna(subset=["fold_id", "sharpe"])
    if frame.empty:
        return {"status": "no_valid_data", "pbo": float("nan")}

    fold_ids = sorted({int(x) for x in frame["fold_id"].tolist()})
    if len(fold_ids) < 3:
        return {"status": "insufficient_folds", "pbo": float("nan"), "fold_count": int(len(fold_ids))}

    n_models = int(frame[combo_cols].drop_duplicates().shape[0])
    if n_models < 2:
        return {"status": "insufficient_models", "pbo": float("nan"), "model_count": n_models}

    fold_rows: list[dict[str, Any]] = []
    for fid in fold_ids:
        train_df = frame[frame["fold_id"] != fid]
        test_df = frame[frame["fold_id"] == fid]
        if train_df.empty or test_df.empty:
            continue

        train_rank = (
            train_df.groupby(combo_cols, as_index=False)
            .agg(train_sharpe=("sharpe", "mean"))
            .sort_values("train_sharpe", ascending=False)
            .reset_index(drop=True)
        )
        if train_rank.empty:
            continue
        best = train_rank.iloc[0]

        best_mask = pd.Series(True, index=test_df.index)
        for c in combo_cols:
            best_mask &= test_df[c] == best[c]
        test_row = test_df[best_mask]
        if test_row.empty:
            continue

        test_rank = test_df.sort_values("sharpe", ascending=False).reset_index(drop=True)
        test_mask = pd.Series(True, index=test_rank.index)
        for c in combo_cols:
            test_mask &= test_rank[c] == best[c]
        pos = np.flatnonzero(test_mask.values)
        if len(pos) == 0:
            continue

        rank = int(pos[0] + 1)
        total = int(len(test_rank))
        if total <= 1:
            continue

        lam = 1.0 - float(rank - 1) / float(total - 1)
        lam = float(min(1.0 - 1e-6, max(1e-6, lam)))
        logit = float(np.log(lam / (1.0 - lam)))

        fold_rows.append(
            {
                "fold_id": int(fid),
                "train_best_sharpe": float(best["train_sharpe"]),
                "oos_sharpe": float(test_row.iloc[0]["sharpe"]),
                "oos_rank": int(rank),
                "oos_total": int(total),
                "lambda": float(lam),
                "logit_lambda": float(logit),
                "is_overfit": bool(logit < 0.0),
            }
        )

    if not fold_rows:
        return {"status": "no_valid_fold_rows", "pbo": float("nan"), "model_count": n_models}

    fold_table = pd.DataFrame(fold_rows)
    pbo = float(fold_table["is_overfit"].mean())
    return {
        "status": "ok",
        "pbo": float(pbo),
        "fold_count": int(len(fold_table)),
        "model_count": int(n_models),
        "median_lambda": float(fold_table["lambda"].median()),
        "mean_selected_oos_sharpe": float(fold_table["oos_sharpe"].mean()),
        "mean_train_best_sharpe": float(fold_table["train_best_sharpe"].mean()),
        "rows": fold_rows,
    }


def _deflated_sharpe_ratio(daily_returns: pd.Series, observed_sharpe: float, n_trials: int) -> dict[str, Any]:
    rr = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if len(rr) < 30:
        return {"status": "insufficient_returns", "dsr": float("nan"), "n_obs": int(len(rr))}

    sr = float(observed_sharpe)
    m = max(2, int(n_trials))
    n = max(2, int(len(rr)))
    skew = float(rr.skew())
    kurt = float(rr.kurt()) + 3.0

    # PSR/DSR denominator from Bailey & Lopez de Prado.
    denom_term = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr**2)
    denom_term = float(max(1e-12, denom_term))
    sr_std = float(np.sqrt(denom_term / max(1, n - 1)))

    norm = NormalDist()
    euler_gamma = 0.5772156649
    z1 = float(norm.inv_cdf(1.0 - 1.0 / m))
    z2 = float(norm.inv_cdf(1.0 - 1.0 / (m * np.e)))
    sr_star = float(((1.0 - euler_gamma) * z1 + euler_gamma * z2) / np.sqrt(max(1, n - 1)))

    psr = float(norm.cdf(sr / max(sr_std, 1e-12)))
    dsr = float(norm.cdf((sr - sr_star) / max(sr_std, 1e-12)))
    return {
        "status": "ok",
        "dsr": float(dsr),
        "psr": float(psr),
        "expected_max_sr_null": float(sr_star),
        "observed_sharpe": float(sr),
        "n_obs": int(n),
        "n_trials": int(m),
        "skew": float(skew),
        "kurtosis": float(kurt),
    }


def _compute_overfit_diagnostics(
    ctx: dict[str, Any],
    base_cfg: bt.BacktestConfigV3,
    baseline_eq: pd.DataFrame,
    overfit_max_combos: int = 8,
    overfit_cv_splits: int = 3,
) -> dict[str, Any]:
    grid = _build_overfit_param_grid(base_cfg, max_combos=int(overfit_max_combos))
    combo_count = _grid_combo_count(grid)
    splits = max(3, int(overfit_cv_splits))

    if combo_count < 4:
        return {
            "status": "insufficient_grid",
            "combo_count": int(combo_count),
            "cv_splits": int(splits),
            "pbo": float("nan"),
            "dsr": float("nan"),
        }

    try:
        folds_df, _ = bt.evaluate_params_cv(
            daily_universe=ctx["daily"],
            price_df=ctx["price_df"],
            returns_df=ctx["returns_df"],
            regime_df=ctx["regime_df"],
            param_grid=grid,
            n_splits=int(splits),
        )
        pbo_info = _estimate_pbo_from_folds(folds_df)
    except Exception as e:
        return {
            "status": "cv_error",
            "combo_count": int(combo_count),
            "cv_splits": int(splits),
            "pbo": float("nan"),
            "dsr": float("nan"),
            "error": str(e),
        }

    eq_frame = baseline_eq if isinstance(baseline_eq, pd.DataFrame) else pd.DataFrame()
    rr = pd.to_numeric(eq_frame.get("daily_return", 0.0), errors="coerce").fillna(0.0)
    obs_sharpe = _sharpe_from_returns(rr)
    dsr_info = _deflated_sharpe_ratio(
        daily_returns=rr,
        observed_sharpe=float(obs_sharpe),
        n_trials=max(int(combo_count), int(pbo_info.get("model_count", 0) or 0)),
    )

    status = "ok" if (str(pbo_info.get("status", "")).lower() == "ok" and str(dsr_info.get("status", "")).lower() == "ok") else "partial"
    return {
        "status": status,
        "cv_splits": int(splits),
        "combo_count": int(combo_count),
        "param_grid": grid,
        "pbo": float(_to_float(pbo_info.get("pbo", float("nan")), float("nan"))),
        "pbo_details": pbo_info,
        "dsr": float(_to_float(dsr_info.get("dsr", float("nan")), float("nan"))),
        "dsr_details": dsr_info,
    }


def _build_stability_zone(
    ctx: dict[str, Any],
    base_cfg: bt.BacktestConfigV3,
    base_metrics: dict[str, float],
    max_combos: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    topk_vals = sorted({max(3, base_cfg.top_k - 2), base_cfg.top_k, base_cfg.top_k + 2})
    invest_vals = sorted(
        {
            max(base_cfg.top_k, base_cfg.invest_more_n - 2),
            base_cfg.invest_more_n,
            base_cfg.invest_more_n + 2,
        }
    )
    stop_vals = sorted({max(0.03, base_cfg.stop_loss_pct - 0.01), base_cfg.stop_loss_pct, base_cfg.stop_loss_pct + 0.01})
    trail_vals = sorted(
        {
            max(0.04, base_cfg.trailing_stop_pct - 0.01),
            base_cfg.trailing_stop_pct,
            base_cfg.trailing_stop_pct + 0.01,
        }
    )
    combos = list(itertools.product(topk_vals, invest_vals, stop_vals, trail_vals))
    combos = [c for c in combos if c[1] >= c[0]]
    combos.sort(key=lambda c: abs(c[0] - base_cfg.top_k) + abs(c[1] - base_cfg.invest_more_n))
    combos = combos[: max(1, int(max_combos))]

    rows: list[dict[str, Any]] = []
    for top_k, invest_n, stop_loss, trailing in combos:
        cfg = _make_cfg(
            base_cfg,
            top_k=int(top_k),
            invest_more_n=int(invest_n),
            stop_loss_pct=float(stop_loss),
            trailing_stop_pct=float(trailing),
        )
        res = _run_once(
            daily_universe=ctx["daily"],
            price_df=ctx["price_df"],
            returns_df=ctx["returns_df"],
            cfg=cfg,
            regime_df=ctx["regime_df"],
            index_filter=ctx["index_filter"],
            start_date=ctx["start_date"],
        )
        m = _extended_metrics(res, initial_capital=cfg.initial_capital)
        robust_score = (
            _to_float(m["sharpe"]) * 1.8
            + _to_float(m["annual_return_pct"]) / 25.0
            - abs(_to_float(m["max_drawdown_pct"])) / 20.0
            + _to_float(m["calmar"]) * 0.3
        )
        rows.append(
            {
                "top_k": int(top_k),
                "invest_more_n": int(invest_n),
                "stop_loss_pct": float(stop_loss),
                "trailing_stop_pct": float(trailing),
                **m,
                "robust_score": float(robust_score),
            }
        )

    df = pd.DataFrame(rows).sort_values("robust_score", ascending=False).reset_index(drop=True)
    if df.empty:
        return df, df

    base_ann = _to_float(base_metrics.get("annual_return_pct", 0.0))
    base_sharpe = _to_float(base_metrics.get("sharpe", 0.0))
    base_dd_abs = abs(_to_float(base_metrics.get("max_drawdown_pct", 0.0)))

    stable = df[
        (df["annual_return_pct"] >= base_ann * 0.85)
        & (df["sharpe"] >= base_sharpe * 0.90)
        & (df["max_drawdown_pct"].abs() <= max(1e-9, base_dd_abs) * 1.15)
    ].copy()
    stable = stable.sort_values("robust_score", ascending=False).reset_index(drop=True)
    return df, stable


def _risk_scenario_table(
    ctx: dict[str, Any],
    base_cfg: bt.BacktestConfigV3,
) -> pd.DataFrame:
    base_cap = float(base_cfg.max_total_position)
    base_brake_cap = float(base_cfg.drawdown_brake_position_cap)
    scenarios = {
        "baseline": base_cfg,
        "conservative": _make_cfg(
            base_cfg,
            max_total_position=min(base_cap, 0.65),
            drawdown_brake_threshold=min(float(base_cfg.drawdown_brake_threshold), 0.10),
            drawdown_brake_position_cap=min(base_brake_cap, 0.30),
            use_adaptive_drawdown_mode=True,
        ),
        "defensive": _make_cfg(
            base_cfg,
            max_total_position=min(base_cap, 0.55),
            drawdown_brake_threshold=min(float(base_cfg.drawdown_brake_threshold), 0.08),
            drawdown_brake_position_cap=min(base_brake_cap, 0.25),
            block_new_in_bear_regime=True,
            use_adaptive_drawdown_mode=True,
        ),
        "aggressive": _make_cfg(
            base_cfg,
            max_total_position=min(0.95, base_cap + 0.10),
            use_portfolio_drawdown_brake=False,
            use_adaptive_drawdown_mode=False,
        ),
    }

    rows: list[dict[str, Any]] = []
    for name, cfg in scenarios.items():
        res = _run_once(
            daily_universe=ctx["daily"],
            price_df=ctx["price_df"],
            returns_df=ctx["returns_df"],
            cfg=cfg,
            regime_df=ctx["regime_df"],
            index_filter=ctx["index_filter"],
            start_date=ctx["start_date"],
        )
        m = _extended_metrics(res, initial_capital=cfg.initial_capital)
        rows.append(
            {
                "scenario": name,
                "max_total_position": float(cfg.max_total_position),
                "dd_brake_threshold": float(cfg.drawdown_brake_threshold),
                "dd_brake_cap": float(cfg.drawdown_brake_position_cap),
                "adaptive_dd": bool(cfg.use_adaptive_drawdown_mode),
                "bear_entry_block": bool(cfg.block_new_in_bear_regime),
                **m,
            }
        )
    return pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)


def _failure_monitor(eq: pd.DataFrame, base_metrics: dict[str, float], monitor_days: int, base_cap: float) -> dict[str, Any]:
    if eq is None or eq.empty:
        return {"status": "no_data", "action": "hold", "suggested_position_cap": base_cap}

    # Backward-compatible single-window interface. Internally we use dual-window monitor.
    window = max(20, int(monitor_days))
    short_days = max(20, int(window // 3))
    return _failure_monitor_dual(
        eq=eq,
        base_metrics=base_metrics,
        short_days=short_days,
        long_days=window,
        base_cap=base_cap,
    )


def _monitor_window_stats(
    r: pd.Series,
    base_metrics: dict[str, float],
    window_days: int,
) -> dict[str, Any]:
    recent = r.tail(int(window_days))
    rec_ann = _safe_annual_from_returns(recent) * 100.0
    rec_sharpe = _sharpe_from_returns(recent)
    rec_dd = _max_drawdown_from_returns(recent) * 100.0

    b_ann = _to_float(base_metrics.get("annual_return_pct", 0.0))
    b_sharpe = _to_float(base_metrics.get("sharpe", 0.0))
    b_dd_abs = abs(_to_float(base_metrics.get("max_drawdown_pct", 0.0)))
    rec_dd_abs = abs(rec_dd)

    severe = (rec_sharpe < 0.0) and (rec_ann < 0.0) and (rec_dd_abs > b_dd_abs * 1.2)
    warning = (rec_sharpe < b_sharpe * 0.6) or (rec_ann < b_ann * 0.6) or (rec_dd_abs > b_dd_abs * 1.3)

    if severe:
        status = "degraded_severe"
    elif warning:
        status = "degraded_warning"
    else:
        status = "healthy"

    return {
        "status": status,
        "window_days": int(window_days),
        "recent_annual_return_pct": float(rec_ann),
        "recent_sharpe": float(rec_sharpe),
        "recent_max_drawdown_pct": float(rec_dd),
        "baseline_annual_return_pct": float(b_ann),
        "baseline_sharpe": float(b_sharpe),
        "baseline_max_drawdown_pct": float(_to_float(base_metrics.get("max_drawdown_pct", 0.0))),
    }


def _failure_monitor_dual(
    eq: pd.DataFrame,
    base_metrics: dict[str, float],
    short_days: int,
    long_days: int,
    base_cap: float,
) -> dict[str, Any]:
    if eq is None or eq.empty:
        return {"status": "no_data", "action": "hold", "suggested_position_cap": float(base_cap)}

    short_days = max(20, int(short_days))
    long_days = max(short_days, int(long_days))

    r = pd.to_numeric(eq.get("daily_return", 0.0), errors="coerce").fillna(0.0)
    if len(r) < long_days:
        return {
            "status": "insufficient_data",
            "action": "hold",
            "suggested_position_cap": float(base_cap),
            "short_window_days": int(short_days),
            "long_window_days": int(long_days),
        }

    short_eval = _monitor_window_stats(r=r, base_metrics=base_metrics, window_days=short_days)
    long_eval = _monitor_window_stats(r=r, base_metrics=base_metrics, window_days=long_days)

    states = [str(short_eval.get("status", "healthy")), str(long_eval.get("status", "healthy"))]
    severe_count = sum(1 for s in states if s == "degraded_severe")
    warn_or_worse_count = sum(1 for s in states if s in {"degraded_warning", "degraded_severe"})

    if severe_count >= 2:
        action = "stop"
        cap = 0.0
        status = "degraded_severe"
    elif severe_count >= 1 and warn_or_worse_count >= 2:
        action = "reduce_hard"
        cap = float(min(base_cap * 0.40, 0.35))
        status = "degraded_warning"
    elif warn_or_worse_count >= 2:
        action = "reduce"
        cap = float(min(base_cap * 0.65, 0.60))
        status = "degraded_warning"
    else:
        action = "normal"
        cap = float(base_cap)
        status = "healthy"

    return {
        "status": status,
        "action": action,
        "suggested_position_cap": float(cap),
        "monitor_days": int(long_days),
        "recent_annual_return_pct": float(_to_float(long_eval.get("recent_annual_return_pct", 0.0))),
        "recent_sharpe": float(_to_float(long_eval.get("recent_sharpe", 0.0))),
        "recent_max_drawdown_pct": float(_to_float(long_eval.get("recent_max_drawdown_pct", 0.0))),
        "baseline_annual_return_pct": float(_to_float(long_eval.get("baseline_annual_return_pct", 0.0))),
        "baseline_sharpe": float(_to_float(long_eval.get("baseline_sharpe", 0.0))),
        "baseline_max_drawdown_pct": float(_to_float(long_eval.get("baseline_max_drawdown_pct", 0.0))),
        "short_window_days": int(short_days),
        "long_window_days": int(long_days),
        "windows": {
            "short": short_eval,
            "long": long_eval,
        },
        "trigger_counts": {
            "severe_count": int(severe_count),
            "warn_or_worse_count": int(warn_or_worse_count),
        },
    }


def _trade_gate_decision(
    quality_gate: dict[str, Any],
    monitor: dict[str, Any],
    base_cap: float,
    warn_cap_min: float = 0.60,
    warn_cap_max: float = 0.80,
) -> dict[str, Any]:
    warn_lo = max(0.0, min(1.0, float(warn_cap_min)))
    warn_hi = max(0.0, min(1.0, float(warn_cap_max)))
    if warn_lo > warn_hi:
        warn_lo, warn_hi = warn_hi, warn_lo

    gate_status = str((quality_gate or {}).get("status", "unknown")).strip().lower()
    monitor_action = str((monitor or {}).get("action", "hold")).strip().lower()
    monitor_cap = _clamp01((monitor or {}).get("suggested_position_cap"), default=None)
    base_cap = float(_clamp01(base_cap, default=1.0) or 1.0)

    if gate_status == "fail" or monitor_action == "stop":
        return {
            "status": "fail",
            "action": "stop",
            "position_cap": 0.0,
            "position_cap_source": "gate_fail_or_monitor_stop",
        }

    if gate_status == "warn" or monitor_action in {"reduce", "reduce_hard"}:
        raw = monitor_cap if monitor_cap is not None else min(base_cap * 0.75, warn_hi)
        cap = float(min(warn_hi, max(warn_lo, raw)))
        return {
            "status": "warn",
            "action": "reduce",
            "position_cap": cap,
            "position_cap_source": "warn_band",
            "warn_cap_min": warn_lo,
            "warn_cap_max": warn_hi,
        }

    pass_cap = base_cap
    if monitor_cap is not None:
        pass_cap = min(pass_cap, float(monitor_cap))
    return {
        "status": "pass",
        "action": "normal",
        "position_cap": float(pass_cap),
        "position_cap_source": "base_or_monitor",
    }


def _quality_gate(
    baseline: dict[str, Any],
    wf_summary: dict[str, Any],
    stable_summary: dict[str, Any],
    monitor: dict[str, Any],
    overfit_diag: dict[str, Any] | None = None,
    min_expectancy: float = 0.0,
    max_drawdown_floor_pct: float = -20.0,
    min_wf_mean_sharpe: float = 1.0,
    min_wf_sharpe_ok_ratio: float = 0.60,
    min_wf_folds_required: int = 3,
    min_stable_ratio: float = 0.30,
    max_avg_turnover: float = 0.45,
    max_slippage_drag_pct: float = 15.0,
    max_pbo: float = 0.45,
    min_dsr: float = 0.55,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def _add_check(name: str, value: Any, rule: str, passed: bool, level: str) -> None:
        checks.append(
            {
                "name": name,
                "value": value,
                "rule": rule,
                "passed": bool(passed),
                "level": level,
            }
        )

    daily_expectancy = _to_float(baseline.get("daily_expectancy", baseline.get("expectancy", 0.0)))
    max_drawdown_pct = _to_float(baseline.get("max_drawdown_pct", 0.0))
    wf_mean_sharpe = _to_float(wf_summary.get("mean_sharpe", 0.0))
    wf_sharpe_ok_ratio = _to_float(wf_summary.get("sharpe_ok_ratio", 0.0))
    wf_fold_count = int(_to_float(wf_summary.get("fold_count", 0), 0.0))
    wf_folds_required = max(1, int(min_wf_folds_required))
    wf_has_enough_folds = bool(wf_fold_count >= wf_folds_required)
    tested_combos = max(1, int(_to_float(stable_summary.get("tested_combos", 0), 0.0)))
    stable_combo_count = int(_to_float(stable_summary.get("stable_combo_count", 0), 0.0))
    stable_ratio = float(stable_combo_count / tested_combos)
    monitor_action = str(monitor.get("action", "hold")).lower().strip()
    avg_turnover = _to_float(baseline.get("avg_turnover", 0.0))
    slippage_drag_pct = _to_float(baseline.get("slippage_drag_pct", 0.0))
    pbo = _to_float((overfit_diag or {}).get("pbo", float("nan")), float("nan"))
    dsr = _to_float((overfit_diag or {}).get("dsr", float("nan")), float("nan"))
    pbo_info = (overfit_diag or {}).get("pbo_details", {}) if isinstance((overfit_diag or {}).get("pbo_details", {}), dict) else {}
    pbo_status = str(pbo_info.get("status", "")).strip().lower()
    pbo_fold_count = int(_to_float(pbo_info.get("fold_count", 0), 0.0))
    pbo_model_count = int(_to_float(pbo_info.get("model_count", 0), 0.0))
    pbo_reliable = bool(
        np.isfinite(pbo)
        and pbo_status == "ok"
        and pbo_fold_count >= 4
        and pbo_model_count >= 6
    )
    pbo_ok = (not pbo_reliable) or (pbo <= float(max_pbo))
    dsr_ok = (not np.isfinite(dsr)) or (dsr >= float(min_dsr))

    _add_check(
        name="expectancy_positive",
        value=daily_expectancy,
        rule=f"daily_expectancy > {min_expectancy:.6f}",
        passed=daily_expectancy > min_expectancy,
        level="hard",
    )
    _add_check(
        name="drawdown_acceptable",
        value=max_drawdown_pct,
        rule=f"max_drawdown_pct >= {max_drawdown_floor_pct:.2f}",
        passed=max_drawdown_pct >= max_drawdown_floor_pct,
        level="hard",
    )
    if wf_has_enough_folds:
        _add_check(
            name="wf_mean_sharpe_ok",
            value=wf_mean_sharpe,
            rule=f"wf_mean_sharpe >= {min_wf_mean_sharpe:.2f}",
            passed=wf_mean_sharpe >= min_wf_mean_sharpe,
            level="hard",
        )
        _add_check(
            name="wf_sharpe_ok_ratio",
            value=wf_sharpe_ok_ratio,
            rule=f"wf_sharpe_ok_ratio >= {min_wf_sharpe_ok_ratio:.2f}",
            passed=wf_sharpe_ok_ratio >= min_wf_sharpe_ok_ratio,
            level="warn",
        )
    else:
        _add_check(
            name="wf_data_sufficient",
            value=wf_fold_count,
            rule=f"wf_fold_count >= {wf_folds_required}",
            passed=False,
            level="warn",
        )
    _add_check(
        name="stable_zone_ratio",
        value=stable_ratio,
        rule=f"stable_combo_ratio >= {min_stable_ratio:.2f}",
        passed=stable_ratio >= min_stable_ratio,
        level="warn",
    )
    _add_check(
        name="failure_monitor_action",
        value=monitor_action,
        rule="monitor_action in ['normal', 'reduce']",
        passed=monitor_action in {"normal", "reduce"},
        level="hard",
    )
    _add_check(
        name="execution_slippage_drag",
        value=slippage_drag_pct,
        rule=f"slippage_drag_pct <= {float(max_slippage_drag_pct):.2f}",
        passed=slippage_drag_pct <= float(max_slippage_drag_pct),
        level="hard",
    )
    _add_check(
        name="avg_turnover_control",
        value=avg_turnover,
        rule=f"avg_turnover <= {float(max_avg_turnover):.3f}",
        passed=avg_turnover <= float(max_avg_turnover),
        level="warn",
    )
    _add_check(
        name="pbo_control",
        value=pbo if pbo_reliable else None,
        rule=f"pbo <= {float(max_pbo):.2f} (if fold_count>=4 and model_count>=6)",
        passed=pbo_ok,
        level="warn",
    )
    _add_check(
        name="dsr_control",
        value=dsr if np.isfinite(dsr) else None,
        rule=f"dsr >= {float(min_dsr):.2f} (if available)",
        passed=dsr_ok,
        level="hard",
    )

    hard_failed = [c for c in checks if (c["level"] == "hard" and not c["passed"])]
    warn_failed = [c for c in checks if (c["level"] == "warn" and not c["passed"])]

    if hard_failed:
        status = "fail"
    elif warn_failed:
        status = "warn"
    else:
        status = "pass"

    return {
        "status": status,
        "hard_failed": [c["name"] for c in hard_failed],
        "warn_failed": [c["name"] for c in warn_failed],
        "checks": checks,
    }


def _build_context(
    base_dir: Path,
    start_date: str,
    cfg: bt.BacktestConfigV3,
    watchlist_file: str | None = None,
) -> dict[str, Any]:
    feats = bt.load_and_prepare_features(
        base_dir,
        start_date=start_date,
        news_sentiment_lag_days=int(getattr(cfg, "news_sentiment_lag_days", 1)),
    )
    if watchlist_file:
        wl_path = base_dir / watchlist_file
        if wl_path.exists():
            try:
                symbols = _load_watchlist_symbols(base_dir, watchlist_file)
                if symbols:
                    before = feats["symbol"].nunique()
                    feats = feats[feats["symbol"].isin(symbols)].copy()
                    after = feats["symbol"].nunique()
                    print(
                        f"[Watchlist] using {watchlist_file} "
                        f"({after}/{before} symbols kept, pool size {len(symbols)})"
                    )
            except Exception as e:
                print(f"[Watchlist] parse failed ({watchlist_file}): {e}")
        else:
            print(f"[Watchlist] file not found: {wl_path}; fallback to full universe")

    industry_map = bt.build_industry_map_from_config(base_dir, feats["symbol"])
    feats["industry"] = feats["symbol"].map(industry_map).fillna("OTHER")
    daily = bt.precompute_daily_universe(feats, cfg=cfg)

    close_df = feats.pivot_table(index="date", columns="symbol", values="close").sort_index()
    if "open" in feats.columns:
        open_df = feats.pivot_table(index="date", columns="symbol", values="open").sort_index()
    else:
        open_df = close_df
    close_ret = close_df.pct_change(fill_method=None).fillna(0.0)
    open_ret = open_df.pct_change(fill_method=None).fillna(0.0)

    exec_mode = str(cfg.execution_price_mode).strip().lower()
    if exec_mode == "next_open":
        price_df, returns_df = open_df, open_ret
    else:
        price_df, returns_df = close_df, close_ret

    regime_df = bt.compute_market_regime(feats, cfg=cfg)
    index_filter = bt.compute_index_filter(base_dir, start_date, cfg)
    return {
        "feats": feats,
        "daily": daily,
        "price_df": price_df,
        "returns_df": returns_df,
        "regime_df": regime_df,
        "index_filter": index_filter,
        "start_date": start_date,
    }


def _render_md(summary: dict[str, Any]) -> str:
    b = summary["baseline"]
    wf = summary["walk_forward"]
    roll = summary["rolling"]
    stable = summary["stable_zone"]
    overfit = summary.get("overfit_diagnostics", {}) or {}
    mon = summary["failure_monitor"]
    gate = summary.get("quality_gate", {})
    trade_gate = summary.get("trade_gate", {})
    hard_failed = gate.get("hard_failed", []) or []
    warn_failed = gate.get("warn_failed", []) or []
    pbo = _to_float(overfit.get("pbo", float("nan")), float("nan"))
    dsr = _to_float(overfit.get("dsr", float("nan")), float("nan"))
    return "\n".join(
        [
            "# Strategy Process Report",
            "",
            f"- Generated at: {summary['generated_at']}",
            f"- Start date: {summary['start_date']}",
            "",
            "## Baseline",
            f"- Annual return: {b['annual_return_pct']:+.2f}%",
            f"- Max drawdown: {b['max_drawdown_pct']:.2f}%",
            f"- Sharpe: {b['sharpe']:.3f}",
            f"- Calmar: {b['calmar']:.3f}",
            f"- Win rate: {b['win_rate_pct']:.2f}%",
            f"- Profit factor: {b['profit_factor']:.3f}",
            f"- Expectancy: {b['expectancy']:.5f}",
            f"- Avg turnover: {b['avg_turnover']:.4f}",
            f"- Avg extra execution cost: {b['avg_extra_exec_cost_bps']:.2f} bps",
            f"- Slippage drag: {b['slippage_drag_pct']:.2f}% annualized",
            "",
            "## Out-Of-Sample",
            f"- Walk-forward folds: {wf.get('fold_count', 0)}",
            f"- WF mean annual return: {wf.get('mean_annual_return_pct', 0.0):+.2f}%",
            f"- WF mean max drawdown: {wf.get('mean_max_drawdown_pct', 0.0):.2f}%",
            f"- WF mean Sharpe: {wf.get('mean_sharpe', 0.0):.3f}",
            f"- Rolling windows: {roll.get('window_count', 0)}",
            f"- Rolling Sharpe mean/std: {roll.get('sharpe_mean', 0.0):.3f}/{roll.get('sharpe_std', 0.0):.3f}",
            f"- Rolling worst window drawdown: {roll.get('worst_window_dd_pct', 0.0):.2f}%",
            "",
            "## Stability Zone",
            f"- Tested parameter combos: {stable.get('tested_combos', 0)}",
            f"- Stable-zone combos: {stable.get('stable_combo_count', 0)}",
            f"- Recommended in stable-zone: {stable.get('recommended_params', {})}",
            "",
            "## Overfit Diagnostics",
            f"- Status: {overfit.get('status', 'unknown')}",
            f"- CV splits: {overfit.get('cv_splits', 0)}",
            f"- Combo count: {overfit.get('combo_count', 0)}",
            f"- PBO: {pbo:.3f}" if np.isfinite(pbo) else "- PBO: N/A",
            f"- DSR: {dsr:.3f}" if np.isfinite(dsr) else "- DSR: N/A",
            "",
            "## Risk Layer",
            f"- Recommended scenario: {summary.get('risk_recommendation', 'baseline')}",
            "",
            "## Failure Monitor",
            f"- Status: {mon.get('status', 'unknown')}",
            f"- Action: {mon.get('action', 'hold')}",
            f"- Suggested position cap: {mon.get('suggested_position_cap', 0.0):.2f}",
            f"- Recent annual return ({mon.get('monitor_days', 0)}d): {mon.get('recent_annual_return_pct', 0.0):+.2f}%",
            f"- Monitor windows: short={mon.get('short_window_days', 0)}d, long={mon.get('long_window_days', 0)}d",
            f"- Short-window Sharpe: {((mon.get('windows', {}) or {}).get('short', {}) or {}).get('recent_sharpe', 0.0):.3f}",
            f"- Long-window Sharpe: {((mon.get('windows', {}) or {}).get('long', {}) or {}).get('recent_sharpe', 0.0):.3f}",
            "",
            "## Quality Gate",
            f"- Status: {gate.get('status', 'unknown')}",
            f"- Hard failed checks: {hard_failed if hard_failed else 'None'}",
            f"- Warn failed checks: {warn_failed if warn_failed else 'None'}",
            "",
            "## Trade Gate",
            f"- Status: {trade_gate.get('status', 'unknown')}",
            f"- Action: {trade_gate.get('action', 'hold')}",
            f"- Position cap: {float(_to_float(trade_gate.get('position_cap', 0.0), 0.0)):.2f}",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Strategy process pipeline for robustness and risk governance.")
    ap.add_argument("--base-dir", default=".", help="project root")
    ap.add_argument("--config", default="", help="config path (default: config.yaml, fallback: config_v31.yaml)")
    ap.add_argument("--watchlist", default="", help="optional watchlist file to limit universe")
    ap.add_argument(
        "--no-default-watchlist",
        action="store_true",
        help="do not auto-use backtest_watchlist.yaml/watchlist_cache.yaml when --watchlist is omitted",
    )
    ap.add_argument("--start-date", default=None, help="backtest start date override")
    ap.add_argument("--monitor-days", type=int, default=60, help="legacy alias for long-window monitor days")
    ap.add_argument("--monitor-short-days", type=int, default=20, help="short monitor window days")
    ap.add_argument("--monitor-long-days", type=int, default=None, help="long monitor window days")
    ap.add_argument("--warn-cap-min", type=float, default=0.60, help="warn-band lower position cap")
    ap.add_argument("--warn-cap-max", type=float, default=0.80, help="warn-band upper position cap")
    ap.add_argument("--gate-max-avg-turnover", type=float, default=0.45, help="quality-gate max average turnover")
    ap.add_argument(
        "--gate-max-slippage-drag-pct",
        type=float,
        default=15.0,
        help="quality-gate max annualized slippage drag pct",
    )
    ap.add_argument("--overfit-max-combos", type=int, default=8, help="max combos for overfit CV diagnostics")
    ap.add_argument("--overfit-cv-splits", type=int, default=3, help="time-series CV splits for overfit diagnostics")
    ap.add_argument("--gate-max-pbo", type=float, default=0.45, help="quality-gate max accepted PBO")
    ap.add_argument("--gate-min-dsr", type=float, default=0.55, help="quality-gate minimum accepted DSR")
    ap.add_argument(
        "--gate-profile",
        choices=["production", "research"],
        default="production",
        help="quality-gate profile: production (strict) or research (looser)",
    )
    ap.add_argument("--gate-min-expectancy", type=float, default=None, help="override min daily expectancy")
    ap.add_argument("--gate-max-drawdown-floor-pct", type=float, default=None, help="override max-drawdown floor pct")
    ap.add_argument("--gate-min-wf-mean-sharpe", type=float, default=None, help="override min WF mean sharpe")
    ap.add_argument(
        "--gate-min-wf-sharpe-ok-ratio",
        type=float,
        default=None,
        help="override min WF sharpe-ok ratio",
    )
    ap.add_argument(
        "--gate-min-wf-folds-required",
        type=int,
        default=None,
        help="override minimum required walk-forward fold count",
    )
    ap.add_argument("--gate-min-stable-ratio", type=float, default=None, help="override min stable-zone ratio")
    ap.add_argument("--rolling-window-days", type=int, default=252, help="rolling metric window")
    ap.add_argument("--rolling-step-days", type=int, default=21, help="rolling metric step")
    ap.add_argument("--max-combos", type=int, default=24, help="max parameter combos for stability zone search")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "data" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = str(args.config).strip() or None
    cfg = bt._load_cfg_from_yaml(base_dir, config_path=cfg_path)
    start_date = _load_start_date(base_dir, args.start_date, config_path=cfg_path)

    watchlist_file = str(args.watchlist).strip() or None
    if watchlist_file is None and not args.no_default_watchlist:
        for cand in ["backtest_watchlist.yaml", "watchlist_cache.yaml"]:
            if (base_dir / cand).exists():
                watchlist_file = cand
                print(f"auto watchlist: {cand} (use --no-default-watchlist to disable)")
                break

    print("[StrategyProcess] building context...")
    ctx = _build_context(base_dir=base_dir, start_date=start_date, cfg=cfg, watchlist_file=watchlist_file)

    print("[StrategyProcess] baseline backtest...")

    baseline_res = _run_once(
        daily_universe=ctx["daily"],
        price_df=ctx["price_df"],
        returns_df=ctx["returns_df"],
        cfg=cfg,
        regime_df=ctx["regime_df"],
        index_filter=ctx["index_filter"],
        start_date=start_date,
    )
    baseline_metrics = _extended_metrics(baseline_res, initial_capital=cfg.initial_capital)

    print("[StrategyProcess] walk-forward...")
    wf_df, wf_summary = _wf_summary(ctx, cfg)
    print("[StrategyProcess] rolling metrics...")
    rolling_df = _rolling_window_metrics(
        baseline_res.get("equity_curve", pd.DataFrame()),
        window_days=int(args.rolling_window_days),
        step_days=int(args.rolling_step_days),
    )
    rolling_summary = {
        "window_count": int(len(rolling_df)),
        "sharpe_mean": _to_float(rolling_df["sharpe"].mean(), 0.0) if not rolling_df.empty else 0.0,
        "sharpe_std": _to_float(rolling_df["sharpe"].std(ddof=1), 0.0) if len(rolling_df) > 1 else 0.0,
        "worst_window_dd_pct": _to_float(rolling_df["max_drawdown_pct"].min(), 0.0) if not rolling_df.empty else 0.0,
    }

    print("[StrategyProcess] stability zone scan...")
    zone_all_df, zone_stable_df = _build_stability_zone(
        ctx=ctx,
        base_cfg=cfg,
        base_metrics=baseline_metrics,
        max_combos=int(args.max_combos),
    )
    requested_overfit_max_combos = int(args.overfit_max_combos)
    requested_overfit_cv_splits = int(args.overfit_cv_splits)
    effective_overfit_max_combos = max(4, requested_overfit_max_combos)
    effective_overfit_cv_splits = max(3, requested_overfit_cv_splits)
    if (
        effective_overfit_max_combos != requested_overfit_max_combos
        or effective_overfit_cv_splits != requested_overfit_cv_splits
    ):
        print(
            "[StrategyProcess] overfit args adjusted to effective minimums: "
            f"max_combos {requested_overfit_max_combos}->{effective_overfit_max_combos}, "
            f"cv_splits {requested_overfit_cv_splits}->{effective_overfit_cv_splits}"
        )

    print("[StrategyProcess] overfit diagnostics...")
    overfit_diag = _compute_overfit_diagnostics(
        ctx=ctx,
        base_cfg=cfg,
        baseline_eq=baseline_res.get("equity_curve", pd.DataFrame()),
        overfit_max_combos=effective_overfit_max_combos,
        overfit_cv_splits=effective_overfit_cv_splits,
    )
    recommended_params = {}
    if not zone_stable_df.empty:
        top = zone_stable_df.iloc[0]
        recommended_params = {
            "top_k": int(top["top_k"]),
            "invest_more_n": int(top["invest_more_n"]),
            "stop_loss_pct": float(top["stop_loss_pct"]),
            "trailing_stop_pct": float(top["trailing_stop_pct"]),
        }

    print("[StrategyProcess] risk scenarios...")
    risk_df = _risk_scenario_table(ctx=ctx, base_cfg=cfg)
    risk_reco = "baseline"
    if not risk_df.empty:
        risk_df = risk_df.copy()
        risk_df["risk_score"] = (
            risk_df["sharpe"] * 1.5
            + risk_df["annual_return_pct"] / 30.0
            - risk_df["max_drawdown_pct"].abs() / 18.0
        )
        risk_reco = str(risk_df.sort_values("risk_score", ascending=False).iloc[0]["scenario"])

    monitor_long_days = int(args.monitor_long_days) if args.monitor_long_days is not None else int(args.monitor_days)
    monitor_short_days = int(args.monitor_short_days)
    if monitor_short_days >= monitor_long_days:
        monitor_short_days = max(20, int(monitor_long_days // 2))

    monitor = _failure_monitor_dual(
        eq=baseline_res.get("equity_curve", pd.DataFrame()),
        base_metrics=baseline_metrics,
        short_days=monitor_short_days,
        long_days=monitor_long_days,
        base_cap=float(cfg.max_total_position),
    )

    gate_profile = str(args.gate_profile).strip().lower()
    profile_defaults: dict[str, dict[str, float]] = {
        "production": {
            "min_expectancy": 0.0,
            "max_drawdown_floor_pct": -20.0,
            "min_wf_mean_sharpe": 1.0,
            "min_wf_sharpe_ok_ratio": 0.60,
            "min_wf_folds_required": 3.0,
            "min_stable_ratio": 0.30,
        },
        "research": {
            "min_expectancy": 0.0,
            "max_drawdown_floor_pct": -25.0,
            "min_wf_mean_sharpe": 0.70,
            "min_wf_sharpe_ok_ratio": 0.45,
            "min_wf_folds_required": 3.0,
            "min_stable_ratio": 0.20,
        },
    }
    gate_thresholds = dict(profile_defaults.get(gate_profile, profile_defaults["production"]))

    cfg_raw: dict[str, Any] = {}
    try:
        resolved_cfg_path = bt._resolve_config_file(base_dir, config_path=cfg_path)
        if resolved_cfg_path and resolved_cfg_path.exists():
            cfg_raw = yaml.safe_load(resolved_cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        cfg_raw = {}
    sp_cfg = cfg_raw.get("strategy_process", {}) if isinstance(cfg_raw, dict) else {}
    profiles = sp_cfg.get("gate_profiles", {}) if isinstance(sp_cfg, dict) else {}
    profile_cfg = profiles.get(gate_profile, {}) if isinstance(profiles, dict) else {}
    if isinstance(profile_cfg, dict):
        for k in [
            "min_expectancy",
            "max_drawdown_floor_pct",
            "min_wf_mean_sharpe",
            "min_wf_sharpe_ok_ratio",
            "min_wf_folds_required",
            "min_stable_ratio",
        ]:
            if profile_cfg.get(k) is not None:
                gate_thresholds[k] = float(profile_cfg.get(k))

    if args.gate_min_expectancy is not None:
        gate_thresholds["min_expectancy"] = float(args.gate_min_expectancy)
    if args.gate_max_drawdown_floor_pct is not None:
        gate_thresholds["max_drawdown_floor_pct"] = float(args.gate_max_drawdown_floor_pct)
    if args.gate_min_wf_mean_sharpe is not None:
        gate_thresholds["min_wf_mean_sharpe"] = float(args.gate_min_wf_mean_sharpe)
    if args.gate_min_wf_sharpe_ok_ratio is not None:
        gate_thresholds["min_wf_sharpe_ok_ratio"] = float(args.gate_min_wf_sharpe_ok_ratio)
    if args.gate_min_wf_folds_required is not None:
        gate_thresholds["min_wf_folds_required"] = float(args.gate_min_wf_folds_required)
    if args.gate_min_stable_ratio is not None:
        gate_thresholds["min_stable_ratio"] = float(args.gate_min_stable_ratio)

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "start_date": start_date,
        "gate_policy": {
            "profile": gate_profile,
            "min_expectancy": float(gate_thresholds["min_expectancy"]),
            "max_drawdown_floor_pct": float(gate_thresholds["max_drawdown_floor_pct"]),
            "min_wf_mean_sharpe": float(gate_thresholds["min_wf_mean_sharpe"]),
            "min_wf_sharpe_ok_ratio": float(gate_thresholds["min_wf_sharpe_ok_ratio"]),
            "min_wf_folds_required": int(max(1, gate_thresholds["min_wf_folds_required"])),
            "min_stable_ratio": float(gate_thresholds["min_stable_ratio"]),
            "max_avg_turnover": float(args.gate_max_avg_turnover),
            "max_slippage_drag_pct": float(args.gate_max_slippage_drag_pct),
            "warn_cap_min": float(args.warn_cap_min),
            "warn_cap_max": float(args.warn_cap_max),
            "monitor_short_days": int(monitor_short_days),
            "monitor_long_days": int(monitor_long_days),
            "overfit_max_combos": int(effective_overfit_max_combos),
            "overfit_cv_splits": int(effective_overfit_cv_splits),
            "overfit_max_combos_requested": int(requested_overfit_max_combos),
            "overfit_cv_splits_requested": int(requested_overfit_cv_splits),
            "max_pbo": float(args.gate_max_pbo),
            "min_dsr": float(args.gate_min_dsr),
        },
        "baseline": baseline_metrics,
        "walk_forward": wf_summary,
        "rolling": rolling_summary,
        "stable_zone": {
            "tested_combos": int(len(zone_all_df)),
            "stable_combo_count": int(len(zone_stable_df)),
            "recommended_params": recommended_params,
        },
        "overfit_diagnostics": overfit_diag,
        "risk_recommendation": risk_reco,
        "failure_monitor": monitor,
    }
    summary["quality_gate"] = _quality_gate(
        baseline=baseline_metrics,
        wf_summary=wf_summary,
        stable_summary=summary["stable_zone"],
        monitor=monitor,
        overfit_diag=overfit_diag,
        min_expectancy=float(gate_thresholds["min_expectancy"]),
        max_drawdown_floor_pct=float(gate_thresholds["max_drawdown_floor_pct"]),
        min_wf_mean_sharpe=float(gate_thresholds["min_wf_mean_sharpe"]),
        min_wf_sharpe_ok_ratio=float(gate_thresholds["min_wf_sharpe_ok_ratio"]),
        min_wf_folds_required=int(max(1, gate_thresholds["min_wf_folds_required"])),
        min_stable_ratio=float(gate_thresholds["min_stable_ratio"]),
        max_avg_turnover=float(args.gate_max_avg_turnover),
        max_slippage_drag_pct=float(args.gate_max_slippage_drag_pct),
        max_pbo=float(args.gate_max_pbo),
        min_dsr=float(args.gate_min_dsr),
    )
    summary["trade_gate"] = _trade_gate_decision(
        quality_gate=summary["quality_gate"],
        monitor=monitor,
        base_cap=float(cfg.max_total_position),
        warn_cap_min=float(args.warn_cap_min),
        warn_cap_max=float(args.warn_cap_max),
    )

    summary_path = out_dir / "strategy_process_summary.json"
    wf_path = out_dir / "strategy_process_walk_forward.csv"
    roll_path = out_dir / "strategy_process_rolling.csv"
    zone_all_path = out_dir / "strategy_process_param_zone_all.csv"
    zone_stable_path = out_dir / "strategy_process_param_zone_stable.csv"
    risk_path = out_dir / "strategy_process_risk_scenarios.csv"
    md_path = out_dir / "strategy_process_report.md"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not wf_df.empty:
        wf_df.to_csv(wf_path, index=False, encoding="utf-8-sig")
    if not rolling_df.empty:
        rolling_df.to_csv(roll_path, index=False, encoding="utf-8-sig")
    if not zone_all_df.empty:
        zone_all_df.to_csv(zone_all_path, index=False, encoding="utf-8-sig")
    if not zone_stable_df.empty:
        zone_stable_df.to_csv(zone_stable_path, index=False, encoding="utf-8-sig")
    if not risk_df.empty:
        risk_df.to_csv(risk_path, index=False, encoding="utf-8-sig")
    md_path.write_text(_render_md(summary), encoding="utf-8")

    print("Strategy process pipeline done.")
    print(f"summary: {summary_path}")
    print(f"report : {md_path}")
    print(f"baseline sharpe: {baseline_metrics['sharpe']:.3f}")
    print(f"baseline max drawdown: {baseline_metrics['max_drawdown_pct']:.2f}%")
    print(
        "gate profile: "
        f"{gate_profile} "
        f"(wf_mean>={gate_thresholds['min_wf_mean_sharpe']:.2f}, "
        f"wf_ratio>={gate_thresholds['min_wf_sharpe_ok_ratio']:.2f}, "
        f"wf_folds>={int(max(1, gate_thresholds['min_wf_folds_required']))})"
    )
    print(f"monitor action: {monitor.get('action')} (cap={monitor.get('suggested_position_cap'):.2f})")
    print(
        f"overfit: status={overfit_diag.get('status', 'unknown')} "
        f"pbo={_to_float(overfit_diag.get('pbo', float('nan')), float('nan')):.3f} "
        f"dsr={_to_float(overfit_diag.get('dsr', float('nan')), float('nan')):.3f}"
    )
    print(f"quality gate: {summary['quality_gate'].get('status', 'unknown')}")
    print(
        f"trade gate: {summary['trade_gate'].get('status', 'unknown')} "
        f"action={summary['trade_gate'].get('action', 'hold')} "
        f"cap={_to_float(summary['trade_gate'].get('position_cap', 0.0), 0.0):.2f}"
    )


if __name__ == "__main__":
    main()
