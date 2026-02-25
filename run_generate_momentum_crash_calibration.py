from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not math.isfinite(x):
        return float(default)
    return float(x)


def _annual_return(nav: pd.Series, periods_per_year: int = 252) -> float:
    n = len(nav)
    if n < 2:
        return 0.0
    start = float(nav.iloc[0])
    end = float(nav.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    return float((end / start) ** (periods_per_year / n) - 1.0)


def _max_drawdown(nav: pd.Series) -> float:
    if nav is None or len(nav) == 0:
        return 0.0
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def _sharpe(r: pd.Series, periods_per_year: int = 252) -> float:
    if r is None or len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if not math.isfinite(sd) or sd < 1e-12:
        return 0.0
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def _parse_float_list(raw: str, default_vals: list[float]) -> list[float]:
    t = str(raw or "").strip()
    if not t:
        return sorted(set(float(x) for x in default_vals))
    out: list[float] = []
    for p in t.replace(";", ",").split(","):
        s = p.strip()
        if not s:
            continue
        try:
            out.append(float(s))
        except Exception:
            continue
    if not out:
        out = list(default_vals)
    return sorted(set(float(x) for x in out))


def _parse_int_list(raw: str, default_vals: list[int]) -> list[int]:
    t = str(raw or "").strip()
    if not t:
        return sorted(set(int(x) for x in default_vals))
    out: list[int] = []
    for p in t.replace(";", ",").split(","):
        s = p.strip()
        if not s:
            continue
        try:
            out.append(int(float(s)))
        except Exception:
            continue
    if not out:
        out = list(default_vals)
    return sorted(set(max(1, int(x)) for x in out))


def _infer_as_of(base_dir: Path, as_of_arg: str) -> str:
    t = str(as_of_arg or "").strip()
    if t:
        return t
    p = base_dir / "data" / "signals" / "latest_daily_rank.csv"
    if p.exists():
        try:
            df = pd.read_csv(p, nrows=1)
            if not df.empty and "date" in df.columns:
                d = str(df["date"].iloc[0]).strip()
                if d:
                    return d
        except Exception:
            pass
    return date.today().isoformat()


def _load_cfg(base_dir: Path, config_arg: str) -> dict[str, Any]:
    candidates: list[Path] = []
    t = str(config_arg or "").strip()
    if t:
        p = Path(t)
        if not p.is_absolute():
            p = base_dir / p
        candidates.append(p)
    candidates.extend([base_dir / "config.yaml", base_dir / "config_v31.yaml"])
    for p in candidates:
        if not p.exists():
            continue
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict):
                return raw
        except Exception:
            continue
    return {}


def _resolve_replay_start_date(cfg_raw: dict[str, Any], start_date_arg: str) -> str:
    t = str(start_date_arg or "").strip()
    if t:
        return t
    bt = cfg_raw.get("backtest", {}) if isinstance(cfg_raw, dict) else {}
    md = cfg_raw.get("market_data", {}) if isinstance(cfg_raw, dict) else {}
    s = str(bt.get("start_date") or md.get("start_date") or "").strip()
    return s if s else "2020-01-01"


def _resolve_replay_watchlist(base_dir: Path, watchlist_arg: str) -> str:
    t = str(watchlist_arg or "").strip()
    if t:
        return t
    for cand in ["watchlist_cache.yaml", "backtest_watchlist.yaml"]:
        if (base_dir / cand).exists():
            return cand
    return ""


def _load_backtest_module(base_dir: Path):
    mod_path = base_dir / "run_backtest_strategy_v3.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"missing backtest module: {mod_path}")
    spec = importlib.util.spec_from_file_location("run_backtest_strategy_v3", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _extract_watchlist_symbols(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    raw_codes = list(set(re.findall(r"(\d{6})", text)))
    if not raw_codes:
        try:
            parsed = yaml.safe_load(text) or {}
            wl_list = parsed.get("watchlist", []) if isinstance(parsed, dict) else []
            raw_codes = [str(c).split(".")[0] for c in wl_list if str(c).strip()]
        except Exception:
            raw_codes = []
    symbols: set[str] = set()
    for code in raw_codes:
        c = str(code).zfill(6)
        if c.startswith(("6", "5")):
            symbols.add(f"{c}.SH")
        else:
            symbols.add(f"{c}.SZ")
    return {s for s in symbols if not s.startswith("399")}


def _candidate_score(annual_return_pct: float, max_drawdown_pct: float, active_day_ratio: float) -> float:
    return float(float(annual_return_pct) - 0.75 * abs(float(max_drawdown_pct)) - 8.0 * float(active_day_ratio))


def _simulate_flags(
    market_ret_1d: pd.Series,
    crash_lookback_days: int,
    crash_drop_threshold: float,
    rebound_lookback_days: int,
    rebound_threshold: float,
    protection_days: int,
) -> pd.DataFrame:
    mret = pd.to_numeric(market_ret_1d, errors="coerce").fillna(0.0)
    lb = max(3, int(crash_lookback_days))
    rebound_lb = max(1, int(rebound_lookback_days))
    protect_days = max(1, int(protection_days))

    crash_lb_ret = (1.0 + mret).rolling(lb, min_periods=lb).apply(np.prod, raw=True) - 1.0
    rebound_lb_ret = (1.0 + mret).rolling(rebound_lb, min_periods=rebound_lb).apply(np.prod, raw=True) - 1.0
    trigger = ((crash_lb_ret <= float(crash_drop_threshold)) & (rebound_lb_ret >= float(rebound_threshold))).fillna(False)
    trigger = trigger.astype(bool)

    active_flags: list[bool] = []
    left = 0
    for is_trigger in trigger.tolist():
        if bool(is_trigger):
            left = protect_days
        active = left > 0
        active_flags.append(active)
        if left > 0:
            left -= 1

    return pd.DataFrame(
        {
            "crash_lb_ret": crash_lb_ret,
            "rebound_lb_ret": rebound_lb_ret,
            "trigger": trigger.astype(int),
            "active": pd.Series(active_flags, dtype=bool),
        }
    )


def _extract_windows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    d = pd.to_datetime(df["date"])
    active = pd.to_numeric(df["active"], errors="coerce").fillna(0).astype(bool).reset_index(drop=True)
    trigger = pd.to_numeric(df.get("trigger", 0), errors="coerce").fillna(0).astype(int).gt(0).reset_index(drop=True)
    crash = pd.to_numeric(df.get("crash_lb_ret", np.nan), errors="coerce").reset_index(drop=True)
    rebound = pd.to_numeric(df.get("rebound_lb_ret", np.nan), errors="coerce").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    i = 0
    wid = 0
    n = len(active)
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
        rows.append(
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
    for k in range(len(rows)):
        if k + 1 < len(rows):
            end_dt = pd.to_datetime(rows[k]["end_date"])
            next_start = pd.to_datetime(rows[k + 1]["start_date"])
            rows[k]["recovery_gap_days_to_next_window"] = int(max(0, (next_start - end_dt).days - 1))
        else:
            rows[k]["recovery_gap_days_to_next_window"] = None
    return rows


def _run_replay_validation(
    base_dir: Path,
    config_arg: str,
    replay_start_date: str,
    replay_watchlist: str,
    replay_dynamic_watchlist: bool,
    replay_dynamic_top_n: int,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[str]], str]:
    if not candidates:
        return [], {}, {}, "no_candidates"

    bt = _load_backtest_module(base_dir)
    cfg = bt._load_cfg_from_yaml(base_dir, config_path=str(config_arg or "").strip() or None)
    start_date = str(replay_start_date).strip() or "2020-01-01"

    feats = bt.load_and_prepare_features(
        base_dir,
        start_date=start_date,
        news_sentiment_lag_days=int(getattr(cfg, "news_sentiment_lag_days", 1)),
    )
    if feats is None or feats.empty:
        return [], {}, {}, "empty_features"

    wl_file = str(replay_watchlist or "").strip()
    if wl_file:
        wl_path = Path(wl_file)
        if not wl_path.is_absolute():
            wl_path = base_dir / wl_path
        if wl_path.exists():
            wl_symbols = _extract_watchlist_symbols(wl_path)
            if wl_symbols:
                feats = feats[feats["symbol"].isin(wl_symbols)].copy()
        if feats.empty:
            return [], {}, {}, "empty_after_watchlist_filter"

    if hasattr(bt, "build_industry_map_from_config"):
        ind_map = bt.build_industry_map_from_config(base_dir, feats["symbol"])
        feats["industry"] = feats["symbol"].map(ind_map).fillna("OTHER")

    daily = bt.precompute_daily_universe(feats, cfg=cfg)
    if daily is None or daily.empty:
        return [], {}, {}, "empty_daily_universe"

    close_price_df = feats.pivot_table(index="date", columns="symbol", values="close").sort_index()
    if "open" in feats.columns:
        open_price_df = feats.pivot_table(index="date", columns="symbol", values="open").sort_index()
    else:
        open_price_df = close_price_df.copy()
    close_returns_df = close_price_df.pct_change(fill_method=None).fillna(0.0)
    open_returns_df = open_price_df.pct_change(fill_method=None).fillna(0.0)

    index_filter = bt.compute_index_filter(base_dir, start_date, cfg)
    allowed_symbols_by_date = None
    if bool(replay_dynamic_watchlist):
        allowed_symbols_by_date = bt.build_dynamic_watchlist_by_date(
            daily_universe=daily,
            max_symbols=max(50, int(replay_dynamic_top_n)),
            rebalance_freq="M",
        )

    rows: list[dict[str, Any]] = []
    windows_map: dict[str, list[dict[str, Any]]] = {}
    recovery_map: dict[str, list[str]] = {}
    date_start_ts = pd.to_datetime(start_date)
    date_end_ts = daily["date"].max()

    for r in candidates:
        dth = float(_safe_float(r.get("crash_drop_threshold", -0.08), -0.08))
        rth = float(_safe_float(r.get("rebound_threshold", 0.03), 0.03))
        pdays = int(_safe_float(r.get("protection_days", 5), 5))
        cap = float(_safe_float(r.get("position_cap", 0.45), 0.45))
        key = f"{dth:.4f}|{rth:.4f}|{pdays}|{cap:.4f}"

        cfg2 = bt.BacktestConfigV3(
            **{
                **asdict(cfg),
                "use_momentum_crash_protection": True,
                "momentum_crash_drop_threshold": dth,
                "momentum_rebound_threshold": rth,
                "momentum_crash_protection_days": int(max(1, pdays)),
                "momentum_crash_position_cap": float(min(1.0, max(0.05, cap))),
            }
        )
        regime_df = bt.compute_market_regime(feats, cfg2)
        exec_mode = str(getattr(cfg2, "execution_price_mode", "close")).strip().lower()
        if exec_mode == "next_open":
            run_price_df = open_price_df
            run_returns_df = open_returns_df
        else:
            run_price_df = close_price_df
            run_returns_df = close_returns_df

        result = bt.run_backtest_v3(
            daily_universe=daily,
            price_df=run_price_df,
            returns_df=run_returns_df,
            cfg=cfg2,
            regime_df=regime_df,
            date_start=date_start_ts,
            date_end=date_end_ts,
            index_filter=index_filter,
            allowed_symbols_by_date=allowed_symbols_by_date,
        )
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        eq2 = result.get("equity_curve", pd.DataFrame()) if isinstance(result, dict) else pd.DataFrame()
        layer = bt.summarize_momentum_crash_layer(cfg=cfg2, regime_df=regime_df, eq=eq2)
        active_ratio = float(_safe_float(layer.get("active_day_ratio", 0.0), 0.0))
        annual = float(_safe_float(metrics.get("annual_return_pct", 0.0), 0.0))
        mdd = float(_safe_float(metrics.get("max_drawdown_pct", 0.0), 0.0))
        shp = float(_safe_float(metrics.get("sharpe", 0.0), 0.0))
        score_replay = _candidate_score(annual_return_pct=annual, max_drawdown_pct=mdd, active_day_ratio=active_ratio)

        rows.append(
            {
                "candidate_key": key,
                "grid_rank": int(_safe_float(r.get("grid_rank", 0), 0.0)),
                "crash_drop_threshold": dth,
                "rebound_threshold": rth,
                "protection_days": int(pdays),
                "position_cap": float(cap),
                "active_day_ratio": active_ratio,
                "trigger_count": int(_safe_float(layer.get("trigger_count", 0), 0.0)),
                "window_count": int(_safe_float(layer.get("window_count", 0), 0.0)),
                "annual_return_pct": annual,
                "max_drawdown_pct": mdd,
                "sharpe": shp,
                "score_proxy": float(_safe_float(r.get("score", 0.0), 0.0)),
                "score_replay": float(score_replay),
                "status": str(layer.get("status", "unknown")),
            }
        )
        windows_map[key] = layer.get("windows", []) if isinstance(layer.get("windows", []), list) else []
        recovery_map[key] = (
            layer.get("recovery_conditions", [])
            if isinstance(layer.get("recovery_conditions", []), list)
            else []
        )

    rows = sorted(rows, key=lambda x: float(x.get("score_replay", -1e12)), reverse=True)
    for i, r in enumerate(rows, start=1):
        r["replay_rank"] = int(i)
    return rows, windows_map, recovery_map, "ok"


def _render_md(payload: dict[str, Any]) -> str:
    reco = payload.get("recommendation", {})
    top_rows = payload.get("grid_top", []) or []
    replay_rows = payload.get("replay_top", []) or []
    recov = payload.get("recovery_conditions", []) or []
    baseline = payload.get("baseline_metrics", {}) or {}
    current = payload.get("current_policy", {}) or {}
    replay_meta = payload.get("replay", {}) or {}
    source = str(payload.get("selection_source", "grid_proxy"))

    lines: list[str] = []
    lines.append(f"# 动量崩盘保护参数校准报告 - {payload.get('as_of', 'N/A')}")
    lines.append("")
    lines.append(f"- 生成时间：{payload.get('generated_at', 'N/A')}")
    lines.append(f"- 模式：`{payload.get('mode', 'unknown')}`")
    lines.append(f"- 推荐来源：`{source}`")
    lines.append("")
    lines.append("## 1. 当前策略口径")
    lines.append(f"- crash_lookback_days: `{current.get('crash_lookback_days', 'N/A')}`")
    lines.append(f"- crash_drop_threshold: `{_safe_float(current.get('crash_drop_threshold', 0.0), 0.0):+.2%}`")
    lines.append(f"- rebound_lookback_days: `{current.get('rebound_lookback_days', 'N/A')}`")
    lines.append(f"- rebound_threshold: `{_safe_float(current.get('rebound_threshold', 0.0), 0.0):+.2%}`")
    lines.append(f"- protection_days: `{current.get('protection_days', 'N/A')}`")
    lines.append(f"- position_cap: `{_safe_float(current.get('position_cap', 1.0), 1.0):.2f}`")
    lines.append("")
    lines.append("## 2. 基线表现")
    lines.append(f"- 年化：`{_safe_float(baseline.get('annual_return_pct', 0.0), 0.0):+.2f}%`")
    lines.append(f"- 最大回撤：`{_safe_float(baseline.get('max_drawdown_pct', 0.0), 0.0):.2f}%`")
    lines.append(f"- Sharpe：`{_safe_float(baseline.get('sharpe', 0.0), 0.0):.3f}`")
    lines.append("")
    lines.append("## 3. 参数网格扫描 Top 10")
    if not top_rows:
        lines.append("- 当前缺少 `market_regime_snapshot`，无法进行完整网格扫描。")
    else:
        lines.append("| drop | rebound | protect_days | cap | active_ratio | annual | max_dd | sharpe | score |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in top_rows[:10]:
            lines.append(
                "| "
                f"{_safe_float(r.get('crash_drop_threshold', 0.0), 0.0):+.2%} | "
                f"{_safe_float(r.get('rebound_threshold', 0.0), 0.0):+.2%} | "
                f"{int(_safe_float(r.get('protection_days', 0), 0))} | "
                f"{_safe_float(r.get('position_cap', 1.0), 1.0):.2f} | "
                f"{_safe_float(r.get('active_day_ratio', 0.0), 0.0) * 100.0:.1f}% | "
                f"{_safe_float(r.get('annual_return_pct', 0.0), 0.0):+.2f}% | "
                f"{_safe_float(r.get('max_drawdown_pct', 0.0), 0.0):.2f}% | "
                f"{_safe_float(r.get('sharpe', 0.0), 0.0):.3f} | "
                f"{_safe_float(r.get('score', 0.0), 0.0):.2f} |"
            )
    lines.append("")
    lines.append("## 4. 二阶段真实回放 Top（如启用）")
    lines.append(
        f"- replay: `enabled={bool(replay_meta.get('enabled', False))}`"
        f", `status={replay_meta.get('status', 'unknown')}`"
        f", `top_k={int(_safe_float(replay_meta.get('top_k', 0), 0))}`"
    )
    if replay_rows:
        lines.append("| replay_rank | grid_rank | drop | rebound | protect_days | cap | annual | max_dd | sharpe | active_ratio | replay_score |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in replay_rows[:10]:
            lines.append(
                "| "
                f"{int(_safe_float(r.get('replay_rank', 0), 0))} | "
                f"{int(_safe_float(r.get('grid_rank', 0), 0))} | "
                f"{_safe_float(r.get('crash_drop_threshold', 0.0), 0.0):+.2%} | "
                f"{_safe_float(r.get('rebound_threshold', 0.0), 0.0):+.2%} | "
                f"{int(_safe_float(r.get('protection_days', 0), 0))} | "
                f"{_safe_float(r.get('position_cap', 1.0), 1.0):.2f} | "
                f"{_safe_float(r.get('annual_return_pct', 0.0), 0.0):+.2f}% | "
                f"{_safe_float(r.get('max_drawdown_pct', 0.0), 0.0):.2f}% | "
                f"{_safe_float(r.get('sharpe', 0.0), 0.0):.3f} | "
                f"{_safe_float(r.get('active_day_ratio', 0.0), 0.0) * 100.0:.1f}% | "
                f"{_safe_float(r.get('score_replay', 0.0), 0.0):.2f} |"
            )
    else:
        lines.append("- 未启用或未完成真实回放，当前推荐基于网格近似评分。")
    lines.append("")

    lines.append("## 5. 推荐参数（候选）")
    lines.append(f"- crash_drop_threshold: `{_safe_float(reco.get('crash_drop_threshold', 0.0), 0.0):+.2%}`")
    lines.append(f"- rebound_threshold: `{_safe_float(reco.get('rebound_threshold', 0.0), 0.0):+.2%}`")
    lines.append(f"- protection_days: `{int(_safe_float(reco.get('protection_days', 0), 0))}`")
    lines.append(f"- position_cap: `{_safe_float(reco.get('position_cap', 1.0), 1.0):.2f}`")
    if "annual_return_pct" in reco:
        lines.append(f"- 推荐年化（对应来源口径）：`{_safe_float(reco.get('annual_return_pct', 0.0), 0.0):+.2f}%`")
    if "max_drawdown_pct" in reco:
        lines.append(f"- 推荐最大回撤（对应来源口径）：`{_safe_float(reco.get('max_drawdown_pct', 0.0), 0.0):.2f}%`")
    if "sharpe" in reco:
        lines.append(f"- 推荐 Sharpe（对应来源口径）：`{_safe_float(reco.get('sharpe', 0.0), 0.0):.3f}`")
    lines.append("")
    lines.append("## 6. 恢复条件")
    if recov:
        for x in recov:
            lines.append(f"- {x}")
    else:
        lines.append("- 暂无恢复条件。")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate momentum-crash protection calibration report.")
    ap.add_argument("--base-dir", default=".", help="project root")
    ap.add_argument("--config", default="", help="config path")
    ap.add_argument("--as-of", default="", help="report date")
    ap.add_argument("--equity", default="data/backtests/backtest_strategy_v3_equity.csv", help="equity csv")
    ap.add_argument(
        "--regime-snapshot",
        default="data/backtests/backtest_strategy_v3_market_regime_snapshot.csv",
        help="market regime snapshot csv",
    )
    ap.add_argument("--drop-thresholds", default="-0.12,-0.10,-0.08,-0.06", help="candidate crash drop thresholds")
    ap.add_argument("--rebound-thresholds", default="0.02,0.03,0.04,0.05", help="candidate rebound thresholds")
    ap.add_argument("--protection-days", default="3,5,7,10", help="candidate protection days")
    ap.add_argument("--position-caps", default="0.30,0.35,0.40,0.45,0.50", help="candidate position caps")
    ap.add_argument("--replay-top-k", type=int, default=3, help="run true backtest replay on top-K proxy candidates (0=disable)")
    ap.add_argument("--replay-start-date", default="", help="start date for replay backtest (default: from config)")
    ap.add_argument("--replay-watchlist", default="", help="watchlist for replay backtest (default: auto cache)")
    ap.add_argument("--replay-dynamic-top-n", type=int, default=300, help="dynamic watchlist top_n for replay")
    ap.add_argument(
        "--replay-dynamic-watchlist",
        dest="replay_dynamic_watchlist",
        action="store_true",
        default=True,
        help="enable dynamic watchlist during replay (default: on)",
    )
    ap.add_argument(
        "--no-replay-dynamic-watchlist",
        dest="replay_dynamic_watchlist",
        action="store_false",
        help="disable dynamic watchlist during replay",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    eq_path = base_dir / args.equity
    regime_path = base_dir / args.regime_snapshot
    if not eq_path.exists():
        raise FileNotFoundError(f"missing equity file: {eq_path}")

    cfg_raw = _load_cfg(base_dir, args.config)
    risk_cfg = cfg_raw.get("risk_control", {}) if isinstance(cfg_raw, dict) else {}
    mom_cfg = risk_cfg.get("momentum_crash_protection", {}) if isinstance(risk_cfg, dict) else {}
    current_policy = {
        "enabled": bool(mom_cfg.get("enabled", False)),
        "crash_lookback_days": int(_safe_float(mom_cfg.get("crash_lookback_days", 8), 8.0)),
        "crash_drop_threshold": float(_safe_float(mom_cfg.get("crash_drop_threshold", -0.08), -0.08)),
        "rebound_lookback_days": int(_safe_float(mom_cfg.get("rebound_lookback_days", 3), 3.0)),
        "rebound_threshold": float(_safe_float(mom_cfg.get("rebound_threshold", 0.03), 0.03)),
        "protection_days": int(_safe_float(mom_cfg.get("protection_days", 5), 5.0)),
        "position_cap": float(_safe_float(mom_cfg.get("position_cap", 0.45), 0.45)),
    }

    eq = pd.read_csv(eq_path)
    eq["date"] = pd.to_datetime(eq["date"])
    eq["daily_return"] = pd.to_numeric(eq.get("daily_return", 0.0), errors="coerce").fillna(0.0)
    eq = eq.sort_values("date").reset_index(drop=True)
    nav = (1.0 + eq["daily_return"]).cumprod()
    baseline_metrics = {
        "annual_return_pct": _annual_return(nav) * 100.0,
        "max_drawdown_pct": _max_drawdown(nav) * 100.0,
        "sharpe": _sharpe(eq["daily_return"]),
    }

    mode = "grid"
    grid_rows: list[dict[str, Any]] = []
    grid_df_out = pd.DataFrame()
    replay_rows: list[dict[str, Any]] = []
    reco: dict[str, Any] = {
        "crash_drop_threshold": float(current_policy["crash_drop_threshold"]),
        "rebound_threshold": float(current_policy["rebound_threshold"]),
        "protection_days": int(current_policy["protection_days"]),
        "position_cap": float(current_policy["position_cap"]),
    }
    selection_source = "grid_proxy"
    trigger_log_df = pd.DataFrame()
    recovery_conditions: list[str] = []
    replay_info = {
        "enabled": bool(int(args.replay_top_k) > 0),
        "top_k": int(max(0, int(args.replay_top_k))),
        "status": "not_run",
        "start_date": _resolve_replay_start_date(cfg_raw, args.replay_start_date),
        "watchlist": _resolve_replay_watchlist(base_dir, args.replay_watchlist),
        "dynamic_watchlist": bool(args.replay_dynamic_watchlist),
        "dynamic_top_n": int(max(50, int(args.replay_dynamic_top_n))),
    }

    if regime_path.exists():
        rr = pd.read_csv(regime_path)
        rr["date"] = pd.to_datetime(rr["date"])
        rr["market_ret_1d"] = pd.to_numeric(rr.get("market_ret_1d", 0.0), errors="coerce").fillna(0.0)
        merged = eq.merge(rr, on="date", how="inner")
        if merged.empty:
            mode = "fallback_no_overlap"
        else:
            current_active = pd.to_numeric(merged.get("momentum_crash_active", False), errors="coerce").fillna(0).astype(bool)
            current_cap_ser = pd.to_numeric(
                merged.get("momentum_crash_position_cap", current_policy["position_cap"]), errors="coerce"
            ).fillna(float(current_policy["position_cap"]))
            current_eff = np.where(current_active, current_cap_ser.clip(lower=0.05, upper=1.0), 1.0)
            proxy_ret = merged["daily_return"].values / current_eff
            proxy_ret = np.clip(proxy_ret, -0.20, 0.20)

            drops = _parse_float_list(args.drop_thresholds, [-0.12, -0.10, -0.08, -0.06])
            rebounds = _parse_float_list(args.rebound_thresholds, [0.02, 0.03, 0.04, 0.05])
            p_days = _parse_int_list(args.protection_days, [3, 5, 7, 10])
            caps = _parse_float_list(args.position_caps, [0.30, 0.35, 0.40, 0.45, 0.50])
            crash_lb = int(current_policy["crash_lookback_days"])
            rebound_lb = int(current_policy["rebound_lookback_days"])

            for dth in drops:
                for rth in rebounds:
                    for pdays in p_days:
                        flags = _simulate_flags(
                            market_ret_1d=merged["market_ret_1d"],
                            crash_lookback_days=crash_lb,
                            crash_drop_threshold=float(dth),
                            rebound_lookback_days=rebound_lb,
                            rebound_threshold=float(rth),
                            protection_days=int(pdays),
                        )
                        active = pd.to_numeric(flags["active"], errors="coerce").fillna(0).astype(bool).values
                        trigger_count = int(pd.to_numeric(flags["trigger"], errors="coerce").fillna(0).sum())
                        active_days = int(np.sum(active))
                        active_ratio = float(active_days / max(1, len(active)))
                        for cap in caps:
                            eff = np.where(active, float(cap), 1.0)
                            sim_ret = proxy_ret * eff
                            nav = pd.Series(1.0 + sim_ret).cumprod()
                            annual = _annual_return(nav) * 100.0
                            mdd = _max_drawdown(nav) * 100.0
                            shp = _sharpe(pd.Series(sim_ret))
                            score = _candidate_score(
                                annual_return_pct=float(annual),
                                max_drawdown_pct=float(mdd),
                                active_day_ratio=float(active_ratio),
                            )
                            grid_rows.append(
                                {
                                    "crash_drop_threshold": float(dth),
                                    "rebound_threshold": float(rth),
                                    "protection_days": int(pdays),
                                    "position_cap": float(cap),
                                    "trigger_count": int(trigger_count),
                                    "active_days": int(active_days),
                                    "active_day_ratio": float(active_ratio),
                                    "annual_return_pct": float(annual),
                                    "max_drawdown_pct": float(mdd),
                                    "sharpe": float(shp),
                                    "score": float(score),
                                }
                            )

            grid_df = pd.DataFrame(grid_rows).sort_values(
                [
                    "score",
                    "crash_drop_threshold",
                    "rebound_threshold",
                    "protection_days",
                    "position_cap",
                ],
                ascending=[False, True, True, True, True],
            ).reset_index(drop=True)
            if not grid_df.empty:
                grid_df_out = grid_df.copy()
                grid_df["grid_rank"] = np.arange(1, len(grid_df) + 1)
                baseline_annual = float(baseline_metrics["annual_return_pct"])
                baseline_mdd = float(baseline_metrics["max_drawdown_pct"])
                feasible = grid_df[
                    (grid_df["annual_return_pct"] >= baseline_annual - 5.0)
                    & (grid_df["max_drawdown_pct"] >= baseline_mdd - 3.0)
                    & (grid_df["active_day_ratio"] <= 0.40)
                ]
                choose_proxy = feasible.iloc[0] if not feasible.empty else grid_df.iloc[0]
                reco = {
                    "crash_drop_threshold": float(choose_proxy["crash_drop_threshold"]),
                    "rebound_threshold": float(choose_proxy["rebound_threshold"]),
                    "protection_days": int(choose_proxy["protection_days"]),
                    "position_cap": float(choose_proxy["position_cap"]),
                    "annual_return_pct": float(choose_proxy["annual_return_pct"]),
                    "max_drawdown_pct": float(choose_proxy["max_drawdown_pct"]),
                    "sharpe": float(choose_proxy["sharpe"]),
                    "trigger_count": int(choose_proxy["trigger_count"]),
                    "active_day_ratio": float(choose_proxy["active_day_ratio"]),
                    "score": float(choose_proxy["score"]),
                }

                reco_flags = _simulate_flags(
                    market_ret_1d=merged["market_ret_1d"],
                    crash_lookback_days=crash_lb,
                    crash_drop_threshold=float(reco["crash_drop_threshold"]),
                    rebound_lookback_days=rebound_lb,
                    rebound_threshold=float(reco["rebound_threshold"]),
                    protection_days=int(reco["protection_days"]),
                )
                log_df = pd.DataFrame(
                    {
                        "date": merged["date"],
                        "crash_lb_ret": pd.to_numeric(reco_flags["crash_lb_ret"], errors="coerce"),
                        "rebound_lb_ret": pd.to_numeric(reco_flags["rebound_lb_ret"], errors="coerce"),
                        "trigger": pd.to_numeric(reco_flags["trigger"], errors="coerce").fillna(0).astype(int),
                        "active": pd.to_numeric(reco_flags["active"], errors="coerce").fillna(0).astype(bool),
                    }
                )
                proxy_windows = _extract_windows(log_df)
                trigger_log_df = pd.DataFrame(proxy_windows)
                gaps = [int(x["recovery_gap_days_to_next_window"]) for x in proxy_windows if x["recovery_gap_days_to_next_window"] is not None]
                hold_n = max(2, min(6, int(reco["protection_days"])))
                recovery_conditions = [
                    f"连续 {hold_n} 天无新触发。",
                    f"crash_lb_ret 回升至 {float(reco['crash_drop_threshold']) * 0.5:+.2%} 以上。",
                    f"rebound_lb_ret 回落到 {float(reco['rebound_threshold']):+.2%} 以下。",
                ]
                if gaps:
                    recovery_conditions.append(
                        f"历史恢复间隔中位数 {int(np.median(gaps))} 天，P75={int(np.percentile(gaps, 75))} 天。"
                    )

                if bool(replay_info.get("enabled", False)):
                    top_k = int(replay_info.get("top_k", 0))
                    top_candidates = grid_df.head(max(1, top_k)).to_dict(orient="records")
                    try:
                        replay_rows, replay_windows_map, replay_recovery_map, replay_status = _run_replay_validation(
                            base_dir=base_dir,
                            config_arg=args.config,
                            replay_start_date=str(replay_info.get("start_date", "")),
                            replay_watchlist=str(replay_info.get("watchlist", "")),
                            replay_dynamic_watchlist=bool(replay_info.get("dynamic_watchlist", True)),
                            replay_dynamic_top_n=int(replay_info.get("dynamic_top_n", 300)),
                            candidates=top_candidates,
                        )
                        replay_info["status"] = replay_status
                        if replay_rows:
                            mode = "grid+replay"
                            replay_df = pd.DataFrame(replay_rows).sort_values("score_replay", ascending=False).reset_index(drop=True)
                            replay_feasible = replay_df[
                                (replay_df["annual_return_pct"] >= baseline_annual - 5.0)
                                & (replay_df["max_drawdown_pct"] >= baseline_mdd - 3.0)
                                & (replay_df["active_day_ratio"] <= 0.40)
                            ]
                            choose_replay = replay_feasible.iloc[0] if not replay_feasible.empty else replay_df.iloc[0]
                            reco = {
                                "crash_drop_threshold": float(choose_replay["crash_drop_threshold"]),
                                "rebound_threshold": float(choose_replay["rebound_threshold"]),
                                "protection_days": int(choose_replay["protection_days"]),
                                "position_cap": float(choose_replay["position_cap"]),
                                "annual_return_pct": float(choose_replay["annual_return_pct"]),
                                "max_drawdown_pct": float(choose_replay["max_drawdown_pct"]),
                                "sharpe": float(choose_replay["sharpe"]),
                                "trigger_count": int(choose_replay["trigger_count"]),
                                "active_day_ratio": float(choose_replay["active_day_ratio"]),
                                "score_replay": float(choose_replay["score_replay"]),
                                "score_proxy": float(choose_replay.get("score_proxy", 0.0)),
                            }
                            selection_source = "replay_backtest"
                            k = str(choose_replay.get("candidate_key", ""))
                            chosen_windows = replay_windows_map.get(k, [])
                            if chosen_windows:
                                trigger_log_df = pd.DataFrame(chosen_windows)
                            replay_recovery = replay_recovery_map.get(k, [])
                            if replay_recovery:
                                recovery_conditions = replay_recovery
                        else:
                            replay_info["status"] = "empty_replay_rows"
                    except Exception as e:
                        replay_info["status"] = f"error: {e}"
            else:
                mode = "fallback_empty_grid"
    else:
        mode = "fallback_missing_snapshot"

    if str(mode).startswith("fallback"):
        selection_source = "current_policy_fallback"
        # Fallback mode: only use existing equity active windows.
        active_col = "momentum_crash_protect_active"
        if active_col in eq.columns:
            tmp = pd.DataFrame(
                {
                    "date": eq["date"],
                    "active": pd.to_numeric(eq[active_col], errors="coerce").fillna(0).astype(bool),
                    "trigger": 0,
                    "crash_lb_ret": np.nan,
                    "rebound_lb_ret": np.nan,
                }
            )
            windows = _extract_windows(tmp)
            trigger_log_df = pd.DataFrame(windows)
        if not recovery_conditions:
            recovery_conditions = [
                "缺少 market_regime_snapshot，无法做完整参数网格回放。",
                "请先运行一次 run_backtest_strategy_v3.py 生成 market_regime_snapshot 后再校准。",
            ]

    as_of = _infer_as_of(base_dir, args.as_of)
    if not grid_df_out.empty:
        tmp = grid_df_out.copy()
        if "grid_rank" not in tmp.columns:
            tmp["grid_rank"] = np.arange(1, len(tmp) + 1)
        top_rows = tmp.head(20).to_dict(orient="records")
    else:
        top_rows_all = sorted(grid_rows, key=lambda x: float(x.get("score", -1e9)), reverse=True)
        for i, r in enumerate(top_rows_all, start=1):
            r["grid_rank"] = int(i)
        top_rows = top_rows_all[:20]
    replay_info["candidate_count"] = int(len(replay_rows))
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": as_of,
        "mode": mode,
        "selection_source": selection_source,
        "inputs": {
            "equity_path": str(eq_path),
            "regime_snapshot_path": str(regime_path),
        },
        "current_policy": current_policy,
        "baseline_metrics": baseline_metrics,
        "replay": replay_info,
        "grid_top": top_rows,
        "replay_top": replay_rows[:20],
        "recommendation": reco,
        "recovery_conditions": recovery_conditions,
    }

    backtests_dir = base_dir / "data" / "backtests"
    reports_dir = base_dir / "data" / "reports"
    backtests_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_out = backtests_dir / "momentum_crash_calibration.json"
    grid_out = backtests_dir / "momentum_crash_calibration_grid.csv"
    replay_out = backtests_dir / "momentum_crash_calibration_replay.csv"
    log_out = backtests_dir / "momentum_crash_trigger_log.csv"
    md_out = reports_dir / f"momentum_crash_calibration_{as_of}.md"
    md_latest_out = reports_dir / "momentum_crash_calibration_latest.md"

    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(top_rows).to_csv(grid_out, index=False, encoding="utf-8-sig")
    if replay_rows:
        pd.DataFrame(replay_rows).to_csv(replay_out, index=False, encoding="utf-8-sig")
    elif replay_out.exists():
        replay_out.unlink()
    if trigger_log_df is not None and not trigger_log_df.empty:
        trigger_log_df.to_csv(log_out, index=False, encoding="utf-8-sig")
    elif log_out.exists():
        log_out.unlink()

    md = _render_md(payload)
    md_out.write_text(md, encoding="utf-8")
    md_latest_out.write_text(md, encoding="utf-8")

    print("Momentum crash calibration report generated.")
    print(f"json  : {json_out}")
    print(f"grid  : {grid_out}")
    print(f"replay: {replay_out}")
    print(f"log   : {log_out}")
    print(f"md    : {md_out}")
    print(f"latest: {md_latest_out}")


if __name__ == "__main__":
    main()
