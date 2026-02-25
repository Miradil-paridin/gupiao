from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

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


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_float_list(raw: str, default_vals: list[float]) -> list[float]:
    t = str(raw or "").strip()
    if not t:
        return sorted(set(float(x) for x in default_vals))
    vals: list[float] = []
    for part in t.replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            vals.append(float(p))
        except Exception:
            continue
    if not vals:
        vals = list(default_vals)
    return sorted(set(float(x) for x in vals))


def _infer_as_of(base_dir: Path, as_of_arg: str) -> str:
    t = str(as_of_arg or "").strip()
    if t:
        return t
    rank_path = base_dir / "data" / "signals" / "latest_daily_rank.csv"
    if rank_path.exists():
        try:
            df = pd.read_csv(rank_path, nrows=1)
            if not df.empty and "date" in df.columns:
                d = str(df["date"].iloc[0]).strip()
                if d:
                    return d
        except Exception:
            pass
    return date.today().isoformat()


def _load_base_cap(base_dir: Path, config_path_arg: str, summary: dict[str, Any]) -> float:
    cfg_candidates: list[Path] = []
    t = str(config_path_arg or "").strip()
    if t:
        p = Path(t)
        if not p.is_absolute():
            p = base_dir / p
        cfg_candidates.append(p)
    cfg_candidates.extend([base_dir / "config.yaml", base_dir / "config_v31.yaml"])

    for p in cfg_candidates:
        if not p.exists():
            continue
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            strategy = raw.get("strategy", {}) if isinstance(raw, dict) else {}
            cap = _safe_float(strategy.get("max_total_position"), default=float("nan"))
            if math.isfinite(cap) and 0 < cap <= 1.0:
                return float(cap)
        except Exception:
            continue

    monitor = summary.get("failure_monitor", {}) if isinstance(summary, dict) else {}
    mon_action = str((monitor or {}).get("action", "")).strip().lower()
    mon_cap = _safe_float((monitor or {}).get("suggested_position_cap"), default=float("nan"))
    if mon_action == "reduce" and math.isfinite(mon_cap):
        est = mon_cap / 0.65
        if 0 < est <= 1.0:
            return float(est)
    return 1.0


def _scan_rolling_thresholds(
    rolling_df: pd.DataFrame,
    sharpe_thresholds: list[float],
    drawdown_floors: list[float],
    annual_return_floor: float,
) -> pd.DataFrame:
    if rolling_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    ann = pd.to_numeric(rolling_df.get("annual_return_pct", 0.0), errors="coerce")
    dd = pd.to_numeric(rolling_df.get("max_drawdown_pct", 0.0), errors="coerce")
    sharpe = pd.to_numeric(rolling_df.get("sharpe", 0.0), errors="coerce")
    valid = ann.notna() & dd.notna() & sharpe.notna()
    if valid.sum() <= 0:
        return pd.DataFrame()

    ann = ann[valid]
    dd = dd[valid]
    sharpe = sharpe[valid]
    total = int(len(ann))

    for s in sharpe_thresholds:
        for d in drawdown_floors:
            pass_mask = (ann > float(annual_return_floor)) & (sharpe >= float(s)) & (dd >= float(d))
            pass_count = int(pass_mask.sum())
            pass_ratio = float(pass_count / total) if total > 0 else 0.0
            if pass_count > 0:
                pass_ann_mean = _safe_float(ann[pass_mask].mean(), 0.0)
                pass_ann_median = _safe_float(ann[pass_mask].median(), 0.0)
                pass_worst_dd = _safe_float(dd[pass_mask].min(), 0.0)
                pass_sharpe_mean = _safe_float(sharpe[pass_mask].mean(), 0.0)
                score = pass_ann_mean - 0.80 * abs(pass_worst_dd) + 8.0 * pass_ratio
            else:
                pass_ann_mean = float("nan")
                pass_ann_median = float("nan")
                pass_worst_dd = float("nan")
                pass_sharpe_mean = float("nan")
                score = -1e9
            rows.append(
                {
                    "min_window_sharpe": float(s),
                    "max_window_drawdown_floor_pct": float(d),
                    "annual_return_floor_pct": float(annual_return_floor),
                    "window_count": int(total),
                    "pass_count": int(pass_count),
                    "pass_ratio": float(pass_ratio),
                    "pass_mean_annual_return_pct": float(pass_ann_mean),
                    "pass_median_annual_return_pct": float(pass_ann_median),
                    "pass_worst_drawdown_pct": float(pass_worst_dd),
                    "pass_mean_sharpe": float(pass_sharpe_mean),
                    "score": float(score),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "pass_ratio"], ascending=[False, False]).reset_index(drop=True)


def _pick_reco_threshold_row(grid: pd.DataFrame, pass_ratio_target: float, min_pass_windows: int) -> dict[str, Any]:
    if grid.empty:
        return {}
    pass_target = float(_clamp(pass_ratio_target, 0.05, 0.95))
    min_pass = max(1, int(min_pass_windows))

    c1 = grid[(grid["pass_ratio"] >= pass_target) & (grid["pass_count"] >= min_pass)]
    if not c1.empty:
        return dict(c1.iloc[0].to_dict())

    c2 = grid[grid["pass_count"] >= min_pass]
    if not c2.empty:
        return dict(c2.iloc[0].to_dict())

    return dict(grid.iloc[0].to_dict())


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
    return float(rr.mean() / sd * (252.0 ** 0.5))


def _monitor_window_status(recent_returns: pd.Series, baseline: dict[str, Any]) -> dict[str, Any]:
    rec_ann = _safe_annual_from_returns(recent_returns) * 100.0
    rec_sharpe = _sharpe_from_returns(recent_returns)
    rec_dd = _max_drawdown_from_returns(recent_returns) * 100.0

    b_ann = _safe_float(baseline.get("annual_return_pct"), 0.0)
    b_sharpe = _safe_float(baseline.get("sharpe"), 0.0)
    b_dd_abs = abs(_safe_float(baseline.get("max_drawdown_pct"), 0.0))
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
        "recent_annual_return_pct": float(rec_ann),
        "recent_sharpe": float(rec_sharpe),
        "recent_max_drawdown_pct": float(rec_dd),
    }


def _monitor_action(short_status: str, long_status: str, base_cap: float) -> tuple[str, float]:
    severe_count = int(sum(1 for s in [short_status, long_status] if s == "degraded_severe"))
    warn_or_worse_count = int(sum(1 for s in [short_status, long_status] if s in {"degraded_warning", "degraded_severe"}))
    base_cap = float(_clamp(base_cap, 0.01, 1.0))
    if severe_count >= 2:
        return "stop", 0.0
    if severe_count >= 1 and warn_or_worse_count >= 2:
        return "reduce_hard", float(min(base_cap * 0.40, 0.35))
    if warn_or_worse_count >= 2:
        return "reduce", float(min(base_cap * 0.65, 0.60))
    return "normal", float(base_cap)


def _replay_monitor(
    eq_df: pd.DataFrame,
    baseline: dict[str, Any],
    short_days: int,
    long_days: int,
    base_cap: float,
) -> pd.DataFrame:
    if eq_df.empty or "daily_return" not in eq_df.columns:
        return pd.DataFrame()

    short_days = max(20, int(short_days))
    long_days = max(short_days, int(long_days))
    if len(eq_df) < long_days:
        return pd.DataFrame()

    r = pd.to_numeric(eq_df["daily_return"], errors="coerce").fillna(0.0)
    dates = eq_df.get("date", pd.Series(range(len(eq_df)), dtype=object)).astype(str)

    rows: list[dict[str, Any]] = []
    for i in range(long_days - 1, len(r)):
        short_slice = r.iloc[max(0, i - short_days + 1) : i + 1]
        long_slice = r.iloc[i - long_days + 1 : i + 1]
        s_eval = _monitor_window_status(short_slice, baseline)
        l_eval = _monitor_window_status(long_slice, baseline)
        action, cap = _monitor_action(
            short_status=str(s_eval["status"]),
            long_status=str(l_eval["status"]),
            base_cap=base_cap,
        )
        rows.append(
            {
                "date": str(dates.iloc[i]),
                "short_status": str(s_eval["status"]),
                "long_status": str(l_eval["status"]),
                "action": str(action),
                "suggested_position_cap": float(cap),
                "long_recent_annual_return_pct": float(l_eval["recent_annual_return_pct"]),
                "long_recent_sharpe": float(l_eval["recent_sharpe"]),
                "long_recent_max_drawdown_pct": float(l_eval["recent_max_drawdown_pct"]),
            }
        )
    return pd.DataFrame(rows)


def _monitor_summary(
    monitor_hist: pd.DataFrame,
    gate_policy: dict[str, Any],
) -> dict[str, Any]:
    if monitor_hist.empty:
        return {
            "window_count": 0,
            "action_counts": {},
            "action_ratios": {},
            "suggested_warn_cap_min": _safe_float(gate_policy.get("warn_cap_min"), 0.60),
            "suggested_warn_cap_max": _safe_float(gate_policy.get("warn_cap_max"), 0.80),
            "note": "insufficient_data",
        }

    action_counts = monitor_hist["action"].value_counts().to_dict()
    total = max(1, int(len(monitor_hist)))
    action_ratios = {str(k): float(v / total) for k, v in action_counts.items()}

    reduce_caps = pd.to_numeric(
        monitor_hist.loc[monitor_hist["action"].isin(["reduce", "reduce_hard"]), "suggested_position_cap"],
        errors="coerce",
    ).dropna()
    if not reduce_caps.empty:
        q25 = _safe_float(reduce_caps.quantile(0.25), 0.55)
        q75 = _safe_float(reduce_caps.quantile(0.75), 0.70)
        warn_cap_min = _clamp(q25 - 0.02, 0.35, 0.85)
        warn_cap_max = _clamp(max(warn_cap_min + 0.08, q75 + 0.02), 0.45, 0.90)
    else:
        warn_cap_min = _safe_float(gate_policy.get("warn_cap_min"), 0.60)
        warn_cap_max = _safe_float(gate_policy.get("warn_cap_max"), 0.80)

    if warn_cap_min > warn_cap_max:
        warn_cap_min, warn_cap_max = warn_cap_max, warn_cap_min

    return {
        "window_count": int(total),
        "action_counts": {str(k): int(v) for k, v in action_counts.items()},
        "action_ratios": action_ratios,
        "suggested_warn_cap_min": float(round(warn_cap_min, 3)),
        "suggested_warn_cap_max": float(round(warn_cap_max, 3)),
        "reduce_cap_q25": _safe_float(reduce_caps.quantile(0.25), float("nan")) if not reduce_caps.empty else None,
        "reduce_cap_q50": _safe_float(reduce_caps.quantile(0.50), float("nan")) if not reduce_caps.empty else None,
        "reduce_cap_q75": _safe_float(reduce_caps.quantile(0.75), float("nan")) if not reduce_caps.empty else None,
    }


def _overfit_sensitivity(
    overfit: dict[str, Any],
    current_max_pbo: float,
    current_min_dsr: float,
    pbo_candidates: list[float],
    dsr_candidates: list[float],
) -> dict[str, Any]:
    pbo = _safe_float(overfit.get("pbo"), float("nan"))
    dsr = _safe_float(overfit.get("dsr"), float("nan"))
    pbo_details = overfit.get("pbo_details", {}) if isinstance(overfit.get("pbo_details", {}), dict) else {}

    pbo_status = str(pbo_details.get("status", "")).strip().lower()
    pbo_fold_count = int(_safe_float(pbo_details.get("fold_count"), 0.0))
    pbo_model_count = int(_safe_float(pbo_details.get("model_count"), 0.0))
    pbo_reliable = bool(math.isfinite(pbo) and pbo_status == "ok" and pbo_fold_count >= 4 and pbo_model_count >= 6)
    dsr_available = bool(math.isfinite(dsr))

    pbo_rows: list[dict[str, Any]] = []
    for t in pbo_candidates:
        if pbo_reliable:
            passed = bool(pbo <= float(t))
            mode = "check"
        else:
            passed = True
            mode = "skip_unreliable"
        pbo_rows.append(
            {
                "max_pbo_threshold": float(t),
                "pbo_value": float(pbo) if math.isfinite(pbo) else None,
                "pbo_reliable": bool(pbo_reliable),
                "evaluation_mode": mode,
                "passed": bool(passed),
                "margin": float(t - pbo) if (math.isfinite(pbo) and pbo_reliable) else None,
            }
        )

    dsr_rows: list[dict[str, Any]] = []
    for t in dsr_candidates:
        if dsr_available:
            passed = bool(dsr >= float(t))
            margin = float(dsr - t)
        else:
            passed = True
            margin = None
        dsr_rows.append(
            {
                "min_dsr_threshold": float(t),
                "dsr_value": float(dsr) if dsr_available else None,
                "dsr_available": bool(dsr_available),
                "passed": bool(passed),
                "margin": margin,
            }
        )

    if pbo_reliable:
        passing_pbo = [r["max_pbo_threshold"] for r in pbo_rows if r["passed"]]
        reco_pbo = float(min(passing_pbo)) if passing_pbo else float(current_max_pbo)
    else:
        reco_pbo = float(current_max_pbo)

    if dsr_available:
        passing_dsr = [r["min_dsr_threshold"] for r in dsr_rows if r["passed"]]
        reco_dsr = float(max(passing_dsr)) if passing_dsr else float(current_min_dsr)
    else:
        reco_dsr = float(current_min_dsr)

    return {
        "pbo_value": float(pbo) if math.isfinite(pbo) else None,
        "dsr_value": float(dsr) if dsr_available else None,
        "pbo_reliable": bool(pbo_reliable),
        "pbo_reliability_details": {
            "status": pbo_status or "unknown",
            "fold_count": int(pbo_fold_count),
            "model_count": int(pbo_model_count),
        },
        "pbo_table": pbo_rows,
        "dsr_table": dsr_rows,
        "recommended_max_pbo": float(round(reco_pbo, 3)),
        "recommended_min_dsr": float(round(reco_dsr, 3)),
    }


def _fmt_pct(v: Any, digits: int = 2) -> str:
    x = _safe_float(v, float("nan"))
    if not math.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}%"


def _fmt_num(v: Any, digits: int = 3) -> str:
    x = _safe_float(v, float("nan"))
    if not math.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}"


def _render_md(
    as_of: str,
    summary: dict[str, Any],
    grid: pd.DataFrame,
    rolling_reco: dict[str, Any],
    monitor_sum: dict[str, Any],
    overfit_sum: dict[str, Any],
    calibrated_gate: dict[str, Any],
) -> str:
    gate_policy = summary.get("gate_policy", {}) if isinstance(summary, dict) else {}
    baseline = summary.get("baseline", {}) if isinstance(summary, dict) else {}
    top_grid = grid.head(10).copy() if not grid.empty else pd.DataFrame()

    lines: list[str] = []
    lines.append(f"# 闸门阈值校准报告 - {as_of}")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Rolling 窗口数：{int(_safe_float(summary.get('rolling', {}).get('window_count', 0), 0.0))}")
    lines.append(f"- 当前 quality_gate：`{str((summary.get('quality_gate', {}) or {}).get('status', 'unknown'))}`")
    lines.append(f"- 当前 trade_gate：`{str((summary.get('trade_gate', {}) or {}).get('status', 'unknown'))}`")
    lines.append("")
    lines.append("## 1. 基线摘要")
    lines.append(f"- 年化：`{_fmt_pct(baseline.get('annual_return_pct', 0.0))}`")
    lines.append(f"- 最大回撤：`{_fmt_pct(baseline.get('max_drawdown_pct', 0.0))}`")
    lines.append(f"- Sharpe：`{_fmt_num(baseline.get('sharpe', 0.0))}`")
    lines.append("")
    lines.append("## 2. Rolling 阈值扫描（Top 10）")
    if top_grid.empty:
        lines.append("- 无可用 rolling 数据。")
    else:
        lines.append("| min_sharpe | dd_floor_pct | pass_ratio | pass_mean_ann | pass_worst_dd | score |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for _, r in top_grid.iterrows():
            lines.append(
                "| "
                f"{_fmt_num(r.get('min_window_sharpe'), 2)} | "
                f"{_fmt_num(r.get('max_window_drawdown_floor_pct'), 1)} | "
                f"{_fmt_num(100.0 * _safe_float(r.get('pass_ratio'), 0.0), 1)}% | "
                f"{_fmt_num(r.get('pass_mean_annual_return_pct'), 2)}% | "
                f"{_fmt_num(r.get('pass_worst_drawdown_pct'), 2)}% | "
                f"{_fmt_num(r.get('score'), 2)} |"
            )
    lines.append("")
    lines.append("## 3. 交易闸门回放（monitor replay）")
    action_ratios = monitor_sum.get("action_ratios", {}) if isinstance(monitor_sum, dict) else {}
    if not action_ratios:
        lines.append("- 无可用 monitor replay 数据。")
    else:
        ratio_text = ", ".join([f"{k}={_fmt_num(100.0 * _safe_float(v, 0.0), 1)}%" for k, v in action_ratios.items()])
        lines.append(f"- action 分布：{ratio_text}")
        lines.append(
            f"- 建议 warn cap：`{_fmt_num(monitor_sum.get('suggested_warn_cap_min', 0.60), 3)}`"
            f" ~ `{_fmt_num(monitor_sum.get('suggested_warn_cap_max', 0.80), 3)}`"
        )
    lines.append("")
    lines.append("## 4. PBO/DSR 敏感性")
    lines.append(
        f"- PBO：`{_fmt_num(overfit_sum.get('pbo_value'), 3)}`，"
        f"可靠性：`{bool(overfit_sum.get('pbo_reliable', False))}`"
    )
    lines.append(f"- DSR：`{_fmt_num(overfit_sum.get('dsr_value'), 3)}`")
    lines.append(
        f"- 推荐 `max_pbo`：`{_fmt_num(overfit_sum.get('recommended_max_pbo'), 3)}`，"
        f"`min_dsr`：`{_fmt_num(overfit_sum.get('recommended_min_dsr'), 3)}`"
    )
    lines.append("")
    lines.append("## 5. 推荐生产阈值（候选）")
    lines.append(f"- gate profile：`{gate_policy.get('profile', 'production')}`")
    lines.append(f"- `max_drawdown_floor_pct`: `{_fmt_num(calibrated_gate.get('max_drawdown_floor_pct'), 2)}`")
    lines.append(f"- `min_wf_mean_sharpe`: `{_fmt_num(calibrated_gate.get('min_wf_mean_sharpe'), 2)}`")
    lines.append(f"- `min_wf_sharpe_ok_ratio`: `{_fmt_num(calibrated_gate.get('min_wf_sharpe_ok_ratio'), 2)}`")
    lines.append(f"- `warn_cap_min`: `{_fmt_num(calibrated_gate.get('warn_cap_min'), 3)}`")
    lines.append(f"- `warn_cap_max`: `{_fmt_num(calibrated_gate.get('warn_cap_max'), 3)}`")
    lines.append(f"- `max_pbo`: `{_fmt_num(calibrated_gate.get('max_pbo'), 3)}`")
    lines.append(f"- `min_dsr`: `{_fmt_num(calibrated_gate.get('min_dsr'), 3)}`")
    lines.append("")
    lines.append("建议先在 `research` 配置跑 1-2 周回放，再冻结到生产默认值。")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate gate threshold calibration report from strategy artifacts.")
    ap.add_argument("--base-dir", default=".", help="project root")
    ap.add_argument("--config", default="", help="config path for loading max_total_position")
    ap.add_argument("--as-of", default="", help="report date, e.g. 2026-02-25")
    ap.add_argument("--summary", default="data/backtests/strategy_process_summary.json", help="strategy summary json")
    ap.add_argument("--rolling", default="data/backtests/strategy_process_rolling.csv", help="rolling csv")
    ap.add_argument("--equity", default="data/backtests/backtest_strategy_v3_equity.csv", help="equity csv for monitor replay")
    ap.add_argument(
        "--sharpe-thresholds",
        default="0.80,0.90,1.00,1.10,1.20,1.30",
        help="candidate min sharpe thresholds",
    )
    ap.add_argument(
        "--drawdown-floors",
        default="-25,-22,-20,-18,-16,-14,-12,-10",
        help="candidate max drawdown floors (pct, negative)",
    )
    ap.add_argument("--annual-return-floor", type=float, default=0.0, help="rolling annual return floor (pct)")
    ap.add_argument("--pass-ratio-target", type=float, default=0.60, help="target pass ratio when selecting recommendation")
    ap.add_argument("--min-pass-windows", type=int, default=18, help="minimum passing windows for recommendation")
    ap.add_argument("--pbo-candidates", default="0.35,0.40,0.45,0.50,0.55", help="candidate max_pbo thresholds")
    ap.add_argument("--dsr-candidates", default="0.45,0.50,0.55,0.60,0.65", help="candidate min_dsr thresholds")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    summary_path = base_dir / args.summary
    rolling_path = base_dir / args.rolling
    equity_path = base_dir / args.equity

    summary = _load_json(summary_path)
    if not summary:
        raise FileNotFoundError(f"missing or invalid summary json: {summary_path}")
    if not rolling_path.exists():
        raise FileNotFoundError(f"missing rolling csv: {rolling_path}")

    rolling_df = pd.read_csv(rolling_path)
    if rolling_df.empty:
        raise RuntimeError(f"rolling csv is empty: {rolling_path}")

    gate_policy = summary.get("gate_policy", {}) if isinstance(summary, dict) else {}
    baseline = summary.get("baseline", {}) if isinstance(summary, dict) else {}
    overfit = summary.get("overfit_diagnostics", {}) if isinstance(summary, dict) else {}

    sharpe_thresholds = _parse_float_list(args.sharpe_thresholds, [0.8, 0.9, 1.0, 1.1, 1.2])
    drawdown_floors = _parse_float_list(args.drawdown_floors, [-25, -22, -20, -18, -16, -14, -12, -10])
    pbo_candidates = _parse_float_list(args.pbo_candidates, [0.35, 0.40, 0.45, 0.50, 0.55])
    dsr_candidates = _parse_float_list(args.dsr_candidates, [0.45, 0.50, 0.55, 0.60, 0.65])

    grid = _scan_rolling_thresholds(
        rolling_df=rolling_df,
        sharpe_thresholds=sharpe_thresholds,
        drawdown_floors=drawdown_floors,
        annual_return_floor=float(args.annual_return_floor),
    )
    rolling_reco = _pick_reco_threshold_row(
        grid=grid,
        pass_ratio_target=float(args.pass_ratio_target),
        min_pass_windows=int(args.min_pass_windows),
    )

    base_cap = _load_base_cap(base_dir, args.config, summary)
    monitor_short_days = int(_safe_float(gate_policy.get("monitor_short_days"), 20.0))
    monitor_long_days = int(_safe_float(gate_policy.get("monitor_long_days"), 60.0))
    if equity_path.exists():
        eq_df = pd.read_csv(equity_path)
    else:
        eq_df = pd.DataFrame()
    monitor_hist = _replay_monitor(
        eq_df=eq_df,
        baseline=baseline,
        short_days=monitor_short_days,
        long_days=monitor_long_days,
        base_cap=base_cap,
    )
    monitor_sum = _monitor_summary(monitor_hist, gate_policy)

    current_max_pbo = _safe_float(gate_policy.get("max_pbo"), 0.45)
    current_min_dsr = _safe_float(gate_policy.get("min_dsr"), 0.55)
    overfit_sum = _overfit_sensitivity(
        overfit=overfit,
        current_max_pbo=current_max_pbo,
        current_min_dsr=current_min_dsr,
        pbo_candidates=pbo_candidates,
        dsr_candidates=dsr_candidates,
    )

    reco_dd_floor = _safe_float(
        rolling_reco.get("max_window_drawdown_floor_pct"),
        _safe_float(gate_policy.get("max_drawdown_floor_pct"), -20.0),
    )
    reco_sharpe = _safe_float(
        rolling_reco.get("min_window_sharpe"),
        _safe_float(gate_policy.get("min_wf_mean_sharpe"), 1.0),
    )
    reco_pass_ratio = _safe_float(
        rolling_reco.get("pass_ratio"),
        _safe_float(gate_policy.get("min_wf_sharpe_ok_ratio"), 0.60),
    )

    calibrated_gate = {
        "profile": str(gate_policy.get("profile", "production")),
        "max_drawdown_floor_pct": float(round(reco_dd_floor, 2)),
        "min_wf_mean_sharpe": float(round(_clamp(reco_sharpe * 0.95, 0.60, 1.60), 2)),
        "min_wf_sharpe_ok_ratio": float(round(_clamp(reco_pass_ratio, 0.40, 0.90), 2)),
        "warn_cap_min": float(_safe_float(monitor_sum.get("suggested_warn_cap_min"), 0.60)),
        "warn_cap_max": float(_safe_float(monitor_sum.get("suggested_warn_cap_max"), 0.80)),
        "max_pbo": float(_safe_float(overfit_sum.get("recommended_max_pbo"), current_max_pbo)),
        "min_dsr": float(_safe_float(overfit_sum.get("recommended_min_dsr"), current_min_dsr)),
    }

    as_of = _infer_as_of(base_dir, args.as_of)
    result = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": as_of,
        "inputs": {
            "summary_path": str(summary_path),
            "rolling_path": str(rolling_path),
            "equity_path": str(equity_path),
            "window_count": int(len(rolling_df)),
        },
        "current_gate_policy": gate_policy,
        "rolling_threshold_scan_top": grid.head(20).to_dict(orient="records") if not grid.empty else [],
        "rolling_recommendation": rolling_reco,
        "monitor_replay_summary": monitor_sum,
        "overfit_sensitivity": overfit_sum,
        "recommended_gate_policy": calibrated_gate,
    }

    backtests_dir = base_dir / "data" / "backtests"
    reports_dir = base_dir / "data" / "reports"
    backtests_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = backtests_dir / "strategy_gate_calibration.json"
    grid_path = backtests_dir / "strategy_gate_calibration_grid.csv"
    monitor_path = backtests_dir / "strategy_gate_monitor_replay.csv"
    md_path = reports_dir / f"strategy_gate_calibration_{as_of}.md"
    md_latest_path = reports_dir / "strategy_gate_calibration_latest.md"

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not grid.empty:
        grid.to_csv(grid_path, index=False, encoding="utf-8-sig")
    if not monitor_hist.empty:
        monitor_hist.to_csv(monitor_path, index=False, encoding="utf-8-sig")

    md = _render_md(
        as_of=as_of,
        summary=summary,
        grid=grid,
        rolling_reco=rolling_reco,
        monitor_sum=monitor_sum,
        overfit_sum=overfit_sum,
        calibrated_gate=calibrated_gate,
    )
    md_path.write_text(md, encoding="utf-8")
    md_latest_path.write_text(md, encoding="utf-8")

    print("Gate calibration report generated.")
    print(f"json   : {json_path}")
    print(f"grid   : {grid_path}")
    print(f"monitor: {monitor_path}")
    print(f"md     : {md_path}")
    print(f"latest : {md_latest_path}")


if __name__ == "__main__":
    main()
