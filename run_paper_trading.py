from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass
class PaperTradingConfig:
    initial_cash: float = 100000.0
    lot_size: int = 100
    slippage_bps: float = 5.0
    buy_fee_bps: float = 2.0
    sell_fee_bps: float = 12.0
    sync_portfolio_yaml: bool = False
    rebalance_band: float = 0.01


def _resolve_config_path(base_dir: Path, config_path: str | None = None) -> Path | None:
    if config_path:
        p = Path(config_path)
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


def _load_project_config(base_dir: Path, config_path: str | None = None) -> dict[str, Any]:
    p = _resolve_config_path(base_dir, config_path=config_path)
    if p and p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        print(f"  📂 配置: {p.name}")
        return raw
    return {}


def _build_cfg(
    base_dir: Path,
    initial_cash_override: float | None,
    sync_override: bool | None,
    config_path: str | None = None,
) -> PaperTradingConfig:
    raw = _load_project_config(base_dir, config_path=config_path)
    p = raw.get("paper_trading", {}) or {}
    cfg = PaperTradingConfig(
        initial_cash=float(p.get("initial_cash", 100000.0)),
        lot_size=int(p.get("lot_size", 100)),
        slippage_bps=float(p.get("slippage_bps", 5.0)),
        buy_fee_bps=float(p.get("buy_fee_bps", 2.0)),
        sell_fee_bps=float(p.get("sell_fee_bps", 12.0)),
        sync_portfolio_yaml=bool(p.get("sync_portfolio_yaml", False)),
        rebalance_band=float(p.get("rebalance_band", 0.01)),
    )
    if initial_cash_override is not None:
        cfg.initial_cash = float(initial_cash_override)
    if sync_override is not None:
        cfg.sync_portfolio_yaml = bool(sync_override)
    return cfg


def _to_date(v: Any) -> pd.Timestamp | None:
    if v is None or v == "":
        return None
    try:
        return pd.to_datetime(v).normalize()
    except Exception:
        return None


def _date_str(v: pd.Timestamp | None) -> str | None:
    if v is None:
        return None
    return pd.to_datetime(v).date().isoformat()


def _load_state(state_path: Path, cfg: PaperTradingConfig) -> dict[str, Any]:
    if not state_path.exists():
        return {
            "version": 1,
            "initial_cash": float(cfg.initial_cash),
            "cash": float(cfg.initial_cash),
            "last_signal_date": None,
            "positions": {},
            "pending_orders": [],
        }
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    raw.setdefault("version", 1)
    raw.setdefault("initial_cash", float(cfg.initial_cash))
    raw.setdefault("cash", float(raw.get("initial_cash", cfg.initial_cash)))
    raw.setdefault("last_signal_date", None)
    raw.setdefault("positions", {})
    raw.setdefault("pending_orders", [])
    return raw


def _save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_signals(signals_path: Path) -> pd.DataFrame:
    if not signals_path.exists():
        raise FileNotFoundError(f"signals file not found: {signals_path}")
    df = pd.read_csv(signals_path)
    if df.empty or "date" not in df.columns or "symbol" not in df.columns:
        raise ValueError("latest_daily_rank.csv is empty or missing required columns.")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    if "target_weight" not in df.columns:
        df["target_weight"] = 0.0
    if "action" not in df.columns:
        df["action"] = "HOLD"
    return df


def _load_market_data(market_path: Path) -> pd.DataFrame:
    if not market_path.exists():
        raise FileNotFoundError(f"market file not found: {market_path}")
    df = pd.read_parquet(market_path, columns=["date", "symbol", "open", "close"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "open"])
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def _next_trading_day(calendar: np.ndarray, d: pd.Timestamp) -> pd.Timestamp | None:
    if len(calendar) == 0:
        return None
    idx = int(np.searchsorted(calendar, d.to_datetime64(), side="right"))
    if idx >= len(calendar):
        return None
    return pd.to_datetime(calendar[idx]).normalize()


def _append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _get_target_weights(signal_df: pd.DataFrame) -> dict[str, float]:
    invest = signal_df[signal_df["action"] == "INVEST_MORE"].copy()
    if invest.empty:
        return {}
    invest["target_weight"] = pd.to_numeric(invest["target_weight"], errors="coerce").fillna(0.0)
    invest = invest[invest["target_weight"] > 0]
    if invest.empty:
        return {}
    tgt = dict(zip(invest["symbol"], invest["target_weight"]))
    total = float(sum(tgt.values()))
    if total > 1.0 and total > 1e-12:
        tgt = {k: float(v / total) for k, v in tgt.items()}
    return {k: float(v) for k, v in tgt.items()}


def _clamp01(v: Any, default: float | None = None) -> float | None:
    try:
        x = float(v)
    except Exception:
        return default
    if not np.isfinite(x):
        return default
    return float(min(1.0, max(0.0, x)))


def _load_cap_from_strategy_summary(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    trade_gate = raw.get("trade_gate", {}) or {}
    trade_status = str(trade_gate.get("status", "")).strip().lower()
    trade_action = str(trade_gate.get("action", "")).strip().lower()
    trade_cap = _clamp01(trade_gate.get("position_cap"), default=None)
    if trade_status in {"pass", "warn", "fail"}:
        if trade_status == "fail" or trade_action == "stop":
            return 0.0
        return trade_cap

    gate_status = str((raw.get("quality_gate", {}) or {}).get("status", "")).strip().lower()
    monitor = raw.get("failure_monitor", {}) or {}
    monitor_action = str(monitor.get("action", "")).strip().lower()
    suggested_cap = _clamp01(monitor.get("suggested_position_cap"), default=None)

    if gate_status == "fail" or monitor_action == "stop":
        return 0.0
    return suggested_cap


def _apply_total_position_cap(target_weights: dict[str, float], cap: float | None) -> tuple[dict[str, float], float]:
    cleaned = {str(k): float(v) for k, v in target_weights.items() if float(v) > 0}
    total_before = float(sum(cleaned.values()))
    if cap is None:
        return cleaned, total_before

    cap = _clamp01(cap, default=1.0)
    if cap is None or cap <= 0:
        return {}, total_before
    if total_before <= 1e-12 or total_before <= cap:
        return cleaned, total_before

    scale = float(cap / total_before)
    scaled = {k: float(v * scale) for k, v in cleaned.items()}
    return scaled, total_before


def _schedule_rebalance_orders(
    state: dict[str, Any],
    signal_date: pd.Timestamp,
    target_weights: dict[str, float],
    calendar: np.ndarray,
) -> bool:
    last_sig = _to_date(state.get("last_signal_date"))
    if last_sig is not None and signal_date <= last_sig:
        return False

    symbols = set(state.get("positions", {}).keys()) | set(target_weights.keys())
    if not symbols:
        state["last_signal_date"] = _date_str(signal_date)
        return True

    exec_date = _next_trading_day(calendar, signal_date)
    exec_date_str = _date_str(exec_date)
    for sym in sorted(symbols):
        state["pending_orders"].append(
            {
                "signal_date": _date_str(signal_date),
                "exec_date": exec_date_str,
                "symbol": sym,
                "target_weight": float(target_weights.get(sym, 0.0)),
            }
        )
    state["last_signal_date"] = _date_str(signal_date)
    return True


def _position_snapshot_value(positions: dict[str, Any], price_map: dict[str, float]) -> float:
    total = 0.0
    for sym, pos in positions.items():
        shares = float(pos.get("shares", 0.0))
        if shares <= 0:
            continue
        px = price_map.get(sym)
        if px is None or not np.isfinite(px) or px <= 0:
            px = float(pos.get("last_price", pos.get("avg_cost", 0.0)))
        total += shares * float(px)
    return float(total)


def _execute_rebalance_for_date(
    state: dict[str, Any],
    orders: list[dict[str, Any]],
    exec_date: pd.Timestamp,
    day_df: pd.DataFrame,
    calendar: np.ndarray,
    cfg: PaperTradingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    positions: dict[str, Any] = state["positions"]
    cash = float(state.get("cash", 0.0))

    day = day_df.set_index("symbol")
    open_map = day["open"].to_dict()
    close_map = day["close"].to_dict() if "close" in day.columns else {}

    equity_price_map: dict[str, float] = {}
    for sym in set(positions.keys()) | {str(o["symbol"]) for o in orders}:
        if sym in open_map and np.isfinite(open_map[sym]) and float(open_map[sym]) > 0:
            equity_price_map[sym] = float(open_map[sym])

    equity_base = cash + _position_snapshot_value(positions, equity_price_map)
    if equity_base <= 0:
        equity_base = float(state.get("initial_cash", cfg.initial_cash))

    lot = max(1, int(cfg.lot_size))
    slip = float(cfg.slippage_bps) / 10000.0
    rebalance_band = max(0.0, float(cfg.rebalance_band))

    carry_orders: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    for od in orders:
        sym = str(od["symbol"]).upper()
        px = open_map.get(sym)
        if px is None or not np.isfinite(px) or float(px) <= 0:
            nxt = _next_trading_day(calendar, exec_date)
            od2 = dict(od)
            od2["exec_date"] = _date_str(nxt)
            carry_orders.append(od2)
            continue

        px = float(px)
        target_w = float(od.get("target_weight", 0.0))
        target_w = max(0.0, min(1.0, target_w))
        current_shares = int(float(positions.get(sym, {}).get("shares", 0.0)))
        current_value = current_shares * px
        current_w = float(current_value / max(equity_base, 1e-12))
        if abs(target_w - current_w) < rebalance_band:
            target_w = current_w
        desired_value = equity_base * target_w
        desired_shares = int(np.floor(desired_value / px / lot) * lot)
        delta = desired_shares - current_shares
        plans.append(
            {
                "symbol": sym,
                "signal_date": str(od.get("signal_date", "")),
                "target_weight": target_w,
                "price": px,
                "current_shares": current_shares,
                "desired_shares": desired_shares,
                "delta": int(delta),
            }
        )

    trades: list[dict[str, Any]] = []
    trade_id = 0

    for p in [x for x in plans if x["delta"] < 0]:
        sym = p["symbol"]
        shares_to_sell = int(-p["delta"])
        current = int(float(positions.get(sym, {}).get("shares", 0.0)))
        if current <= 0:
            continue
        if shares_to_sell < lot and p["desired_shares"] == 0:
            shares_to_sell = current
        shares_to_sell = min(shares_to_sell, current)
        if shares_to_sell <= 0:
            continue

        sell_px = float(p["price"]) * (1.0 - slip)
        notional = shares_to_sell * sell_px
        fee = notional * (float(cfg.sell_fee_bps) / 10000.0)
        net = notional - fee
        cash += net

        remaining = current - shares_to_sell
        if remaining <= 0:
            positions.pop(sym, None)
        else:
            pos = positions[sym]
            pos["shares"] = float(remaining)
            pos["last_price"] = float(close_map.get(sym, sell_px))

        trade_id += 1
        trades.append(
            {
                "trade_id": trade_id,
                "date": _date_str(exec_date),
                "signal_date": p["signal_date"],
                "symbol": sym,
                "side": "SELL",
                "shares": int(shares_to_sell),
                "price": round(float(sell_px), 4),
                "notional": round(float(notional), 4),
                "fee": round(float(fee), 4),
                "cash_after": round(float(cash), 4),
                "target_weight": round(float(p["target_weight"]), 6),
            }
        )

    for p in [x for x in plans if x["delta"] > 0]:
        sym = p["symbol"]
        shares_to_buy = int(p["delta"])
        if shares_to_buy < lot:
            continue

        buy_px = float(p["price"]) * (1.0 + slip)
        unit_cost = buy_px * (1.0 + float(cfg.buy_fee_bps) / 10000.0)
        max_buy = int(np.floor(cash / unit_cost / lot) * lot)
        shares = min(shares_to_buy, max_buy)
        if shares < lot:
            continue

        notional = shares * buy_px
        fee = notional * (float(cfg.buy_fee_bps) / 10000.0)
        total_cost = notional + fee
        cash -= total_cost

        pos = positions.get(sym, {"shares": 0.0, "avg_cost": 0.0, "last_price": float(buy_px)})
        old_shares = float(pos.get("shares", 0.0))
        old_avg = float(pos.get("avg_cost", 0.0))
        new_shares = old_shares + float(shares)
        new_avg = (old_shares * old_avg + total_cost) / max(new_shares, 1e-12)
        pos["shares"] = float(new_shares)
        pos["avg_cost"] = float(new_avg)
        pos["last_price"] = float(close_map.get(sym, buy_px))
        positions[sym] = pos

        trade_id += 1
        trades.append(
            {
                "trade_id": trade_id,
                "date": _date_str(exec_date),
                "signal_date": p["signal_date"],
                "symbol": sym,
                "side": "BUY",
                "shares": int(shares),
                "price": round(float(buy_px), 4),
                "notional": round(float(notional), 4),
                "fee": round(float(fee), 4),
                "cash_after": round(float(cash), 4),
                "target_weight": round(float(p["target_weight"]), 6),
            }
        )

    for sym, pos in list(positions.items()):
        if sym in close_map and np.isfinite(close_map[sym]) and float(close_map[sym]) > 0:
            pos["last_price"] = float(close_map[sym])
        elif sym in open_map and np.isfinite(open_map[sym]) and float(open_map[sym]) > 0:
            pos["last_price"] = float(open_map[sym])
        positions[sym] = pos

    pos_value = _position_snapshot_value(positions, {s: float(p.get("last_price", 0.0)) for s, p in positions.items()})
    equity = float(cash + pos_value)
    equity_row = {
        "date": _date_str(exec_date),
        "cash": round(float(cash), 4),
        "position_value": round(float(pos_value), 4),
        "equity": round(float(equity), 4),
        "n_positions": int(sum(1 for p in positions.values() if float(p.get("shares", 0)) > 0)),
        "pending_orders": int(len(carry_orders)),
    }
    state["cash"] = float(cash)
    state["positions"] = positions
    return trades, equity_row, carry_orders


def _execute_due_pending_orders(
    state: dict[str, Any],
    market_df: pd.DataFrame,
    cfg: PaperTradingConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pending = state.get("pending_orders", []) or []
    if not pending:
        return [], []
    if market_df.empty:
        return [], []

    calendar = np.array(sorted(pd.to_datetime(market_df["date"].unique())))
    if len(calendar) == 0:
        return [], []
    max_date = pd.to_datetime(calendar[-1]).normalize()

    by_date: dict[pd.Timestamp, pd.DataFrame] = {
        pd.to_datetime(d).normalize(): g.copy()
        for d, g in market_df.groupby("date", sort=True)
    }

    due_groups: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    remaining: list[dict[str, Any]] = []
    for od in pending:
        sig_d = _to_date(od.get("signal_date"))
        if sig_d is None:
            continue
        exec_d = _to_date(od.get("exec_date"))
        if exec_d is None:
            exec_d = _next_trading_day(calendar, sig_d)
        if exec_d is None or exec_d > max_date:
            od2 = dict(od)
            od2["exec_date"] = _date_str(exec_d)
            remaining.append(od2)
            continue
        due_groups.setdefault(exec_d, []).append(od)

    all_trades: list[dict[str, Any]] = []
    all_equity_rows: list[dict[str, Any]] = []
    for exec_date in sorted(due_groups.keys()):
        day_df = by_date.get(exec_date)
        if day_df is None or day_df.empty:
            for od in due_groups[exec_date]:
                od2 = dict(od)
                od2["exec_date"] = _date_str(_next_trading_day(calendar, exec_date))
                remaining.append(od2)
            continue

        trades, equity_row, carry = _execute_rebalance_for_date(
            state=state,
            orders=due_groups[exec_date],
            exec_date=exec_date,
            day_df=day_df,
            calendar=calendar,
            cfg=cfg,
        )
        all_trades.extend(trades)
        all_equity_rows.append(equity_row)
        remaining.extend(carry)

    state["pending_orders"] = remaining
    return all_trades, all_equity_rows


def _write_portfolio_yaml(path: Path, state: dict[str, Any]) -> None:
    positions = state.get("positions", {}) or {}
    cash = float(state.get("cash", 0.0))
    holdings = {}
    total_pos_value = 0.0
    for sym, pos in positions.items():
        shares = float(pos.get("shares", 0.0))
        if shares <= 0:
            continue
        px = float(pos.get("last_price", pos.get("avg_cost", 0.0)))
        val = shares * px
        total_pos_value += val
        holdings[sym] = {
            "shares": int(round(shares)),
            "cost": round(float(pos.get("avg_cost", 0.0)), 4),
            "last_price": round(px, 4),
            "market_value": round(val, 4),
        }
    equity = cash + total_pos_value
    for sym in list(holdings.keys()):
        holdings[sym]["weight"] = round(holdings[sym]["market_value"] / max(equity, 1e-12), 6)

    payload = {
        "updated": pd.Timestamp.now().date().isoformat(),
        "cash": round(cash, 4),
        "equity": round(equity, 4),
        "total_positions": len(holdings),
        "holdings": holdings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-trading ledger from latest signal ranking.")
    ap.add_argument("--base-dir", default=".", help="project root")
    ap.add_argument("--config", default="", help="config path (default: config.yaml, fallback: config_v31.yaml)")
    ap.add_argument("--initial-cash", type=float, default=None, help="override initial cash on first run")
    ap.add_argument("--sync-portfolio", action="store_true", help="also write data/portfolio.yaml from paper state")
    ap.add_argument("--no-sync-portfolio", action="store_true", help="do not write data/portfolio.yaml")
    ap.add_argument(
        "--max-total-position-cap",
        type=float,
        default=None,
        help="cap total target exposure in [0,1], e.g. 0.92",
    )
    ap.add_argument(
        "--strategy-summary",
        default="",
        help="optional strategy_process_summary.json path for fallback cap",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    sync_override = None
    if args.sync_portfolio:
        sync_override = True
    if args.no_sync_portfolio:
        sync_override = False
    cfg = _build_cfg(
        base_dir,
        initial_cash_override=args.initial_cash,
        sync_override=sync_override,
        config_path=(str(args.config).strip() or None),
    )

    signals_path = base_dir / "data" / "signals" / "latest_daily_rank.csv"
    market_path = base_dir / "data" / "clean" / "market_daily_all.parquet"
    state_path = base_dir / "data" / "paper" / "state.json"
    trades_path = base_dir / "data" / "paper" / "trades.csv"
    equity_path = base_dir / "data" / "paper" / "equity.csv"
    paper_portfolio_path = base_dir / "data" / "paper_portfolio.yaml"
    live_portfolio_path = base_dir / "data" / "portfolio.yaml"

    signal_df = _load_signals(signals_path)
    signal_date = pd.to_datetime(signal_df["date"].max()).normalize()

    # 信号新鲜度检查
    days_stale = (pd.Timestamp.now().normalize() - signal_date).days
    if days_stale > 5:
        print(f"  ⚠ 信号过期 {days_stale} 天 (signal_date={signal_date.strftime('%Y-%m-%d')})，建议先运行 run_make_daily_rank.py")
    elif days_stale > 2:
        print(f"  ⚠ 信号已 {days_stale} 天未更新 (signal_date={signal_date.strftime('%Y-%m-%d')})")

    signal_df = signal_df[signal_df["date"] == signal_date].copy()
    target_weights_raw = _get_target_weights(signal_df)

    # 诊断信息
    invest_more = signal_df[signal_df.get("action", pd.Series()) == "INVEST_MORE"] if "action" in signal_df.columns else pd.DataFrame()
    print(f"  📊 信号诊断: 日期={signal_date.strftime('%Y-%m-%d')}, 总信号={len(signal_df)}, "
          f"INVEST_MORE={len(invest_more)}, 有权重={sum(1 for w in target_weights_raw.values() if w > 0)}")

    summary_cap = None
    if str(args.strategy_summary).strip():
        summary_path = Path(str(args.strategy_summary).strip())
        if not summary_path.is_absolute():
            summary_path = (base_dir / summary_path).resolve()
        summary_cap = _load_cap_from_strategy_summary(summary_path)

    cli_cap = _clamp01(args.max_total_position_cap, default=None)
    if cli_cap is not None:
        effective_cap = cli_cap
        cap_source = "cli"
    elif summary_cap is not None:
        effective_cap = summary_cap
        cap_source = "summary"
    else:
        effective_cap = None
        cap_source = "none"

    target_weights, raw_total_target = _apply_total_position_cap(target_weights_raw, effective_cap)
    final_total_target = float(sum(target_weights.values()))

    market_df = _load_market_data(market_path)
    calendar = np.array(sorted(pd.to_datetime(market_df["date"].unique())))

    state = _load_state(state_path, cfg)
    scheduled = _schedule_rebalance_orders(state, signal_date=signal_date, target_weights=target_weights, calendar=calendar)
    trades, equity_rows = _execute_due_pending_orders(state, market_df=market_df, cfg=cfg)
    _save_state(state_path, state)

    _append_csv(
        trades_path,
        trades,
        ["trade_id", "date", "signal_date", "symbol", "side", "shares", "price", "notional", "fee", "cash_after", "target_weight"],
    )
    _append_csv(
        equity_path,
        equity_rows,
        ["date", "cash", "position_value", "equity", "n_positions", "pending_orders"],
    )

    _write_portfolio_yaml(paper_portfolio_path, state)
    if cfg.sync_portfolio_yaml:
        _write_portfolio_yaml(live_portfolio_path, state)

    print("=" * 60)
    print("Paper Trading Ledger")
    print("=" * 60)
    print(f"signal_date: {_date_str(signal_date)}")
    print(f"scheduled_new_signal: {'yes' if scheduled else 'no'}")
    print(f"target_symbols: {len(target_weights)}")
    print(f"target_exposure_raw: {raw_total_target:.4f}")
    print(f"target_exposure_final: {final_total_target:.4f}")
    print(f"position_cap_source: {cap_source}")
    print(f"rebalance_band: {float(cfg.rebalance_band):.4f}")
    if effective_cap is not None:
        print(f"position_cap: {float(effective_cap):.4f}")
    print(f"executed_trades: {len(trades)}")
    print(f"pending_orders: {len(state.get('pending_orders', []))}")
    print(f"cash: {float(state.get('cash', 0.0)):.2f}")
    pos_count = sum(1 for p in (state.get('positions', {}) or {}).values() if float(p.get('shares', 0.0)) > 0)
    print(f"positions: {pos_count}")
    print(f"state: {state_path}")
    print(f"trades: {trades_path}")
    print(f"equity: {equity_path}")
    print(f"paper_portfolio: {paper_portfolio_path}")
    if cfg.sync_portfolio_yaml:
        print(f"portfolio_synced: {live_portfolio_path}")


if __name__ == "__main__":
    main()
