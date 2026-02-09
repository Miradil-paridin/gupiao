from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    positions: pd.DataFrame
    stats: dict


def _max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1
    return float(dd.min())


def _annualized_return(equity_final: float, n_days: int) -> float:
    if n_days <= 0 or not np.isfinite(equity_final) or equity_final <= 0:
        return float("nan")
    return float(equity_final ** (252 / n_days) - 1)


def _sharpe(daily_ret: np.ndarray) -> float:
    if len(daily_ret) < 30:
        return float("nan")
    mu = float(np.mean(daily_ret))
    sd = float(np.std(daily_ret, ddof=1))
    return float(mu / sd * np.sqrt(252)) if sd > 1e-12 else float("nan")


def _prep(features: pd.DataFrame, min_history_days: int) -> tuple[pd.Index, list[str], pd.DataFrame]:
    """
    Prepare minimal panel columns for backtesting, and enforce a minimum history length per symbol.
    """
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    need = {"symbol", "date", "open", "close", "ma_20"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"features missing columns: {miss}")

    df["rank"] = df.groupby("symbol")["date"].rank(method="first")
    df = df[df["rank"] >= min_history_days].copy()
    df = df.drop(columns=["rank"])

    df = df[["date", "symbol", "open", "close", "ma_20"]].copy()
    dates = pd.Index(sorted(df["date"].unique()))
    symbols = sorted(df["symbol"].unique().tolist())
    return dates, symbols, df


def backtest_ma20_voltarget_ddstop(
    features: pd.DataFrame,
    min_history_days: int = 60,
    buy_cost_bps: float = 15.0,
    sell_cost_bps: float = 20.0,

    # Vol targeting
    vol_lookback: int = 30,
    target_vol: float = 0.18,     # annualized
    max_leverage: float = 1.0,    # long-only cap
    min_leverage: float = 0.25,   # keep minimum exposure

    # Drawdown regime
    risk_off_leverage: float = 0.25,  # when risk_off, cap leverage to this
    dd_stop: float = -0.35,           # enter risk-off when dd <= dd_stop
    dd_recover: float = -0.20,        # exit risk-off when dd >= dd_recover
    dd_cooldown_days: int = 20,       # prevent flip-flop
) -> BacktestResult:
    """
    Tradability-aware MA20 strategy traded at next-day open, with:
      - volatility targeting (portfolio vol -> target_vol)
      - drawdown risk-off regime (NOT full cash by default)
      - cooldown to reduce regime churning

    Execution:
      - signal computed at day t close: close(t) > ma_20(t)
      - target weights applied at day t+1 open (lagged by 1)
      - returns are open-to-open
      - tradability-aware: if no bar today, cannot trade that symbol

    Notes:
      - This is a deliberately conservative (but realistic) simulator.
      - For small baskets, volatility can be spiky; min_leverage prevents "never invested" behavior.
    """
    dates, symbols, df = _prep(features, min_history_days=min_history_days)

    open_w = df.pivot(index="date", columns="symbol", values="open").reindex(dates)
    close_w = df.pivot(index="date", columns="symbol", values="close").reindex(dates)
    ma20_w = df.pivot(index="date", columns="symbol", values="ma_20").reindex(dates)

    # tradable today if both open and close exist
    tradable = open_w.notna() & close_w.notna()

    # signal at day t close, trade at day t+1 open => lag by 1 for target at day t
    signal_t = (close_w > ma20_w) & ma20_w.notna() & close_w.notna()
    signal_lag = signal_t.shift(1).fillna(False)

    # equal weight among signaled names
    n_sig = signal_lag.sum(axis=1)
    target = signal_lag.div(n_sig.replace(0, np.nan), axis=0).fillna(0.0)

    # open-to-open returns using forward-filled opens (continuity), but trading is limited by tradability
    open_ff = open_w.ffill()
    ret_oo = open_ff.shift(-1) / open_ff - 1
    ret_oo = ret_oo.fillna(0.0)

    # state
    w = pd.Series(0.0, index=symbols, dtype=float)
    equity = 1.0
    peak = 1.0
    risk_off = False
    cooldown = 0
    net_hist: list[float] = []

    rec = []
    pos_records = []

    for i in range(len(dates) - 1):
        d = dates[i]
        trad_today = tradable.loc[d].reindex(symbols).fillna(False)
        tgt_today = target.loc[d].reindex(symbols).fillna(0.0)

        # drawdown
        peak = max(peak, equity)
        dd = equity / peak - 1.0

        if cooldown > 0:
            cooldown -= 1

        # regime transitions with cooldown
        if (not risk_off) and (dd <= dd_stop) and cooldown == 0:
            risk_off = True
            cooldown = dd_cooldown_days
        elif risk_off and (dd >= dd_recover) and cooldown == 0:
            risk_off = False
            cooldown = dd_cooldown_days

        # realized portfolio vol from recent net returns
        if len(net_hist) >= vol_lookback:
            window = np.array(net_hist[-vol_lookback:], dtype=float)
            sd = float(np.std(window, ddof=1))
            rv = float(sd * np.sqrt(252)) if sd > 1e-12 else 0.0
        else:
            rv = 0.0

        # volatility targeting leverage
        if rv > 1e-12:
            lev = target_vol / rv
        else:
            lev = max_leverage

        lev = float(np.clip(lev, min_leverage, max_leverage))

        # apply risk-off cap (not zero by default)
        leverage = min(lev, risk_off_leverage) if risk_off else lev

        # locked holdings: cannot trade on non-tradable symbols
        locked = ~trad_today
        locked_weight = float(w[locked].sum())

        # investable budget on tradable symbols
        investable = max(0.0, (1.0 - locked_weight) * leverage)

        # robust tradable symbol list
        trad_syms = trad_today[trad_today].index.tolist()

        desired = w.copy().astype(float)
        tgt_trad_sum = float(tgt_today.loc[trad_syms].sum()) if trad_syms else 0.0

        if trad_syms and tgt_trad_sum > 0 and investable > 0:
            desired.loc[trad_syms] = tgt_today.loc[trad_syms].astype(float) * (investable / tgt_trad_sum)
        else:
            if trad_syms:
                desired.loc[trad_syms] = 0.0

        # transaction costs on tradable deltas only
        delta = desired - w
        buy_amt = float(delta.loc[trad_syms].clip(lower=0).sum()) if trad_syms else 0.0
        sell_amt = float((-delta.loc[trad_syms]).clip(lower=0).sum()) if trad_syms else 0.0
        cost = buy_amt * (buy_cost_bps / 10000.0) + sell_amt * (sell_cost_bps / 10000.0)

        equity_after_cost = equity * (1.0 - cost)

        # allocate
        vals = desired * equity_after_cost
        cash = equity_after_cost * (1.0 - float(desired.sum()))

        # apply returns
        r = ret_oo.loc[d].reindex(symbols).fillna(0.0)
        vals_next = vals * (1.0 + r)
        equity_next = float(vals_next.sum() + cash)

        net_ret = equity_next / equity - 1.0 if equity > 0 else 0.0
        net_hist.append(net_ret)

        rec.append({
            "date": d,
            "net_ret": net_ret,
            "equity": equity_next,
            "cost": cost,
            "buy_amt": buy_amt,
            "sell_amt": sell_amt,
            "leverage": leverage,
            "risk_off": int(risk_off),
            "cooldown": int(cooldown),
            "drawdown": dd,
            "n_sig": int(n_sig.loc[d]) if d in n_sig.index else 0,
            "rv_ann": rv,   # realized vol (annualized) estimate
        })

        # update weights at next open
        total = equity_next
        if total > 0:
            w = (vals_next / total).astype(float)
        else:
            w[:] = 0.0

        equity = equity_next

        # sparse positions
        nz = w[w.abs() > 1e-9]
        for sym, wt in nz.items():
            pos_records.append({"date": d, "symbol": sym, "weight": float(wt)})

    equity_df = pd.DataFrame(rec)
    positions_df = pd.DataFrame(pos_records)

    stats = {
        "days": int(len(equity_df)),
        "final_equity": float(equity_df["equity"].iloc[-1]) if len(equity_df) else float("nan"),
        "annualized_return": _annualized_return(float(equity_df["equity"].iloc[-1]), int(len(equity_df))) if len(equity_df) else float("nan"),
        "sharpe_approx": _sharpe(equity_df["net_ret"].values),
        "max_drawdown": _max_drawdown(equity_df["equity"]),
        "avg_leverage": float(equity_df["leverage"].mean()) if len(equity_df) else float("nan"),
        "risk_off_days": int(equity_df["risk_off"].sum()) if len(equity_df) else 0,
        "buy_cost_bps": float(buy_cost_bps),
        "sell_cost_bps": float(sell_cost_bps),
        "target_vol": float(target_vol),
        "vol_lookback": int(vol_lookback),
        "max_leverage": float(max_leverage),
        "min_leverage": float(min_leverage),
        "risk_off_leverage": float(risk_off_leverage),
        "dd_stop": float(dd_stop),
        "dd_recover": float(dd_recover),
        "dd_cooldown_days": int(dd_cooldown_days),
    }

    return BacktestResult(equity=equity_df, positions=positions_df, stats=stats)


def save_backtest_outputs(base_dir: Path, result: BacktestResult, name: str) -> Path:
    out_dir = base_dir / "data" / "backtests" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_path = out_dir / "equity_curve.csv"
    pos_path = out_dir / "positions.parquet"
    stats_path = out_dir / "stats.json"

    result.equity.to_csv(equity_path, index=False, encoding="utf-8-sig")
    result.positions.to_parquet(pos_path, index=False)

    import json
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(result.stats, f, ensure_ascii=False, indent=2)

    return out_dir
