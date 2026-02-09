from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def max_drawdown(nav: pd.Series) -> float:
    nav = nav.astype(float)
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="backtest csv path")
    ap.add_argument("--rf", type=float, default=0.02, help="annual risk-free rate, e.g. 0.02 = 2%")
    ap.add_argument("--days", type=int, default=252, help="trading days per year")
    args = ap.parse_args()

    p = Path(args.file).resolve()
    df = pd.read_csv(p)
    need = {"date", "strategy_return", "nav"}
    if not need.issubset(df.columns):
        raise ValueError(f"缺列：需要 {need}，实际 {list(df.columns)}")

    r = pd.to_numeric(df["strategy_return"], errors="coerce").dropna()
    nav = pd.to_numeric(df["nav"], errors="coerce").dropna()
    if len(r) < 30:
        print("⚠️ 样本天数太少（<30），指标会很不稳定。")

    n = len(r)
    nav0, nav1 = float(nav.iloc[0]), float(nav.iloc[-1])
    ann_ret = (nav1 / nav0) ** (args.days / max(n, 1)) - 1.0

    mu = float(r.mean())
    vol = float(r.std(ddof=1))
    ann_vol = vol * np.sqrt(args.days)

    rf_daily = args.rf / args.days
    sharpe = np.nan if vol < 1e-12 else (mu - rf_daily) / vol * np.sqrt(args.days)

    mdd = max_drawdown(nav)

    win = float((r > 0).mean())
    print("======== METRICS ========")
    print(f"file: {p}")
    print(f"days: {n}")
    print(f"nav : {nav1:.4f}")
    print(f"ann_return : {ann_ret:.2%}")
    print(f"ann_vol    : {ann_vol:.2%}")
    print(f"sharpe(rf={args.rf:.2%}): {sharpe:.3f}")
    print(f"max_drawdown: {mdd:.2%}")
    print(f"win_rate    : {win:.2%}")
    print("=========================")


if __name__ == "__main__":
    main()
