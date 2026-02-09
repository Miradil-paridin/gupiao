from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None


# -----------------------------
# Config helpers
# -----------------------------
def _normalize_code(code: str) -> str:
    c = str(code).strip().upper()
    if c.endswith(".SH") or c.endswith(".SZ"):
        return c
    if c.startswith("688") or c.startswith("6"):
        return c + ".SH"
    return c + ".SZ"


def load_watchlist(base_dir: Path) -> list[str]:
    cfg_path = base_dir / "config.yaml"
    if not cfg_path.exists():
        return []
    if yaml is None:
        print("❌ 缺少依赖 pyyaml：pip install pyyaml", file=sys.stderr)
        return []
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    wl = cfg.get("watchlist", []) or []
    return [_normalize_code(x) for x in wl]


# -----------------------------
# Score model (same as yours)
# -----------------------------
def _zscore(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    mu = x.mean()
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sd


@dataclass(frozen=True)
class ScoreParams:
    # weights for (ma, r20, r60, vol, atr, vr)
    w: tuple[float, float, float, float, float, float] = (2.0, 1.0, 0.5, -1.0, -0.5, 0.3)
    zclip: float = 6.0


@dataclass(frozen=True)
class PortfolioParams:
    top_k: int = 3
    cost_bps: float = 10.0
    risk_vol_20d_threshold: float = 0.55  # filter very high vol names


def compute_scores_for_day(day: pd.DataFrame, sp: ScoreParams) -> pd.DataFrame:
    need = ["symbol", "date", "close", "ma_dist_20", "ret_20d", "ret_60d", "vol_20d", "atr_14", "vol_ratio_20"]
    miss = [c for c in need if c not in day.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    w_ma, w_r20, w_r60, w_vol, w_atr, w_vr = sp.w
    d = day.copy()
    d["atr_pct"] = d["atr_14"] / d["close"]

    z_ma = _zscore(d["ma_dist_20"]).clip(-sp.zclip, sp.zclip)
    z_r20 = _zscore(d["ret_20d"]).clip(-sp.zclip, sp.zclip)
    z_r60 = _zscore(d["ret_60d"]).clip(-sp.zclip, sp.zclip)
    z_vol = _zscore(d["vol_20d"]).clip(-sp.zclip, sp.zclip)
    z_atr = _zscore(d["atr_pct"]).clip(-sp.zclip, sp.zclip)
    z_vr = _zscore(d["vol_ratio_20"]).clip(-sp.zclip, sp.zclip)

    d["score"] = (
        w_ma * z_ma +
        w_r20 * z_r20 +
        w_r60 * z_r60 +
        w_vol * z_vol +
        w_atr * z_atr +
        w_vr * z_vr
    )

    d["code"] = d["symbol"].astype(str).str.upper()
    out = d[["date", "code", "score", "vol_20d"]].copy()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["date", "code", "score"])
    # stable sort: score desc, code asc
    out = out.sort_values(["score", "code"], ascending=[False, True])
    return out


# -----------------------------
# Data loading & returns
# -----------------------------
def load_features(base_dir: Path) -> pd.DataFrame:
    p = base_dir / "data" / "features" / "features_daily.parquet"
    if not p.exists():
        raise FileNotFoundError(f"features not found: {p}")
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def add_next_day_return(feats: pd.DataFrame) -> pd.DataFrame:
    # ret1 aligned to signal date: return of next trading day (close-to-close approx)
    f = feats.sort_values(["symbol", "date"]).copy()
    f["close_next"] = f.groupby("symbol")["close"].shift(-1)
    f["ret1"] = (f["close_next"] / f["close"]) - 1.0
    return f.dropna(subset=["ret1"])


# -----------------------------
# Portfolio construction & backtest
# -----------------------------
def build_weights_for_date(scores_day: pd.DataFrame, pp: PortfolioParams) -> pd.DataFrame:
    d = scores_day.copy()
    # risk filter
    d = d[d["vol_20d"] < pp.risk_vol_20d_threshold]
    if d.empty:
        return pd.DataFrame(columns=["date", "code", "w"])
    pick = d.head(pp.top_k).copy()
    k = len(pick)
    if k <= 0:
        return pd.DataFrame(columns=["date", "code", "w"])
    pick["w"] = 1.0 / k
    pick = pick[["date", "code", "w"]]
    return pick


def backtest_from_features(
    feats: pd.DataFrame,
    sp: ScoreParams,
    pp: PortfolioParams,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    f = feats.copy()
    if start:
        s = pd.to_datetime(start).date()
        f = f[f["date"] >= s]
    if end:
        e = pd.to_datetime(end).date()
        f = f[f["date"] <= e]
    if f.empty:
        raise ValueError("No rows after date filtering.")

    f = add_next_day_return(f)

    dates = sorted(f["date"].unique())
    weights_list = []

    for d in dates:
        day = f[f["date"] == d]
        sc = compute_scores_for_day(day, sp)
        w = build_weights_for_date(sc, pp)
        if not w.empty:
            weights_list.append(w)

    if not weights_list:
        raise ValueError("No weights generated (maybe risk filter too strict or too few symbols).")

    weights = pd.concat(weights_list, ignore_index=True)
    weights["date"] = pd.to_datetime(weights["date"]).dt.strftime("%Y-%m-%d")

    # merge returns
    rets = f[["date", "symbol", "ret1"]].copy()
    rets.rename(columns={"symbol": "code"}, inplace=True)
    rets["date"] = pd.to_datetime(rets["date"]).dt.strftime("%Y-%m-%d")

    df = weights.merge(rets, on=["date", "code"], how="left")
    df["ret1"] = df["ret1"].fillna(0.0)

    # portfolio gross return by date
    gross = df.groupby("date", as_index=False).apply(lambda x: float((x["w"] * x["ret1"]).sum()))
    gross.columns = ["date", "gross_return"]

    # turnover-based cost
    wmat = weights.pivot_table(index="date", columns="code", values="w", fill_value=0.0).sort_index()
    turnover = wmat.diff().abs().sum(axis=1) * 0.5
    turnover = turnover.reset_index()
    turnover.columns = ["date", "turnover"]

    out = gross.merge(turnover, on="date", how="left")
    out["turnover"] = out["turnover"].fillna(0.0)
    out["cost"] = out["turnover"] * (pp.cost_bps / 10000.0)
    out["daily_return"] = out["gross_return"] - out["cost"]
    out["nav"] = (1.0 + out["daily_return"]).cumprod()
    return out


# -----------------------------
# Metrics
# -----------------------------
def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = (nav / peak) - 1.0
    return float(dd.min())


def calc_metrics(bt: pd.DataFrame, rf_annual: float = 0.0) -> dict:
    r = bt["daily_return"].astype(float)
    nav = bt["nav"].astype(float)
    n = len(r)
    if n < 10:
        # too short -> metrics not stable
        return {
            "n_days": n,
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_dd": float("nan"),
            "calmar": float("nan"),
        }

    ann = 252.0
    ann_ret = float(nav.iloc[-1] ** (ann / n) - 1.0)
    ann_vol = float(r.std(ddof=1) * math.sqrt(ann))
    rf_daily = rf_annual / ann
    denom = r.std(ddof=1)
    sharpe = float(((r.mean() - rf_daily) / denom) * math.sqrt(ann)) if denom > 1e-12 else float("nan")
    mdd = max_drawdown(nav)
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 else float("inf")
    return {
        "n_days": n,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": mdd,
        "calmar": calmar,
    }


# -----------------------------
# Regime analysis (market proxy = equal-weight of universe)
# -----------------------------
def regime_table(feats: pd.DataFrame, bt: pd.DataFrame) -> pd.DataFrame:
    f = add_next_day_return(feats)
    # market proxy: equal-weight average ret1 across symbols each day
    m = f.groupby("date", as_index=False)["ret1"].mean()
    m["date"] = pd.to_datetime(m["date"]).dt.strftime("%Y-%m-%d")
    m = m.sort_values("date")
    m["m_ret60"] = (1.0 + m["ret1"]).rolling(60).apply(np.prod, raw=True) - 1.0
    m["m_vol20"] = m["ret1"].rolling(20).std(ddof=1)

    # label regimes
    # bull/bear by 60d return sign; high-vol by 20d vol above its median
    vol_med = float(m["m_vol20"].median(skipna=True))
    def lab(row):
        if not np.isfinite(row["m_ret60"]) or not np.isfinite(row["m_vol20"]):
            return "warmup"
        trend = "bull" if row["m_ret60"] > 0 else "bear"
        hv = "highvol" if row["m_vol20"] > vol_med else "lowvol"
        return f"{trend}_{hv}"

    m["regime"] = m.apply(lab, axis=1)

    x = bt.merge(m[["date", "regime"]], on="date", how="left")
    g = x.groupby("regime", as_index=False).agg(
        n=("daily_return", "count"),
        mean=("daily_return", "mean"),
        vol=("daily_return", "std"),
        sharpe=("daily_return", lambda s: (s.mean() / (s.std(ddof=1) + 1e-12)) * math.sqrt(252.0)),
        ann_return=("nav", lambda s: float(s.iloc[-1] ** (252.0 / len(s)) - 1.0) if len(s) > 10 else float("nan")),
    )
    return g.sort_values("n", ascending=False)


# -----------------------------
# Robust evaluations
# -----------------------------
def cost_stress(feats: pd.DataFrame, sp: ScoreParams, pp: PortfolioParams, costs: list[float], start: str | None, end: str | None) -> pd.DataFrame:
    rows = []
    for c in costs:
        pp2 = PortfolioParams(top_k=pp.top_k, cost_bps=float(c), risk_vol_20d_threshold=pp.risk_vol_20d_threshold)
        bt = backtest_from_features(feats, sp, pp2, start=start, end=end)
        met = calc_metrics(bt, rf_annual=0.0)
        rows.append({"cost_bps": c, **met})
    return pd.DataFrame(rows).sort_values("cost_bps")


def param_perturbation(
    feats: pd.DataFrame,
    sp: ScoreParams,
    pp: PortfolioParams,
    n: int = 50,
    sigma_w: float = 0.15,
    seed: int = 7,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.array(sp.w, dtype=float)
    rows = []
    for i in range(n):
        w = base + rng.normal(0.0, sigma_w, size=base.shape)
        sp2 = ScoreParams(w=tuple(map(float, w.tolist())), zclip=sp.zclip)
        bt = backtest_from_features(feats, sp2, pp, start=start, end=end)
        met = calc_metrics(bt, rf_annual=0.0)
        rows.append({"i": i, "w": str(tuple(round(x, 3) for x in w.tolist())), **met})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def walk_forward(
    feats: pd.DataFrame,
    start: str,
    end: str,
    train_days: int = 252,
    test_days: int = 21,
    step_days: int = 21,
    grid_topk: list[int] = [2, 3, 5],
    grid_cost_bps: list[float] = [10.0],
    grid_risk_vol: list[float] = [0.50, 0.55, 0.60],
) -> pd.DataFrame:
    # prepare date list (signal dates)
    f = feats.copy()
    s = pd.to_datetime(start).date()
    e = pd.to_datetime(end).date()
    f = f[(f["date"] >= s) & (f["date"] <= e)]
    if f.empty:
        raise ValueError("No rows in walk-forward date range.")

    dates = sorted(f["date"].unique())
    dates_str = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates]

    def backtest_range(d1: str, d2: str, sp: ScoreParams, pp: PortfolioParams) -> dict:
        bt = backtest_from_features(f, sp, pp, start=d1, end=d2)
        met = calc_metrics(bt, rf_annual=0.0)
        met["nav_end"] = float(bt["nav"].iloc[-1])
        return met

    rows = []
    # rolling
    i0 = train_days
    while i0 < len(dates_str):
        train_start = dates_str[max(0, i0 - train_days)]
        train_end = dates_str[i0 - 1]
        test_start = dates_str[i0]
        test_end = dates_str[min(len(dates_str) - 1, i0 + test_days - 1)]

        best = None
        best_cfg = None

        # small grid search (keep it sane)
        for topk in grid_topk:
            for cost in grid_cost_bps:
                for rv in grid_risk_vol:
                    sp = ScoreParams()  # keep score weights fixed here (grid explodes otherwise)
                    pp = PortfolioParams(top_k=topk, cost_bps=cost, risk_vol_20d_threshold=rv)
                    met = backtest_range(train_start, train_end, sp, pp)
                    key = met.get("sharpe", float("nan"))
                    if not np.isfinite(key):
                        continue
                    if (best is None) or (key > best):
                        best = key
                        best_cfg = (topk, cost, rv, met)

        if best_cfg is None:
            # couldn't pick anything -> skip window
            i0 += step_days
            continue

        topk, cost, rv, train_met = best_cfg
        sp = ScoreParams()
        pp = PortfolioParams(top_k=topk, cost_bps=cost, risk_vol_20d_threshold=rv)
        test_met = backtest_range(test_start, test_end, sp, pp)

        rows.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "pick_topk": topk,
            "pick_cost_bps": cost,
            "pick_risk_vol": rv,
            "train_sharpe": train_met.get("sharpe"),
            "test_sharpe": test_met.get("sharpe"),
            "test_ann_return": test_met.get("ann_return"),
            "test_max_dd": test_met.get("max_dd"),
            "test_days": test_met.get("n_days"),
        })

        i0 += step_days

    return pd.DataFrame(rows)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", help="project root")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")

    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--risk-vol", type=float, default=0.55)

    ap.add_argument("--do-cost-stress", action="store_true")
    ap.add_argument("--cost-list", default="0,5,10,20,50")

    ap.add_argument("--do-perturb", action="store_true")
    ap.add_argument("--perturb-n", type=int, default=50)
    ap.add_argument("--perturb-sigma", type=float, default=0.15)

    ap.add_argument("--do-walk-forward", action="store_true")
    ap.add_argument("--train-days", type=int, default=252)
    ap.add_argument("--test-days", type=int, default=21)
    ap.add_argument("--step-days", type=int, default=21)

    ap.add_argument("--do-regime", action="store_true")

    args = ap.parse_args()
    base_dir = Path(args.base_dir).resolve()

    feats = load_features(base_dir)

    wl = load_watchlist(base_dir)
    if wl:
        feats = feats[feats["symbol"].isin([x.upper() for x in wl])].copy()

    sp = ScoreParams()
    pp = PortfolioParams(top_k=args.topk, cost_bps=args.cost_bps, risk_vol_20d_threshold=args.risk_vol)

    out_dir = base_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # baseline backtest
    bt = backtest_from_features(feats, sp, pp, start=args.start, end=args.end)
    met = calc_metrics(bt, rf_annual=0.0)
    bt_path = out_dir / f"robust_bt_top{pp.top_k}_cost{pp.cost_bps:g}_rv{pp.risk_vol_20d_threshold:g}.csv"
    bt.to_csv(bt_path, index=False, encoding="utf-8-sig")
    print(f"✅ baseline backtest saved: {bt_path}")
    print("baseline metrics:", met)

    # cost stress
    if args.do_cost_stress:
        costs = [float(x) for x in str(args.cost_list).split(",") if str(x).strip()]
        cs = cost_stress(feats, sp, pp, costs, start=args.start, end=args.end)
        cs_path = out_dir / "robust_cost_stress.csv"
        cs.to_csv(cs_path, index=False, encoding="utf-8-sig")
        print(f"✅ cost stress saved: {cs_path}")

    # param perturbation
    if args.do_perturb:
        pr = param_perturbation(
            feats, sp, pp,
            n=args.perturb_n,
            sigma_w=args.perturb_sigma,
            seed=7,
            start=args.start,
            end=args.end
        )
        pr_path = out_dir / "robust_param_perturb.csv"
        pr.to_csv(pr_path, index=False, encoding="utf-8-sig")
        print(f"✅ perturbation saved: {pr_path}")

    # walk-forward
    if args.do_walk_forward:
        if not args.start or not args.end:
            raise ValueError("walk-forward 需要 --start 和 --end 指定区间")
        wf = walk_forward(
            feats,
            start=args.start,
            end=args.end,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
        )
        wf_path = out_dir / "robust_walk_forward.csv"
        wf.to_csv(wf_path, index=False, encoding="utf-8-sig")
        print(f"✅ walk-forward saved: {wf_path}")

    # regime analysis
    if args.do_regime:
        rg = regime_table(feats, bt)
        rg_path = out_dir / "robust_regime.csv"
        rg.to_csv(rg_path, index=False, encoding="utf-8-sig")
        print(f"✅ regime table saved: {rg_path}")


if __name__ == "__main__":
    main()
