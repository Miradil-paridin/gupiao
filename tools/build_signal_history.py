from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None


@dataclass
class SignalConfig:
    # 只是预留：你后面要做 withdraw / 风控标签可以加进来
    withdraw_score_threshold: float = -0.5
    risk_vol_20d_threshold: float = 0.55


# ----------------------------
# Utilities
# ----------------------------
def _zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    mu = x.mean()
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sd


def _normalize_code(code: str) -> str:
    """支持 config.yaml 写 '600519'，自动补成 '600519.SH' / '000001.SZ' """
    c = str(code).strip().upper()
    if not c:
        return c
    if c.endswith(".SH") or c.endswith(".SZ"):
        return c
    # 6 / 688 -> 上交所
    if c.startswith("6") or c.startswith("688"):
        return c + ".SH"
    # 0/2/3 -> 深交所（含 300）
    return c + ".SZ"


def _parse_weights(s: str) -> tuple[float, float, float, float, float, float]:
    """
    解析 weights 字符串： "2,1,0.5,-1,-0.5,0.3"
    顺序：ma_dist_20, ret_20d, ret_60d, vol_20d, atr_pct, vol_ratio_20
    """
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    if len(parts) != 6:
        raise ValueError(f"weights 需要 6 个数，如: 2,1,0.5,-1,-0.5,0.3  你给的是: {s}")
    vals = tuple(float(x) for x in parts)
    return vals  # type: ignore


def _ensure_date_str(df: pd.DataFrame, col: str = "date") -> pd.Series:
    return pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")


# ----------------------------
# Load config + data
# ----------------------------
def load_config_watchlist(base_dir: Path) -> list[str]:
    cfg_path = base_dir / "config.yaml"
    if not cfg_path.exists():
        return []
    if yaml is None:
        print("❌ 缺少依赖 pyyaml：请先 pip install pyyaml", file=sys.stderr)
        sys.exit(2)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    wl = cfg.get("watchlist", []) or []
    wl_norm = [_normalize_code(x) for x in wl if str(x).strip() != ""]
    # 去重且稳定
    wl_norm = sorted(set(wl_norm))
    return wl_norm


def load_features(base_dir: Path) -> pd.DataFrame:
    p = base_dir / "data" / "features" / "features_daily.parquet"
    if not p.exists():
        raise FileNotFoundError(f"features not found: {p}")
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        raise RuntimeError(
            f"读取 parquet 失败：{p}\n"
            f"常见原因：没装 pyarrow。\n"
            f"解决：在当前虚拟环境执行 pip install pyarrow\n\n"
            f"原始错误：{repr(e)}"
        ) from e

    df["date"] = pd.to_datetime(df["date"]).dt.date
    # 统一 symbol 格式
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    return df


# ----------------------------
# Score computation (one day)
# ----------------------------
def compute_score_for_one_day(
    day: pd.DataFrame,
    weights: tuple[float, float, float, float, float, float] = (2.0, 1.0, 0.5, -1.0, -0.5, 0.3),
    zclip: float = 6.0,
) -> pd.DataFrame:
    """
    输入：某一天的 features（多股票）
    输出：date, code, score
    注意：score 是横截面 zscore 组合，所以同一天必须一起算（不能按股票单独增量）
    """
    need = [
        "symbol", "date", "close",
        "ma_dist_20", "ret_20d", "ret_60d",
        "vol_20d", "atr_14", "vol_ratio_20"
    ]
    miss = [c for c in need if c not in day.columns]
    if miss:
        raise ValueError(f"Missing columns for score: {miss}")

    w_ma, w_r20, w_r60, w_vol, w_atr, w_vr = weights

    day = day.copy()
    day["close"] = pd.to_numeric(day["close"], errors="coerce")
    day["atr_14"] = pd.to_numeric(day["atr_14"], errors="coerce")
    day = day.dropna(subset=["close", "atr_14"])
    if day.empty:
        # 这一日全是坏数据，返回空
        return pd.DataFrame(columns=["date", "code", "score"])

    day["atr_pct"] = day["atr_14"] / day["close"]

    z_ma = _zscore(day["ma_dist_20"]).clip(-zclip, zclip)
    z_r20 = _zscore(day["ret_20d"]).clip(-zclip, zclip)
    z_r60 = _zscore(day["ret_60d"]).clip(-zclip, zclip)
    z_vol = _zscore(day["vol_20d"]).clip(-zclip, zclip)
    z_atr = _zscore(day["atr_pct"]).clip(-zclip, zclip)
    z_vr = _zscore(day["vol_ratio_20"]).clip(-zclip, zclip)

    day["score"] = (
        w_ma * z_ma +
        w_r20 * z_r20 +
        w_r60 * z_r60 +
        w_vol * z_vol +
        w_atr * z_atr +
        w_vr * z_vr
    )

    # 稳定排序：score 相同按 code 排，保证可重复
    day["code"] = day["symbol"].astype(str).str.upper().str.strip()
    out = day[["date", "code", "score"]].copy()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["date", "code", "score"])
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out = out.sort_values(["date", "score", "code"], ascending=[True, False, True])
    return out


# ----------------------------
# Build signal history (multi-day)
# ----------------------------
def build_signal_history(
    base_dir: Path,
    start: str | None,
    end: str | None,
    append: bool = True,
    out_name: str = "signal_history.csv",
    weights: tuple[float, float, float, float, float, float] = (2.0, 1.0, 0.5, -1.0, -0.5, 0.3),
    zclip: float = 6.0,
    verbose: bool = True,
) -> Path:
    feats = load_features(base_dir)

    # watchlist 过滤（若 config.yaml 没有 watchlist，则不过滤）
    watchlist = load_config_watchlist(base_dir)
    if watchlist:
        feats = feats[feats["symbol"].isin(watchlist)].copy()

    # 日期范围过滤
    if start:
        s = pd.to_datetime(start).date()
        feats = feats[feats["date"] >= s]
    if end:
        e = pd.to_datetime(end).date()
        feats = feats[feats["date"] <= e]

    if feats.empty:
        raise ValueError("No feature rows after filtering (watchlist/start/end).")

    out_path = base_dir / "out" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取旧结果（用于增量判断与替换）
    old_df = None
    if append and out_path.exists():
        try:
            old_df = pd.read_csv(out_path)
            if not {"date", "code", "score"}.issubset(old_df.columns):
                if verbose:
                    print(f"⚠️ 旧文件列不符合要求，将重建：{out_path}")
                old_df = None
        except Exception:
            old_df = None

    # 把 feats 分组的“预期股票集合”准备好（用于判断 watchlist 变更导致需要重算某天）
    # expected_codes[date] = set(codes in feats for that date)
    feats["code"] = feats["symbol"].astype(str).str.upper().str.strip()
    expected_codes: dict[str, set[str]] = {}
    for d, g in feats.groupby("date"):
        ds = str(d)
        expected_codes[ds] = set(g["code"].unique())

    # 已有每一天的 codes 集合
    existing_codes: dict[str, set[str]] = {}
    if old_df is not None and len(old_df) > 0:
        tmp = old_df.copy()
        tmp["date"] = tmp["date"].astype(str)
        tmp["code"] = tmp["code"].astype(str).str.upper().str.strip()
        for d, g in tmp.groupby("date"):
            existing_codes[str(d)] = set(g["code"].unique())

    # 决定哪些日期需要重算
    # 规则：
    # - 没有旧文件 / 不 append => 全部重算
    # - 有旧文件：如果某天不存在，或该天 codes 集合与 expected 不一致 => 重算该天
    all_dates = sorted(set(str(x) for x in feats["date"].unique()))
    to_recalc: list[str] = []
    if (not append) or (old_df is None):
        to_recalc = all_dates
    else:
        for ds in all_dates:
            exp = expected_codes.get(ds, set())
            have = existing_codes.get(ds, set())
            if (ds not in existing_codes) or (have != exp):
                to_recalc.append(ds)

    if verbose:
        print(f"✅ build_signal_history: out={out_path}")
        print(f"   feats date range: {min(all_dates)} -> {max(all_dates)}  dates={len(all_dates)}")
        if watchlist:
            print(f"   watchlist size: {len(watchlist)}")
        print(f"   weights={weights}, zclip={zclip}")
        print(f"   append={append}, need_recalc_dates={len(to_recalc)}")

    if len(to_recalc) == 0:
        if verbose:
            print("✅ 没有需要重算的日期（包括 watchlist 变更检查）。")
        return out_path

    # 重算需要的日期
    chunks = []
    for ds in to_recalc:
        d = pd.to_datetime(ds).date()
        day = feats[feats["date"] == d].copy()
        if day.empty:
            continue
        scored = compute_score_for_one_day(day, weights=weights, zclip=zclip)
        if len(scored) > 0:
            chunks.append(scored)

    new_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["date", "code", "score"])
    new_df["date"] = new_df["date"].astype(str)
    new_df["code"] = new_df["code"].astype(str).str.upper().str.strip()
    new_df["score"] = pd.to_numeric(new_df["score"], errors="coerce")
    new_df = new_df.dropna(subset=["date", "code", "score"])

    # 合并：对“重算的日期”采用替换策略（删除旧的同日数据，再加新的）
    if append and old_df is not None and len(old_df) > 0:
        old_df2 = old_df.copy()
        old_df2["date"] = old_df2["date"].astype(str)
        old_df2["code"] = old_df2["code"].astype(str).str.upper().str.strip()
        old_df2["score"] = pd.to_numeric(old_df2["score"], errors="coerce")
        old_df2 = old_df2.dropna(subset=["date", "code", "score"])

        # 删除需要重算日期的旧记录
        mask_keep = ~old_df2["date"].isin(set(to_recalc))
        kept = old_df2.loc[mask_keep, ["date", "code", "score"]]
        all_df = pd.concat([kept, new_df], ignore_index=True)
    else:
        all_df = new_df

    # 最终去重排序（稳定）
    all_df["date"] = pd.to_datetime(all_df["date"]).dt.strftime("%Y-%m-%d")
    all_df["code"] = all_df["code"].astype(str).str.upper().str.strip()
    all_df["score"] = pd.to_numeric(all_df["score"], errors="coerce")
    all_df = all_df.dropna(subset=["date", "code", "score"])
    all_df = all_df.drop_duplicates(subset=["date", "code"], keep="last")
    all_df = all_df.sort_values(["date", "code"])
    all_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    if verbose:
        print(f"✅ signal history saved: {out_path}")
        print(f"   rows={len(all_df)}  dates={all_df['date'].nunique()}")
        print(f"   range: {all_df['date'].min()} -> {all_df['date'].max()}")

    return out_path


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build long-range signal history (date,code,score) from features_daily.parquet")
    ap.add_argument("--base-dir", default=".", help="project root")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--no-append", action="store_true", help="rebuild from scratch (ignore existing output)")
    ap.add_argument("--out-name", default="signal_history.csv", help="output file name under ./out/")
    ap.add_argument("--weights", default="2,1,0.5,-1,-0.5,0.3", help="6 weights: ma,ret20,ret60,vol,atr,vr")
    ap.add_argument("--zclip", type=float, default=6.0, help="clip zscore to [-zclip, zclip]")
    ap.add_argument("--quiet", action="store_true", help="less logs")

    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    weights = _parse_weights(args.weights)

    build_signal_history(
        base_dir=base_dir,
        start=args.start,
        end=args.end,
        append=(not args.no_append),
        out_name=args.out_name,
        weights=weights,
        zclip=float(args.zclip),
        verbose=(not args.quiet),
    )


if __name__ == "__main__":
    main()
