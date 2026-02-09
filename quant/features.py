"""
特征计算模块 (升级版)

原有特征：
- 收益率: ret_1d, ret_5d, ret_20d, ret_60d, fwd_ret_1d
- 均线: ma_5, ma_20, ma_60, ma_dist_20, ma_dist_60
- 波动: vol_20d, vol_ratio_20
- 技术指标: rsi_14, atr_14

新增特征（涨停回调策略需要）：
- 涨跌停: limit_up_flag, limit_down_flag, near_limit_up
- 涨停追踪: days_since_limit_up, limit_up_price, limit_up_count_10d
- 回调: pullback_pct, pullback_valid
- 突破: ma5_slope, vol_ma5, volume_breakout, price_above_ma5
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# 原有函数（保持不变）
# =============================================================================

def load_market_daily_all(base_dir: Path) -> pd.DataFrame:
    qc_path = base_dir / "data" / "clean_qc" / "market_daily_all_qc.parquet"
    clean_path = base_dir / "data" / "clean" / "market_daily_all.parquet"

    def _max_date(path: Path) -> pd.Timestamp | None:
        if not path.exists():
            return None
        try:
            df_tmp = pd.read_parquet(path, columns=["date"])
        except Exception:
            df_tmp = pd.read_parquet(path)
        if df_tmp is None or df_tmp.empty or "date" not in df_tmp.columns:
            return None
        return pd.to_datetime(df_tmp["date"]).max()

    qc_max = _max_date(qc_path)
    clean_max = _max_date(clean_path)

    if qc_max is not None and clean_max is not None:
        if qc_max >= clean_max:
            df = pd.read_parquet(qc_path)
        else:
            df = pd.read_parquet(clean_path)
    elif qc_max is not None:
        df = pd.read_parquet(qc_path)
    elif clean_max is not None:
        df = pd.read_parquet(clean_path)
    else:
        raise FileNotFoundError("No market_daily_all parquet found (qc or clean).")

    if df is None or df.empty:
        raise RuntimeError("Loaded market data is empty.")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr


# =============================================================================
# 新增：涨跌停阈值判断
# =============================================================================

def get_limit_threshold(symbol: str) -> tuple[float, float]:
    """
    根据股票代码判断涨跌停阈值

    规则：
    - 科创板 (688xxx): ±20%
    - 创业板 (300xxx, 301xxx): ±20%
    - 北交所 (8xxxxx, 4xxxxx): ±30%
    - 主板其他: ±10%
    """
    code = str(symbol).replace("sh.", "").replace("sz.", "").replace(".SH", "").replace(".SZ", "").strip()

    if code.startswith("688"):
        return (0.20, -0.20)
    if code.startswith("300") or code.startswith("301"):
        return (0.20, -0.20)
    if code.startswith("8") or code.startswith("4"):
        return (0.30, -0.30)
    return (0.10, -0.10)


# =============================================================================
# 新增：涨停相关特征计算
# =============================================================================

def _compute_limit_up_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算单只股票的涨停相关特征

    输入：单只股票的 DataFrame，已按日期排序
    输出：新增涨停相关列
    """
    df = df.copy()
    close = df["close"]

    # 日收益率
    df["prev_close"] = close.shift(1)
    df["ret1"] = close / df["prev_close"] - 1

    # 获取涨跌停阈值
    symbol = df["symbol"].iloc[0]
    up_thresh, down_thresh = get_limit_threshold(symbol)
    df["limit_up_threshold"] = up_thresh
    df["limit_down_threshold"] = down_thresh

    # 涨跌停标记
    df["limit_up_flag"] = (df["ret1"] >= up_thresh * 0.98).astype(int)
    df["limit_down_flag"] = (df["ret1"] <= down_thresh * 0.98).astype(int)
    df["near_limit_up"] = (df["ret1"] >= up_thresh * 0.90).astype(int)

    # 一字板标记
    df["one_line_board"] = (df["high"] == df["low"]).astype(int)

    # 计算距离最近涨停的天数
    limit_up_indices = df.index[df["limit_up_flag"] == 1].tolist()

    days_since = []
    limit_up_prices = []

    for idx in df.index:
        past_lu = [i for i in limit_up_indices if i < idx]
        if past_lu:
            last_lu_idx = max(past_lu)
            days_diff = idx - last_lu_idx  # 简化：用索引差代替交易日差
            days_since.append(days_diff)
            limit_up_prices.append(df.loc[last_lu_idx, "close"])
        else:
            days_since.append(np.nan)
            limit_up_prices.append(np.nan)

    df["days_since_limit_up"] = days_since
    df["limit_up_price"] = limit_up_prices

    # 过去10天涨停次数
    df["limit_up_count_10d"] = df["limit_up_flag"].rolling(10, min_periods=1).sum()

    # 是否有近期涨停（3-10天内）
    df["has_recent_limit_up"] = (
        (df["days_since_limit_up"] >= 1) &
        (df["days_since_limit_up"] <= 10)
    ).astype(int)

    # 回调幅度
    df["pullback_pct"] = (df["limit_up_price"] - df["close"]) / df["limit_up_price"]
    df["pullback_pct"] = df["pullback_pct"].clip(lower=0)

    # 回调是否在合理范围 (5%-25%)
    df["pullback_valid"] = (
        (df["pullback_pct"] >= 0.05) &
        (df["pullback_pct"] <= 0.25)
    ).astype(int)

    # MA5 斜率
    if "ma_5" in df.columns:
        df["ma5"] = df["ma_5"]
    else:
        df["ma5"] = close.rolling(5, min_periods=3).mean()
    df["ma5_slope"] = df["ma5"] - df["ma5"].shift(1)

    # 5日均量
    df["vol_ma5"] = df["volume"].rolling(5, min_periods=3).mean()

    # 放量标记
    df["volume_breakout"] = (df["volume"] > df["vol_ma5"] * 1.3).astype(int)

    # 价格在MA5上方
    df["price_above_ma5"] = (df["close"] > df["ma5"]).astype(int)

    # 前一日最高价
    df["prev_high"] = df["high"].shift(1)

    # 突破前高
    df["high_breakout"] = (df["high"] > df["prev_high"]).astype(int)

    # 近5日最低价
    df["low_5d_min"] = df["low"].rolling(5, min_periods=3).min()

    # 不创新低
    df["no_new_low"] = (df["low"] >= df["low_5d_min"].shift(1)).astype(int)

    return df


# =============================================================================
# 主函数：构建完整特征
# =============================================================================

def build_features_daily(
    mkt: pd.DataFrame,
    include_limit_up: bool = True,
    include_tdx: bool = True,
) -> pd.DataFrame:
    """
    构建每日特征

    Args:
        mkt: 原始行情数据
        include_limit_up: 是否包含涨停相关特征（默认True）
        include_tdx: 是否包含通达信指标（默认True）

    Returns:
        完整特征 DataFrame
    """
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(mkt.columns)
    if missing:
        raise ValueError(f"Market data missing required columns: {missing}")

    mkt = mkt.copy()
    mkt["date"] = pd.to_datetime(mkt["date"]).dt.date
    mkt = mkt.sort_values(["symbol", "date"]).reset_index(drop=True)

    feats = []

    for sym, df in mkt.groupby("symbol", sort=False):
        df = df.sort_values("date").reset_index(drop=True)
        close = df["close"]
        vol = df["volume"]

        # =====================
        # 原有特征
        # =====================

        # past returns (no lookahead)
        df["ret_1d"] = close.pct_change(1)
        df["ret_5d"] = close.pct_change(5)
        df["ret_20d"] = close.pct_change(20)
        df["ret_60d"] = close.pct_change(60)

        # forward return for backtests (uses future close; only used as label/target)
        df["fwd_ret_1d"] = close.shift(-1) / close - 1

        # moving averages
        df["ma_5"] = close.rolling(5, min_periods=5).mean()
        df["ma_20"] = close.rolling(20, min_periods=20).mean()
        df["ma_60"] = close.rolling(60, min_periods=60).mean()
        df["ma_dist_20"] = close / df["ma_20"] - 1
        df["ma_dist_60"] = close / df["ma_60"] - 1

        # volatility (annualized, based on log returns)
        logret = np.log(close).diff()
        df["vol_20d"] = logret.rolling(20, min_periods=20).std() * np.sqrt(252)

        # volume surprise
        vol_ma20 = vol.rolling(20, min_periods=20).mean()
        df["vol_ratio_20"] = vol / vol_ma20

        # RSI / ATR
        df["rsi_14"] = _rsi_wilder(close, 14)
        df["atr_14"] = _atr_wilder(df["high"], df["low"], close, 14)

        # =====================
        # 新增：涨停相关特征
        # =====================
        if include_limit_up:
            df = _compute_limit_up_features(df)

        feats.append(df)

    out = pd.concat(feats, ignore_index=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    # 选择输出列
    base_cols = [
        "symbol", "date",
        "open", "high", "low", "close", "volume",
        "ret_1d", "ret_5d", "ret_20d", "ret_60d",
        "fwd_ret_1d",
        "ma_5", "ma_20", "ma_60",
        "ma_dist_20", "ma_dist_60",
        "vol_20d", "vol_ratio_20",
        "rsi_14", "atr_14",
    ]

    limit_up_cols = [
        "ret1", "prev_close",
        "limit_up_threshold", "limit_down_threshold",
        "limit_up_flag", "limit_down_flag", "near_limit_up",
        "one_line_board",
        "days_since_limit_up", "limit_up_price", "limit_up_count_10d",
        "has_recent_limit_up",
        "pullback_pct", "pullback_valid",
        "ma5", "ma5_slope",
        "vol_ma5", "volume_breakout",
        "price_above_ma5", "prev_high", "high_breakout",
        "low_5d_min", "no_new_low",
    ]

    if include_limit_up:
        keep_cols = base_cols + [c for c in limit_up_cols if c in out.columns]
    else:
        keep_cols = base_cols

    # 只保留存在的列
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols]

    # =====================
    # 新增：通达信指标
    # =====================
    if include_tdx:
        try:
            from quant.tdx_indicators import add_tdx_indicators
            print("计算通达信指标...")
            out = add_tdx_indicators(out)
        except ImportError:
            print("  ⚠️ tdx_indicators 模块不存在，跳过通达信指标")
        except Exception as e:
            print(f"  ⚠️ 通达信指标计算失败: {e}")

    return out


def save_features(base_dir: Path, features: pd.DataFrame) -> Path:
    out_dir = base_dir / "data" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features_daily.parquet"
    features.to_parquet(out_path, index=False)
    return out_path


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    # 简单测试
    print("Testing features module...")

    # 创建模拟数据
    dates = pd.date_range("2025-01-01", periods=30, freq="B")

    data = []
    for symbol in ["600519", "300750"]:
        base_price = 100
        for i, d in enumerate(dates):
            ret = np.random.normal(0, 0.03)

            # 模拟涨停
            if i == 10 and symbol == "300750":
                ret = 0.20  # 创业板涨停
            elif i == 15 and symbol == "600519":
                ret = 0.10  # 主板涨停

            close = base_price * (1 + ret)
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))

            data.append({
                "date": d,
                "symbol": symbol,
                "open": base_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(100000, 1000000),
            })

            base_price = close

    mkt = pd.DataFrame(data)

    # 构建特征
    feats = build_features_daily(mkt, include_limit_up=True)

    print(f"特征数量: {len(feats.columns)}")
    print(f"特征列: {list(feats.columns)}")

    # 显示涨停情况
    limit_ups = feats[feats["limit_up_flag"] == 1][["date", "symbol", "ret1", "close"]]
    print(f"\n涨停日: {len(limit_ups)}")
    print(limit_ups)