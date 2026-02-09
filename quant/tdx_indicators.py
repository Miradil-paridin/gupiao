"""
通达信指标 Python 实现

包含指标：
1. 高30突破 (high30_breakout) - **月线级别**趋势
2. 主力控盘 (main_force_control) - **月线级别**主力控盘程度
3. 涨停30日 (limit_up_30d) - 日线级别近期涨停频率
4. 起爆点 (explosion_point) - 双重EMA

注意：高30突破和主力控盘使用月线数据计算，涨停30日使用日线数据

使用方法：
    from quant.tdx_indicators import add_tdx_indicators
    df = add_tdx_indicators(df)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均 EMA"""
    return series.ewm(span=period, adjust=False).mean()


def hhv(series: pd.Series, period: int) -> pd.Series:
    """最高值 HHV"""
    return series.rolling(period, min_periods=1).max()


def llv(series: pd.Series, period: int) -> pd.Series:
    """最低值 LLV"""
    return series.rolling(period, min_periods=1).min()


def ma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均 MA"""
    return series.rolling(period, min_periods=1).mean()


def ref(series: pd.Series, n: int) -> pd.Series:
    """引用前N日数据 REF"""
    return series.shift(n)


# =============================================================================
# 日线转月线
# =============================================================================

def resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    将日线数据转换为月线数据

    每月取：
    - open: 月初第一个交易日的开盘价
    - high: 月内最高价
    - low: 月内最低价
    - close: 月末最后一个交易日的收盘价
    - volume: 月成交量之和
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])

    # 按股票和月份分组
    df["year_month"] = df["date"].dt.to_period("M")

    monthly_data = []

    for symbol in df["symbol"].unique():
        sym_df = df[df["symbol"] == symbol].copy()

        for ym in sym_df["year_month"].unique():
            month_df = sym_df[sym_df["year_month"] == ym]

            if month_df.empty:
                continue

            monthly_data.append({
                "symbol": symbol,
                "year_month": ym,
                "date": month_df["date"].iloc[-1],  # 月末日期
                "open": month_df["open"].iloc[0],
                "high": month_df["high"].max(),
                "low": month_df["low"].min(),
                "close": month_df["close"].iloc[-1],
                "volume": month_df["volume"].sum(),
            })

    monthly_df = pd.DataFrame(monthly_data)
    monthly_df = monthly_df.sort_values(["symbol", "year_month"]).reset_index(drop=True)

    return monthly_df


# =============================================================================
# 指标1: 高30突破（月线级别）
# =============================================================================

def compute_high30_breakout_monthly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    高30突破指标（月线级别）

    公式：
        X1 = (C + L + H) / 1.5
        X2 = EMA(X1, 3)  -- 3个月的EMA
        高30 = HHV(X2, 30)  -- 30个月内最高值
        条件：高30 > REF(高30, 1)  即创新高

    返回：
        high30_monthly: 30月内X2最高值
        high30_breakout_monthly: 1=创新高，0=否
    """
    df = monthly_df.copy()

    # X1 = (C + L + H) / 1.5
    df["x1"] = (df["close"] + df["low"] + df["high"]) / 1.5

    # X2 = EMA(X1, 3) -- 3个月
    df["x2"] = df.groupby("symbol")["x1"].transform(lambda x: ema(x, 3))

    # 高30 = HHV(X2, 30) -- 30个月
    df["high30_monthly"] = df.groupby("symbol")["x2"].transform(lambda x: hhv(x, 30))

    # 前高30
    df["high30_prev"] = df.groupby("symbol")["high30_monthly"].shift(1)

    # 突破条件：高30 > 前高30
    df["high30_breakout_monthly"] = (df["high30_monthly"] > df["high30_prev"]).astype(int)

    # 清理中间列
    df = df.drop(columns=["x1", "x2", "high30_prev"], errors="ignore")

    return df


# =============================================================================
# 指标2: 主力控盘（月线级别）
# =============================================================================

def compute_main_force_control_monthly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    主力控盘指标（月线级别）

    公式：
        GU1 = (CLOSE*2 + HIGH + LOW) / 4  (加权收盘价)
        GU2 = EMA(GU1, 13) - EMA(GU1, 34)  -- 13个月和34个月的EMA
        GU3 = EMA(GU2, 5)  -- 5个月
        主力线 = 2 * (GU2 - GU3) * 3.8
        起爆 = EMA(EMA(CLOSE, 9), 9)  -- 9个月的双重EMA
        主力控盘 = (GU1 - REF(起爆, 1)) / REF(起爆, 1)

        条件：主力控盘 > 0.5（即价格比起爆点高50%以上）

    返回：
        gu1_monthly: 月线加权收盘价
        main_force_line_monthly: 月线主力线
        explosion_point_monthly: 月线起爆点
        main_force_control_monthly: 月线主力控盘度
        main_force_strong_monthly: 1=强控盘(>0.5), 0=否
    """
    df = monthly_df.copy()

    # GU1 = (CLOSE*2 + HIGH + LOW) / 4
    df["gu1_monthly"] = (df["close"] * 2 + df["high"] + df["low"]) / 4

    # GU2 = EMA(GU1, 13) - EMA(GU1, 34)
    df["ema_gu1_13"] = df.groupby("symbol")["gu1_monthly"].transform(lambda x: ema(x, 13))
    df["ema_gu1_34"] = df.groupby("symbol")["gu1_monthly"].transform(lambda x: ema(x, 34))
    df["gu2"] = df["ema_gu1_13"] - df["ema_gu1_34"]

    # GU3 = EMA(GU2, 5)
    df["gu3"] = df.groupby("symbol")["gu2"].transform(lambda x: ema(x, 5))

    # 主力线 = 2 * (GU2 - GU3) * 3.8
    df["main_force_line_monthly"] = 2 * (df["gu2"] - df["gu3"]) * 3.8

    # 起爆 = EMA(EMA(CLOSE, 9), 9)
    df["ema_close_9"] = df.groupby("symbol")["close"].transform(lambda x: ema(x, 9))
    df["explosion_point_monthly"] = df.groupby("symbol")["ema_close_9"].transform(lambda x: ema(x, 9))

    # REF(起爆, 1)
    df["explosion_prev"] = df.groupby("symbol")["explosion_point_monthly"].shift(1)

    # 主力控盘 = (GU1 - REF(起爆, 1)) / REF(起爆, 1)
    df["main_force_control_monthly"] = (df["gu1_monthly"] - df["explosion_prev"]) / df["explosion_prev"]
    df["main_force_control_monthly"] = df["main_force_control_monthly"].replace([np.inf, -np.inf], np.nan)

    # 强控盘条件：主力控盘 > 0.5（月线级别用原始阈值）
    df["main_force_strong_monthly"] = (df["main_force_control_monthly"] > 0.5).astype(int)

    # 清理中间列
    drop_cols = ["ema_gu1_13", "ema_gu1_34", "gu2", "gu3", "ema_close_9", "explosion_prev"]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def merge_monthly_to_daily(daily_df: pd.DataFrame, monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    将月线指标合并到日线数据

    每个交易日使用其所在月份的月线指标值
    """
    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["year_month"] = daily_df["date"].dt.to_period("M")

    # 选择需要合并的月线指标列
    monthly_cols = [
        "symbol", "year_month",
        "high30_monthly", "high30_breakout_monthly",
        "gu1_monthly", "main_force_line_monthly",
        "explosion_point_monthly", "main_force_control_monthly",
        "main_force_strong_monthly"
    ]
    monthly_cols = [c for c in monthly_cols if c in monthly_df.columns]

    merge_df = monthly_df[monthly_cols].copy()

    # 合并
    result = daily_df.merge(merge_df, on=["symbol", "year_month"], how="left")

    # 删除临时列
    result = result.drop(columns=["year_month"], errors="ignore")

    # 重命名列（去掉 _monthly 后缀，保持兼容）
    rename_map = {
        "high30_monthly": "high30",
        "high30_breakout_monthly": "high30_breakout",
        "main_force_control_monthly": "main_force_control",
        "main_force_strong_monthly": "main_force_strong",
        "main_force_line_monthly": "main_force_line",
        "explosion_point_monthly": "explosion_point",
        "gu1_monthly": "gu1",
    }
    result = result.rename(columns=rename_map)

    return result


# =============================================================================
# 指标3: 涨停30日（日线级别）
# =============================================================================

def compute_limit_up_30d(df: pd.DataFrame) -> pd.DataFrame:
    """
    涨停30日指标（日线级别）

    公式：
        VAR1 = IF(CLOSE/REF(CLOSE, 1) > 1.095, 1, 0)  (涨幅>9.5%标记为涨停)
        涨停30日 = MA(VAR1, 30) * 10

        条件：涨停30日 > 0.5

    说明：
        MA(VAR1, 30) * 10 的含义：
        - 如果30天内有1次涨停，MA=1/30=0.033，*10=0.33
        - 如果30天内有2次涨停，MA=2/30=0.067，*10=0.67
        - 所以 >0.5 意味着30天内至少有1-2次涨停

    返回：
        limit_up_mark: 当日是否涨停
        limit_up_30d: 30日涨停频率指标
        has_limit_up_30d: 1=近期有涨停(>0.5), 0=否
    """
    df = df.copy()

    # 计算日涨幅
    df["ret_1d_calc"] = df.groupby("symbol")["close"].pct_change()

    # VAR1: 涨幅 > 9.5% 标记为1（接近涨停）
    # 考虑到科创板/创业板是20%，这里用更通用的判断
    def is_limit_up(row):
        ret = row.get("ret_1d_calc", 0)
        if pd.isna(ret):
            return 0
        symbol = str(row.get("symbol", ""))
        # 科创板/创业板
        if symbol.startswith("688") or symbol.startswith("300") or symbol.startswith("301"):
            return 1 if ret > 0.195 else 0
        # 主板
        return 1 if ret > 0.095 else 0

    # 如果已经有 limit_up_flag 列，直接使用
    if "limit_up_flag" in df.columns:
        df["limit_up_mark"] = df["limit_up_flag"]
    else:
        df["limit_up_mark"] = df.apply(is_limit_up, axis=1)

    # 涨停30日 = MA(VAR1, 30) * 10
    df["limit_up_30d"] = df.groupby("symbol")["limit_up_mark"].transform(
        lambda x: ma(x, 30) * 10
    )

    # 条件：涨停30日 > 0.5 (30天内至少有1-2次涨停)
    df["has_limit_up_30d"] = (df["limit_up_30d"] > 0.5).astype(int)

    # 清理临时列
    df = df.drop(columns=["ret_1d_calc"], errors="ignore")

    return df


# =============================================================================
# 综合评分
# =============================================================================

def compute_tdx_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于通达信指标计算综合评分

    评分规则：
    - high30_breakout = 1: +1分（月线趋势向上）
    - main_force_strong = 1: +1.5分（月线主力控盘）
    - has_limit_up_30d = 1: +0.5分（日线近期有涨停）
    - main_force_control > 0: +0.5分（月线主力介入）

    返回：
        tdx_score: 通达信指标综合得分
        tdx_eligible: 是否满足入场条件（至少满足2个）
    """
    df = df.copy()

    # 初始化得分
    df["tdx_score"] = 0.0

    # 高30突破 +1分（月线级别）
    if "high30_breakout" in df.columns:
        df["tdx_score"] += df["high30_breakout"].fillna(0) * 1.0

    # 主力强控盘 +1.5分（月线级别）
    if "main_force_strong" in df.columns:
        df["tdx_score"] += df["main_force_strong"].fillna(0) * 1.5

    # 近期有涨停 +0.5分（日线级别）
    if "has_limit_up_30d" in df.columns:
        df["tdx_score"] += df["has_limit_up_30d"].fillna(0) * 0.5

    # 主力控盘 > 0 +0.5分（月线级别）
    if "main_force_control" in df.columns:
        df["tdx_score"] += (df["main_force_control"].fillna(0) > 0).astype(int) * 0.5

    # 入场条件：tdx_score >= 2（至少满足高30突破+主力控盘）
    df["tdx_eligible"] = (df["tdx_score"] >= 2.0).astype(int)

    return df


# =============================================================================
# 主入口函数
# =============================================================================

def add_tdx_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    一键添加所有通达信指标

    处理流程：
    1. 将日线数据转换为月线
    2. 在月线上计算：高30突破、主力控盘（月线级别指标）
    3. 将月线指标合并回日线
    4. 在日线上计算：涨停30日（日线级别指标）
    5. 计算综合评分

    新增列：
    - high30: 30月内X2最高值（月线）
    - high30_breakout: 高30突破（月线，1=创新高）
    - gu1: 月线加权收盘价
    - main_force_line: 月线主力线
    - explosion_point: 月线起爆点
    - main_force_control: 月线主力控盘度
    - main_force_strong: 月线强控盘（1=控盘>50%）
    - limit_up_mark: 当日涨停标记（日线）
    - limit_up_30d: 30日涨停频率（日线）
    - has_limit_up_30d: 近期有涨停（日线，1=是）
    - tdx_score: 通达信综合得分
    - tdx_eligible: 是否满足TDX入场条件
    """
    print("计算通达信指标...")

    # 确保按 symbol, date 排序
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Step 1: 日线转月线
    print("  - 转换为月线数据...")
    monthly_df = resample_to_monthly(df)

    if monthly_df.empty:
        print("  ⚠️ 月线数据为空，跳过月线指标")
        # 只计算日线指标
        print("  - 涨停30日（日线）...")
        df = compute_limit_up_30d(df)
        print("  - 综合评分...")
        df = compute_tdx_score(df)
        return df

    # Step 2: 月线指标计算
    print("  - 高30突破（月线）...")
    monthly_df = compute_high30_breakout_monthly(monthly_df)

    print("  - 主力控盘（月线）...")
    monthly_df = compute_main_force_control_monthly(monthly_df)

    # Step 3: 月线指标合并回日线
    print("  - 合并月线指标到日线...")
    df = merge_monthly_to_daily(df, monthly_df)

    # Step 4: 日线指标计算
    print("  - 涨停30日（日线）...")
    df = compute_limit_up_30d(df)

    # Step 5: 综合评分
    print("  - 综合评分...")
    df = compute_tdx_score(df)

    print("  完成！")

    return df


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    # 创建测试数据（需要足够多的月份）
    np.random.seed(42)

    # 生成3年的日线数据（约750个交易日）
    dates = pd.date_range("2022-01-01", periods=750, freq="B")

    data = []
    for symbol in ["600519", "300750"]:
        base_price = 100
        for i, d in enumerate(dates):
            ret = np.random.normal(0.001, 0.02)  # 微涨趋势

            # 模拟涨停
            if i % 60 == 30 and symbol == "300750":  # 每隔2个月涨停一次
                ret = 0.20
            elif i % 90 == 45 and symbol == "600519":  # 每隔3个月涨停一次
                ret = 0.10

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

    df = pd.DataFrame(data)

    # 添加指标
    df = add_tdx_indicators(df)

    # 显示结果
    print("\n最后10行（300750）:")
    cols = ["date", "symbol", "close", "high30_breakout", "main_force_control",
            "limit_up_30d", "tdx_score", "tdx_eligible"]
    cols = [c for c in cols if c in df.columns]
    print(df[df["symbol"] == "300750"][cols].tail(10).to_string(index=False))

    print("\n月线指标统计:")
    print(f"  高30突破次数: {df['high30_breakout'].sum() if 'high30_breakout' in df.columns else 'N/A'}")
    print(f"  主力强控盘次数: {df['main_force_strong'].sum() if 'main_force_strong' in df.columns else 'N/A'}")
    print(f"  日线涨停次数: {df['limit_up_mark'].sum() if 'limit_up_mark' in df.columns else 'N/A'}")

    print("\n入场条件满足的行:")
    if "tdx_eligible" in df.columns:
        eligible = df[df["tdx_eligible"] == 1][["date", "symbol", "tdx_score"]]
        print(f"  总数: {len(eligible)}")
        print(eligible.tail(10))