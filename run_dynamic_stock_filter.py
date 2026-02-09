"""
动态股票筛选模块

功能：
1. 获取全A股股票列表（约5600只）
2. 排除科创板(688)和创业板(300/301)
3. 根据通达信指标筛选：
   - 高30突破（月线）
   - 主力控盘 > 50%（月线）
   - 涨停30日 > 0.5（日线）
4. 输出符合条件的股票列表

使用方法：
    python run_dynamic_stock_filter.py
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Set
import pandas as pd
import numpy as np

# 尝试导入 baostock
try:
    import baostock as bs

    HAS_BAOSTOCK = True
except ImportError:
    HAS_BAOSTOCK = False
    print("⚠️ 请安装 baostock: pip install baostock")


def get_all_a_stocks() -> pd.DataFrame:
    """
    获取全部A股股票列表

    返回：
        DataFrame with columns: code, code_name, ipoDate, outDate, type, status
    """
    if not HAS_BAOSTOCK:
        raise ImportError("需要安装 baostock")

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock登录失败: {lg.error_msg}")

    try:
        # 获取股票列表
        rs = bs.query_stock_basic()

        if rs.error_code != "0":
            raise RuntimeError(f"获取股票列表失败: {rs.error_msg}")

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        df = pd.DataFrame(data_list, columns=rs.fields)

        return df

    finally:
        bs.logout()


def filter_main_board_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤主板股票（排除科创板和创业板）

    规则：
    - 排除 688xxx（科创板）
    - 排除 300xxx、301xxx（创业板）
    - 只保留 A股（type=1）
    - 只保留 上市状态（status=1）
    """
    df = df.copy()

    # 提取纯股票代码
    df["pure_code"] = df["code"].str.replace("sh.", "").str.replace("sz.", "")

    # 过滤条件
    # 1. 只保留A股
    df = df[df["type"] == "1"]

    # 2. 只保留上市状态
    df = df[df["status"] == "1"]

    # 3. 排除科创板 (688)
    df = df[~df["pure_code"].str.startswith("688")]

    # 4. 排除创业板 (300, 301)
    df = df[~df["pure_code"].str.startswith("300")]
    df = df[~df["pure_code"].str.startswith("301")]

    # 5. 排除ST股票
    df = df[~df["code_name"].str.contains("ST", case=False, na=False)]

    # 6. 排除B股
    df = df[~df["pure_code"].str.startswith("9")]
    df = df[~df["pure_code"].str.startswith("2")]

    print(f"过滤后剩余: {len(df)} 只主板股票")

    return df


def fetch_stock_monthly_data(
        code: str,
        start_date: str = "2021-01-01",
        end_date: str = None,
) -> pd.DataFrame:
    """
    获取单只股票的月线数据
    """
    if not HAS_BAOSTOCK:
        return pd.DataFrame()

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    lg = bs.login()
    if lg.error_code != "0":
        return pd.DataFrame()

    try:
        rs = bs.query_history_k_data_plus(
            code=code,
            fields="date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="m",  # 月线
            adjustflag="2",  # 前复权
        )

        if rs.error_code != "0":
            return pd.DataFrame()

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)

        # 转换数据类型
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = code.replace("sh.", "").replace("sz.", "")

        return df

    finally:
        bs.logout()


def fetch_stock_daily_data(
        code: str,
        start_date: str = "2023-01-01",
        end_date: str = None,
) -> pd.DataFrame:
    """
    获取单只股票的日线数据
    """
    if not HAS_BAOSTOCK:
        return pd.DataFrame()

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    lg = bs.login()
    if lg.error_code != "0":
        return pd.DataFrame()

    try:
        rs = bs.query_history_k_data_plus(
            code=code,
            fields="date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",  # 日线
            adjustflag="2",  # 前复权
        )

        if rs.error_code != "0":
            return pd.DataFrame()

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = code.replace("sh.", "").replace("sz.", "")

        return df

    finally:
        bs.logout()


# =============================================================================
# 通达信指标计算（简化版，用于筛选）
# =============================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def hhv(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).max()


def ma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()


def check_high30_breakout(monthly_df: pd.DataFrame) -> bool:
    """
    检查高30突破条件（月线）

    条件：高30 > 前高30（创新高）
    """
    if monthly_df.empty or len(monthly_df) < 3:
        return False

    df = monthly_df.copy()
    df = df.sort_values("date")

    # X1 = (C + L + H) / 1.5
    df["x1"] = (df["close"] + df["low"] + df["high"]) / 1.5

    # X2 = EMA(X1, 3)
    df["x2"] = ema(df["x1"], 3)

    # 高30 = HHV(X2, 30)
    df["high30"] = hhv(df["x2"], 30)

    # 检查最后一个月是否突破
    if len(df) < 2:
        return False

    latest = df.iloc[-1]["high30"]
    prev = df.iloc[-2]["high30"]

    return latest > prev


def check_main_force_control(monthly_df: pd.DataFrame, threshold: float = 0.5) -> bool:
    """
    检查主力控盘条件（月线）

    条件：主力控盘 > 50%
    """
    if monthly_df.empty or len(monthly_df) < 10:
        return False

    df = monthly_df.copy()
    df = df.sort_values("date")

    # GU1 = (CLOSE*2 + HIGH + LOW) / 4
    df["gu1"] = (df["close"] * 2 + df["high"] + df["low"]) / 4

    # 起爆 = EMA(EMA(CLOSE, 9), 9)
    df["ema_close_9"] = ema(df["close"], 9)
    df["explosion"] = ema(df["ema_close_9"], 9)

    # 主力控盘 = (GU1 - REF(起爆, 1)) / REF(起爆, 1)
    df["explosion_prev"] = df["explosion"].shift(1)
    df["main_force_control"] = (df["gu1"] - df["explosion_prev"]) / df["explosion_prev"]

    # 检查最后一个月
    latest_mfc = df.iloc[-1]["main_force_control"]

    if pd.isna(latest_mfc):
        return False

    return latest_mfc > threshold


def check_limit_up_30d(daily_df: pd.DataFrame, threshold: float = 0.5) -> bool:
    """
    检查涨停30日条件（日线）

    条件：30天内至少1-2次涨停
    """
    if daily_df.empty or len(daily_df) < 30:
        return False

    df = daily_df.copy()
    df = df.sort_values("date")

    # 日涨幅
    df["ret"] = df["close"].pct_change()

    # 涨停标记（主板 > 9.5%）
    df["limit_up"] = (df["ret"] > 0.095).astype(int)

    # 涨停30日 = MA(涨停标记, 30) * 10
    df["limit_up_30d"] = ma(df["limit_up"], 30) * 10

    # 检查最后一天
    latest = df.iloc[-1]["limit_up_30d"]

    if pd.isna(latest):
        return False

    return latest > threshold


def filter_stocks_by_tdx_rules(
        stock_list: pd.DataFrame,
        start_date: str = "2021-01-01",
        daily_start_date: str = "2023-01-01",
        require_all: bool = False,
        max_stocks: int = None,
) -> List[str]:
    """
    根据通达信规则筛选股票

    Args:
        stock_list: 股票列表 DataFrame
        start_date: 月线数据开始日期
        daily_start_date: 日线数据开始日期
        require_all: True=必须满足全部3个条件，False=满足任一即可
        max_stocks: 最多处理多少只股票（用于测试）

    Returns:
        符合条件的股票代码列表
    """
    from tqdm import tqdm

    codes = stock_list["code"].tolist()

    if max_stocks:
        codes = codes[:max_stocks]

    qualified_stocks = []

    print(f"\n开始筛选 {len(codes)} 只股票...")
    print(f"筛选规则: {'全部满足' if require_all else '满足任一'}")

    for code in tqdm(codes, desc="筛选进度"):
        try:
            # 获取月线数据
            monthly_df = fetch_stock_monthly_data(code, start_date)

            # 获取日线数据
            daily_df = fetch_stock_daily_data(code, daily_start_date)

            if monthly_df.empty and daily_df.empty:
                continue

            # 检查三个条件
            high30_ok = check_high30_breakout(monthly_df) if not monthly_df.empty else False
            main_force_ok = check_main_force_control(monthly_df) if not monthly_df.empty else False
            limit_up_ok = check_limit_up_30d(daily_df) if not daily_df.empty else False

            # 判断是否符合条件
            if require_all:
                qualified = high30_ok and main_force_ok and limit_up_ok
            else:
                qualified = high30_ok or main_force_ok or limit_up_ok

            if qualified:
                symbol = code.replace("sh.", "").replace("sz.", "")
                qualified_stocks.append(symbol)

                # 显示符合条件的股票
                name = stock_list[stock_list["code"] == code]["code_name"].values
                name = name[0] if len(name) > 0 else ""
                conditions = []
                if high30_ok:
                    conditions.append("高30突破")
                if main_force_ok:
                    conditions.append("主力控盘")
                if limit_up_ok:
                    conditions.append("涨停30日")

                print(f"  ✅ {symbol} {name}: {', '.join(conditions)}")

        except Exception as e:
            # 跳过出错的股票
            continue

    print(f"\n筛选完成! 符合条件: {len(qualified_stocks)} 只")

    return qualified_stocks


def save_dynamic_watchlist(stocks: List[str], output_path: Path) -> Path:
    """
    保存动态股票池
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存为简单的文本文件
    with open(output_path, "w", encoding="utf-8") as f:
        for stock in stocks:
            f.write(f"{stock}\n")

    return output_path


def update_config_watchlist(config_path: Path, stocks: List[str]) -> None:
    """
    更新 config.yaml 中的 watchlist
    """
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["watchlist"] = stocks

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"已更新 {config_path} 中的 watchlist ({len(stocks)} 只股票)")


def main():
    """
    主函数：动态筛选全A股
    """
    base_dir = Path(__file__).resolve().parent

    print("=" * 60)
    print("🔍 动态股票筛选系统")
    print("=" * 60)

    # Step 1: 获取全A股列表
    print("\n📊 Step 1: 获取全A股列表...")
    all_stocks = get_all_a_stocks()
    print(f"   全部股票: {len(all_stocks)}")

    # Step 2: 过滤主板股票
    print("\n📊 Step 2: 过滤主板股票（排除科创板/创业板）...")
    main_board = filter_main_board_stocks(all_stocks)

    # Step 3: 根据TDX规则筛选
    print("\n📊 Step 3: 根据通达信指标筛选...")
    print("   规则1: 高30突破（月线）- 30个月内创新高")
    print("   规则2: 主力控盘 > 50%（月线）")
    print("   规则3: 涨停30日 > 0.5（日线）- 30天内有涨停")

    # 筛选（满足任一条件即可）
    qualified = filter_stocks_by_tdx_rules(
        main_board,
        start_date="2021-01-01",  # 月线需要更早的数据
        daily_start_date="2023-01-01",  # 日线从2023年开始
        require_all=False,  # 满足任一条件即可
        max_stocks=None,  # 不限制，筛选全部
    )

    # Step 4: 保存结果
    print("\n📊 Step 4: 保存结果...")

    # 保存到文件
    watchlist_path = base_dir / "data" / "dynamic_watchlist.txt"
    save_dynamic_watchlist(qualified, watchlist_path)
    print(f"   保存到: {watchlist_path}")

    # 更新 config.yaml
    config_path = base_dir / "config.yaml"
    if config_path.exists():
        update_config_watchlist(config_path, qualified)

    # 显示结果
    print("\n" + "=" * 60)
    print(f"✅ 筛选完成！共 {len(qualified)} 只股票符合条件")
    print("=" * 60)
    print("\n符合条件的股票:")
    for i, stock in enumerate(qualified, 1):
        print(f"  {i:3d}. {stock}")

    print(f"\n下一步: 运行 python run_all_daily.py 进行量化分析")

    return qualified


if __name__ == "__main__":
    main()