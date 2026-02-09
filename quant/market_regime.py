"""
市场环境判断模块 (Market Regime)

功能：
1. 获取沪深300指数数据
2. 判断市场环境（牛市/熊市/震荡）
3. 决定是否允许开仓

使用方法：
    from quant.market_regime import MarketRegime

    regime = MarketRegime(base_dir)
    regime.update_index_data()  # 更新指数数据

    # 判断某天是否允许开仓
    can_open = regime.can_open_position("2025-01-30")
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


def fetch_hs300_from_baostock(
        start_date: str = "2015-01-01",
        end_date: str | None = None,
) -> pd.DataFrame:
    """
    从 BaoStock 获取沪深300指数日线数据

    Args:
        start_date: 开始日期 "YYYY-MM-DD"
        end_date: 结束日期，None 则为今天

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    try:
        import baostock as bs
    except ImportError:
        raise ImportError("请安装 baostock: pip install baostock")

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # 登录
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")

    try:
        # 获取沪深300指数 (sh.000300)
        rs = bs.query_history_k_data_plus(
            code="sh.000300",
            fields="date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",  # 不复权（指数不需要复权）
        )

        if rs.error_code != "0":
            raise RuntimeError(f"获取沪深300失败: {rs.error_msg}")

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)

        # 类型转换
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        return df.dropna(subset=["close"]).reset_index(drop=True)

    finally:
        bs.logout()


def compute_index_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算指数的技术特征

    新增列：
    - ma5, ma20, ma60, ma200: 均线
    - ma60_slope: MA60 斜率（MA60[t] - MA60[t-1]）
    - above_ma200: 是否在 MA200 上方
    - above_ma60: 是否在 MA60 上方
    - ma60_up: MA60 是否向上
    - regime: 市场环境 (BULL/BEAR/NEUTRAL)
    - can_open: 是否允许开仓
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # 均线
    df["ma5"] = df["close"].rolling(5, min_periods=3).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=10).mean()
    df["ma60"] = df["close"].rolling(60, min_periods=30).mean()
    df["ma200"] = df["close"].rolling(200, min_periods=100).mean()

    # MA60 斜率
    df["ma60_slope"] = df["ma60"] - df["ma60"].shift(1)

    # 位置判断
    df["above_ma200"] = (df["close"] > df["ma200"]).astype(int)
    df["above_ma60"] = (df["close"] > df["ma60"]).astype(int)
    df["ma60_up"] = (df["ma60_slope"] > 0).astype(int)

    # 市场环境分类
    def classify_regime(row):
        """
        市场环境判断规则：
        - BULL: 收盘价 > MA200 且 > MA60 且 MA60 向上
        - BEAR: 收盘价 < MA200 且 < MA60
        - NEUTRAL: 其他情况
        """
        if pd.isna(row["ma200"]) or pd.isna(row["ma60"]):
            return "NEUTRAL"

        if row["above_ma200"] == 1 and row["above_ma60"] == 1 and row["ma60_up"] == 1:
            return "BULL"
        elif row["above_ma200"] == 0 and row["above_ma60"] == 0:
            return "BEAR"
        else:
            return "NEUTRAL"

    df["regime"] = df.apply(classify_regime, axis=1)

    # 是否允许开仓（两种规则可选）
    # 规则1（保守）：只在 BULL 时开仓
    # 规则2（适中）：在 BULL 或 NEUTRAL 时开仓
    # 这里用规则2
    df["can_open"] = df["regime"].isin(["BULL", "NEUTRAL"]).astype(int)

    return df


class MarketRegime:
    """
    市场环境管理器

    Example:
        regime = MarketRegime(base_dir)
        regime.update_index_data()

        if regime.can_open_position("2025-01-30"):
            # 执行买入逻辑
            pass
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data" / "index"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.data_dir / "hs300_daily.parquet"
        self._df: Optional[pd.DataFrame] = None

    def update_index_data(
            self,
            start_date: str = "2015-01-01",
            force: bool = False
    ) -> pd.DataFrame:
        """
        更新沪深300指数数据

        Args:
            start_date: 起始日期
            force: 是否强制重新下载

        Returns:
            更新后的 DataFrame
        """
        # 检查是否需要更新
        if not force and self.index_path.exists():
            df = pd.read_parquet(self.index_path)
            last_date = pd.to_datetime(df["date"]).max()
            today = pd.Timestamp.now().normalize()

            # 如果数据已经是今天或昨天的，不需要更新
            if (today - last_date).days <= 1:
                print(f"沪深300数据已是最新: {last_date.strftime('%Y-%m-%d')}")
                self._df = df
                return df

            # 增量更新
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"增量更新沪深300: {start_date} -> today")

        # 获取数据
        print(f"从 BaoStock 获取沪深300数据...")
        new_df = fetch_hs300_from_baostock(start_date=start_date)

        if new_df.empty:
            print("未获取到新数据")
            if self.index_path.exists():
                self._df = pd.read_parquet(self.index_path)
                return self._df
            return pd.DataFrame()

        # 合并旧数据
        if not force and self.index_path.exists():
            old_df = pd.read_parquet(self.index_path)
            df = pd.concat([old_df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["date"], keep="last")
        else:
            df = new_df

        # 计算特征
        df = compute_index_features(df)

        # 保存
        df.to_parquet(self.index_path, index=False)
        print(f"沪深300数据已保存: {self.index_path}")
        print(f"  日期范围: {df['date'].min()} -> {df['date'].max()}")
        print(f"  总行数: {len(df)}")

        self._df = df
        return df

    def load(self) -> pd.DataFrame:
        """加载已保存的指数数据"""
        if self._df is not None:
            return self._df

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"沪深300数据不存在: {self.index_path}\n"
                "请先运行 regime.update_index_data()"
            )

        self._df = pd.read_parquet(self.index_path)
        return self._df

    def get_regime(self, as_of: str) -> str:
        """
        获取指定日期的市场环境

        Args:
            as_of: 日期 "YYYY-MM-DD"

        Returns:
            "BULL" / "BEAR" / "NEUTRAL"
        """
        df = self.load()
        d = pd.to_datetime(as_of).normalize()

        row = df[df["date"] == d]
        if row.empty:
            # 如果该日期没有数据，取最近的交易日
            df_before = df[df["date"] <= d]
            if df_before.empty:
                return "NEUTRAL"
            row = df_before.iloc[-1:]

        return row["regime"].values[0]

    def can_open_position(self, as_of: str) -> bool:
        """
        判断指定日期是否允许开仓

        Args:
            as_of: 日期 "YYYY-MM-DD"

        Returns:
            True = 允许开仓, False = 禁止开仓
        """
        df = self.load()
        d = pd.to_datetime(as_of).normalize()

        row = df[df["date"] == d]
        if row.empty:
            df_before = df[df["date"] <= d]
            if df_before.empty:
                return True  # 无数据时默认允许
            row = df_before.iloc[-1:]

        return bool(row["can_open"].values[0])

    def get_regime_df(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取一段时间的市场环境数据

        Returns:
            DataFrame with: date, regime, can_open
        """
        df = self.load()

        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        return df[["date", "close", "ma60", "ma200", "regime", "can_open"]].copy()


def main():
    """测试脚本"""
    import sys

    # 假设在项目根目录运行
    base_dir = Path(__file__).resolve().parent.parent

    regime = MarketRegime(base_dir)

    # 更新数据
    df = regime.update_index_data()

    # 显示最近的市场环境
    print("\n最近 10 个交易日的市场环境:")
    recent = df.tail(10)[["date", "close", "ma60", "ma200", "regime", "can_open"]]
    print(recent.to_string(index=False))

    # 统计
    print("\n市场环境统计:")
    print(df["regime"].value_counts())


if __name__ == "__main__":
    main()