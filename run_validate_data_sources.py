"""
数据交叉验证脚本

比对多个数据源的数据一致性，确保回测数据准确。
主要检查：
1. 收盘价差异（复权后）
2. 成交量差异
3. 缺失交易日
"""
from __future__ import annotations
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant.providers import get_provider, PROVIDERS, ProviderError


def fetch_sample_data(
        provider_name: str,
        code6: str,
        start: dt.date,
        end: dt.date,
        adjust: str = "qfq"
) -> pd.DataFrame | None:
    """从指定 provider 获取数据"""
    try:
        provider = get_provider(provider_name)
        df = provider.fetch_daily(code6=code6, start=start, end=end, adjust=adjust)
        if df is not None and not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.sort_values("date").reset_index(drop=True)
            return df
    except Exception as e:
        print(f"  ⚠️ {provider_name} 获取失败: {e}")
    return None


def compare_providers(
        code6: str,
        start: dt.date,
        end: dt.date,
        adjust: str = "qfq",
        tolerance_pct: float = 0.5,  # 允许的价格差异百分比
) -> dict:
    """
    比对多个数据源的数据

    Args:
        code6: 股票代码
        start: 开始日期
        end: 结束日期
        adjust: 复权类型
        tolerance_pct: 允许的价格差异百分比

    Returns:
        比对结果字典
    """
    result = {
        "code": code6,
        "providers_tested": [],
        "providers_success": [],
        "price_diff_max_pct": 0,
        "volume_diff_max_pct": 0,
        "date_mismatches": [],
        "issues": [],
        "data": {},
    }

    print(f"\n{'=' * 60}")
    print(f"验证股票: {code6} ({start} -> {end})")
    print("=" * 60)

    # 获取各数据源数据
    for name in ["baostock", "akshare", "sina"]:
        result["providers_tested"].append(name)
        df = fetch_sample_data(name, code6, start, end, adjust)
        if df is not None and not df.empty:
            result["providers_success"].append(name)
            result["data"][name] = df
            print(f"  ✅ {name}: {len(df)} 行")
        else:
            print(f"  ❌ {name}: 无数据")

    # 如果少于2个数据源成功，无法比对
    if len(result["providers_success"]) < 2:
        result["issues"].append("数据源不足，无法交叉验证")
        print("  ⚠️ 数据源不足，无法交叉验证")
        return result

    # 使用第一个成功的作为基准
    base_name = result["providers_success"][0]
    base_df = result["data"][base_name]

    print(f"\n  基准数据源: {base_name}")

    # 与其他数据源比对
    for other_name in result["providers_success"][1:]:
        other_df = result["data"][other_name]

        # 合并比对（按日期）
        merged = pd.merge(
            base_df[["date", "close", "volume"]].rename(
                columns={"close": "close_base", "volume": "volume_base"}
            ),
            other_df[["date", "close", "volume"]].rename(
                columns={"close": "close_other", "volume": "volume_other"}
            ),
            on="date",
            how="outer",
            indicator=True,
        )

        # 检查日期缺失
        left_only = merged[merged["_merge"] == "left_only"]
        right_only = merged[merged["_merge"] == "right_only"]

        if len(left_only) > 0:
            result["date_mismatches"].append(
                f"{other_name} 缺少 {len(left_only)} 个交易日"
            )
        if len(right_only) > 0:
            result["date_mismatches"].append(
                f"{base_name} 缺少 {len(right_only)} 个交易日"
            )

        # 只比较都有的日期
        both = merged[merged["_merge"] == "both"].copy()

        if len(both) == 0:
            result["issues"].append(f"{base_name} 和 {other_name} 没有共同交易日")
            continue

        # 计算价格差异
        both["price_diff_pct"] = (
                (both["close_base"] - both["close_other"]).abs()
                / both["close_base"] * 100
        )
        max_price_diff = both["price_diff_pct"].max()
        result["price_diff_max_pct"] = max(
            result["price_diff_max_pct"], max_price_diff
        )

        # 计算成交量差异
        both["volume_diff_pct"] = (
                (both["volume_base"] - both["volume_other"]).abs()
                / (both["volume_base"] + 1) * 100
        )
        max_volume_diff = both["volume_diff_pct"].max()
        result["volume_diff_max_pct"] = max(
            result["volume_diff_max_pct"], max_volume_diff
        )

        print(f"\n  {base_name} vs {other_name}:")
        print(f"    共同交易日: {len(both)}")
        print(f"    收盘价最大差异: {max_price_diff:.4f}%")
        print(f"    成交量最大差异: {max_volume_diff:.2f}%")

        # 检查是否超过容忍度
        if max_price_diff > tolerance_pct:
            result["issues"].append(
                f"收盘价差异 {max_price_diff:.4f}% 超过阈值 {tolerance_pct}%"
            )

            # 显示差异最大的几天
            worst = both.nlargest(3, "price_diff_pct")
            print(f"\n    ⚠️ 价格差异超过阈值！最大差异日期:")
            for _, row in worst.iterrows():
                print(
                    f"      {row['date']}: {base_name}={row['close_base']:.2f}, "
                    f"{other_name}={row['close_other']:.2f}, "
                    f"差异={row['price_diff_pct']:.4f}%"
                )

    return result


def main():
    print("=" * 60)
    print("数据交叉验证")
    print("=" * 60)

    # 测试参数
    end = dt.date.today()
    start = end - dt.timedelta(days=60)  # 最近60天

    # 测试几只代表性股票
    test_codes = [
        ("600519", "贵州茅台（主板）"),
        ("000921", "海信家电（深市主板）"),
        ("688981", "中芯国际（科创板）"),
    ]

    all_results = []

    for code6, name in test_codes:
        print(f"\n📊 {name}")
        result = compare_providers(code6, start, end, adjust="qfq")
        all_results.append(result)

    # 汇总报告
    print("\n" + "=" * 60)
    print("验证汇总")
    print("=" * 60)

    has_issues = False
    for result in all_results:
        code = result["code"]
        if result["issues"]:
            has_issues = True
            print(f"\n❌ {code}:")
            for issue in result["issues"]:
                print(f"   - {issue}")
        else:
            print(f"✅ {code}: 数据一致 (最大价格差异: {result['price_diff_max_pct']:.4f}%)")

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)

    if has_issues:
        print("""
⚠️ 发现数据差异！

可能原因：
1. 复权算法差异（不同数据源复权方式略有不同）
2. 除权除息日期对齐问题
3. 数据源更新时间差

建议：
- 如果差异 < 1%，通常可以接受
- 如果差异 > 1%，建议人工检查具体日期
- 坚持使用同一个数据源进行回测和实盘
        """)
    else:
        print("""
✅ 所有数据源数据一致！

BaoStock 数据可以放心使用。
        """)


if __name__ == "__main__":
    main()