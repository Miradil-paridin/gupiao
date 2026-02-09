"""
数据质量检查脚本 v2

检查内容：
1. 日期范围是否正确（是否有早于 start_date 的数据）
2. 是否有缺失交易日
3. 是否有异常值（负价格、成交量为0等）
4. 复权数据是否正确
5. 科创板特殊检查（688开头）
"""
from __future__ import annotations
import datetime as dt
from pathlib import Path
import pandas as pd
import yaml


def load_config(base_dir: Path) -> dict:
    """Load config.yaml"""
    cfg_path = base_dir / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_kcb(code: str) -> bool:
    """判断是否是科创板（688开头）"""
    return code.startswith("688")


def check_date_range(df: pd.DataFrame, expected_start: str, symbol: str) -> list[str]:
    """检查日期范围是否正确"""
    issues = []
    expected_start_date = dt.date.fromisoformat(expected_start)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    actual_min = df["date"].min()
    actual_max = df["date"].max()

    if actual_min < expected_start_date:
        issues.append(
            f"❌ 数据早于配置的 start_date！"
            f"实际最早: {actual_min}, 配置: {expected_start_date}"
        )

    # 检查科创板上市日期
    code = symbol.split(".")[0]
    if is_kcb(code):
        # 科创板2019年7月22日开板
        kcb_start = dt.date(2019, 7, 22)
        if actual_min < kcb_start:
            issues.append(
                f"⚠️ 科创板数据早于开板日期（2019-07-22），请检查"
            )

    return issues


def check_missing_days(df: pd.DataFrame, symbol: str) -> list[str]:
    """检查是否有缺失的交易日（简化版，只检查连续性）"""
    issues = []
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # 计算日期间隔
    df["gap"] = df["date"].diff().dt.days

    # 超过 10 天的间隔可能有问题（考虑春节等长假）
    large_gaps = df[df["gap"] > 10]

    for _, row in large_gaps.iterrows():
        issues.append(
            f"⚠️ 大间隔 {int(row['gap'])} 天 @ {row['date'].date()}"
        )

    return issues


def check_data_quality(df: pd.DataFrame, symbol: str) -> list[str]:
    """检查数据质量（异常值）"""
    issues = []
    code = symbol.split(".")[0]

    # 检查负价格
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                issues.append(f"❌ {col} 有 {neg_count} 个负值")

    # 检查 high < low
    if "high" in df.columns and "low" in df.columns:
        invalid = (df["high"] < df["low"]).sum()
        if invalid > 0:
            issues.append(f"❌ {invalid} 行 high < low")

    # 检查 volume = 0 的天数（可能是停牌，但太多就有问题）
    if "volume" in df.columns:
        zero_vol = (df["volume"] == 0).sum()
        zero_pct = zero_vol / len(df) * 100
        if zero_pct > 20:
            issues.append(f"⚠️ {zero_pct:.1f}% 的天数成交量为0（可能有停牌）")

    # 检查涨跌幅异常
    if "pct_chg" in df.columns:
        # 科创板涨跌幅限制是 ±20%，主板是 ±10%
        # 但上市前5天无限制，所以用 30% 作为警告阈值
        limit = 35 if is_kcb(code) else 25
        extreme = df[df["pct_chg"].abs() > limit]
        if len(extreme) > 0:
            issues.append(
                f"⚠️ {len(extreme)} 天涨跌幅超过{limit}%（可能是上市初期或复权）"
            )

    return issues


def check_duplicates(df: pd.DataFrame, symbol: str) -> list[str]:
    """检查重复数据"""
    issues = []

    dup_count = df.duplicated(subset=["date"], keep=False).sum()
    if dup_count > 0:
        issues.append(f"❌ {dup_count} 行重复日期")

    return issues


def check_price_continuity(df: pd.DataFrame, symbol: str) -> list[str]:
    """检查价格连续性（发现复权问题）"""
    issues = []
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    if "close" not in df.columns:
        return issues

    # 计算日收益率
    df["ret"] = df["close"].pct_change()

    # 超过 50% 的单日变动可能是复权问题
    extreme_ret = df[df["ret"].abs() > 0.5]

    for _, row in extreme_ret.iterrows():
        issues.append(
            f"⚠️ 单日变动 {row['ret']*100:.1f}% @ {row['date']}（可能是复权断点）"
        )

    return issues


def main():
    base_dir = Path(__file__).resolve().parent
    cfg = load_config(base_dir)

    expected_start = cfg["market_data"]["start_date"]
    watchlist = cfg["watchlist"]

    print("=" * 60)
    print("数据质量检查报告 v2")
    print(f"配置的 start_date: {expected_start}")
    print("=" * 60)

    clean_dir = base_dir / "data" / "clean" / "market_daily"
    all_issues = []

    summary_data = []

    for code in watchlist:
        # 尝试找到对应的 parquet 文件
        possible_files = [
            clean_dir / f"{code}.SH.parquet",
            clean_dir / f"{code}.SZ.parquet",
            clean_dir / f"{code}.parquet",
        ]

        found = False
        for fpath in possible_files:
            if fpath.exists():
                df = pd.read_parquet(fpath)
                found = True

                symbol = fpath.stem
                board = "科创板" if is_kcb(code) else "主板/创业板"

                print(f"\n📊 {symbol} ({board}): {len(df)} 行")

                # 日期范围
                df_dates = pd.to_datetime(df["date"]).dt.date
                date_min = df_dates.min()
                date_max = df_dates.max()
                print(f"   日期范围: {date_min} -> {date_max}")

                # 各项检查
                issues = []
                issues.extend(check_date_range(df, expected_start, symbol))
                issues.extend(check_duplicates(df, symbol))
                issues.extend(check_data_quality(df, symbol))
                issues.extend(check_price_continuity(df, symbol))
                issues.extend(check_missing_days(df, symbol))

                if issues:
                    for issue in issues:
                        print(f"   {issue}")
                    all_issues.extend([(symbol, issue) for issue in issues])
                else:
                    print(f"   ✅ 数据质量良好")

                # 收集汇总数据
                summary_data.append({
                    "symbol": symbol,
                    "rows": len(df),
                    "start": date_min,
                    "end": date_max,
                    "issues": len(issues),
                })

                break

        if not found:
            print(f"\n❌ {code}: 未找到数据文件")
            all_issues.append((code, "未找到数据文件"))

    # 汇总表格
    print("\n" + "=" * 60)
    print("数据汇总")
    print("=" * 60)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        print(f"\n总行数: {summary_df['rows'].sum():,}")
        print(f"日期范围: {summary_df['start'].min()} -> {summary_df['end'].max()}")

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)

    if all_issues:
        # 统计问题类型
        errors = [i for i in all_issues if "❌" in i[1]]
        warnings = [i for i in all_issues if "⚠️" in i[1]]

        print(f"❌ 错误: {len(errors)} 个")
        print(f"⚠️ 警告: {len(warnings)} 个")

        if errors:
            print("\n必须修复的错误:")
            for symbol, issue in errors:
                print(f"  {symbol}: {issue}")

        if warnings and not errors:
            print("\n⚠️ 有警告但无严重错误，数据可以使用。")
            print("   警告通常是正常现象（停牌、复权、科创板特殊规则等）")
    else:
        print("✅ 所有数据质量良好！可以放心进行回测。")


if __name__ == "__main__":
    main()