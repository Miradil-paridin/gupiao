from pathlib import Path
import pandas as pd
import numpy as np

def sharpe_from_returns(r: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 5:
        return float("nan")

    rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1
    ex = r - rf_period

    sd = ex.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")

    return (ex.mean() / sd) * np.sqrt(periods_per_year)

def find_backtest_files(base_dir: Path):
    """
    在常见输出目录里找回测结果文件（CSV/Parquet），按修改时间倒序返回。
    """
    candidates = [
        base_dir / "data" / "backtests",
        base_dir / "data" / "backtest",
        base_dir / "data" / "bt",
        base_dir / "out" / "backtests",
        base_dir / "out",
        base_dir / "reports",
        base_dir / "data",
    ]

    files = []
    for d in candidates:
        if d.exists():
            files += list(d.glob("**/*.csv"))
            files += list(d.glob("**/*.parquet"))

    files = sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)
    return files

def load_returns_from_file(p: Path) -> pd.Series:
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file type: {p}")

    # 常见列名：收益率
    ret_cols = ["daily_return", "strategy_return", "ret", "return", "returns"]
    # 常见列名：净值
    nav_cols = ["nav", "equity", "portfolio_value", "value", "portfolio"]

    for c in ret_cols:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")

    nav = None
    for c in nav_cols:
        if c in df.columns:
            nav = pd.to_numeric(df[c], errors="coerce")
            break

    if nav is None:
        raise ValueError(
            f"Can't find returns or NAV columns in: {p}\n"
            f"Columns: {list(df.columns)}\n"
            f"Need one of returns={ret_cols} OR nav={nav_cols}"
        )

    return nav.pct_change()

def main():
    base_dir = Path(__file__).resolve().parents[1]

    files = find_backtest_files(base_dir)
    if not files:
        print("❌ 没找到任何 CSV/Parquet 回测结果文件。")
        print("你需要先运行回测脚本，生成净值/收益结果。")
        print("\n📌 你项目根目录下常见的回测脚本可能是：")
        for p in sorted(base_dir.glob("run_backtest*.py")):
            print("  -", p.name)

        print("\n✅ 你可以先试着运行其中一个，例如：")
        print("  python run_backtest_v2.py")
        print("或：")
        print("  python run_backtest_ma20_equal_weight.py")
        return

    # 默认取最新一个
    p = files[0]

    try:
        r = load_returns_from_file(p)
    except Exception as e:
        print(f"❌ 找到了文件但无法解析收益率：{p}")
        print("原因：", e)
        return

    s0 = sharpe_from_returns(r, rf_annual=0.0, periods_per_year=252)
    s2 = sharpe_from_returns(r, rf_annual=0.02, periods_per_year=252)

    print(f"✅ Using file: {p}")
    print(f"Sharpe (rf=0%):  {s0:.4f}")
    print(f"Sharpe (rf=2%):  {s2:.4f}")

if __name__ == "__main__":
    main()
