import os
from pathlib import Path
import pandas as pd

import qlib
from qlib.constant import REG_CN
from qlib.utils.time import Freq
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy


def to_qlib_code(code: str) -> str:
    """
    支持输入：
    - 600000.SH / 000001.SZ
    - SH600000 / SZ000001
    - 600000 / 000001（按首位推断交易所）
    输出统一为：SH600000 / SZ000001
    """
    c = str(code).strip().upper()

    if c.endswith(".SH") and len(c) >= 7:
        return "SH" + c[:6]
    if c.endswith(".SZ") and len(c) >= 7:
        return "SZ" + c[:6]

    if (c.startswith("SH") or c.startswith("SZ")) and len(c) == 8:
        return c

    if len(c) == 6 and c.isdigit():
        return ("SH" if c[0] in ("6", "9") else "SZ") + c

    raise ValueError(f"无法识别的股票代码格式: {code}")


def load_pred_score(signal_csv: str) -> pd.Series:
    """
    读取 out/signal.csv (date, code, score) -> pred_score: pd.Series
    index: MultiIndex(datetime, instrument)
    """
    df = pd.read_csv(signal_csv)

    required = {"date", "code", "score"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"{signal_csv} 缺少列: {sorted(miss)}，需要 {sorted(required)}")

    df["datetime"] = pd.to_datetime(df["date"])
    df["instrument"] = df["code"].map(to_qlib_code)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    pred_score = (
        df[["datetime", "instrument", "score"]]
        .dropna()
        .groupby(["datetime", "instrument"])["score"]
        .mean()
        .sort_index()
    )
    return pred_score


def main():
    # 1) init qlib
    data_dir = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    provider_uri = os.environ.get("QLIB_PROVIDER_URI", str(data_dir))
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # 2) load your signal
    signal_csv = "out/signal.csv"  # 你只需要保证这个文件存在且格式正确
    pred_score = load_pred_score(signal_csv)

    if pred_score.empty:
        raise ValueError("pred_score 为空：检查 out/signal.csv 是否有数据、日期是否正确、score 是否是数字")

    # 3) 自动用 signal 覆盖的日期范围做回测（更省事）
    start_time = pred_score.index.get_level_values(0).min().strftime("%Y-%m-%d")
    end_time = pred_score.index.get_level_values(0).max().strftime("%Y-%m-%d")

    # 4) 策略：TopK + Drop（注意参数名是 signal）
    STRATEGY_CONFIG = {
        "topk": 50,
        "n_drop": 5,
        "signal": pred_score,  # 关键：把你的打分喂给策略
    }
    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)

    # 5) 执行器：按天回测
    EXECUTOR_CONFIG = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    }
    executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)

    # 6) 回测参数（A股常用）
    FREQ = "day"
    backtest_config = {
        "start_time": start_time,
        "end_time": end_time,
        "account": 1_000_000,
        "benchmark": "SH000300",
        "exchange_kwargs": {
            "freq": FREQ,
            "trade_unit": 100,
            "limit_threshold": 0.095,
            "deal_price": "open",   # 次日开盘成交（更保守）
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }

    portfolio_metric_dict, indicator_dict = backtest(
        executor=executor_obj,
        strategy=strategy_obj,
        **backtest_config,
    )

    analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))  # 通常是 "1day"
    report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

    # 7) 风险分析（超额收益）
    analysis = {}
    analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"], freq=analysis_freq)
    analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq)
    analysis_df = pd.concat(analysis)

    # 8) 保存结果
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    report_normal.to_csv(out_dir / "qlib_report.csv", encoding="utf-8-sig")
    analysis_df.to_csv(out_dir / "qlib_risk.csv", encoding="utf-8-sig")

    print("\n=== 回测完成 ===")
    print(f"signal  : {signal_csv}")
    print(f"range   : {start_time} -> {end_time}")
    print(f"report  : {out_dir / 'qlib_report.csv'}")
    print(f"risk    : {out_dir / 'qlib_risk.csv'}")
    print("\n--- 风险指标（最后几行）---")
    print(analysis_df.tail())


if __name__ == "__main__":
    main()
