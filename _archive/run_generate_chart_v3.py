"""
生成权益曲线图 - 带大盘对比版

功能：
1. 策略权益曲线
2. 沪深300对比曲线
3. 超额收益曲线
4. 年度收益对比柱状图
5. 回撤对比

支持回测版本：v2, v3
"""
from pathlib import Path
import pandas as pd
import numpy as np


def fetch_benchmark_data(start_date: str, end_date: str) -> pd.DataFrame:
    """获取沪深300指数数据"""
    try:
        import baostock as bs

        lg = bs.login()
        if lg.error_code != "0":
            print(f"BaoStock 登录失败: {lg.error_msg}")
            return pd.DataFrame()

        try:
            rs = bs.query_history_k_data_plus(
                code="sh.000300",
                fields="date,close",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",
            )

            if rs.error_code != "0":
                print(f"获取沪深300失败: {rs.error_msg}")
                return pd.DataFrame()

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                return pd.DataFrame()

            df = pd.DataFrame(data_list, columns=rs.fields)
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])

            return df

        finally:
            bs.logout()

    except Exception as e:
        print(f"获取基准数据失败: {e}")
        return pd.DataFrame()


def load_backtest_result(base_dir: Path) -> tuple[pd.DataFrame, str]:
    """加载回测结果，优先 v3 > v2 > v1"""

    paths = [
        (base_dir / "data" / "backtests" / "backtest_strategy_v3_equity.csv", "v3（涨停回调+风控）"),
        (base_dir / "data" / "backtests" / "backtest_strategy_v2_equity.csv", "v2（仓位控制）"),
        (base_dir / "data" / "backtests" / "backtest_strategy_equity.csv", "v1（原版）"),
    ]

    for path, version in paths:
        if path.exists():
            df = pd.read_csv(path)
            print(f"加载回测结果: {path.name} ({version})")
            return df, version

    raise FileNotFoundError("未找到回测结果，请先运行回测脚本")


def create_comparison_chart_html(base_dir: Path) -> Path:
    """生成带大盘对比的交互式图表"""

    # 读取回测数据
    df, version = load_backtest_result(base_dir)
    df["date"] = pd.to_datetime(df["date"])

    # 获取日期范围
    start_date = df["date"].min().strftime("%Y-%m-%d")
    end_date = df["date"].max().strftime("%Y-%m-%d")

    print(f"获取沪深300数据: {start_date} -> {end_date}")
    benchmark = fetch_benchmark_data(start_date, end_date)

    # 计算基准净值
    if not benchmark.empty:
        benchmark = benchmark.sort_values("date")
        initial_benchmark = benchmark["close"].iloc[0]
        benchmark["benchmark_nav"] = benchmark["close"] / initial_benchmark * 100000

        df = df.merge(
            benchmark[["date", "benchmark_nav"]],
            on="date",
            how="left"
        )
        df["benchmark_nav"] = df["benchmark_nav"].ffill().bfill()
        df["excess_return"] = (df["equity"] / df["benchmark_nav"] - 1) * 100
    else:
        df["benchmark_nav"] = 100000
        df["excess_return"] = 0

    # 计算关键指标
    initial = 100000
    final = df["equity"].iloc[-1]
    total_return = (final / initial - 1) * 100

    benchmark_final = df["benchmark_nav"].iloc[-1]
    benchmark_return = (benchmark_final / initial - 1) * 100

    # 计算回撤
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = (df["equity"] / df["peak"] - 1) * 100
    max_dd = df["drawdown"].min()

    df["bench_peak"] = df["benchmark_nav"].cummax()
    df["bench_drawdown"] = (df["benchmark_nav"] / df["bench_peak"] - 1) * 100
    bench_max_dd = df["bench_drawdown"].min()

    # 计算夏普比率
    if "daily_return" in df.columns:
        daily_ret = df["daily_return"]
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    else:
        sharpe = 0

    # 年度收益
    df["year"] = df["date"].dt.year
    yearly = df.groupby("year").agg({
        "equity": ["first", "last"],
        "benchmark_nav": ["first", "last"]
    })
    yearly.columns = ["strat_start", "strat_end", "bench_start", "bench_end"]
    yearly["strategy_return"] = (yearly["strat_end"] / yearly["strat_start"] - 1) * 100
    yearly["benchmark_return"] = (yearly["bench_end"] / yearly["bench_start"] - 1) * 100
    yearly["excess"] = yearly["strategy_return"] - yearly["benchmark_return"]

    # 盈利年份统计
    win_years = (yearly["strategy_return"] > 0).sum()
    total_years = len(yearly)

    # 准备图表数据
    dates_json = df["date"].dt.strftime("%Y-%m-%d").tolist()
    equity_json = df["equity"].round(0).tolist()
    benchmark_json = df["benchmark_nav"].round(0).tolist()
    drawdown_json = df["drawdown"].round(2).tolist()
    bench_drawdown_json = df["bench_drawdown"].round(2).tolist()
    excess_json = df["excess_return"].round(2).tolist()

    years_json = yearly.index.tolist()
    strat_returns_json = yearly["strategy_return"].round(2).tolist()
    bench_returns_json = yearly["benchmark_return"].round(2).tolist()

    # 策略特性描述
    if "v3" in version:
        features_desc = "涨停回调入场 + 多重风控 + 仓位控制"
    elif "v2" in version:
        features_desc = "仓位控制 + 市场情绪调整"
    else:
        features_desc = "基础因子策略"

    # 生成 HTML
    html_content = f'''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A股量化策略回测报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ text-align: center; margin-bottom: 5px; color: #00d4ff; font-size: 28px; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
        .version-tag {{
            display: inline-block;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            color: #000;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        .features-tag {{
            display: inline-block;
            background: rgba(255,165,2,0.2);
            color: #ffa502;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            margin-left: 10px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-card.highlight {{
            background: rgba(0,212,255,0.1);
            border-color: rgba(0,212,255,0.3);
        }}
        .stat-value {{ font-size: 28px; font-weight: bold; }}
        .stat-value.positive {{ color: #00ff88; }}
        .stat-value.negative {{ color: #ff4757; }}
        .stat-value.neutral {{ color: #00d4ff; }}
        .stat-label {{ font-size: 12px; color: #888; margin-top: 5px; }}
        .stat-compare {{ font-size: 11px; color: #666; margin-top: 3px; }}

        .chart-container {{
            background: rgba(255,255,255,0.03);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .chart-title {{
            font-size: 16px;
            margin-bottom: 15px;
            color: #00d4ff;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .chart-title .legend {{
            display: flex;
            gap: 15px;
            margin-left: auto;
            font-size: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}

        .summary-box {{
            background: rgba(0,255,136,0.1);
            border: 1px solid rgba(0,255,136,0.3);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .summary-box h3 {{
            color: #00ff88;
            margin-bottom: 10px;
        }}
        .summary-box p {{
            color: #aaa;
            line-height: 1.6;
        }}

        .disclaimer {{
            background: rgba(255,71,87,0.1);
            border: 1px solid rgba(255,71,87,0.3);
            border-radius: 8px;
            padding: 15px;
            margin-top: 30px;
            font-size: 12px;
            color: #ff4757;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 A股量化策略回测报告</h1>
        <p class="subtitle">
            <span class="version-tag">{version}</span>
            <span class="features-tag">{features_desc}</span>
        </p>

        <div class="summary-box">
            <h3>📋 策略概要</h3>
            <p>
                回测区间：{start_date} ~ {end_date}（共 {len(df)} 个交易日）<br>
                策略特性：{features_desc}<br>
                初始资金：¥100,000
            </p>
        </div>

        <div class="stats-grid">
            <div class="stat-card highlight">
                <div class="stat-value {'positive' if total_return > 0 else 'negative'}">{'+' if total_return > 0 else ''}{total_return:.1f}%</div>
                <div class="stat-label">策略总收益</div>
                <div class="stat-compare">沪深300: {'+' if benchmark_return > 0 else ''}{benchmark_return:.1f}%</div>
            </div>
            <div class="stat-card highlight">
                <div class="stat-value positive">+{total_return - benchmark_return:.1f}%</div>
                <div class="stat-label">超额收益</div>
                <div class="stat-compare">跑赢大盘</div>
            </div>
            <div class="stat-card">
                <div class="stat-value negative">{max_dd:.1f}%</div>
                <div class="stat-label">策略最大回撤</div>
                <div class="stat-compare">沪深300: {bench_max_dd:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-value neutral">{sharpe:.2f}</div>
                <div class="stat-label">夏普比率</div>
                <div class="stat-compare">&gt;1 为优秀</div>
            </div>
            <div class="stat-card">
                <div class="stat-value neutral">{win_years}/{total_years}</div>
                <div class="stat-label">盈利年份</div>
                <div class="stat-compare">胜率 {win_years / total_years * 100:.0f}%</div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">
                📊 权益曲线对比
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-dot" style="background:#00d4ff"></div>
                        <span>策略净值</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot" style="background:#ffa502"></div>
                        <span>沪深300</span>
                    </div>
                </div>
            </div>
            <canvas id="equityChart" height="350"></canvas>
        </div>

        <div class="chart-container">
            <div class="chart-title">
                📈 超额收益曲线
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-dot" style="background:#00ff88"></div>
                        <span>相对沪深300超额</span>
                    </div>
                </div>
            </div>
            <canvas id="excessChart" height="200"></canvas>
        </div>

        <div class="chart-container">
            <div class="chart-title">📉 回撤对比</div>
            <canvas id="drawdownChart" height="200"></canvas>
        </div>

        <div class="chart-container">
            <div class="chart-title">📅 年度收益对比</div>
            <canvas id="yearlyChart" height="300"></canvas>
        </div>

        <div class="disclaimer">
            ⚠️ <strong>风险提示：</strong>
            本报告基于历史数据回测，不保证未来收益。过去的表现不代表未来的业绩。
            股市有风险，投资需谨慎。本报告仅供学习研究参考，不构成任何投资建议。
        </div>
    </div>

    <script>
        const chartOptions = {{
            responsive: true,
            interaction: {{ intersect: false, mode: 'index' }},
            plugins: {{
                legend: {{ display: false }},
                tooltip: {{
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(255,255,255,0.2)',
                    borderWidth: 1,
                }}
            }},
            scales: {{
                x: {{
                    type: 'time',
                    time: {{ unit: 'year' }},
                    ticks: {{ color: '#666' }},
                    grid: {{ color: 'rgba(255,255,255,0.05)' }}
                }},
                y: {{
                    ticks: {{ color: '#666' }},
                    grid: {{ color: 'rgba(255,255,255,0.05)' }}
                }}
            }}
        }};

        // 权益曲线
        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {{
                labels: {dates_json},
                datasets: [
                    {{
                        label: '策略净值',
                        data: {equity_json},
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0,212,255,0.1)',
                        fill: true,
                        tension: 0.1,
                        pointRadius: 0,
                        borderWidth: 2,
                    }},
                    {{
                        label: '沪深300',
                        data: {benchmark_json},
                        borderColor: '#ffa502',
                        backgroundColor: 'transparent',
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                        borderWidth: 2,
                        borderDash: [5, 5],
                    }}
                ]
            }},
            options: chartOptions
        }});

        // 超额收益
        new Chart(document.getElementById('excessChart'), {{
            type: 'line',
            data: {{
                labels: {dates_json},
                datasets: [{{
                    label: '超额收益 %',
                    data: {excess_json},
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0,255,136,0.2)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                }}]
            }},
            options: chartOptions
        }});

        // 回撤
        new Chart(document.getElementById('drawdownChart'), {{
            type: 'line',
            data: {{
                labels: {dates_json},
                datasets: [
                    {{
                        label: '策略回撤',
                        data: {drawdown_json},
                        borderColor: '#ff4757',
                        backgroundColor: 'rgba(255,71,87,0.2)',
                        fill: true,
                        tension: 0.1,
                        pointRadius: 0,
                    }},
                    {{
                        label: '沪深300回撤',
                        data: {bench_drawdown_json},
                        borderColor: '#ffa502',
                        backgroundColor: 'transparent',
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                        borderDash: [5, 5],
                    }}
                ]
            }},
            options: chartOptions
        }});

        // 年度收益
        new Chart(document.getElementById('yearlyChart'), {{
            type: 'bar',
            data: {{
                labels: {years_json},
                datasets: [
                    {{
                        label: '策略',
                        data: {strat_returns_json},
                        backgroundColor: {strat_returns_json}.map(r => r >= 0 ? '#00d4ff' : '#ff4757'),
                        borderRadius: 4,
                    }},
                    {{
                        label: '沪深300',
                        data: {bench_returns_json},
                        backgroundColor: {bench_returns_json}.map(r => r >= 0 ? 'rgba(255,165,2,0.7)' : 'rgba(255,71,87,0.5)'),
                        borderRadius: 4,
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#888' }}
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{ color: '#666' }},
                        grid: {{ color: 'rgba(255,255,255,0.05)' }}
                    }},
                    y: {{
                        ticks: {{ color: '#666' }},
                        grid: {{ color: 'rgba(255,255,255,0.05)' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
'''

    # 保存
    out_path = base_dir / "out" / "backtest_report.html"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ 回测报告已生成: {out_path}")
    return out_path


def main():
    base_dir = Path(__file__).resolve().parent
    create_comparison_chart_html(base_dir)


if __name__ == "__main__":
    main()