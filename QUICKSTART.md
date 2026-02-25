# A股量化项目 Quick Start

> 与 `README.md` 同步口径。  
> 这份文档只讲最短可执行路径。

## 1. 一次性环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. 每日运行（推荐）

```powershell
python run_all_daily.py
```

这条命令会按顺序执行：

1. 数据更新与特征构建  
2. 信号生成  
3. quick backtest  
4. 策略治理流程（`strategy_process_pipeline`）  
5. 读取 `quality_gate` + `failure_monitor`  
6. 将建议仓位上限自动传给 `run_paper_trading.py`  
7. 生成日报与仪表盘

## 3. 快速模式（仅做快检查）

```powershell
python run_all_daily.py --fast
```

`--fast` 会跳过部分耗时步骤，适合临时检查，不适合完整复盘。

## 4. Gate 相关常用命令

### 4.1 手动跑策略治理

```powershell
python tools/strategy_process_pipeline.py --base-dir . --max-combos 24 --monitor-days 60
```

### 4.2 临时忽略 gate（仅调试）

```powershell
python run_all_daily.py --paper-ignore-gate
```

不建议长期使用该参数。

## 5. 先看哪些结果

1. 信号：`data/signals/latest_daily_rank.csv`  
2. 策略治理汇总：`data/backtests/strategy_process_summary.json`  
3. 策略治理报告：`data/backtests/strategy_process_report.md`  
4. Paper 状态：`data/paper/state.json`  
5. Paper 交易：`data/paper/trades.csv`  
6. Paper 净值：`data/paper/equity.csv`

## 6. 当前判定逻辑（核心）

1. `quality_gate = pass`：正常  
2. `quality_gate = warn`：降仓（按监控建议 cap）  
3. `quality_gate = fail` 或 `monitor_action = stop`：强降仓/停机逻辑

## 7. 常见问题

### Q1: 报 `No module named 'baostock'`

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install baostock
```

### Q2: 想确认今天 gate 结果

直接看：

- `data/backtests/strategy_process_summary.json` 的 `quality_gate`
- `data/backtests/strategy_process_summary.json` 的 `failure_monitor`

---

更多背景、当前进度、优化路线图请看 `README.md`。

