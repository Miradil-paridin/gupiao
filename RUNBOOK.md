# 量化流程运行手册（Daily / Weekly）

本手册用于日常执行本项目流程，包含：
- 每天运行什么
- 每周运行什么
- 常用参数（你说的“后缀”）

当前默认行为（已调整）：
- `run_all_daily.py` 回测池自动优先使用 `watchlist_cache.yaml`
- 若不存在，再回退到 `backtest_watchlist.yaml`

---

## 1. 每天执行（推荐）

```powershell
python run_all_daily.py
```

说明：
- 会跑抓数、特征、信号、新闻、快速回测、策略治理、日报/仪表盘。
- 默认配置解析顺序：`--config` > `PIPELINE_CONFIG` > `config.yaml` > `config_v31.yaml`。

如果你想显式指定配置：

```powershell
python run_all_daily.py --config config.yaml
```

---

## 2. 每周执行（更新股票池）

每周（建议周末或夜间）执行一次：

```powershell
python run_update_watchlist.py --days 180
python sync_watchlist.py
```

然后再跑一次全流程：

```powershell
python run_all_daily.py --config config.yaml
```

说明：
- `run_update_watchlist.py` 会更新 `watchlist_cache.yaml`。
- `sync_watchlist.py` 会把最新池同步进 `config.yaml` / `config_v31.yaml`（便于数据抓取和显示一致）。

可选：周末顺手补历史新闻（建议限流）：

```powershell
python run_backfill_news_history.py --start-date 2025-01-01 --end-date 2026-02-23 --watchlist watchlist_cache.yaml --max-days 5 --max-symbols 80 --latest-first
```

---

## 3. 常用“后缀参数”（按需）

## 3.1 快速模式（省时间）

```powershell
python run_all_daily.py --fast
```

会跳过新闻、图表、部分报告步骤。

## 3.2 指定回测股票池文件

```powershell
python run_all_daily.py --backtest-watchlist watchlist_cache.yaml
```

用于强制回测使用指定池，不走自动选择。

## 3.3 指定动态池规模（TopN）

```powershell
python run_all_daily.py --backtest-top-n 300
```

默认已经是 `300`，此参数用于临时改动。

## 3.4 跳过某些步骤

```powershell
python run_all_daily.py --skip-news
python run_all_daily.py --skip-backtest
python run_all_daily.py --skip-backtest-report
python run_all_daily.py --skip-paper-trade
python run_all_daily.py --skip-health-card
python run_all_daily.py --skip-gate-calibration
python run_all_daily.py --skip-crash-calibration
```

## 3.5 在 run_all_daily 里直接触发新闻回补

```powershell
python run_all_daily.py --news-backfill-start 2025-01-01 --news-backfill-end 2026-02-23 --news-backfill-max-days 5 --news-backfill-max-symbols 80 --news-backfill-latest-first
```

## 3.6 数据后端切换（Parquet / DuckDB）

```powershell
# 默认兼容模式
$env:QUANT_DATA_BACKEND='parquet'

# 迁移推荐：双写
$env:QUANT_DATA_BACKEND='hybrid'

# 先把历史 parquet 灌入 DuckDB
python run_migrate_market_to_duckdb.py --base-dir .

# 完成后可切纯 DuckDB
$env:QUANT_DATA_BACKEND='duckdb'
```

---

## 4. 常用单独回测命令

## 4.1 更新后静态池（推荐核验口径）

```powershell
python run_backtest_strategy_v3.py --base-dir . --config config.yaml --start-date 2020-01-01 --watchlist watchlist_cache.yaml --dynamic-watchlist --dynamic-top-n 300
```

## 4.2 全市场动态池（不使用静态池）

```powershell
python run_backtest_strategy_v3.py --base-dir . --config config.yaml --start-date 2020-01-01 --no-default-watchlist --dynamic-watchlist --dynamic-top-n 300
```

## 4.3 单独生成策略健康诊断卡

```powershell
python run_generate_strategy_health_card.py
```

输出：
- `data/backtests/strategy_health_card.json`
- `data/reports/strategy_health_card_latest.md`

## 4.4 单独生成闸门阈值校准报告

```powershell
python run_generate_gate_calibration.py
```

输出：
- `data/backtests/strategy_gate_calibration.json`
- `data/backtests/strategy_gate_calibration_grid.csv`
- `data/backtests/strategy_gate_monitor_replay.csv`
- `data/reports/strategy_gate_calibration_latest.md`

## 4.5 单独生成动量崩盘保护参数校准报告

```powershell
python run_generate_momentum_crash_calibration.py --base-dir . --config config.yaml
# 两阶段（含真实回放）示例：
python run_generate_momentum_crash_calibration.py --base-dir . --config config.yaml --replay-top-k 1 --replay-start-date 2025-01-01
```

输出：
- `data/backtests/momentum_crash_calibration.json`
- `data/backtests/momentum_crash_calibration_grid.csv`
- `data/backtests/momentum_crash_calibration_replay.csv`
- `data/backtests/momentum_crash_trigger_log.csv`
- `data/reports/momentum_crash_calibration_latest.md`

---

## 5. 建议节奏

- 工作日白天：`run_all_daily.py`（默认）
- 每周一次：更新股票池 + 同步 + 跑一次全流程
- 重大参数变更后：跑一次 `run_backtest_strategy_v3.py` 做独立核验

---

## 6. 常见问题

## 6.1 为什么我更新了股票池，结果还像旧池？

请确认是否执行了：
1. `python run_update_watchlist.py --days 180`
2. `python sync_watchlist.py`

并在 `run_all_daily.py` 中不要强制指定旧池文件。

## 6.2 为什么日志里显示编码报错（GBK/emoji）？

可用：

```powershell
$env:PYTHONIOENCODING='utf-8'
python -X utf8 run_all_daily.py
```
