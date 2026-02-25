# A股量化交易系统（个人自用版）

> 本项目定位为个人研究与实盘前验证工具，不是售卖产品。  
> 目标不是“调一个神参数”，而是建立一套可持续迭代、可风控、可验证的策略工程流程。

## 1. 这个程序是干什么的

这是一个 A 股日度量化流水线，核心用途是：

1. 自动更新行情与特征数据  
2. 生成每日选股与目标权重（`latest_daily_rank.csv`）  
3. 执行回测与样本外稳定性验证  
4. 进行 paper trading 记账与持仓跟踪  
5. 产出日报 / 仪表盘 / 回测报告，形成闭环

一句话：`run_all_daily.py` 会把“数据 -> 信号 -> 回测 -> 风控 -> 报告”串成一条可重复执行的日常流程。

---

## 2. 当前进度（截至 2026-02-23）

### 2.1 已落地能力（主流程 + 回测 + Paper）

1. 策略治理流程（`tools/strategy_process_pipeline.py`）  
方案：固定评估框架 + Walk-forward + rolling + 参数稳定区 + 风险情景层 + 双窗口失效监控。

2. 交易闸门（`trade_gate`）  
方案：`pass / warn / fail -> normal / reduce / stop`，并输出仓位上限，主流程与 paper 自动联动。

3. 主流程联动（`run_all_daily.py`）  
方案：优先读取新 `trade_gate`，兼容旧 `quality_gate/failure_monitor` 回退逻辑。

4. 再平衡死区（`rebalance_band`）  
方案：已在 paper 与回测双端生效，降低无效换手和摩擦成本。

5. 执行现实性约束已接入治理  
方案：`execution_slippage_drag`（硬约束）+ `avg_turnover_control`（告警约束）。

6. 过拟合防护（PBO/DSR）已接入治理  
方案：输出 `overfit_diagnostics`，并纳入 `quality_gate` 检查（`pbo_control`、`dsr_control`）。

7. 动量崩盘保护已接入回测主仓位闸门  
方案：识别“急跌后反抽”阶段并自动降仓，和 `index_filter/drawdown_brake` 叠加取更严上限。

### 2.2 配置与参数联动状态

1. `config.yaml` / `config_v31.yaml` 已接入：  
`strategy.rebalance_band`、`paper_trading.rebalance_band`、`risk_control.momentum_crash_protection`。

2. `run_all_daily.py` 已透传治理参数：  
`monitor-short/long`、`warn-cap`、`gate-max-avg-turnover`、`gate-max-slippage-drag-pct`、`gate-max-pbo`、`gate-min-dsr`。

3. `tools/strategy_process_pipeline.py` 已支持过拟合诊断参数：  
`--overfit-max-combos`、`--overfit-cv-splits`、`--gate-max-pbo`、`--gate-min-dsr`。

### 2.3 已完成验证（可复现实测）

1. 语法检查通过：  
`run_all_daily.py`、`run_backtest_strategy_v3.py`、`run_paper_trading.py`、`tools/strategy_process_pipeline.py`。

2. 参数联动检查通过：  
上述新增 CLI 参数在 `--help` 中可见，并可透传运行。

3. 轻量治理实跑通过：  
可产出 `trade_gate`、`overfit_diagnostics`、`quality_gate` 新检查项。

4. 短区间回测实跑通过：  
日志显示 `Momentum crash protection: on`，净值文件新增相关列。

### 2.4 进度结论（当前阶段）

1. 原 Phase A 的核心项已基本完成（交易闸门、双窗口监控、执行现实化）。  
2. 现阶段重点已从“功能补齐”转为“阈值校准 + 稳定性增强 + 回归防漂移”。

### 2.5 本次新增进度（2026-02-23，已落地）

1. 退出逻辑优化已接入  
- `rank_exit` 增加“仅趋势转弱才触发”过滤（`rank_exit_refine.only_when_trend_down: true`）。  
- `ma_stop` 与 `trailing_stop` 同时命中时支持优先级仲裁（`exit_conflict_policy.ma_vs_trailing`）。

2. 新闻情绪因子链路已打通（默认低权重关闭）  
- 支持将新闻情绪并入打分并做 A/B。  
- LLM provider 默认走 DeepSeek（环境变量可切换）。

3. 新闻历史回补能力已接入主流程  
- 新增脚本：`run_backfill_news_history.py`（区间回补、跳过已完成日期、强制重抓）。  
- `run_all_daily.py` 已支持回补参数透传，并支持按配置自动周回补。  
- 新增限流参数：`--max-days`、`--max-symbols`、`--latest-first`（避免一次跑太久超时）。

4. 股票池工作流已明确  
- 日常建议使用较小静态池跑主流程（速度快、结果稳定）。  
- 夜间/每周更新 `watchlist_cache.yaml`，主流程自动用新池继续运行。

5. 历史新闻回补实测进度（当前）  
- 已完成新增日期：`2025-01-21/22/23/24/27`（以及此前已有日期）。  
- 在 `2025-01-01 ~ 2026-02-23` 区间内，当前有 `manifest` 的交易日约 `17` 天，剩余约 `256` 天待补（按当前本地数据口径统计）。

---

## 3. 当前默认策略口径（config.yaml，v31 为可切换基线）

- `top_k: 10`
- `invest_more_n: 14`
- `max_total_position: 0.92`
- `stop_loss_pct: 0.08`
- `trailing_stop_pct: 0.10`
- `use_index_filter: true`
- `index_filter_hard_gate: false`
- `index_filter_block_position_cap: 0.30`

说明：指数门控当前是“软门控降仓”而非“硬拦截禁开仓”。
补充：`run_all_daily.py` 默认读取 `config.yaml`，若你要对比历史口径可用 `--config config_v31.yaml` 切换。

---

## 4. 你现在如何日常使用

### 4.1 常规日跑

```bash
python run_all_daily.py
```

### 4.2 快速模式（跳过部分耗时步骤）

```bash
python run_all_daily.py --fast
```

### 4.3 临时忽略 gate（仅调试用）

```bash
python run_all_daily.py --paper-ignore-gate
```

### 4.4 指定配置文件运行（对比 v31）

```bash
python run_all_daily.py --config config_v31.yaml
```

---

## 5. 主要输出文件

- 信号：`data/signals/latest_daily_rank.csv`
- 回测统计：`data/backtests/backtest_strategy_v3_stats.json`
- 回测净值：`data/backtests/backtest_strategy_v3_equity.csv`
- 策略治理汇总：`data/backtests/strategy_process_summary.json`
- 策略治理报告：`data/backtests/strategy_process_report.md`
- 策略健康诊断卡：`data/backtests/strategy_health_card.json`
- 策略健康诊断卡（Markdown）：`data/reports/strategy_health_card_latest.md`
- 闸门阈值校准：`data/backtests/strategy_gate_calibration.json`
- 闸门阈值校准网格：`data/backtests/strategy_gate_calibration_grid.csv`
- 闸门阈值校准报告：`data/reports/strategy_gate_calibration_latest.md`
- Paper 状态：`data/paper/state.json`
- Paper 交易：`data/paper/trades.csv`
- Paper 净值：`data/paper/equity.csv`
- 日度仪表盘：`out/daily_dashboard.html`
- AI 研报仪表盘：`out/ai_report_dashboard.html`
- 交易日志仪表盘：`out/trade_journal_dashboard.html`
- 回测 HTML 报告：`out/backtest_report.html`

---

## 6. 下一步优化方向（主线）

你的主目标已经明确为 3 条：

1. 长期正期望（expectancy > 0）  
2. 可接受回撤（你睡得着）  
3. 不同市场阶段不崩（鲁棒性）

围绕这 3 条，后续只做“提升确定性”的优化，不做堆代码的花活。

---

## 7. 后续优化建议与方案（从“能跑”到“可长期维护”）

### Phase 1（优先级 P1，近期 1-2 周）

1. 治理阈值校准（交易闸门）
方案：  
按滚动时间段回放，校准 `warn/fail` 阈值，输出“阈值-收益-回撤”对照表，冻结一版生产默认值。

2. PBO/DSR 稳定化
方案：  
扩大样本窗口并做敏感性分析，区分“研究模式阈值”和“生产模式阈值”，降低因短样本导致的误杀。

3. 动量崩盘保护参数标定
方案：  
对 `drop/rebound/protection_days/position_cap` 做网格回放，形成“崩盘保护触发日志 + 恢复条件”。

### Phase 2（优先级 P2，中期 2-4 周）

1. 组合层风控增强
方案：  
增加行业暴露上限、相关性拥挤约束、单票波动约束，避免同因子抱团导致回撤共振。

2. 执行仿真与实盘一致性进一步提高
方案：  
细化成交约束与冲击成本分层，统一回测与 paper 的成交假设，减少“回测可做、实盘做不到”。

3. 样本外分层报告
方案：  
按牛/熊/震荡分段输出回测与治理结果，明确策略在不同市场状态下的边界。

### Phase 3（优先级 P3，持续工程化）

1. 自动化健康诊断卡
方案：  
每日输出策略健康状态、降仓原因、恢复条件和建议动作，支持快速复盘。

2. 回归测试与防漂移
方案：  
建立关键指标基线，新增 CI 检查（阈值漂移、输出字段漂移、关键脚本参数漂移）。

3. 策略分层（核心 + 卫星）
方案：  
核心仓位维持低换手稳健收益，卫星仓位承接事件驱动与高弹性机会。

---

## 8. 下一步执行方案（可直接续跑）

### 8.1 短期（本周）

1. 持续回补历史新闻（分批限流）  
建议命令：  
`python run_backfill_news_history.py --start-date 2025-01-01 --end-date 2026-02-23 --watchlist watchlist_cache.yaml --max-days 10 --max-symbols 80 --latest-first`

2. 每补完一批，做一次快速核验  
- 检查 `data/news/<date>/manifest.json` 是否新增。  
- 检查 `data/backtests/backtest_strategy_v3_three_layer_report.json` 指标是否有边际改善。

3. 新闻因子低权重 A/B（先验证再放大）  
- 先用 `weight=0.05~0.08` 小步测试，重点看 WF Sharpe 与回撤变化。  
- 若提升稳定再考虑提高权重。

### 8.2 中期（下周）

1. 组合构建升级（轻量风险预算约束优先，不上重优化器）。  
2. 治理阈值二次校准（`trade_gate` / `PBO` / `DSR`）。  
3. 固化一版“生产参数快照”并记录回归基线。

---

## 9. 近几次会话纪要（摘要）

1. 会话 A（策略退出优化）  
- 完成 `rank_exit` 趋势过滤与 `ma/trailing` 冲突仲裁。  
- 做了固定 A/B 对照，结论是“趋势过滤保留更优”。

2. 会话 B（股票池与实盘口径）  
- 明确 `watchlist_cache.yaml` 是更新后的动态缓存池，`backtest_watchlist.yaml` 是较稳定的回测池。  
- 建议“日常静态池 + 每周更新股票池”的运行节奏。

3. 会话 C（新闻情绪因子接入）  
- 完成因子并入打分链路（默认可关），并以低权重方式进入 A/B。  
- DeepSeek 作为当前默认 LLM provider。

4. 会话 D（新闻历史回补）  
- 完成 `run_backfill_news_history.py` 与 `run_all_daily.py` 联动。  
- 新增限流与最新优先参数，支持长期后台分批补齐。

5. 会话 E（运行文档）  
- 已补充 `RUNBOOK.md`，覆盖每日/每周运行命令与常用参数。

注：会话中提到的收益指标以对应回测输出文件为准，不同股票池、日期区间、成本参数会导致结果差异。

---

## 10. 开发与运行建议

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

建议始终使用项目虚拟环境 `.venv` 运行，避免系统 Python 与依赖不一致。

---

## 11. 风险声明（必须看）

1. 量化策略不存在“稳赚”，只能追求长期正期望与可控回撤。  
2. 回测结果不代表未来收益，尤其在结构切换阶段。  
3. 实盘前必须保留风控闸门，不建议关闭。  
4. 任何新增策略都应先过样本外和稳定区，再考虑纳入主流程。
