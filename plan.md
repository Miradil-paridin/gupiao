# 当前程序策略全景（选股 / 买卖 / 风控 / 治理）

更新时间：2026-02-22  
适用入口：`run_all_daily.py`（主流程）、`run_backtest_strategy_v3.py`（回测）、`run_make_daily_rank.py`（日信号）、`run_paper_trading.py`（纸上交易）、`tools/strategy_process_pipeline.py`（治理）

## 1. 总体框架

程序不是“单一买卖点模型”，而是一个分层策略系统：

1. 股票池与数据策略（抓哪些标的、用哪些数据）
2. 日度评分与候选策略（哪些票更优先）
3. 入场策略（什么时候能开仓）
4. 仓位与再平衡策略（买多少、何时调）
5. 出场策略（怎么卖）
6. 执行现实策略（成交限制、冲击成本）
7. 治理闸门策略（质量门、交易门）
8. Paper 交易策略（把信号变成记账交易）

---

## 2. 股票池与数据策略

### 2.1 股票池策略

1. 静态 watchlist
- 用途：限定回测与信号的基础股票池。
- 当前：`config.yaml` 中维护大名单，默认运行时会优先使用 `backtest_watchlist.yaml` / `watchlist_cache.yaml`（若存在）。

2. 动态 watchlist（可选）
- 用途：按月更新池子，降低静态池幸存者偏差。
- 入口：`run_backtest_strategy_v3.py --dynamic-watchlist --dynamic-top-n ...`

### 2.2 数据源策略

1. 多源行情抓取 + fallback
- 用途：提升可用性，避免单数据源失败导致停摆。
- 当前链路：`baostock -> akshare -> sina`（见 `config.yaml/providers.daily`）。

2. 特征增强（策略特征）
- 用途：构建打分和入场条件所需因子。
- 主要包含：`TDX 相关特征`、`主力强度(main_force_pct)`、`30日涨停/新高特征`、`流动性与市值特征`、`机构持仓特征（可选）`。

---

## 3. 选股评分策略（谁进入候选池）

### 3.1 核心评分策略（每日横截面）

基础分由以下项合成（`precompute_daily_universe`）：

1. 趋势与动量：`ma_dist_20`、`ret_20d`、`ret_60d`
2. 风险与波动惩罚：`vol_20d`、`atr_pct`
3. 交易活跃度：`vol_ratio_20`
4. 事件加分：`high30_breakout`、`main_force_strong`、`has_limit_up_30d`
5. 月线状态加分：`monthly_bullish`

### 3.2 Alpha 增强策略（可开关）

用途：在基础分上叠加低相关因子，提高区分度。

因子包括：

1. 行业相对强度：`alpha_industry_rs`
2. 资金流持续性：`alpha_flow_persistence`
3. 质量因子：`alpha_quality`
4. 短期反转：`alpha_short_reversal`
5. 换手反转：`alpha_turnover_reversal`
6. 价值代理：`alpha_value_proxy`

当前 `config.yaml` 中 `alpha_enhancement.enabled: true`，权重已配置。

### 3.3 候选过滤策略（Enhancement Filters）

用途：在“高分股”中先做可交易与风格约束，避免不可执行和拥挤。

1. 可交易性过滤：`tradeable == 1`
2. 市值过滤：`float_mkt_cap_20d` 区间
3. 流动性过滤：`amount_20d`、`turnover_20d`
4. 机构持仓过滤（可选）
- `data` 模式：按机构持仓真实数据分位过滤
- `proxy` 模式：按 ROE/CFO/YOY/波动/成交额代理分过滤
5. 行业分散：`max_per_industry`
6. 相关性拥挤控制：`max_pairwise_corr`
7. 板块总仓位上限：`max_sector_weight`（在持仓层再次约束）

---

## 4. 入场策略（什么时候买）

### 4.1 入场模式策略（entry_mode）

支持四种模式：

1. `custom`
- 严格自定义组合：近30日涨停 + 主力强 + 新高 + 回调区间 + TDX分数门槛

2. `strict`
- `(涨停回调+放量突破) or TDX强`，且 `trend_up`

3. `normal`（当前主模式）
- 三路入场并行，满足任一路：
  - `path_pullback`: 涨停回调 + 额外条件达到 `2-of-3`
  - `path_tdx`: TDX强 + 趋势向上
  - `path_breakout`: 放量突破 + 趋势向上 + 主力强
- 并产出信号强度：`strong=2 / normal=1 / weak=0.5`

4. `loose`
- 仅要求 `trend_up`

### 4.2 双层入场策略（Dual-layer Entry）

用途：把候选分成“可直接开仓”与“观察/微仓”。

1. Regime Gate（按市场状态设强度门槛）
- 牛/震荡/熊分别用不同最小信号强度阈值。

2. 信号分层开仓
- 强/中信号直接开仓
- 弱信号按 `weak_entry_mode`
  - `observe`: 不开仓，仅观察
  - `micro`: 允许少量弱信号开仓（数量和权重受限）

### 4.3 市场状态门控策略

1. 指数过滤（Index Filter）
- `hard gate`: 指数不允许时禁止新开仓
- `soft gate`: 不禁开，但组合总仓位上限降到 `index_filter_block_position_cap`

2. 熊市禁开仓（可选）
- `block_new_in_bear_regime`

3. 回撤预刹车（Adaptive Drawdown）
- 触发后提高入场门槛并缩小 `top_k` 与 `invest_more_n`

---

## 5. 仓位与再平衡策略（买多少）

### 5.1 总仓位策略

1. 基础上限：`max_total_position`
2. 市场状态动态上限：`bull/neutral/bear cap`
3. 指数软门控下的额外上限
4. 动量崩盘保护上限（触发时强制 cap）
5. 回撤刹车期上限
6. 弱信号去风险：当平均信号强度低于阈值时再乘以 `cap_multiplier`

### 5.2 单票权重策略

1. 基础等权：`max_pos_today / n_targets`
2. 单票上限：`max_single_weight`
3. 信号分层权重：强/中/弱对应不同倍数
4. 微仓模式下弱信号再缩放

### 5.3 再平衡死区策略（Rebalance Band）

用途：过滤微小调仓，降低无效换手和摩擦成本。  
实现：目标权重与当前权重差值小于 `rebalance_band` 时不调仓。

---

## 6. 卖出策略（怎么卖）

交易循环中的主要出场规则：

1. 硬止损：`hard_stop_loss`
- 跌幅超过 `stop_loss_pct` 触发
- TDX保护：强主力票可放宽（不低于固定保护阈值）

2. ATR 止损：`atr_stop`
- 价格跌破 `entry - N*ATR`

3. 失效止损：`failure_stop`
- 开仓后前 `N` 天若涨幅不足阈值则退出
- 降噪规则（当前已启用）：
- 仅对弱信号仓位生效（`weak_signal_only`）
- 仅在负收益时生效（`require_negative_pnl`）
- 若仍在目标池可跳过（`skip_if_still_target`）
- 若趋势仍向上可跳过（`skip_if_trend_up`）

4. 移动止损：`trailing_stop`
- 按持仓峰值回撤触发
- 盈利较高时自动收紧 trailing 阈值

5. MA 止损：`ma_stop`
- 连续若干天低于 MA20 触发

6. 时间止损：`time_stop`
- 持仓天数超上限且收益未达最低要求触发

7. 止盈：`take_profit`
- ATR止盈 / 固定百分比止盈
- 可选分级止盈（当前默认关闭）
- 阈值按牛熊状态动态调整

8. 排名退出：`rank_exit`
- 固定频率（当前周频）检查
- 不在目标池且持仓已过最短天数则退出
- 新增缓冲带（当前已启用）：`rank_exit_refine.rank_buffer`
- 当持仓排名仍处于 `top_k + buffer` 内时，不触发 rank_exit
- 新增趋势过滤（当前已启用）：`rank_exit_refine.only_when_trend_down`
- 当 `trend_up=1` 时，rank_exit 可跳过，降低趋势内抖动卖出

9. 强制退出
- 缺失价格、异常价格等情况直接清仓记录

10. 盈利加仓（可选）
- `profit pyramiding`，达到收益阈值后按比例追加（当前默认关闭）

补充：`MA 止损` 与 `Trailing 止损` 采用可配置优先级仲裁（`risk_control.exit_conflict_policy.ma_vs_trailing`），避免同日多规则冲突。

---

## 7. 执行现实策略（把回测变得可落地）

1. 涨跌停成交约束
- 买入触及涨停、卖出触及跌停时可被阻塞
- 按主板/ST/创业板/科创板/北交所及新股阶段处理涨跌幅限制

2. 成交额参与率约束
- 单笔目标仓位受 `max_participation_rate * amount / equity` 限制

3. 整手约束
- 按 `lot_size` 做手数取整
- 买卖差异化处理：买入严格整手；卖出允许清理微小残仓，避免“残仓永远卖不掉”造成虚假阻塞

4. 交易成本模型
- 显式手续费：`cost_bps * turnover`
- 额外执行成本：`slippage + impact(participation^exponent)`

---

## 8. 治理策略（质量门 + 交易门）

治理在 `tools/strategy_process_pipeline.py` 执行，核心流程：

1. Baseline 回测指标
2. Walk-forward 样本外稳定性
3. Rolling 窗口稳定性
4. 参数稳定区扫描（Stable Zone）
5. 过拟合诊断（PBO / DSR）
6. 双窗口失效监控（short/long）
7. `quality_gate` 判定（pass/warn/fail）
8. `trade_gate` 产出动作（normal/reduce/stop）与仓位上限

### 8.1 quality_gate 主要检查项

1. 日度期望是否为正
2. 最大回撤是否在阈值内
3. Walk-forward 均值 Sharpe 是否达标
4. Walk-forward 达标比例是否达标
5. 稳定区比例是否达标
6. 失效监控动作是否允许
7. 换手率与执行拖累是否超限
8. PBO 是否超限
9. DSR 是否低于下限
10. Walk-forward 折数是否充足（折数不足时记为 `warn`，避免误判成 `fail`）

### 8.2 trade_gate 映射规则

1. `pass -> normal`
2. `warn -> reduce`（按 warn 区间与监控建议给 cap）
3. `fail 或 monitor=stop -> stop`（cap=0）

`run_all_daily.py` 优先读取 `trade_gate`，若缺失再回退旧口径。

---

## 9. Paper 交易策略（信号如何落到账本）

1. 读取 `latest_daily_rank.csv` 中 `INVEST_MORE` 与 `target_weight`
2. 叠加治理仓位上限（默认读取 `strategy_process_summary.json`）
3. 生成“下一交易日开盘执行”的再平衡订单
4. 执行顺序：先卖后买
5. 执行参数：滑点、手续费、整手、再平衡死区
6. 无法成交则滚动挂到下一交易日
7. 输出：`state.json`、`trades.csv`、`equity.csv`，可选同步 `portfolio.yaml`

---

## 10. 当前默认口径（来自 `config.yaml`）

1. 选股规模：`top_k=10`，`invest_more_n=14`
2. 总仓位：`max_total_position=0.92`
3. 入场模式：`entry_mode=normal`，`normal_min_conditions=2`
4. 指数门控：`use_index_filter=true`，`index_filter_hard_gate=false`（软门控）
5. 动量崩盘保护：开启，`cap=0.45`
6. 止损止盈：`stop_loss=8%`，`trailing=10%`，`max_hold=20d`，`take_profit=on`
7. 组合风控：行业分散/相关性控制开启
8. Alpha 增强：开启
9. 回测执行现实：开启（参与率、滑点、冲击、涨跌停、整手）

---

## 11. 你后续可重点研究的扩展点

1. 入场强度阈值与 `normal_min_conditions` 的鲁棒区间
2. `failure_stop` 与 `trailing_stop` 的联动冲突（过早止损 vs 留住趋势）
3. `index_filter` 软门控阈值与 `momentum_crash` 的叠加关系
4. 行业/相关性约束是否过紧导致收益被压制
5. PBO/DSR 的样本窗口与阈值在研究模式/生产模式的分层
