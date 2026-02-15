# A股智能量化交易系统 V3.2

> 两阶段动态选股 · 涨停回调 · 主力控盘 · 五重风控 · 信号分级 · AI研报

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📈 回测业绩

| 指标 | V3.1（50只） | V3.2（450只） | 变化 |
|------|-------------|--------------|------|
| **年化收益** | 16.44% | **26.16%** | +10% ↑ |
| **总收益** | — | **292.52%** | — |
| **超额收益** | — | **278.80%** | — |
| **夏普比率** | 1.27 | **1.67** | +0.4 ↑ |
| **交易次数** | — | 1121笔 | — |
| 回测区间 | 2020-01 → 2026-02 | 2020-01 → 2026-02 | — |

---

## ✨ 系统亮点

| 特性 | 说明 |
|------|------|
| 🔍 **两阶段选股** | 全市场5000只 → 通达信公式筛450只 → 日线策略选20只 |
| 🎯 **精准入场** | 涨停回调 + 主力控盘 + 2-of-3多路径验证 |
| 📊 **信号分级** | 强/普通/弱三级信号，自动调整仓位 |
| 🛡️ **五重风控** | 硬止损/ATR止损/失败形态/动态止盈/时间止损 |
| 💰 **盈利加仓** | 浮盈>5%自动加仓50%，让利润奔跑 |
| 📰 **精准新闻** | 只抓信号前30只的新闻，不全覆盖 |
| 🤖 **AI研报** | MiMo 大模型生成研报 + 新闻摘要 + 可视化图表 |
| ⚡ **高性能** | 并行抓取 + LRU缓存 + 增量更新 |

> **说明：** 交易决策（买卖、仓位）完全由量化策略（规则引擎+因子评分）做出，AI（MiMo）的角色是生成研报和分析新闻情绪，不参与交易决策。

---

## 🚀 快速开始

### 1. 环境配置

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows

pip install -r requirements.txt
pip install pyarrow matplotlib  # 可选
```

### 2. 配置文件

**`.env` 环境变量：**
```bash
# LLM 选择（用于AI研报和新闻摘要，不影响交易决策）
LLM_PROVIDER=mimo

# MiMo 配置
MIMO_API_KEY=your_key
MIMO_MODEL=MiMo-7B-RL
```

**`config_v31.yaml` 策略配置：** 见下方策略说明。

### 3. 首次运行（建立股票池+数据）

```bash
python run_update_watchlist.py        # 全市场扫描 → 450只
python sync_watchlist.py              # 同步到配置文件
python run_fetch_daily.py             # 抓取历史数据（首次15-30分钟）
python run_build_market_daily_all.py  # 构建市场数据
python run_build_features_daily.py    # 生成特征
python run_backtest_strategy_v3.py    # 回测验证
python run_generate_report.py         # 生成报告
```

### 4. 每日运行

```bash
python run_all_daily.py               # 一键完成全部流程
python run_all_daily.py --fast        # 快速模式（跳过新闻+图表）
python run_all_daily.py --skip-ai     # 跳过AI研报
```

---

## 🔄 运行流程

### 🗓️ 每周一次（更新股票池，约20分钟）

股票池会变化，需要定期从全市场重新筛选：

```bash
# Step 1: 全市场扫描 → 筛选出约500只
python run_update_watchlist.py

# Step 2: 同步到配置文件
python sync_watchlist.py

# Step 3: 抓取新股票的历史数据（首次约15分钟，之后增量秒过）
python run_fetch_daily.py

# Step 4: 构建市场数据
python run_build_market_daily_all.py

# Step 5: 计算特征（含通达信指标）
python run_build_features_daily.py
```

### 📅 每天一次（生成信号+报告，约5-10分钟）

```bash
# 一键完成全部流程
python run_all_daily.py
```

`run_all_daily.py` 内部自动执行：
1. 增量抓取今日行情（500只，约2分钟）
2. 构建市场数据
3. 更新特征
4. **生成今日信号** → `data/signals/latest_daily_rank.csv`
5. 只抓信号前30只的新闻（不全覆盖，省时间）
6. AI简报
7. Markdown日报 + AI研报

快捷选项：
```bash
python run_all_daily.py --fast        # 跳过新闻+图表（约3分钟）
python run_all_daily.py --skip-ai     # 跳过AI研报
python run_all_daily.py --skip-news   # 跳过新闻
python run_all_daily.py --news-top-n 50  # 新闻覆盖前50只
```

### 🔧 偶尔运行（修改策略参数后）

```bash
python run_backtest_strategy_v3.py    # 重新跑回测
python run_generate_report.py         # 生成回测HTML报告
```

### ⏰ 建议时间表

| 时间 | 操作 | 耗时 |
|------|------|------|
| **每天 15:30后** | `python run_all_daily.py` | 5-10分钟 |
| **每周末** | 跑上面的5步更新股票池 | 约20分钟 |
| **调参后** | `python run_backtest_strategy_v3.py` | 几分钟 |

---

## 📁 项目结构

```
├── ── 日常运行 ──
├── run_all_daily.py                 ← 每日一键入口
├── run_generate_report.py           ← 回测HTML报告（含交易统计）
│
├── ── 股票池管理（每周/月）──
├── run_update_watchlist.py          ← 全市场扫描建池
├── sync_watchlist.py                ← 同步缓存到配置文件
│
├── ── 数据流水线 ──
├── run_fetch_daily.py               ← 获取日线数据
├── run_build_market_daily_all.py    ← 合并行情
├── run_build_features_daily.py      ← 构建特征
├── run_make_daily_rank.py           ← 生成交易信号
│
├── ── 回测与优化 ──
├── run_backtest_strategy_v3.py      ← 回测引擎（含参数优化）
├── run_generate_chart_v3.py         ← 回测图表
│
├── ── 新闻与AI ──
├── run_fetch_news.py                ← 新闻抓取（支持精准覆盖）
├── run_build_ai_briefs.py           ← AI简报
├── run_ai_report_v2.py              ← AI研报（优化版）
│
├── ── 配置 ──
├── config_v31.yaml                  ← 主配置文件
├── watchlist_cache.yaml             ← 股票池缓存（自动生成）
├── .env                             ← API密钥
│
├── quant/                           ← 核心库
│   ├── providers/                   ← 数据源（BaoStock/AKShare/Sina）
│   ├── news_providers/              ← 新闻源（东财/财联社/新浪/同花顺）
│   ├── features.py                  ← 特征工程
│   ├── signals.py                   ← 信号计算
│   ├── tdx_indicators.py            ← 通达信指标
│   └── market_regime.py             ← 市场环境判断
│
├── data/
│   ├── clean/market_daily/          ← 清洗后的日线
│   ├── features/                    ← 特征数据
│   ├── backtests/                   ← 回测结果+交易记录
│   ├── signals/                     ← 交易信号
│   └── reports/                     ← AI研报
│
└── out/                             ← 输出目录
    ├── backtest_report.html         ← 回测报告
    ├── charts/                      ← 图表
    └── latest_ai_report.md          ← 最新研报
```

---

## 📈 输出文件

| 路径 | 说明 |
|------|------|
| `data/signals/latest_daily_rank.csv` | 每日交易信号 |
| `data/backtests/backtest_strategy_v3_trades.csv` | 交易记录（每笔买卖） |
| `data/backtests/backtest_strategy_v3_equity.csv` | 权益曲线 |
| `out/backtest_report.html` | 回测报告（含8项交易统计） |
| `out/latest_ai_report.md` | AI研报 |
| `watchlist_cache.yaml` | 股票池缓存 |

---

## 🔧 命令速查

```bash
# ── 每天（收盘后跑）──
python run_all_daily.py              # 完整流程（推荐）
python run_all_daily.py --fast       # 快速（跳过新闻+图表）

# ── 每周（周末跑）──
python run_update_watchlist.py       # 更新股票池
python sync_watchlist.py             # 同步到配置
python run_fetch_daily.py            # 抓取数据
python run_build_market_daily_all.py # 构建市场数据
python run_build_features_daily.py   # 计算特征

# ── 偶尔（改策略后跑）──
python run_backtest_strategy_v3.py   # 回测
python run_generate_report.py        # 生成报告

# ── 单步调试 ──
python run_make_daily_rank.py        # 只生成信号（看诊断输出）
python run_fetch_news.py             # 只抓新闻
```

---

## ⚠️ 风险提示

**本系统仅供学习研究，不构成任何投资建议！**

- 过去表现不代表未来收益
- 回测结果可能存在过拟合
- 股市有风险，投资需谨慎

---

## 📋 版本历史

| 版本 | 年化 | 回撤 | 夏普 | 核心变化 |
|------|------|------|------|----------|
| **V3.2** | **26.16%** | — | **1.67** | 两阶段450只选股、五重风控、信号分级、盈利加仓 |
| V3.1 | 16.44% | -13.6% | 1.27 | 行业分散、流动性过滤、TDX保护 |
| V3.0 | ~20% | -18% | ~1.3 | 涨停回调+TDX指标+风控 |
| V2.0 | ~18% | -22% | ~1.2 | 仓位控制+月线过滤 |
| V1.0 | ~27% | -38% | ~1.18 | 纯因子排序，满仓运行 |

---

*配置文件：config_v31.yaml · 数据来源：BaoStock / AKShare / 新浪*

---

## 📄 License

MIT License