# A股智能量化交易系统 V3.0

> 多因子选股 · 涨停回调 · 主力控盘 · 智能风控 · AI研报

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ 系统亮点

| 特性 | 说明 |
|------|------|
| 🎯 **精准入场** | 涨停回调 + 主力控盘（月线）双重验证 |
| 🛡️ **多重风控** | 止损/止盈/时间止损三重保护 |
| 📊 **科学选股** | 多因子模型 + 通达信月线指标 |
| 🤖 **AI研报** | DeepSeek/MiMo 大模型 + 可视化图表 |
| ⚡ **高性能** | 并行抓取 + LRU缓存 + 增量更新 |
| 📈 **自动化** | 一键运行，每日自动生成信号 |

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Linux/Mac

# 安装依赖
pip install -r requirements.txt
pip install pyarrow matplotlib  # 可选：图表支持
```

### 2. 配置文件

**`.env` 环境变量：**
```bash
# LLM 选择
LLM_PROVIDER=deepseek  # deepseek 或 mimo

# DeepSeek 配置
DEEPSEEK_API_KEY=your_key
DEEPSEEK_MODEL=deepseek-reasoner
```

**`config.yaml` 策略配置：**
```yaml
watchlist:
  - "600519"  # 贵州茅台
  - "000858"  # 五粮液
  # ... 50只精选龙头股

strategy:
  top_k: 10                   # 每日推荐数
  use_tdx_indicators: true    # 通达信月线指标
  use_limit_up_pullback: true # 涨停回调入场

market_data:
  start_date: "2023-01-01"    # 数据起始日期
```

### 3. 一键运行

```bash
# 完整运行
python run_all_daily.py

# 快速模式（跳过新闻+图表）
python run_all_daily.py --fast

# 跳过AI研报
python run_all_daily.py --skip-ai
```

---

## 📁 项目结构

```
├── quant/                          # 核心模块
│   ├── providers/                  # 数据源（BaoStock/AkShare/Sina）
│   ├── news_providers/             # 新闻源（东财/财联社/新浪/同花顺）
│   ├── features.py                 # 特征工程（含涨停+TDX）
│   ├── signals.py                  # 信号计算
│   ├── tdx_indicators.py           # 通达信指标（月线）
│   ├── market_regime.py            # 市场环境判断
│   ├── fetch_daily.py              # 数据抓取（并行+缓存）
│   ├── report_charts.py            # 研报图表生成
│   ├── report_generator.py         # 研报内容生成
│   └── report_quality.py           # 研报质量评估
├── tools/                          # 工具脚本
│   ├── build_signal_history.py     # 历史信号构建
│   ├── run_backtest_from_signal.py # 信号回测
│   └── report_metrics.py           # 回测指标
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   ├── clean/                      # 清洗数据
│   ├── features/                   # 特征数据
│   ├── signals/                    # 信号数据
│   └── reports/                    # AI研报
├── out/                            # 输出目录
│   ├── charts/                     # 图表
│   └── latest_ai_report.md         # 最新研报
├── run_all_daily.py                # 🚀 一键运行入口
├── run_ai_report_v2.py             # AI研报（优化版）
├── run_backtest_strategy_v3.py     # 历史回测+参数优化
├── run_generate_chart_v3.py        # 回测图表生成
├── config.yaml                     # 主配置文件
├── config_optimized.yaml           # 优化后的参数配置
└── .env                            # 环境变量
```

---

## 📊 策略说明

### 选股流程（5层过滤）

```
┌─────────────────────────────────────────────────────────────┐
│  第一层  │ 市场环境过滤：沪深300 > MA200 才开仓              │
├─────────────────────────────────────────────────────────────┤
│  第二层  │ 可交易性过滤：排除一字板、涨停股（买不进）         │
├─────────────────────────────────────────────────────────────┤
│  第三层  │ 入场信号过滤：涨停回调 / 主力控盘 / TDX指标       │
├─────────────────────────────────────────────────────────────┤
│  第四层  │ 因子排序：多因子综合评分，选 Top 10               │
├─────────────────────────────────────────────────────────────┤
│  第五层  │ 仓位分配：波动反比 + 行业分散                     │
└─────────────────────────────────────────────────────────────┘
```

### 入场条件（满足任一）

#### 条件一：涨停回调入场
- 近期（3-10天内）出现过涨停
- 从涨停价回调 5%-25%
- 当日放量突破（量 > 5日均量×1.3）
- 收盘价站上 MA5

#### 条件二：通达信指标入场（月线级别）
| 指标 | 公式 | 条件 |
|------|------|------|
| **高30突破** | `HHV(EMA(X1, 3), 30)` | 30**月**内创新高 |
| **主力控盘** | `(GU1 - REF(起爆,1)) / REF(起爆,1)` | > **50%** |
| **涨停30日** | `MA(涨停标记, 30) × 10` | > 0.5（日线） |

#### 条件三：简单趋势
- 收盘价 > MA20

### 风控体系

| 风控类型 | 触发条件 | 说明 |
|----------|----------|------|
| 🔴 硬止损 | 亏损 -8% | 无条件止损 |
| 🟡 移动止盈 | 从峰值回撤 10% | 锁定利润 |
| 🔵 时间止损 | 持有超 15 天 | 避免长期套牢 |

### 仓位控制

| 市场状态 | 判断条件 | 建议仓位 |
|----------|----------|----------|
| 🟢 牛市 | >60%股票趋势向上 | 70-80% |
| 🟡 震荡 | 中性 | 50-60% |
| 🔴 熊市 | <40%股票趋势向上 | 30-40% |

### 策略增强（v3新增）

| 功能 | 说明 |
|------|------|
| 行业分散控制 | 单行业最多2只 |
| 市值过滤 | 80亿-8000亿流通市值 |
| 流动性过滤 | 20日成交额>6000万 |
| 相关性控制 | 候选股相关性<0.75 |

---

## 🔧 运行命令

### 日常运行

```bash
# 完整流程
python run_all_daily.py

# 快速模式（约3-5分钟）
python run_all_daily.py --fast

# 跳过新闻
python run_all_daily.py --skip-news

# 跳过AI研报
python run_all_daily.py --skip-ai

# 跳过图表
python run_all_daily.py --skip-charts
```

### 历史回测

```bash
# 普通回测
python run_backtest_strategy_v3.py

# 深度优化（网格搜索 + 5折交叉验证）
python run_backtest_strategy_v3.py --optimize

# 生成图表
python run_generate_chart_v3.py
```

### 分步运行

```bash
python run_fetch_daily.py           # 1. 抓取数据
python run_build_market_daily_all.py # 2. 构建市场数据
python run_build_features_daily.py   # 3. 生成特征
python run_make_daily_rank.py        # 4. 生成信号
python run_fetch_news.py             # 5. 抓取新闻
python run_ai_report_v2.py           # 6. AI研报
```

---

## 📈 输出文件

| 路径 | 说明 |
|------|------|
| `data/signals/latest_daily_rank.csv` | **每日信号** |
| `out/latest_ai_report.md` | **AI研报** |
| `out/charts/` | 可视化图表 |
| `data/features/features_daily.parquet` | 特征数据 |
| `data/backtests/` | 回测结果 |
| `config_optimized.yaml` | 优化后的参数 |

---

## 📊 信号输出示例

```
╔══════════════════════════════════════════════════════════════╗
║     🚀 A股智能量化交易系统 v3.0                              ║
╚══════════════════════════════════════════════════════════════╝

📋 今日信号 (Top 10)
============================================================

🎯 建议买入 (INVEST_MORE):
代码      名称       得分   目标仓位   TDX分   高30突破   主力控盘
600519   贵州茅台   3.52     15%      2.5       ✅         ✅
000858   五粮液     3.21     12%      2.0       ✅         ✅
300750   宁德时代   2.81     10%      1.5       ✅         ❌

📊 目标仓位:
   总仓位: 65%
   现金: 35%
```

---

## 📝 AI研报功能（v2优化版）

### 研报内容

| 章节 | 内容 |
|------|------|
| 📈 市场概览 | 大盘状态、风险等级、热门板块 |
| 🔥 行业轮动 | 行业涨跌幅、资金流向判断 |
| 🎯 今日推荐 | 个股详细分析（技术面+TDX指标） |
| ⚠️ 风险提示 | 风险因素识别 |
| 📋 投资建议 | 操作建议、仓位建议 |

### 可视化图表

| 图表 | 说明 |
|------|------|
| 价格走势图 | 收盘价 + MA5/MA20 + 成交量 |
| 技术指标图 | 布林带 + RSI + ATR |
| 行业热力图 | 行业涨跌幅对比 |
| 收益曲线 | 策略净值 + 回撤 |

### 质量评估

| 维度 | 说明 |
|------|------|
| 事实性 | 数据准确性 |
| 一致性 | 格式规范性 |
| 相关性 | 内容相关性 |
| 完整性 | 章节完整性 |
| 可读性 | 易读性 |

---

## ❓ 常见问题

### Q: 运行太慢？
```bash
# 使用快速模式
python run_all_daily.py --fast
```

### Q: Parquet 读取报错？
```bash
pip install pyarrow
```

### Q: 图表中文乱码？
```bash
pip install matplotlib
# 然后安装中文字体（如 SimHei）
```

### Q: LLM API 调用失败？
```bash
# 检查 .env 配置
DEEPSEEK_API_KEY=sk-xxx
```

### Q: `config_optimized.yaml` 是什么？
这是运行参数优化后自动生成的配置文件，保存了最优参数。

---

## 📋 版本历史

### v3.0（当前）
- ✅ 通达信指标改为**月线级别**
- ✅ AI研报优化（可视化+质量评估）
- ✅ 数据抓取优化（并行+缓存+增量）
- ✅ 参数优化（网格搜索+交叉验证）
- ✅ 策略增强（行业分散、市值过滤、流动性过滤、相关性控制）

### v2.0
- ✅ 涨停回调入场策略
- ✅ 市场环境过滤
- ✅ 多重风控
- ✅ DeepSeek/MiMo 多模型

### v1.0
- 基础因子排序策略
- 新闻聚合
- DeepSeek AI研报

---

## ⚠️ 风险提示

**本系统仅供学习研究，不构成任何投资建议！**

- 过去表现不代表未来收益
- 回测结果可能存在过拟合
- 股市有风险，投资需谨慎
- 请根据自身风险承受能力合理配置

---

## 📄 License

MIT License