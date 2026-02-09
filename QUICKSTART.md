# 🚀 A股量化助手 - 快速使用指南

## 你每天只需要做的事

```powershell
# 每天开盘前运行（约2-3分钟）
python run_all_daily.py --skip-deepseek
```

然后查看：`data/signals/latest_daily_rank.csv`

**就这么简单。**

---

## 📊 输出文件说明

| 文件 | 位置 | 用途 |
|------|------|------|
| **每日信号** | `data/signals/latest_daily_rank.csv` | ⭐ 最重要！告诉你今天该买/卖什么 |
| AI 报告 | `data/reports/daily_report_*.md` | 简单的 Markdown 汇总 |
| DeepSeek 报告 | `data/reports/deepseek_report_*.md` | AI 深度分析（可选） |

---

## 🎯 如何使用每日信号

打开 `latest_daily_rank.csv`，看这几列：

| 列名 | 含义 | 操作建议 |
|------|------|---------|
| `action` | 系统建议 | **INVEST_MORE** = 可以买，**WITHDRAW** = 回避 |
| `score` | 综合得分 | 越高越好，>2 比较好 |
| `rank` | 排名 | 1-3 名优先考虑 |
| `trend_up` | 趋势向上 | 1 = 是，0 = 否，**只买 trend_up=1 的** |
| `ma_dist_20` | 离20日均线距离 | >5% = 超买别追，<-5% = 可能超卖 |

### 简单操作规则

```
如果 action = INVEST_MORE 且 trend_up = 1 且 ma_dist_20 < 5%
  → 可以考虑买入
  
如果 action = WITHDRAW 或 trend_up = 0
  → 不要碰
```

---

## 🤖 什么时候用 DeepSeek AI 报告？

**不是每天都需要！** 只在以下情况运行：

1. 周末复盘时
2. 有重大新闻/事件时
3. 想深入分析某只股票时

```powershell
# 完整流程（包含 DeepSeek，约5-10分钟，消耗 API 额度）
python run_all_daily.py
```

---

## ❌ 你不需要关心的文件

- `run_qlib_backtest.py` - 回测用，新手阶段不需要
- `run_backtest_*.py` - 同上
- `tools/` 目录 - 开发工具，不用管
- `data/features/` - 中间数据，不用看
- `data/raw/` - 原始数据，不用管

---

## 📈 因子/指标说明

当前使用的因子：

| 因子 | 含义 | 用途 |
|------|------|------|
| `ma_dist_20` | 价格离20日均线的距离 | 判断超买/超卖 |
| `ret_20d` | 20日收益率 | 短期动量 |
| `ret_60d` | 60日收益率 | 中期动量 |
| `vol_20d` | 20日波动率 | 风险指标 |
| `atr_pct` | ATR百分比 | 波动幅度 |
| `vol_ratio_20` | 成交量比率 | 量能变化 |
| `trend_up` | 是否趋势向上 | 趋势判断 |
| `mom_bad` | 动量是否走坏 | 警告信号 |
| `risk_high` | 风险是否过高 | 警告信号 |

**这些因子够用吗？**

对于你的目标（月赚 300 SGD）：**够了。**

因子不是越多越好。简单、稳定、可理解的因子 > 复杂的黑箱因子。

---

## ⏰ 推荐的每日流程

### 交易日早上 9:00 前
```powershell
python run_all_daily.py --skip-deepseek
```
查看信号，决定今天的操作

### 周末（可选）
```powershell
python run_all_daily.py
```
查看 DeepSeek 报告，做周复盘

---

## 🆘 常见问题

**Q: 数据抓取失败怎么办？**
```powershell
# 重新运行，BaoStock 一般很稳定
python run_fetch_daily.py
```

**Q: 如何只看某一天的历史信号？**
```powershell
# 查看 data/signals/ 目录下的历史文件
dir data\signals\
```

**Q: 如何添加新股票到监控列表？**
编辑 `config.yaml` 的 `watchlist`

---

## 💡 记住

1. **简单就是力量** - 不要追求复杂
2. **信号只是参考** - 最终决策靠你自己
3. **控制风险第一** - 单只股票不超过 20% 仓位
4. **耐心等待** - 没有好信号就空仓