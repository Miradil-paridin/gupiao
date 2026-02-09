"""
AI研报模板和内容生成器

功能：
- 结构化研报模板
- 市场热点分析
- 行业轮动分析
- 个股深度分析
- 质量控制
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import json
import numpy as np
import pandas as pd


@dataclass
class StockAnalysis:
    """个股分析结果"""
    symbol: str
    name: str
    action: str
    score: float
    target_weight: float

    # 技术面
    price: float
    change_pct: float
    ma_dist_20: float
    rsi_14: float
    vol_ratio: float

    # TDX指标
    high30_breakout: bool
    main_force_strong: bool
    has_limit_up_30d: bool
    tdx_score: float

    # 置信度
    confidence: str  # HIGH/MEDIUM/LOW
    confidence_score: float  # 0-1

    # 风险提示
    risks: List[str]


@dataclass
class MarketOverview:
    """市场概览"""
    date: str
    market_regime: str  # BULL/NEUTRAL/BEAR
    hs300_change: float
    trend_up_pct: float
    avg_return_20d: float
    hot_industries: List[str]
    risk_level: str  # LOW/MEDIUM/HIGH


@dataclass
class ReportContent:
    """研报内容"""
    title: str
    date: str
    market_overview: MarketOverview
    stock_analyses: List[StockAnalysis]
    industry_analysis: Dict[str, float]
    investment_summary: str
    risk_warnings: List[str]
    charts: Dict[str, Path]


# =============================================================================
# 研报模板
# =============================================================================

REPORT_TEMPLATE = """# 📊 {title}

> 📅 **报告日期**: {date}
> 🤖 **生成方式**: AI智能分析
> ⚠️ **风险提示**: 本报告仅供参考，不构成投资建议

---

## 📈 一、市场概览

### 1.1 市场状态

| 指标 | 数值 | 说明 |
|------|------|------|
| 市场环境 | {market_regime_emoji} **{market_regime}** | {market_regime_desc} |
| 沪深300 | {hs300_change:+.2f}% | 大盘表现 |
| 趋势向上占比 | {trend_up_pct:.1f}% | 价格>MA20的股票比例 |
| 20日平均收益 | {avg_return_20d:+.2f}% | 市场动量 |
| 风险等级 | {risk_level_emoji} {risk_level} | 综合评估 |

### 1.2 热门板块

{hot_industries_section}

---

## 🔥 二、行业轮动分析

### 2.1 行业涨跌幅

{industry_table}

### 2.2 资金流向判断

{fund_flow_analysis}

{industry_heatmap_section}

---

## 🎯 三、今日推荐

### 3.1 推荐汇总

{recommendation_summary}

### 3.2 个股详细分析

{stock_analyses_section}

---

## ⚠️ 四、风险提示

{risk_warnings_section}

---

## 📋 五、投资建议

{investment_summary}

### 5.1 操作建议

{action_suggestions}

### 5.2 仓位建议

{position_suggestions}

---

## 📊 六、附录：图表

{charts_section}

---

## 📜 免责声明

1. 本报告基于量化模型和AI分析生成，仅供参考
2. 股市有风险，投资需谨慎
3. 过去表现不代表未来收益
4. 请根据自身风险承受能力做出投资决策

---

*报告生成时间: {generated_at}*
*数据来源: BaoStock, 东方财富, 财联社*
*AI模型: {ai_model}*
"""

STOCK_ANALYSIS_TEMPLATE = """
#### {idx}. {symbol} {name} {action_emoji}

> **操作建议**: {action} | **目标仓位**: {target_weight:.1f}% | **置信度**: {confidence_emoji} {confidence}

**📊 核心指标**

| 指标 | 数值 | 评价 |
|------|------|------|
| 当前价格 | ¥{price:.2f} | {price_comment} |
| 今日涨跌 | {change_pct:+.2f}% | {change_comment} |
| 综合得分 | {score:.2f} | 排名靠前 |
| MA20偏离 | {ma_dist_20:+.2f}% | {ma_comment} |
| RSI(14) | {rsi_14:.1f} | {rsi_comment} |
| 量比 | {vol_ratio:.2f} | {vol_comment} |

**🔍 通达信指标**

| 指标 | 状态 | 含义 |
|------|------|------|
| 高30突破 | {high30_status} | 月线趋势 |
| 主力控盘 | {main_force_status} | 主力介入 |
| 涨停30日 | {limit_up_status} | 近期活跃度 |
| TDX综合分 | {tdx_score:.1f} | 综合评估 |

**⚠️ 风险提示**

{risks_section}

---
"""


# =============================================================================
# 内容生成函数
# =============================================================================

def get_market_regime_info(regime: str) -> tuple:
    """获取市场状态信息"""
    info = {
        "BULL": ("🟢", "牛市", "市场情绪乐观，可适当提高仓位"),
        "NEUTRAL": ("🟡", "震荡", "市场方向不明，建议保持中性仓位"),
        "BEAR": ("🔴", "熊市", "市场情绪悲观，建议降低仓位防守"),
    }
    return info.get(regime, ("⚪", "未知", "请谨慎操作"))


def get_risk_level_info(level: str) -> str:
    """获取风险等级emoji"""
    return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(level, "⚪")


def get_action_emoji(action: str) -> str:
    """获取操作建议emoji"""
    return {
        "INVEST_MORE": "🟢 买入",
        "HOLD": "🟡 持有",
        "REDUCE": "🟠 减仓",
        "WITHDRAW": "🔴 卖出",
        "LEAST": "⚫ 回避",
    }.get(action, "⚪ 观望")


def get_confidence_emoji(confidence: str) -> str:
    """获取置信度emoji"""
    return {"HIGH": "⭐⭐⭐", "MEDIUM": "⭐⭐", "LOW": "⭐"}.get(confidence, "")


def calculate_confidence(row: pd.Series) -> tuple:
    """计算置信度"""
    score = 0.0

    # TDX指标贡献
    if row.get("high30_breakout", 0) == 1:
        score += 0.25
    if row.get("main_force_strong", 0) == 1:
        score += 0.25
    if row.get("has_limit_up_30d", 0) == 1:
        score += 0.15

    # 趋势贡献
    if row.get("ma_dist_20", 0) > 0:
        score += 0.15
    if row.get("ret_20d", 0) > 0:
        score += 0.10

    # RSI贡献
    rsi = row.get("rsi_14", 50)
    if 30 < rsi < 70:
        score += 0.10

    # 分类
    if score >= 0.7:
        return "HIGH", score
    elif score >= 0.4:
        return "MEDIUM", score
    else:
        return "LOW", score


def identify_risks(row: pd.Series) -> List[str]:
    """识别风险因素"""
    risks = []

    # RSI风险
    rsi = row.get("rsi_14", 50)
    if rsi > 80:
        risks.append("⚠️ RSI超买(>80)，短期可能回调")
    elif rsi < 20:
        risks.append("⚠️ RSI超卖(<20)，可能继续下跌")

    # 波动风险
    vol = row.get("vol_20d", 0)
    if vol > 0.5:
        risks.append("⚠️ 波动率较高，注意控制仓位")

    # 涨停风险
    if row.get("limit_up_flag", 0) == 1:
        risks.append("⚠️ 当日涨停，追高风险")

    # 一字板风险
    if row.get("one_line_board", 0) == 1:
        risks.append("⚠️ 一字板，流动性风险")

    # MA距离风险
    ma_dist = row.get("ma_dist_20", 0)
    if ma_dist > 0.15:
        risks.append("⚠️ 偏离MA20过大(>15%)，注意回调风险")
    elif ma_dist < -0.10:
        risks.append("⚠️ 跌破MA20较多，趋势可能转弱")

    if not risks:
        risks.append("✅ 未发现明显风险")

    return risks


def generate_price_comment(price: float, change_pct: float) -> str:
    """生成价格评论"""
    if change_pct > 3:
        return "大涨"
    elif change_pct > 0:
        return "上涨"
    elif change_pct > -3:
        return "下跌"
    else:
        return "大跌"


def generate_ma_comment(ma_dist: float) -> str:
    """生成MA评论"""
    if ma_dist > 0.05:
        return "强势（高于均线）"
    elif ma_dist > 0:
        return "偏强"
    elif ma_dist > -0.05:
        return "偏弱"
    else:
        return "弱势（低于均线）"


def generate_rsi_comment(rsi: float) -> str:
    """生成RSI评论"""
    if rsi > 80:
        return "超买⚠️"
    elif rsi > 60:
        return "偏强"
    elif rsi > 40:
        return "中性"
    elif rsi > 20:
        return "偏弱"
    else:
        return "超卖⚠️"


def generate_vol_comment(vol_ratio: float) -> str:
    """生成量比评论"""
    if vol_ratio > 2:
        return "放量🔥"
    elif vol_ratio > 1.3:
        return "温和放量"
    elif vol_ratio > 0.7:
        return "正常"
    else:
        return "缩量"


def generate_industry_analysis(industry_data: Dict[str, float]) -> str:
    """生成行业分析"""
    if not industry_data:
        return "暂无行业数据"

    sorted_data = sorted(industry_data.items(), key=lambda x: x[1], reverse=True)

    lines = []
    for industry, change in sorted_data:
        emoji = "🔴" if change < 0 else "🟢"
        lines.append(f"| {industry} | {emoji} {change:+.2f}% |")

    table = "| 行业 | 涨跌幅 |\n|------|--------|\n" + "\n".join(lines)

    # 分析文字
    top3 = sorted_data[:3]
    bottom3 = sorted_data[-3:]

    analysis = f"""
**领涨行业**: {', '.join([x[0] for x in top3])}

**领跌行业**: {', '.join([x[0] for x in bottom3])}

**轮动判断**: 
"""

    if top3[0][1] > 2:
        analysis += "- 市场热点集中，可关注领涨板块龙头\n"
    if bottom3[0][1] < -2:
        analysis += "- 部分行业承压明显，建议规避\n"

    return table + "\n" + analysis


def generate_fund_flow_analysis(industry_data: Dict[str, float]) -> str:
    """生成资金流向分析"""
    if not industry_data:
        return "暂无资金流向数据"

    sorted_data = sorted(industry_data.items(), key=lambda x: x[1], reverse=True)

    # 简单判断
    positive_count = sum(1 for _, v in sorted_data if v > 0)
    total_count = len(sorted_data)

    if positive_count > total_count * 0.7:
        return """
> 📈 **资金流入**: 多数行业上涨，市场资金活跃
> 
> 建议：可积极参与，但注意控制仓位
"""
    elif positive_count < total_count * 0.3:
        return """
> 📉 **资金流出**: 多数行业下跌，市场资金谨慎
> 
> 建议：以防守为主，等待企稳信号
"""
    else:
        return """
> ⚖️ **资金分化**: 行业涨跌参半，资金轮动明显
> 
> 建议：精选个股，关注资金流入的板块
"""


def generate_action_suggestions(stocks: List[StockAnalysis]) -> str:
    """生成操作建议"""
    invest_more = [s for s in stocks if s.action == "INVEST_MORE"]
    withdraw = [s for s in stocks if s.action == "WITHDRAW"]

    suggestions = []

    if invest_more:
        suggestions.append(f"**买入建议**: {', '.join([s.symbol for s in invest_more])}")
        suggestions.append(f"- 总计 {len(invest_more)} 只股票符合买入条件")
        suggestions.append(f"- 建议分批建仓，每只不超过 15%")
    else:
        suggestions.append("**买入建议**: 今日无符合条件的买入标的")
        suggestions.append("- 建议保持观望，等待更好的入场机会")

    if withdraw:
        suggestions.append(f"\n**卖出建议**: {', '.join([s.symbol for s in withdraw])}")
        suggestions.append(f"- 总计 {len(withdraw)} 只股票建议卖出")

    return "\n".join(suggestions)


def generate_position_suggestions(market: MarketOverview, stocks: List[StockAnalysis]) -> str:
    """生成仓位建议"""
    regime = market.market_regime
    risk = market.risk_level

    if regime == "BULL" and risk == "LOW":
        return """
| 仓位类型 | 建议比例 | 说明 |
|----------|----------|------|
| 总仓位 | 70-80% | 市场环境良好 |
| 单只上限 | 15% | 分散风险 |
| 现金 | 20-30% | 保留机动资金 |
"""
    elif regime == "BEAR" or risk == "HIGH":
        return """
| 仓位类型 | 建议比例 | 说明 |
|----------|----------|------|
| 总仓位 | 30-40% | 市场风险较高 |
| 单只上限 | 10% | 严格控制 |
| 现金 | 60-70% | 防守为主 |
"""
    else:
        return """
| 仓位类型 | 建议比例 | 说明 |
|----------|----------|------|
| 总仓位 | 50-60% | 市场中性 |
| 单只上限 | 12% | 适度分散 |
| 现金 | 40-50% | 保持灵活 |
"""


# =============================================================================
# 主生成函数
# =============================================================================

def generate_report_content(
        signals: pd.DataFrame,
        features: pd.DataFrame,
        market_overview: MarketOverview,
        industry_data: Dict[str, float],
        stock_names: Dict[str, str] = None,
) -> ReportContent:
    """
    生成研报内容

    Args:
        signals: 信号数据
        features: 特征数据
        market_overview: 市场概览
        industry_data: 行业数据
        stock_names: 股票名称字典

    Returns:
        ReportContent 对象
    """
    if stock_names is None:
        stock_names = {}

    # 生成个股分析
    stock_analyses = []
    invest_more = signals[signals["action"] == "INVEST_MORE"].head(10)

    for _, row in invest_more.iterrows():
        symbol = row["symbol"]
        name = stock_names.get(symbol, "")

        confidence, conf_score = calculate_confidence(row)
        risks = identify_risks(row)

        analysis = StockAnalysis(
            symbol=symbol,
            name=name,
            action=row.get("action", "HOLD"),
            score=row.get("score", 0),
            target_weight=row.get("target_weight", 0) * 100,
            price=row.get("close", 0),
            change_pct=row.get("ret_1d", 0) * 100 if "ret_1d" in row else 0,
            ma_dist_20=row.get("ma_dist_20", 0) * 100,
            rsi_14=row.get("rsi_14", 50),
            vol_ratio=row.get("vol_ratio_20", 1),
            high30_breakout=row.get("high30_breakout", 0) == 1,
            main_force_strong=row.get("main_force_strong", 0) == 1,
            has_limit_up_30d=row.get("has_limit_up_30d", 0) == 1,
            tdx_score=row.get("tdx_score", 0),
            confidence=confidence,
            confidence_score=conf_score,
            risks=risks,
        )
        stock_analyses.append(analysis)

    # 风险警告
    risk_warnings = []
    if market_overview.risk_level == "HIGH":
        risk_warnings.append("🔴 **市场风险等级: 高** - 建议降低仓位，以防守为主")
    if market_overview.market_regime == "BEAR":
        risk_warnings.append("🔴 **熊市环境** - 大盘趋势向下，谨慎操作")
    if market_overview.trend_up_pct < 30:
        risk_warnings.append("⚠️ **弱势格局** - 趋势向上的股票占比过低")

    # 投资总结
    if stock_analyses:
        high_conf = sum(1 for s in stock_analyses if s.confidence == "HIGH")
        investment_summary = f"""
今日共筛选出 **{len(stock_analyses)}** 只符合条件的股票，其中：
- 高置信度: {high_conf} 只
- 中置信度: {sum(1 for s in stock_analyses if s.confidence == "MEDIUM")} 只
- 低置信度: {sum(1 for s in stock_analyses if s.confidence == "LOW")} 只

{get_market_regime_info(market_overview.market_regime)[2]}
"""
    else:
        investment_summary = """
今日无符合条件的推荐标的，建议：
- 保持观望，等待更好的入场机会
- 已持仓的按风控规则执行
- 关注市场变化，等待信号
"""

    return ReportContent(
        title="A股量化策略每日研报",
        date=market_overview.date,
        market_overview=market_overview,
        stock_analyses=stock_analyses,
        industry_analysis=industry_data,
        investment_summary=investment_summary,
        risk_warnings=risk_warnings,
        charts={},
    )


def render_report(content: ReportContent, ai_model: str = "DeepSeek") -> str:
    """
    渲染研报为Markdown
    """
    mo = content.market_overview
    regime_emoji, regime_name, regime_desc = get_market_regime_info(mo.market_regime)
    risk_emoji = get_risk_level_info(mo.risk_level)

    # 热门板块
    hot_industries_section = ""
    if mo.hot_industries:
        hot_industries_section = "今日热门板块：**" + "、".join(mo.hot_industries) + "**"

    # 行业分析
    industry_table = generate_industry_analysis(content.industry_analysis)
    fund_flow_analysis = generate_fund_flow_analysis(content.industry_analysis)

    # 推荐汇总
    if content.stock_analyses:
        rec_rows = []
        for s in content.stock_analyses:
            rec_rows.append(f"| {s.symbol} | {s.name} | {get_action_emoji(s.action)} | "
                            f"{s.target_weight:.1f}% | {get_confidence_emoji(s.confidence)} |")
        recommendation_summary = """
| 代码 | 名称 | 建议 | 目标仓位 | 置信度 |
|------|------|------|----------|--------|
""" + "\n".join(rec_rows)
    else:
        recommendation_summary = "> 今日无符合条件的推荐标的"

    # 个股详细分析
    stock_analyses_section = ""
    for idx, s in enumerate(content.stock_analyses, 1):
        risks_section = "\n".join([f"- {r}" for r in s.risks])

        stock_analyses_section += STOCK_ANALYSIS_TEMPLATE.format(
            idx=idx,
            symbol=s.symbol,
            name=s.name,
            action_emoji=get_action_emoji(s.action),
            action=s.action,
            target_weight=s.target_weight,
            confidence_emoji=get_confidence_emoji(s.confidence),
            confidence=s.confidence,
            price=s.price,
            price_comment=generate_price_comment(s.price, s.change_pct),
            change_pct=s.change_pct,
            change_comment="涨" if s.change_pct > 0 else "跌",
            score=s.score,
            ma_dist_20=s.ma_dist_20,
            ma_comment=generate_ma_comment(s.ma_dist_20 / 100),
            rsi_14=s.rsi_14,
            rsi_comment=generate_rsi_comment(s.rsi_14),
            vol_ratio=s.vol_ratio,
            vol_comment=generate_vol_comment(s.vol_ratio),
            high30_status="✅ 突破" if s.high30_breakout else "❌ 未突破",
            main_force_status="✅ 控盘" if s.main_force_strong else "❌ 未控盘",
            limit_up_status="✅ 有" if s.has_limit_up_30d else "❌ 无",
            tdx_score=s.tdx_score,
            risks_section=risks_section,
        )

    # 风险提示
    if content.risk_warnings:
        risk_warnings_section = "\n".join([f"- {w}" for w in content.risk_warnings])
    else:
        risk_warnings_section = "- ✅ 当前无重大风险提示"

    # 操作建议
    action_suggestions = generate_action_suggestions(content.stock_analyses)
    position_suggestions = generate_position_suggestions(mo, content.stock_analyses)

    # 图表
    charts_section = ""
    if content.charts:
        for name, path in content.charts.items():
            charts_section += f"### {name}\n\n![{name}]({path})\n\n"
    else:
        charts_section = "> 图表生成中..."

    # 行业热力图
    industry_heatmap_section = ""
    if "industry_heatmap" in content.charts:
        industry_heatmap_section = f"\n![行业热力图]({content.charts['industry_heatmap']})\n"

    # 渲染
    return REPORT_TEMPLATE.format(
        title=content.title,
        date=content.date,
        market_regime_emoji=regime_emoji,
        market_regime=regime_name,
        market_regime_desc=regime_desc,
        hs300_change=mo.hs300_change,
        trend_up_pct=mo.trend_up_pct,
        avg_return_20d=mo.avg_return_20d,
        risk_level_emoji=risk_emoji,
        risk_level=mo.risk_level,
        hot_industries_section=hot_industries_section,
        industry_table=industry_table,
        fund_flow_analysis=fund_flow_analysis,
        industry_heatmap_section=industry_heatmap_section,
        recommendation_summary=recommendation_summary,
        stock_analyses_section=stock_analyses_section,
        risk_warnings_section=risk_warnings_section,
        investment_summary=content.investment_summary,
        action_suggestions=action_suggestions,
        position_suggestions=position_suggestions,
        charts_section=charts_section,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ai_model=ai_model,
    )