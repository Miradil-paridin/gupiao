"""
AI研报质量评估工具

功能：
- 内容质量评估（事实性、一致性、相关性）
- 置信度计算
- 数据完整性检查
- 研报格式验证
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import re
import json


@dataclass
class QualityScore:
    """质量评分"""
    factuality: float  # 事实性 0-1
    consistency: float  # 一致性 0-1
    relevance: float  # 相关性 0-1
    completeness: float  # 完整性 0-1
    readability: float  # 可读性 0-1
    overall: float  # 综合评分 0-1

    issues: List[str]  # 发现的问题
    suggestions: List[str]  # 改进建议


@dataclass
class DataIntegrityResult:
    """数据完整性检查结果"""
    is_valid: bool
    missing_fields: List[str]
    invalid_values: Dict[str, str]
    warnings: List[str]


def check_data_integrity(
        signals: Any,
        features: Any,
        required_signal_cols: List[str] = None,
        required_feature_cols: List[str] = None,
) -> DataIntegrityResult:
    """
    检查数据完整性

    Args:
        signals: 信号数据
        features: 特征数据
        required_signal_cols: 必需的信号列
        required_feature_cols: 必需的特征列

    Returns:
        DataIntegrityResult
    """
    if required_signal_cols is None:
        required_signal_cols = ["symbol", "date", "action", "score"]

    if required_feature_cols is None:
        required_feature_cols = ["symbol", "date", "close", "volume"]

    missing = []
    invalid = {}
    warnings = []

    # 检查信号数据
    if signals is None or (hasattr(signals, 'empty') and signals.empty):
        missing.append("signals (整个数据集)")
    else:
        for col in required_signal_cols:
            if col not in signals.columns:
                missing.append(f"signals.{col}")

        # 检查空值
        for col in required_signal_cols:
            if col in signals.columns:
                null_count = signals[col].isnull().sum()
                if null_count > 0:
                    warnings.append(f"signals.{col} 有 {null_count} 个空值")

    # 检查特征数据
    if features is None or (hasattr(features, 'empty') and features.empty):
        missing.append("features (整个数据集)")
    else:
        for col in required_feature_cols:
            if col not in features.columns:
                missing.append(f"features.{col}")

    is_valid = len(missing) == 0 and len(invalid) == 0

    return DataIntegrityResult(
        is_valid=is_valid,
        missing_fields=missing,
        invalid_values=invalid,
        warnings=warnings,
    )


def evaluate_report_quality(
        report_content: str,
        signals_count: int = 0,
        has_charts: bool = False,
        has_industry_data: bool = False,
) -> QualityScore:
    """
    评估研报质量

    Args:
        report_content: 研报Markdown内容
        signals_count: 信号数量
        has_charts: 是否有图表
        has_industry_data: 是否有行业数据

    Returns:
        QualityScore
    """
    issues = []
    suggestions = []

    # === 1. 事实性检查 (factuality) ===
    factuality = 1.0

    # 检查是否有具体数据
    has_numbers = bool(re.search(r'\d+\.\d+%', report_content))
    has_prices = bool(re.search(r'¥[\d,]+\.?\d*', report_content))

    if not has_numbers:
        factuality -= 0.2
        issues.append("缺少具体数值数据")
        suggestions.append("添加更多具体的百分比和数值")

    if not has_prices:
        factuality -= 0.1
        suggestions.append("添加股票价格信息")

    # 检查是否有数据来源标注
    has_source = "数据来源" in report_content or "BaoStock" in report_content
    if not has_source:
        factuality -= 0.1
        suggestions.append("标注数据来源")

    # === 2. 一致性检查 (consistency) ===
    consistency = 1.0

    # 检查标题层级
    h1_count = len(re.findall(r'^# ', report_content, re.MULTILINE))
    h2_count = len(re.findall(r'^## ', report_content, re.MULTILINE))

    if h1_count > 1:
        consistency -= 0.1
        issues.append("存在多个一级标题")

    if h2_count < 3:
        consistency -= 0.1
        suggestions.append("增加更多章节结构")

    # 检查表格格式
    tables = re.findall(r'\|[^\n]+\|', report_content)
    if tables:
        for table in tables:
            if table.count('|') < 3:
                consistency -= 0.05

    # === 3. 相关性检查 (relevance) ===
    relevance = 1.0

    # 检查是否有推荐内容
    has_recommendation = "推荐" in report_content or "INVEST_MORE" in report_content
    if not has_recommendation and signals_count == 0:
        relevance -= 0.2
        issues.append("缺少投资推荐内容")

    # 检查是否有风险提示
    has_risk = "风险" in report_content or "⚠️" in report_content
    if not has_risk:
        relevance -= 0.2
        issues.append("缺少风险提示")
        suggestions.append("添加风险提示部分")

    # 检查是否有操作建议
    has_action = "建议" in report_content or "操作" in report_content
    if not has_action:
        relevance -= 0.1
        suggestions.append("添加具体操作建议")

    # === 4. 完整性检查 (completeness) ===
    completeness = 1.0

    # 必要章节
    required_sections = [
        ("市场概览", 0.15),
        ("推荐", 0.15),
        ("风险", 0.15),
        ("建议", 0.15),
    ]

    for section, weight in required_sections:
        if section not in report_content:
            completeness -= weight
            issues.append(f"缺少「{section}」章节")

    # 图表
    if not has_charts:
        completeness -= 0.1
        suggestions.append("添加可视化图表")

    # 行业分析
    if not has_industry_data:
        completeness -= 0.1
        suggestions.append("添加行业分析数据")

    # === 5. 可读性检查 (readability) ===
    readability = 1.0

    # 检查是否使用emoji
    has_emoji = bool(re.search(r'[📊📈📉🔴🟢🟡⚠️✅❌⭐]', report_content))
    if not has_emoji:
        readability -= 0.1
        suggestions.append("使用emoji增强可读性")

    # 检查段落长度
    paragraphs = [p for p in report_content.split('\n\n') if len(p) > 100]
    long_paragraphs = [p for p in paragraphs if len(p) > 500]
    if len(long_paragraphs) > 3:
        readability -= 0.1
        suggestions.append("拆分过长的段落")

    # 检查表格使用
    has_table = '|' in report_content and '---' in report_content
    if not has_table:
        readability -= 0.1
        suggestions.append("使用表格展示结构化数据")

    # === 计算综合分数 ===
    # 确保分数在 0-1 范围内
    factuality = max(0, min(1, factuality))
    consistency = max(0, min(1, consistency))
    relevance = max(0, min(1, relevance))
    completeness = max(0, min(1, completeness))
    readability = max(0, min(1, readability))

    # 加权平均
    overall = (
            factuality * 0.25 +
            consistency * 0.15 +
            relevance * 0.25 +
            completeness * 0.20 +
            readability * 0.15
    )

    return QualityScore(
        factuality=factuality,
        consistency=consistency,
        relevance=relevance,
        completeness=completeness,
        readability=readability,
        overall=overall,
        issues=issues,
        suggestions=suggestions,
    )


def format_quality_report(score: QualityScore) -> str:
    """
    格式化质量报告
    """

    def score_emoji(s: float) -> str:
        if s >= 0.9:
            return "🟢 优秀"
        elif s >= 0.7:
            return "🟡 良好"
        elif s >= 0.5:
            return "🟠 一般"
        else:
            return "🔴 待改进"

    report = f"""
## 📋 研报质量评估报告

### 评分明细

| 维度 | 得分 | 评级 |
|------|------|------|
| 事实性 | {score.factuality:.0%} | {score_emoji(score.factuality)} |
| 一致性 | {score.consistency:.0%} | {score_emoji(score.consistency)} |
| 相关性 | {score.relevance:.0%} | {score_emoji(score.relevance)} |
| 完整性 | {score.completeness:.0%} | {score_emoji(score.completeness)} |
| 可读性 | {score.readability:.0%} | {score_emoji(score.readability)} |
| **综合** | **{score.overall:.0%}** | **{score_emoji(score.overall)}** |

### 发现的问题

"""

    if score.issues:
        for issue in score.issues:
            report += f"- ⚠️ {issue}\n"
    else:
        report += "- ✅ 未发现明显问题\n"

    report += "\n### 改进建议\n\n"

    if score.suggestions:
        for suggestion in score.suggestions:
            report += f"- 💡 {suggestion}\n"
    else:
        report += "- 👍 研报质量良好，继续保持\n"

    return report


def validate_report_format(content: str) -> Dict[str, Any]:
    """
    验证研报格式

    Returns:
        验证结果字典
    """
    result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # 统计信息
    result["stats"]["char_count"] = len(content)
    result["stats"]["line_count"] = content.count('\n')
    result["stats"]["h1_count"] = len(re.findall(r'^# ', content, re.MULTILINE))
    result["stats"]["h2_count"] = len(re.findall(r'^## ', content, re.MULTILINE))
    result["stats"]["h3_count"] = len(re.findall(r'^### ', content, re.MULTILINE))
    result["stats"]["table_count"] = len(re.findall(r'^\|.+\|$', content, re.MULTILINE)) // 3
    result["stats"]["image_count"] = len(re.findall(r'!\[.+\]\(.+\)', content))

    # 格式检查
    if result["stats"]["h1_count"] == 0:
        result["errors"].append("缺少一级标题")
        result["is_valid"] = False

    if result["stats"]["h1_count"] > 1:
        result["warnings"].append("存在多个一级标题")

    if result["stats"]["char_count"] < 500:
        result["warnings"].append("内容较短，建议补充更多分析")

    if result["stats"]["char_count"] > 50000:
        result["warnings"].append("内容过长，建议精简")

    # Markdown语法检查
    unclosed_bold = len(re.findall(r'\*\*[^*]+$', content, re.MULTILINE))
    if unclosed_bold > 0:
        result["errors"].append(f"存在 {unclosed_bold} 处未闭合的加粗标记")
        result["is_valid"] = False

    return result


# =============================================================================
# 置信度标注
# =============================================================================

def add_confidence_annotations(content: str, confidence_map: Dict[str, float]) -> str:
    """
    为研报内容添加置信度标注

    Args:
        content: 研报内容
        confidence_map: {股票代码: 置信度分数}

    Returns:
        添加标注后的内容
    """
    for symbol, score in confidence_map.items():
        if score >= 0.8:
            badge = f"【置信度: ⭐⭐⭐ 高 ({score:.0%})】"
        elif score >= 0.5:
            badge = f"【置信度: ⭐⭐ 中 ({score:.0%})】"
        else:
            badge = f"【置信度: ⭐ 低 ({score:.0%})】"

        # 在股票代码后添加置信度标注
        content = content.replace(
            f"| {symbol} |",
            f"| {symbol} {badge} |"
        )

    return content


def add_data_source_annotations(content: str, sources: List[str]) -> str:
    """
    添加数据来源标注

    Args:
        content: 研报内容
        sources: 数据来源列表

    Returns:
        添加标注后的内容
    """
    source_section = "\n\n---\n\n### 📚 数据来源\n\n"
    for source in sources:
        source_section += f"- {source}\n"

    # 在免责声明前添加
    if "免责声明" in content:
        content = content.replace("## 📜 免责声明", source_section + "\n## 📜 免责声明")
    else:
        content += source_section

    return content


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    # 测试质量评估
    test_content = """
# 📊 A股量化策略每日研报

> 📅 报告日期: 2024-01-15

## 📈 市场概览

| 指标 | 数值 |
|------|------|
| 沪深300 | +1.25% |

## 🎯 今日推荐

| 代码 | 名称 | 建议 |
|------|------|------|
| 600519 | 贵州茅台 | 买入 |

## ⚠️ 风险提示

- 市场有风险，投资需谨慎

## 📜 免责声明

本报告仅供参考。
"""

    score = evaluate_report_quality(
        test_content,
        signals_count=1,
        has_charts=False,
        has_industry_data=False,
    )

    print(format_quality_report(score))

    # 测试格式验证
    validation = validate_report_format(test_content)
    print("\n格式验证结果:")
    print(json.dumps(validation, indent=2, ensure_ascii=False))