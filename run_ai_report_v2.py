"""
AI研报生成主程序（优化版 v2）

功能：
- 结构化研报生成
- 市场热点分析
- 行业轮动分析
- 个股深度分析
- 可视化图表
- 质量控制

使用方法：
    python run_ai_report_v2.py
    python run_ai_report_v2.py --skip-charts  # 跳过图表生成
    python run_ai_report_v2.py --skip-ai      # 跳过AI增强
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import json

import pandas as pd
import numpy as np

# 导入自定义模块
from quant.report_generator import (
    generate_report_content,
    render_report,
    MarketOverview,
    ReportContent,
)
from quant.report_charts import (
    create_price_chart,
    create_indicator_chart,
    create_industry_heatmap,
    create_equity_curve,
    create_signal_summary_chart,
    HAS_MATPLOTLIB,
)
from quant.report_quality import (
    evaluate_report_quality,
    format_quality_report,
    check_data_integrity,
    add_confidence_annotations,
    add_data_source_annotations,
)


def load_signals(base_dir: Path) -> pd.DataFrame:
    """加载最新信号"""
    signals_path = base_dir / "data" / "signals" / "latest_daily_rank.csv"
    if not signals_path.exists():
        raise FileNotFoundError(f"信号文件不存在: {signals_path}")

    df = pd.read_csv(signals_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_features(base_dir: Path) -> pd.DataFrame:
    """加载特征数据"""
    feats_path = base_dir / "data" / "features" / "features_daily.parquet"
    if not feats_path.exists():
        raise FileNotFoundError(f"特征文件不存在: {feats_path}")

    df = pd.read_parquet(feats_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_stock_names(base_dir: Path) -> Dict[str, str]:
    """加载股票名称（从config.yaml读取注释）"""
    import yaml

    config_path = base_dir / "config.yaml"
    names = {}

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 解析注释中的股票名称
            import re
            for match in re.finditer(r'"(\d{6})"\s*#\s*([^\n]+)', content):
                code = match.group(1)
                name = match.group(2).split('-')[0].strip()
                names[code] = name
                # 添加带后缀的版本
                if code.startswith(('6', '5')):
                    names[f"{code}.SH"] = name
                else:
                    names[f"{code}.SZ"] = name
        except Exception as e:
            print(f"  ⚠️ 解析股票名称失败: {e}")

    return names


def compute_market_overview(features: pd.DataFrame, signals: pd.DataFrame) -> MarketOverview:
    """计算市场概览"""
    latest_date = signals["date"].max()

    # 最新日期的特征
    latest_feats = features[features["date"] == latest_date]

    # 趋势向上占比
    if "ma_dist_20" in latest_feats.columns:
        trend_up_pct = (latest_feats["ma_dist_20"] > 0).mean() * 100
    else:
        trend_up_pct = 50.0

    # 平均收益
    if "ret_20d" in latest_feats.columns:
        avg_return_20d = latest_feats["ret_20d"].mean() * 100
    else:
        avg_return_20d = 0.0

    # 市场状态判断
    if trend_up_pct > 60 and avg_return_20d > 3:
        regime = "BULL"
        risk_level = "LOW"
    elif trend_up_pct < 40 and avg_return_20d < -3:
        regime = "BEAR"
        risk_level = "HIGH"
    else:
        regime = "NEUTRAL"
        risk_level = "MEDIUM"

    # 沪深300（如果有）
    hs300_change = 0.0
    hs300_path = Path(__file__).parent / "data" / "index" / "hs300_daily.parquet"
    if hs300_path.exists():
        try:
            hs300 = pd.read_parquet(hs300_path)
            hs300["date"] = pd.to_datetime(hs300["date"]).dt.date
            hs300 = hs300.sort_values("date")
            if len(hs300) >= 2:
                hs300_change = (hs300["close"].iloc[-1] / hs300["close"].iloc[-2] - 1) * 100
        except:
            pass

    # 热门行业（简化版）
    hot_industries = ["白酒", "新能源", "科技"]  # TODO: 从数据计算

    return MarketOverview(
        date=str(latest_date),
        market_regime=regime,
        hs300_change=hs300_change,
        trend_up_pct=trend_up_pct,
        avg_return_20d=avg_return_20d,
        hot_industries=hot_industries,
        risk_level=risk_level,
    )



def compute_industry_performance(features: pd.DataFrame, config_path: Path) -> Dict[str, float]:
    """计算行业表现"""
    if not config_path.exists():
        return {}

    # 读取配置中的行业分组
    industry_map = {}
    current_industry = "其他"

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    for line in content.split("\n"):
        if "==========" in line:
            if "消费" in line:
                current_industry = "消费"
            elif "医药" in line:
                current_industry = "医药"
            elif "科技" in line:
                current_industry = "科技"
            elif "金融" in line:
                current_industry = "金融"
            elif "新能源" in line:
                current_industry = "新能源"
            else:
                current_industry = "其他"

        if '"' in line:
            try:
                code = line.split('"')[1]
                if code.isdigit():
                    industry_map[code] = current_industry
            except:
                pass

    if not industry_map:
        return {}

    # 最新日期
    latest_date = features["date"].max()
    latest_feats = features[features["date"] == latest_date].copy()

    # 分配行业
    latest_feats["industry"] = latest_feats["symbol"].astype(str).map(
        lambda s: industry_map.get(s, "其他")
    )

    # 计算行业平均收益
    industry_returns = latest_feats.groupby("industry")["ret_20d"].mean() * 100

    return industry_returns.to_dict()


def generate_charts(base_dir: Path, signals: pd.DataFrame, features: pd.DataFrame) -> Dict[str, str]:
    """生成图表"""
    if not HAS_MATPLOTLIB:
        print("  ⚠️ matplotlib 未安装，跳过图表")
        return {}

    charts = {}
    out_dir = base_dir / "out" / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 选取前3只股票
    top_stocks = signals[signals["action"] == "INVEST_MORE"].head(3)["symbol"].tolist()

    # 1. 价格走势 + 均线
    for symbol in top_stocks:
        chart_path = create_price_chart(
            features,
            symbol,
            days=60,
            output_path=out_dir / f"price_{symbol}.png"
        )
        if chart_path:
            charts[f"价格走势_{symbol}"] = chart_path

    # 2. 技术指标图
    for symbol in top_stocks:
        indicator_path = create_indicator_chart(
            features,
            symbol,
            days=30,
            output_path=out_dir / f"indicator_{symbol}.png"
        )
        if indicator_path:
            charts[f"技术指标_{symbol}"] = indicator_path

    # 3. 行业热力图
    try:
        # create_industry_heatmap 需要 {行业:涨跌幅} 字典，而不是 DataFrame
        config_path = base_dir / "config.yaml"
        if not config_path.exists():
            config_path = base_dir / "config.yaml"
        industry_data = compute_industry_performance(features, config_path)
        heatmap_path = create_industry_heatmap(
            industry_data,
            output_path=out_dir / "industry_heatmap.png"
        )
        if heatmap_path:
            charts["行业热力图"] = heatmap_path
    except Exception as e:
        print(f"  ⚠️ 行业热力图生成失败: {e}")

    # 4. 收益曲线（如果有回测数据）
    equity_path = base_dir / "data" / "backtests" / "backtest_strategy_v3_equity.csv"
    if equity_path.exists():
        print("  - 生成收益曲线图...")
        equity_df = pd.read_csv(equity_path)
        chart_path = create_equity_curve(
            equity_df,
            title="策略收益曲线",
            output_path=out_dir / "equity_curve.png"
        )
        if chart_path:
            charts["收益曲线"] = chart_path

    return charts


def call_llm_enhance(
        base_content: str,
        market_overview: MarketOverview,
        provider: str = "deepseek",
) -> str:
    """
    调用LLM增强研报内容
    """
    try:
        # 尝试导入LLM模块
        if provider == "deepseek":
            from quant.llm_deepseek import chat_complete

            def call_deepseek(prompt_text: str) -> str:
                resp = chat_complete(messages=[{"role": "user", "content": prompt_text}])
                return resp.content.strip()

            prompt = f"""请基于以下研报草稿，进行内容增强和优化。

要求：
1. 保持原有结构和数据
2. 增加专业的市场分析观点
3. 对推荐的股票给出更详细的分析理由
4. 增加行业前景分析
5. 给出更具体的操作建议

当前市场环境：{market_overview.market_regime}
沪深300涨跌：{market_overview.hs300_change:+.2f}%

研报草稿：
{base_content[:8000]}  # 限制长度

请直接输出增强后的研报内容（Markdown格式）：
"""
            enhanced = call_deepseek(prompt)
            if enhanced and len(enhanced) > 500:
                return enhanced

    except Exception as e:
        print(f"  ⚠️ LLM增强失败: {e}")

    return base_content


def save_report(
        base_dir: Path,
        content: str,
        report_date: str,
        quality_report: str = "",
) -> tuple:
    """保存研报"""
    reports_dir = base_dir / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_dir = base_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存Markdown
    md_path = reports_dir / f"ai_report_{report_date}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
        if quality_report:
            f.write("\n\n---\n\n")
            f.write(quality_report)

    # 保存到 out 目录（最新版）
    latest_md = out_dir / "latest_ai_report.md"
    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(content)

    return md_path, latest_md


def main():
    parser = argparse.ArgumentParser(description="AI研报生成（优化版）")
    parser.add_argument("--skip-charts", action="store_true", help="跳过图表生成")
    parser.add_argument("--skip-ai", action="store_true", help="跳过AI增强")
    parser.add_argument("--skip-quality", action="store_true", help="跳过质量评估")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    print("=" * 60)
    print("📊 AI研报生成系统 v2.0")
    print("=" * 60)

    # Step 1: 加载数据
    print("\n📂 Step 1: 加载数据...")

    try:
        signals = load_signals(base_dir)
        features = load_features(base_dir)
        stock_names = load_stock_names(base_dir)

        print(f"   信号数: {len(signals)}")
        print(f"   特征数: {len(features)}")
        print(f"   股票名: {len(stock_names)}")
    except FileNotFoundError as e:
        print(f"   ❌ 错误: {e}")
        print("   请先运行 python run_all_daily.py 生成数据")
        return

    # 数据完整性检查
    integrity = check_data_integrity(signals, features)
    if not integrity.is_valid:
        print(f"   ⚠️ 数据完整性问题: {integrity.missing_fields}")

    # Step 2: 计算市场概览
    print("\n📈 Step 2: 计算市场概览...")
    market_overview = compute_market_overview(features, signals)
    print(f"   市场状态: {market_overview.market_regime}")
    print(f"   沪深300: {market_overview.hs300_change:+.2f}%")
    print(f"   趋势向上: {market_overview.trend_up_pct:.1f}%")
    print(f"   风险等级: {market_overview.risk_level}")

    # Step 3: 计算行业表现
    print("\n📊 Step 3: 计算行业表现...")
    config_path = base_dir / "config.yaml"
    industry_data = compute_industry_performance(features, config_path)
    if industry_data:
        print(f"   行业数: {len(industry_data)}")
        top3 = sorted(industry_data.items(), key=lambda x: x[1], reverse=True)[:3]
        for ind, ret in top3:
            print(f"   {ind}: {ret:+.2f}%")

    # Step 4: 生成基础研报
    print("\n📝 Step 4: 生成基础研报...")
    base_content = generate_report_content(
        signals=signals,
        features=features,
        stock_names=stock_names,
        market_overview=market_overview,
        industry_data=industry_data,
    )
    report_md = render_report(base_content)

    # Step 5: 图表生成
    charts = {}
    if not args.skip_charts:
        print("\n📉 Step 5: 生成图表...")
        charts = generate_charts(base_dir, signals, features)
        print(f"   图表数: {len(charts)}")
    else:
        print("\n📉 Step 5: 跳过图表生成")

    # Step 6: LLM增强
    if not args.skip_ai:
        print("\n🤖 Step 6: LLM增强...")
        report_md = call_llm_enhance(report_md, market_overview)
    else:
        print("\n🤖 Step 6: 跳过AI增强")

    # Step 7: 添加信心标注 & 数据来源
    print("\n✅ Step 7: 添加信心标注...")
    confidence_map = {}
    invest_more = signals[signals["action"] == "INVEST_MORE"]
    for _, row in invest_more.iterrows():
        symbol = str(row["symbol"])
        score = row.get("score", 0)
        score = float(score) if not np.isnan(score) else 0
        score = min(score / 3.0, 1.0)
        if row.get("high30_breakout", 0) == 1:
            score += 0.2
        if row.get("main_force_strong", 0) == 1:
            score += 0.2
        if row.get("has_limit_up_30d", 0) == 1:
            score += 0.1
        confidence_map[symbol] = min(1.0, score)

    report_md = add_confidence_annotations(report_md, confidence_map)

    sources = [
        "BaoStock - 行情数据",
        "东方财富 - 新闻资讯",
        "财联社 - 快讯",
        "量化模型 - 信号生成",
    ]
    report_md = add_data_source_annotations(report_md, sources)

    # Step 8: 质量评估
    quality_report = ""
    if not args.skip_quality:
        print("\n✅ Step 8: 质量评估...")
        quality_score = evaluate_report_quality(
            report_md,
            signals_count=len(invest_more),
            has_charts=len(charts) > 0,
            has_industry_data=len(industry_data) > 0,
        )
        quality_report = format_quality_report(quality_score)
        print(f"   综合评分: {quality_score.overall:.0%}")
        if quality_score.issues:
            print(f"   发现问题: {len(quality_score.issues)} 个")
    else:
        print("\n✅ Step 8: 跳过质量评估")

    # Step 9: 保存研报
    print("\n💾 Step 9: 保存研报...")
    report_date = market_overview.date
    md_path, latest_path = save_report(base_dir, report_md, report_date, quality_report)
    print(f"   保存到: {md_path}")
    print(f"   最新: {latest_path}")

    # 完成
    print("\n" + "=" * 60)
    print("✅ 研报生成完成！")
    print("=" * 60)

    # 显示摘要
    invest_count = len(signals[signals["action"] == "INVEST_MORE"])
    print(f"\n📋 研报摘要:")
    print(f"   日期: {report_date}")
    print(f"   市场: {market_overview.market_regime}")
    print(f"   推荐: {invest_count} 只股票")
    print(f"   图表: {len(charts)} 个")

    if invest_count > 0:
        print(f"\n🎯 今日推荐:")
        for _, row in invest_more.head(5).iterrows():
            symbol = row["symbol"]
            name = stock_names.get(str(symbol), "")
            score = row.get("score", 0)
            print(f"   - {symbol} {name}: 得分 {score:.2f}")


if __name__ == "__main__":
    main()
