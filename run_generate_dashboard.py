"""
每日信号仪表盘 HTML 生成器

生成一个精美的单页HTML报告，包含：
- 今日买入/卖出信号
- 持仓变动
- 买入时机建议
- 行业分布图表
- 风控参数

使用方法：
    python run_generate_dashboard.py

输出：
    out/daily_dashboard.html
    out/daily_dashboard_{date}.html
"""
from __future__ import annotations
from pathlib import Path
from datetime import date
import pandas as pd
import yaml
import json


def load_ranking(base_dir: Path) -> pd.DataFrame:
    p = base_dir / "data" / "signals" / "latest_daily_rank.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_portfolio(base_dir: Path) -> dict:
    p = base_dir / "data" / "portfolio.yaml"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_config(base_dir: Path) -> dict:
    for name in ["config_v31.yaml", "config.yaml"]:
        p = base_dir / name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def generate_dashboard_html(
    ranking: pd.DataFrame,
    portfolio: dict,
    config: dict,
    signal_date: str,
) -> str:
    """生成仪表盘HTML"""

    holdings = portfolio.get("holdings", {})
    held_symbols = set(holdings.keys())

    # 分类
    invest_more = ranking[ranking["action"] == "INVEST_MORE"] if not ranking.empty else pd.DataFrame()
    withdraws = ranking[ranking["action"] == "WITHDRAW"] if not ranking.empty else pd.DataFrame()
    reduces = ranking[ranking["action"] == "REDUCE"] if not ranking.empty else pd.DataFrame()

    invest_symbols = set(invest_more["symbol"].tolist()) if not invest_more.empty else set()
    new_buys = invest_symbols - held_symbols
    keeps = invest_symbols & held_symbols
    must_sell = held_symbols & set(withdraws["symbol"].tolist()) if not withdraws.empty else set()
    may_sell = held_symbols - invest_symbols - must_sell

    # 统计
    total_stocks = len(ranking)
    tradeable = int(ranking["tradeable"].sum()) if "tradeable" in ranking.columns else 0
    eligible = int(ranking["eligible"].sum()) if "eligible" in ranking.columns else 0
    selected = len(invest_more)
    total_weight = invest_more["target_weight"].sum() if not invest_more.empty else 0

    # 行业分布JSON
    industry_data = []
    if not invest_more.empty and "industry" in invest_more.columns:
        for ind, count in invest_more["industry"].value_counts().items():
            # 简化行业名
            short_name = str(ind).replace("C", "").replace("I", "").replace("G", "")
            if len(short_name) > 8:
                short_name = short_name[:8] + "…"
            industry_data.append({"name": short_name, "value": int(count)})

    # 信号强度分布
    strength_data = {"strong": 0, "normal": 0, "weak": 0}
    if not invest_more.empty and "_signal_strength" in invest_more.columns:
        for _, row in invest_more.iterrows():
            s = float(row.get("_signal_strength", 1))
            if s >= 2.0:
                strength_data["strong"] += 1
            elif s >= 1.0:
                strength_data["normal"] += 1
            else:
                strength_data["weak"] += 1

    # 买入卡片数据
    buy_cards = []
    if not invest_more.empty:
        for _, row in invest_more.iterrows():
            sym = str(row["symbol"])
            s = float(row.get("_signal_strength", 1))
            if s >= 2.0:
                tier = "strong"
                tier_label = "强信号"
            elif s >= 1.0:
                tier = "normal"
                tier_label = "普通"
            else:
                tier = "weak"
                tier_label = "弱信号"

            status = "keep" if sym in held_symbols else "new"
            buy_cards.append({
                "symbol": sym,
                "industry": str(row.get("industry", "")),
                "score": round(float(row.get("score", 0)), 2),
                "weight": round(float(row.get("target_weight", 0)) * 100, 1),
                "tdx": round(float(row.get("tdx_score", 0)), 1),
                "strength": s,
                "tier": tier,
                "tier_label": tier_label,
                "entry_path": str(row.get("entry_path", "")),
                "timing": str(row.get("timing", "")),
                "volatility": round(float(row.get("vol_20d", 0)) * 100, 1),
                "close": round(float(row.get("close", 0)), 2),
                "status": status,
                "rank": int(row.get("rank", 0)),
            })

    # 卖出列表
    sell_list = []
    if not withdraws.empty:
        for _, row in withdraws.head(20).iterrows():
            sell_list.append({
                "symbol": str(row["symbol"]),
                "industry": str(row.get("industry", "")),
                "score": round(float(row.get("score", 0)), 2),
                "in_portfolio": str(row["symbol"]) in held_symbols,
            })

    # 策略参数
    strategy = config.get("strategy", {})
    risk = config.get("risk_control", {})

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A股量化信号仪表盘 | {signal_date}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Noto+Sans+SC:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {{
    --bg-primary: #0a0e17;
    --bg-card: #111827;
    --bg-card-hover: #1a2332;
    --bg-accent: #1e293b;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: #1e293b;
    --accent-buy: #10b981;
    --accent-buy-glow: rgba(16, 185, 129, 0.15);
    --accent-sell: #ef4444;
    --accent-sell-glow: rgba(239, 68, 68, 0.15);
    --accent-hold: #f59e0b;
    --accent-strong: #8b5cf6;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --gradient-main: linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #3b82f6 100%);
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: 'DM Sans', 'Noto Sans SC', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    line-height: 1.6;
}}

.container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px;
}}

/* ── Header ── */
.header {{
    text-align: center;
    padding: 48px 0 36px;
    position: relative;
}}

.header::before {{
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(16,185,129,0.08) 0%, transparent 70%);
    pointer-events: none;
}}

.header h1 {{
    font-size: 2.4rem;
    font-weight: 700;
    background: var(--gradient-main);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}}

.header .date {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: var(--text-muted);
}}

.header .subtitle {{
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 4px;
}}

/* ── Stats Grid ── */
.stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
}}

.stat-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s;
}}

.stat-card:hover {{
    border-color: var(--accent-cyan);
    transform: translateY(-2px);
}}

.stat-card .value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-cyan);
}}

.stat-card .value.buy {{ color: var(--accent-buy); }}
.stat-card .value.sell {{ color: var(--accent-sell); }}
.stat-card .value.hold {{ color: var(--accent-hold); }}

.stat-card .label {{
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 4px;
}}

/* ── Section ── */
.section {{
    margin-bottom: 36px;
}}

.section-title {{
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}}

.section-title .icon {{
    font-size: 1.4rem;
}}

/* ── Portfolio Changes ── */
.changes-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
    margin-bottom: 32px;
}}

.change-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}}

.change-card.new {{ border-left: 3px solid var(--accent-buy); }}
.change-card.keep {{ border-left: 3px solid var(--accent-hold); }}
.change-card.sell {{ border-left: 3px solid var(--accent-sell); }}

.change-badge {{
    font-size: 0.75rem;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
    white-space: nowrap;
}}

.change-badge.new {{ background: var(--accent-buy-glow); color: var(--accent-buy); }}
.change-badge.keep {{ background: rgba(245,158,11,0.15); color: var(--accent-hold); }}
.change-badge.sell {{ background: var(--accent-sell-glow); color: var(--accent-sell); }}

/* ── Buy Cards ── */
.buy-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px;
}}

.buy-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    transition: all 0.3s;
    position: relative;
    overflow: hidden;
}}

.buy-card:hover {{
    border-color: var(--accent-buy);
    background: var(--bg-card-hover);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(16,185,129,0.08);
}}

.buy-card .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 12px;
}}

.buy-card .symbol {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--accent-buy);
}}

.buy-card .industry {{
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 2px;
}}

.buy-card .status-badge {{
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
}}

.buy-card .status-badge.new {{
    background: var(--accent-buy-glow);
    color: var(--accent-buy);
}}

.buy-card .status-badge.keep {{
    background: rgba(245,158,11,0.15);
    color: var(--accent-hold);
}}

.buy-card .metrics {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin: 12px 0;
}}

.buy-card .metric {{
    text-align: center;
    padding: 8px 4px;
    background: var(--bg-accent);
    border-radius: 8px;
}}

.buy-card .metric .val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
}}

.buy-card .metric .lbl {{
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 2px;
}}

.buy-card .timing-bar {{
    background: var(--bg-accent);
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.buy-card .timing-bar .clock {{ font-size: 1rem; }}

.buy-card .timing-bar .path {{
    font-size: 0.8rem;
    color: var(--accent-cyan);
    font-weight: 500;
}}

.buy-card .timing-bar .advice {{
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-left: auto;
}}

/* ── Sell Table ── */
.sell-table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-card);
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
}}

.sell-table th {{
    background: var(--bg-accent);
    padding: 12px 16px;
    text-align: left;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.sell-table td {{
    padding: 10px 16px;
    border-top: 1px solid var(--border);
    font-size: 0.9rem;
}}

.sell-table tr:hover td {{
    background: var(--bg-card-hover);
}}

.sell-table .sym {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: var(--accent-sell);
}}

.sell-table .held {{
    color: var(--accent-sell);
    font-weight: 600;
}}

/* ── Charts ── */
.charts-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 32px;
}}

@media (max-width: 768px) {{
    .charts-row {{ grid-template-columns: 1fr; }}
    .buy-grid {{ grid-template-columns: 1fr; }}
    .stats-grid {{ grid-template-columns: repeat(3, 1fr); }}
}}

.chart-box {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
}}

.chart-box h3 {{
    font-size: 0.95rem;
    color: var(--text-secondary);
    margin-bottom: 16px;
}}

.chart-bar {{
    display: flex;
    align-items: center;
    margin-bottom: 8px;
    gap: 10px;
}}

.chart-bar .bar-label {{
    font-size: 0.8rem;
    color: var(--text-secondary);
    width: 120px;
    text-align: right;
    flex-shrink: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}

.chart-bar .bar-track {{
    flex: 1;
    height: 24px;
    background: var(--bg-accent);
    border-radius: 6px;
    overflow: hidden;
}}

.chart-bar .bar-fill {{
    height: 100%;
    border-radius: 6px;
    background: var(--gradient-main);
    transition: width 1s ease;
    display: flex;
    align-items: center;
    padding-left: 8px;
}}

.chart-bar .bar-fill span {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: #fff;
}}

/* Signal strength donut */
.donut-container {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 30px;
}}

.donut-legend {{
    display: flex;
    flex-direction: column;
    gap: 10px;
}}

.donut-legend .item {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
}}

.donut-legend .dot {{
    width: 12px; height: 12px;
    border-radius: 50%;
}}

.dot.strong {{ background: var(--accent-strong); }}
.dot.normal {{ background: var(--accent-buy); }}
.dot.weak {{ background: var(--text-muted); }}

/* ── Config Panel ── */
.config-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 12px;
}}

.config-item {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    display: flex;
    justify-content: space-between;
}}

.config-item .key {{
    font-size: 0.85rem;
    color: var(--text-muted);
}}

.config-item .val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent-cyan);
}}

/* ── Footer ── */
.footer {{
    text-align: center;
    padding: 40px 0 20px;
    color: var(--text-muted);
    font-size: 0.8rem;
}}

/* ── Animations ── */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.animate {{
    animation: fadeInUp 0.5s ease forwards;
    opacity: 0;
}}
</style>
</head>
<body>

<div class="container">

<!-- Header -->
<div class="header animate" style="animation-delay:0.1s">
    <h1>A股量化信号仪表盘</h1>
    <div class="date">{signal_date}</div>
    <div class="subtitle">多因子选股 · 涨停回调 · 主力控盘 · 智能风控</div>
</div>

<!-- Stats -->
<div class="stats-grid animate" style="animation-delay:0.2s">
    <div class="stat-card">
        <div class="value">{total_stocks}</div>
        <div class="label">股票池</div>
    </div>
    <div class="stat-card">
        <div class="value">{tradeable}</div>
        <div class="label">可交易</div>
    </div>
    <div class="stat-card">
        <div class="value">{eligible}</div>
        <div class="label">符合入场</div>
    </div>
    <div class="stat-card">
        <div class="value buy">{selected}</div>
        <div class="label">今日推荐</div>
    </div>
    <div class="stat-card">
        <div class="value hold">{total_weight*100:.0f}%</div>
        <div class="label">总仓位</div>
    </div>
    <div class="stat-card">
        <div class="value sell">{len(withdraws)}</div>
        <div class="label">建议卖出</div>
    </div>
</div>

<!-- Portfolio Changes -->
<div class="section animate" style="animation-delay:0.3s">
    <div class="section-title"><span class="icon">📦</span> 持仓变动（持有 {len(holdings)} → {len(invest_symbols)} 只）</div>
    <div class="changes-grid">
        {"".join(f'<div class="change-card new"><span class="change-badge new">🆕 新买</span><span>{s}</span></div>' for s in sorted(new_buys)) if new_buys else ""}
        {"".join(f'<div class="change-card keep"><span class="change-badge keep">🟢 续持</span><span>{s}</span></div>' for s in sorted(keeps)) if keeps else ""}
        {"".join(f'<div class="change-card sell"><span class="change-badge sell">🔴 卖出</span><span>{s}</span></div>' for s in sorted(must_sell | may_sell)) if must_sell or may_sell else ""}
    </div>
</div>

<!-- Charts -->
<div class="charts-row animate" style="animation-delay:0.35s">
    <div class="chart-box">
        <h3>行业分布</h3>
        {"".join(f'''<div class="chart-bar">
            <span class="bar-label">{d["name"]}</span>
            <div class="bar-track"><div class="bar-fill" style="width:{d["value"]/max(1,selected)*100:.0f}%"><span>{d["value"]}</span></div></div>
        </div>''' for d in industry_data) if industry_data else '<div style="color:var(--text-muted);text-align:center;padding:20px">暂无数据</div>'}
    </div>
    <div class="chart-box">
        <h3>信号强度</h3>
        <div class="donut-container">
            <svg width="140" height="140" viewBox="0 0 140 140">
                {_donut_svg(strength_data)}
            </svg>
            <div class="donut-legend">
                <div class="item"><span class="dot strong"></span> 强信号 {strength_data["strong"]}</div>
                <div class="item"><span class="dot normal"></span> 普通 {strength_data["normal"]}</div>
                <div class="item"><span class="dot weak"></span> 弱信号 {strength_data["weak"]}</div>
            </div>
        </div>
    </div>
</div>

<!-- Buy Signals -->
<div class="section animate" style="animation-delay:0.4s">
    <div class="section-title"><span class="icon">🎯</span> 今日买入信号 ({selected} 只)</div>
    <div class="buy-grid">
        {"".join(_buy_card_html(c) for c in buy_cards)}
    </div>
</div>

<!-- Sell Signals -->
{"" if withdraws.empty else f'''
<div class="section animate" style="animation-delay:0.5s">
    <div class="section-title"><span class="icon">⛔</span> 建议卖出 ({len(withdraws)} 只)</div>
    <table class="sell-table">
        <tr><th>代码</th><th>行业</th><th>评分</th><th>持仓中</th></tr>
        {"".join(f'<tr><td class="sym">{s["symbol"]}</td><td>{s["industry"]}</td><td>{s["score"]:.2f}</td><td class="held">{"⚠️ 是" if s["in_portfolio"] else "—"}</td></tr>' for s in sell_list)}
    </table>
    {"<p style='color:var(--text-muted);margin-top:8px;font-size:0.85rem'>显示前20只，共" + str(len(withdraws)) + "只</p>" if len(withdraws) > 20 else ""}
</div>
'''}

<!-- Strategy Config -->
<div class="section animate" style="animation-delay:0.55s">
    <div class="section-title"><span class="icon">⚙️</span> 策略参数</div>
    <div class="config-grid">
        <div class="config-item"><span class="key">入场模式</span><span class="val">{strategy.get("entry_mode","normal")}</span></div>
        <div class="config-item"><span class="key">选股数</span><span class="val">{strategy.get("top_k",20)}</span></div>
        <div class="config-item"><span class="key">TDX最低分</span><span class="val">{strategy.get("tdx_min_score",1.0)}</span></div>
        <div class="config-item"><span class="key">回调范围</span><span class="val">{strategy.get("pullback_min_pct",0.05)*100:.0f}%-{strategy.get("pullback_max_pct",0.35)*100:.0f}%</span></div>
        <div class="config-item"><span class="key">止损</span><span class="val">{risk.get("stop_loss_pct",0.08)*100:.0f}%</span></div>
        <div class="config-item"><span class="key">止盈回撤</span><span class="val">{risk.get("trailing_stop_pct",0.10)*100:.0f}%</span></div>
        <div class="config-item"><span class="key">行业上限</span><span class="val">{strategy.get("industry_diversification", dict()).get("max_per_industry", 3)} 只/行业</span></div>
        <div class="config-item"><span class="key">最大持仓</span><span class="val">{strategy.get("max_total_position",0.9)*100:.0f}%</span></div>
    </div>
</div>

<div class="footer">
    A股智能量化交易系统 v3.2 · 生成时间 {date.today().isoformat()}
</div>

</div>

<script>
// Animate bars on load
document.addEventListener('DOMContentLoaded', () => {{
    document.querySelectorAll('.bar-fill').forEach(el => {{
        const w = el.style.width;
        el.style.width = '0%';
        setTimeout(() => el.style.width = w, 300);
    }});
    // Stagger card animations
    document.querySelectorAll('.buy-card').forEach((el, i) => {{
        el.style.animation = `fadeInUp 0.4s ease forwards`;
        el.style.animationDelay = `${{0.4 + i * 0.05}}s`;
        el.style.opacity = '0';
    }});
}});
</script>
</body>
</html>"""

    return html


def _donut_svg(data: dict) -> str:
    """生成简单的环形图SVG路径"""
    total = sum(data.values()) or 1
    colors = {"strong": "#8b5cf6", "normal": "#10b981", "weak": "#64748b"}
    segments = []
    offset = 0
    cx, cy, r = 70, 70, 50

    for key in ["strong", "normal", "weak"]:
        val = data[key]
        if val == 0:
            continue
        pct = val / total
        dash = pct * 314  # 2 * pi * 50
        gap = 314 - dash
        segments.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
            f'stroke="{colors[key]}" stroke-width="20" '
            f'stroke-dasharray="{dash:.1f} {gap:.1f}" '
            f'stroke-dashoffset="{-offset:.1f}" '
            f'transform="rotate(-90 {cx} {cy})" />'
        )
        offset += dash

    # Center text
    segments.append(
        f'<text x="{cx}" y="{cy}" text-anchor="middle" dy="0.35em" '
        f'fill="#e2e8f0" font-family="JetBrains Mono" font-size="24" font-weight="700">'
        f'{total}</text>'
    )
    return "\n".join(segments)


def _buy_card_html(c: dict) -> str:
    """生成单个买入卡片HTML"""
    status_class = c["status"]
    status_text = "🆕 新买" if c["status"] == "new" else "📌 续持"

    return f"""
    <div class="buy-card">
        <div class="card-header">
            <div>
                <div class="symbol">{c["symbol"]}</div>
                <div class="industry">{c["industry"]}</div>
            </div>
            <span class="status-badge {status_class}">{status_text}</span>
        </div>
        <div class="metrics">
            <div class="metric"><div class="val">{c["weight"]}%</div><div class="lbl">仓位</div></div>
            <div class="metric"><div class="val">{c["score"]}</div><div class="lbl">评分</div></div>
            <div class="metric"><div class="val">{c["tdx"]}</div><div class="lbl">TDX</div></div>
        </div>
        <div class="metrics">
            <div class="metric"><div class="val">{c["close"]}</div><div class="lbl">现价</div></div>
            <div class="metric"><div class="val">{c["volatility"]}%</div><div class="lbl">波动率</div></div>
            <div class="metric"><div class="val">{c["tier_label"]}</div><div class="lbl">信号</div></div>
        </div>
        <div class="timing-bar">
            <span class="clock">⏰</span>
            <span class="path">{c["entry_path"]}</span>
            <span class="advice">{c["timing"]}</span>
        </div>
    </div>"""


def main():
    base_dir = Path(__file__).resolve().parent

    print("📊 生成每日信号仪表盘...")

    ranking = load_ranking(base_dir)
    portfolio = load_portfolio(base_dir)
    config = load_config(base_dir)

    if ranking.empty:
        print("❌ 未找到信号数据")
        return

    signal_date = str(ranking["date"].iloc[0]) if "date" in ranking.columns else date.today().isoformat()

    html = generate_dashboard_html(ranking, portfolio, config, signal_date)

    # 保存
    out_dir = base_dir / "out"
    out_dir.mkdir(exist_ok=True)

    latest_path = out_dir / "daily_dashboard.html"
    dated_path = out_dir / f"daily_dashboard_{signal_date}.html"

    latest_path.write_text(html, encoding="utf-8")
    dated_path.write_text(html, encoding="utf-8")

    print(f"✅ {latest_path}")
    print(f"✅ {dated_path}")


if __name__ == "__main__":
    main()