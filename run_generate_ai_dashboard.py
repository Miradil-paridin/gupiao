"""
AI研报仪表盘 HTML 生成器

将 out/latest_ai_report.md 渲染为网页，并融合：
- 今日推荐摘要
- 回测关键指标
- 图表（行业热力图、权益曲线、个股图）

输出：
- out/ai_report_dashboard.html
- out/ai_report_dashboard_{date}.html
"""
from __future__ import annotations

import html
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd


def _fmt_inline(text: str) -> str:
    s = html.escape(text)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<a href="\2" target="_blank">\1</a>', s)
    return s


def markdown_to_html_lite(md_text: str) -> str:
    lines = md_text.splitlines()
    out: list[str] = []
    i = 0
    in_ul = False
    in_ol = False
    in_quote = False
    para: list[str] = []

    def flush_para() -> None:
        nonlocal para
        if para:
            out.append(f"<p>{_fmt_inline(' '.join(para).strip())}</p>")
            para = []

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def close_quote() -> None:
        nonlocal in_quote
        if in_quote:
            out.append("</blockquote>")
            in_quote = False

    def is_table_separator(row: str) -> bool:
        s = row.strip()
        return bool(re.match(r"^\|?[\s:-]+\|[\s|:-]*\|?$", s))

    while i < len(lines):
        line = lines[i].rstrip("\n")
        s = line.strip()

        if not s:
            flush_para()
            close_lists()
            close_quote()
            i += 1
            continue

        # Markdown table
        if "|" in s and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            flush_para()
            close_lists()
            close_quote()
            headers = [c.strip() for c in s.strip("|").split("|")]
            out.append("<table><thead><tr>" + "".join(f"<th>{_fmt_inline(h)}</th>" for h in headers) + "</tr></thead><tbody>")
            i += 2
            while i < len(lines):
                row = lines[i].strip()
                if not row or "|" not in row:
                    break
                cols = [c.strip() for c in row.strip("|").split("|")]
                if len(cols) < len(headers):
                    cols += [""] * (len(headers) - len(cols))
                out.append("<tr>" + "".join(f"<td>{_fmt_inline(c)}</td>" for c in cols[: len(headers)]) + "</tr>")
                i += 1
            out.append("</tbody></table>")
            continue

        # Headings
        if s.startswith("### "):
            flush_para()
            close_lists()
            close_quote()
            out.append(f"<h3>{_fmt_inline(s[4:])}</h3>")
            i += 1
            continue
        if s.startswith("## "):
            flush_para()
            close_lists()
            close_quote()
            out.append(f"<h2>{_fmt_inline(s[3:])}</h2>")
            i += 1
            continue
        if s.startswith("# "):
            flush_para()
            close_lists()
            close_quote()
            out.append(f"<h1>{_fmt_inline(s[2:])}</h1>")
            i += 1
            continue

        # Horizontal line
        if re.match(r"^-{3,}$", s):
            flush_para()
            close_lists()
            close_quote()
            out.append("<hr />")
            i += 1
            continue

        # Quote
        if s.startswith(">"):
            flush_para()
            close_lists()
            if not in_quote:
                out.append("<blockquote>")
                in_quote = True
            out.append(f"<p>{_fmt_inline(s.lstrip('> ').strip())}</p>")
            i += 1
            continue
        else:
            close_quote()

        # List
        if re.match(r"^\s*[-*]\s+", line):
            flush_para()
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            item = re.sub(r"^\s*[-*]\s+", "", line).strip()
            out.append(f"<li>{_fmt_inline(item)}</li>")
            i += 1
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            flush_para()
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            item = re.sub(r"^\s*\d+\.\s+", "", line).strip()
            out.append(f"<li>{_fmt_inline(item)}</li>")
            i += 1
            continue

        para.append(s)
        i += 1

    flush_para()
    close_lists()
    close_quote()
    return "\n".join(out)


def load_latest_ai_markdown(base_dir: Path) -> str:
    latest = base_dir / "out" / "latest_ai_report.md"
    if latest.exists():
        return latest.read_text(encoding="utf-8", errors="ignore")

    reports_dir = base_dir / "data" / "reports"
    if reports_dir.exists():
        candidates = sorted(reports_dir.glob("ai_report_*.md"))
        if candidates:
            return candidates[-1].read_text(encoding="utf-8", errors="ignore")
    return "# AI研报暂不可用\n\n暂无可展示内容。"


def extract_report_date(md_text: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", md_text)
    if m:
        return m.group(1)
    return date.today().isoformat()


def load_signals(base_dir: Path) -> pd.DataFrame:
    p = base_dir / "data" / "signals" / "latest_daily_rank.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_backtest_stats(base_dir: Path) -> dict:
    p = base_dir / "data" / "backtests" / "backtest_strategy_v3_stats.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_chart_tiles(base_dir: Path, symbols: list[str]) -> list[dict]:
    out_dir = base_dir / "out"
    charts_dir = out_dir / "charts"
    tiles: list[dict] = []

    def add_chart(label: str, rel_path: str) -> None:
        p = out_dir / rel_path
        if p.exists():
            tiles.append({"label": label, "path": rel_path.replace("\\", "/")})

    add_chart("行业热力图", "charts/industry_heatmap.png")
    add_chart("策略收益曲线", "charts/equity_curve.png")

    for sym in symbols[:3]:
        add_chart(f"{sym} 价格走势", f"charts/price_{sym}.png")
        add_chart(f"{sym} 技术指标", f"charts/indicator_{sym}.png")
    return tiles


def generate_ai_dashboard_html(
    report_date: str,
    report_html: str,
    invest_more: pd.DataFrame,
    stats: dict,
    chart_tiles: list[dict],
) -> str:
    market_regime = "N/A"
    if not invest_more.empty and "market_regime" in invest_more.columns:
        market_regime = str(invest_more["market_regime"].iloc[0])

    picks = len(invest_more)
    avg_score = float(invest_more["score"].mean()) if (not invest_more.empty and "score" in invest_more.columns) else 0.0
    top_symbols = ", ".join(invest_more["symbol"].astype(str).head(5).tolist()) if picks else "暂无"

    annual = float(stats.get("annual_return_pct", 0.0))
    mdd = float(stats.get("max_drawdown_pct", 0.0))
    sharpe = float(stats.get("sharpe", 0.0))

    cards = "".join(
        f"""
        <div class="pick-card">
            <div class="sym">{html.escape(str(row.get("symbol","")))}</div>
            <div class="meta">{html.escape(str(row.get("industry","")))}</div>
            <div class="row"><span>评分</span><b>{float(row.get("score",0)):.2f}</b></div>
            <div class="row"><span>目标仓位</span><b>{float(row.get("target_weight",0))*100:.1f}%</b></div>
            <div class="row"><span>信号强度</span><b>{float(row.get("_signal_strength",1)):.1f}</b></div>
            <div class="row"><span>路径</span><b>{html.escape(str(row.get("entry_path","")))}</b></div>
        </div>
        """
        for _, row in invest_more.head(6).iterrows()
    )
    if not cards:
        cards = '<div class="empty">今日暂无推荐</div>'

    chart_blocks = "".join(
        f"""
        <figure class="chart-item">
            <img src="{html.escape(tile["path"])}" alt="{html.escape(tile["label"])}" loading="lazy" />
            <figcaption>{html.escape(tile["label"])}</figcaption>
        </figure>
        """
        for tile in chart_tiles
    )
    if not chart_blocks:
        chart_blocks = '<div class="empty">暂无图表</div>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>AI研报仪表盘 | {report_date}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:#06121f; --panel:#0d1d30; --panel2:#122740; --line:#1f3550;
  --txt:#e9f2ff; --muted:#96afca; --brand:#14b8a6; --brand2:#22d3ee; --warn:#f59e0b;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; color:var(--txt);
  font-family:'Manrope','Noto Sans SC',sans-serif;
  background:
    radial-gradient(900px 400px at 10% -10%, rgba(20,184,166,.18), transparent 70%),
    radial-gradient(800px 360px at 90% -20%, rgba(34,211,238,.16), transparent 70%),
    var(--bg);
}}
.wrap {{ max-width:1400px; margin:0 auto; padding:24px; }}
.head {{ padding:22px 0 12px; }}
.head h1 {{ margin:0; font-size:34px; font-weight:800; letter-spacing:.3px; }}
.sub {{ margin-top:8px; color:var(--muted); font-size:14px; }}
.grid {{ display:grid; gap:16px; }}
.stats {{ grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); margin-top:16px; }}
.card {{
  background:linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.00)), var(--panel);
  border:1px solid var(--line); border-radius:14px; padding:14px 16px;
}}
.k {{ color:var(--muted); font-size:12px; }}
.v {{ margin-top:6px; font-size:24px; font-weight:800; }}
.v.small {{ font-size:18px; font-weight:700; }}
.main {{ grid-template-columns: 360px 1fr; margin-top:16px; align-items:start; }}
.section-title {{ font-size:16px; font-weight:700; margin-bottom:10px; }}
.pick-card {{
  border:1px solid var(--line); background:var(--panel2);
  border-radius:12px; padding:12px; margin-bottom:10px;
}}
.pick-card .sym {{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:15px; }}
.pick-card .meta {{ color:var(--muted); font-size:12px; margin-top:4px; }}
.pick-card .row {{ display:flex; justify-content:space-between; font-size:13px; margin-top:6px; }}
.charts {{ display:grid; gap:12px; grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); }}
.chart-item {{ margin:0; border:1px solid var(--line); border-radius:12px; overflow:hidden; background:#0b1828; }}
.chart-item img {{ width:100%; display:block; }}
.chart-item figcaption {{ color:var(--muted); font-size:12px; padding:8px 10px; border-top:1px solid var(--line); }}
.report {{ margin-top:16px; }}
.report article {{
  border:1px solid var(--line); background:var(--panel);
  border-radius:14px; padding:20px; line-height:1.7;
}}
.report h1,.report h2,.report h3 {{ margin:18px 0 10px; }}
.report h1 {{ font-size:24px; }} .report h2 {{ font-size:20px; }} .report h3 {{ font-size:16px; }}
.report p {{ margin:8px 0; color:#dbe8f8; }}
.report table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:13px; }}
.report th,.report td {{ border:1px solid var(--line); padding:8px; text-align:left; }}
.report th {{ background:#10253d; }}
.report code {{ font-family:'JetBrains Mono', monospace; background:#10253d; padding:2px 6px; border-radius:6px; }}
.report blockquote {{ margin:10px 0; padding:8px 12px; border-left:3px solid var(--brand2); background:#0c2036; color:#cce2ff; }}
.empty {{ color:var(--muted); font-size:13px; padding:12px; border:1px dashed var(--line); border-radius:10px; }}
.foot {{ color:var(--muted); font-size:12px; text-align:center; margin:16px 0 6px; }}
@media (max-width: 980px) {{ .main {{ grid-template-columns: 1fr; }} .head h1 {{ font-size:28px; }} }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <h1>AI研报仪表盘</h1>
      <div class="sub">报告日期 {report_date} · 市场状态 {market_regime} · 推荐股票 {picks} 只（{top_symbols}）</div>
    </div>

    <div class="grid stats">
      <div class="card"><div class="k">市场状态</div><div class="v">{market_regime}</div></div>
      <div class="card"><div class="k">今日推荐</div><div class="v">{picks}</div></div>
      <div class="card"><div class="k">平均评分</div><div class="v">{avg_score:.2f}</div></div>
      <div class="card"><div class="k">回测年化</div><div class="v">{annual:+.2f}%</div></div>
      <div class="card"><div class="k">最大回撤</div><div class="v">{mdd:.2f}%</div></div>
      <div class="card"><div class="k">夏普比率</div><div class="v">{sharpe:.2f}</div></div>
    </div>

    <div class="grid main">
      <section class="card">
        <div class="section-title">推荐摘要</div>
        {cards}
      </section>
      <section class="card">
        <div class="section-title">研报图表</div>
        <div class="charts">{chart_blocks}</div>
      </section>
    </div>

    <section class="report">
      <article>{report_html}</article>
    </section>

    <div class="foot">A股智能量化交易系统 · AI报告仪表盘</div>
  </div>
</body>
</html>"""


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    print("📘 生成 AI 研报仪表盘...")

    md_text = load_latest_ai_markdown(base_dir)
    report_date = extract_report_date(md_text)
    report_html = markdown_to_html_lite(md_text)

    ranking = load_signals(base_dir)
    invest_more = ranking[ranking["action"] == "INVEST_MORE"] if (not ranking.empty and "action" in ranking.columns) else pd.DataFrame()
    stats = load_backtest_stats(base_dir)
    chart_tiles = collect_chart_tiles(base_dir, invest_more["symbol"].astype(str).tolist() if not invest_more.empty else [])

    html_text = generate_ai_dashboard_html(
        report_date=report_date,
        report_html=report_html,
        invest_more=invest_more,
        stats=stats,
        chart_tiles=chart_tiles,
    )

    out_dir = base_dir / "out"
    out_dir.mkdir(exist_ok=True)
    latest_path = out_dir / "ai_report_dashboard.html"
    dated_path = out_dir / f"ai_report_dashboard_{report_date}.html"
    latest_path.write_text(html_text, encoding="utf-8")
    dated_path.write_text(html_text, encoding="utf-8")

    print(f"✅ {latest_path}")
    print(f"✅ {dated_path}")


if __name__ == "__main__":
    main()

