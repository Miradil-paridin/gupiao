from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_backtest_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"trades file not found: {path}")
    df = pd.read_csv(path)
    need = [
        "symbol",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "hold_days",
        "pnl_pct",
        "weight",
        "entry_reason",
        "exit_reason",
    ]
    for c in need:
        if c not in df.columns:
            df[c] = ""

    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
    df["hold_days"] = pd.to_numeric(df["hold_days"], errors="coerce").fillna(0).astype(int)
    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce").fillna(0.0)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df["symbol"] = df["symbol"].astype(str)
    return df


def _filter_date_range(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df.copy()
    start_dt = pd.to_datetime(start) if start else None
    end_dt = pd.to_datetime(end) if end else None
    if start_dt is not None:
        out = out[(out["entry_date"] >= start_dt) | (out["exit_date"] >= start_dt)]
    if end_dt is not None:
        out = out[(out["entry_date"] <= end_dt) | (out["exit_date"] <= end_dt)]
    return out


def _fmt_date(x: Any) -> str:
    if pd.isna(x):
        return ""
    return pd.to_datetime(x).strftime("%Y-%m-%d")


def _build_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "n_trades": 0,
            "n_symbols": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "sum_pnl": 0.0,
        }
    wins = int((df["pnl_pct"] > 0).sum())
    return {
        "n_trades": int(len(df)),
        "n_symbols": int(df["symbol"].nunique()),
        "win_rate": float(wins / max(len(df), 1) * 100.0),
        "avg_pnl": float(df["pnl_pct"].mean()),
        "sum_pnl": float(df["pnl_pct"].sum()),
    }


def _build_price_ref_lookup(base_dir: Path, symbols: set[str], dates: set[str]) -> dict[tuple[str, str], float]:
    """
    Build non-HFQ reference prices from raw daily files.
    Current implementation uses amount/volume as an unadjusted-price proxy.
    """
    lookup: dict[tuple[str, str], float] = {}
    raw_dir = base_dir / "data" / "raw" / "market_daily"
    if not raw_dir.exists():
        return lookup

    for sym in symbols:
        p = raw_dir / f"{sym}.csv"
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p, usecols=["date", "volume", "amount", "close"])
        except Exception:
            continue

        if d.empty:
            continue

        d["date"] = d["date"].astype(str)
        d = d[d["date"].isin(dates)]
        if d.empty:
            continue

        d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
        d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        d["ref_price"] = d["amount"] / d["volume"]
        # Some providers mix volume units (shares vs lots). If ref is far above HFQ close,
        # treat volume as "lots" and scale down by 100 as a practical fallback.
        lot_mask = pd.notna(d["close"]) & pd.notna(d["ref_price"]) & (d["close"] > 0) & (d["ref_price"] > d["close"] * 2.0)
        d.loc[lot_mask, "ref_price"] = d.loc[lot_mask, "ref_price"] / 100.0
        d.loc[(d["volume"] <= 0) | (~pd.notna(d["ref_price"])) | (d["ref_price"] <= 0), "ref_price"] = pd.NA

        for _, r in d.iterrows():
            if pd.notna(r["ref_price"]):
                lookup[(sym, str(r["date"]))] = float(r["ref_price"])
    return lookup


def _attach_reference_prices(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["entry_ref_price"] = pd.Series(dtype="float64")
        out["exit_ref_price"] = pd.Series(dtype="float64")
        return out

    out["entry_date_key"] = out["entry_date"].dt.strftime("%Y-%m-%d").fillna("")
    out["exit_date_key"] = out["exit_date"].dt.strftime("%Y-%m-%d").fillna("")

    symbols = set(out["symbol"].dropna().astype(str).unique().tolist())
    dates = set(out["entry_date_key"].tolist()) | set(out["exit_date_key"].tolist())
    dates = {d for d in dates if d}
    lookup = _build_price_ref_lookup(base_dir=base_dir, symbols=symbols, dates=dates)

    out["entry_ref_price"] = [
        lookup.get((str(sym), str(d)), float("nan")) for sym, d in zip(out["symbol"], out["entry_date_key"])
    ]
    out["exit_ref_price"] = [
        lookup.get((str(sym), str(d)), float("nan")) for sym, d in zip(out["symbol"], out["exit_date_key"])
    ]
    out = out.drop(columns=["entry_date_key", "exit_date_key"], errors="ignore")
    return out


def _build_symbol_panels(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class='empty'>该区间没有交易记录。</p>"

    panels: list[str] = []
    grouped = df.sort_values(["symbol", "entry_date", "exit_date"]).groupby("symbol", sort=True)
    for sym, g in grouped:
        g = g.reset_index(drop=True)
        wins = int((g["pnl_pct"] > 0).sum())
        win_rate = wins / max(len(g), 1) * 100.0
        avg_pnl = float(g["pnl_pct"].mean())
        sum_pnl = float(g["pnl_pct"].sum())
        stat = f"交易 {len(g)} 笔 | 胜率 {win_rate:.1f}% | 平均 {avg_pnl:+.2f}% | 累计 {sum_pnl:+.2f}%"

        rows: list[str] = []
        for i, row in g.iterrows():
            pnl = _safe_float(row["pnl_pct"])
            pnl_cls = "pnl-pos" if pnl > 0 else ("pnl-neg" if pnl < 0 else "pnl-flat")
            rows.append(
                "<tr>"
                f"<td>{i + 1}</td>"
                f"<td>{_fmt_date(row['entry_date'])}</td>"
                f"<td>{_fmt_date(row['exit_date'])}</td>"
                f"<td>¥{_safe_float(row['entry_price']):.2f}</td>"
                f"<td>¥{_safe_float(row['exit_price']):.2f}</td>"
                f"<td>{('¥' + format(_safe_float(row.get('entry_ref_price', float('nan'))), '.2f')) if pd.notna(row.get('entry_ref_price', pd.NA)) else '--'}</td>"
                f"<td>{('¥' + format(_safe_float(row.get('exit_ref_price', float('nan'))), '.2f')) if pd.notna(row.get('exit_ref_price', pd.NA)) else '--'}</td>"
                f"<td>{_safe_float(row['weight']):.1f}%</td>"
                f"<td>{int(row['hold_days'])}</td>"
                f"<td class='{pnl_cls}'>{pnl:+.2f}%</td>"
                f"<td>{str(row.get('entry_reason', ''))}</td>"
                f"<td>{str(row.get('exit_reason', ''))}</td>"
                "</tr>"
            )

        panels.append(
            "<details class='symbol-card' open>"
            f"<summary><span class='sym'>{sym}</span><span class='stat'>{stat}</span></summary>"
            "<div class='table-wrap'>"
            "<table>"
            "<thead><tr>"
            "<th>#</th><th>买入日期</th><th>卖出日期</th>"
            "<th>买入价(HFQ, 元/股)</th><th>卖出价(HFQ, 元/股)</th>"
            "<th>买入参考价(不复权近似)</th><th>卖出参考价(不复权近似)</th>"
            "<th>仓位(%)</th><th>持有天数</th><th>收益率</th><th>买入理由</th><th>卖出原因</th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
            "</div>"
            "</details>"
        )
    return "\n".join(panels)


def _load_three_layer_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _fmt_pct(x: Any, digits: int = 2) -> str:
    v = _safe_float(x, default=float("nan"))
    if not pd.notna(v):
        return "--"
    return f"{v:.{digits}f}%"


def _fmt_num(x: Any, digits: int = 2) -> str:
    v = _safe_float(x, default=float("nan"))
    if not pd.notna(v):
        return "--"
    return f"{v:.{digits}f}"


def _fmt_ratio_pct(x: Any, digits: int = 1) -> str:
    v = _safe_float(x, default=float("nan"))
    if not pd.notna(v):
        return "--"
    return f"{v * 100.0:.{digits}f}%"


def _status_label(status: str) -> str:
    s = str(status or "").lower()
    if s in {"ok", "pass", "good"}:
        return "通过"
    if s in {"warning", "warn"}:
        return "预警"
    if s in {"fail", "bad", "risk"}:
        return "风险"
    return "未知"


def _status_class(status: str) -> str:
    s = str(status or "").lower()
    if s in {"ok", "pass", "good"}:
        return "st-ok"
    if s in {"warning", "warn"}:
        return "st-warn"
    if s in {"fail", "bad", "risk"}:
        return "st-risk"
    return "st-unknown"


def _build_three_layer_section(report: dict[str, Any] | None) -> str:
    if not report:
        return ""

    overall_status = str(report.get("overall_status", "unknown"))
    start_txt = str(report.get("period_start", "N/A"))
    end_txt = str(report.get("period_end", "N/A"))

    layer1 = report.get("layer_1_backtest") or report.get("strategy_quality_layer") or {}
    layer2 = report.get("layer_2_execution") or report.get("execution_realism_layer") or {}
    layer3 = report.get("layer_3_stability") or report.get("stability_layer") or {}

    l1_metrics = layer1.get("metrics") if isinstance(layer1, dict) else {}
    if not isinstance(l1_metrics, dict):
        l1_metrics = {}

    l2_mode = layer2.get("mode_sensitivity") if isinstance(layer2, dict) else {}
    if not isinstance(l2_mode, dict):
        l2_mode = {}

    l3_wf = layer3.get("walk_forward") if isinstance(layer3, dict) else {}
    if not isinstance(l3_wf, dict):
        l3_wf = {}

    return (
        "<section class='three-layer'>"
        "<div class='three-head'>"
        "<div class='three-title'>三层评估快照</div>"
        f"<div class='three-meta'>评估区间：{start_txt} ~ {end_txt}</div>"
        f"<div class='overall {_status_class(overall_status)}'>总体状态：{_status_label(overall_status)}</div>"
        "</div>"
        "<div class='three-grid'>"
        "<article class='three-card'>"
        f"<h3>Layer 1 回测质量 <span class='tag {_status_class(layer1.get('status', 'unknown'))}'>{_status_label(layer1.get('status', 'unknown'))}</span></h3>"
        "<p class='mini'>核心 KPI（HFQ 回测口径）</p>"
        "<ul>"
        f"<li>年化收益：{_fmt_pct(l1_metrics.get('annual_return_pct'))}</li>"
        f"<li>最大回撤：{_fmt_pct(l1_metrics.get('max_drawdown_pct'))}</li>"
        f"<li>夏普：{_fmt_num(l1_metrics.get('sharpe'))}</li>"
        f"<li>胜率：{_fmt_pct(l1_metrics.get('win_rate_pct'))}</li>"
        "</ul>"
        "</article>"
        "<article class='three-card'>"
        f"<h3>Layer 2 执行可实现性 <span class='tag {_status_class(layer2.get('status', 'unknown'))}'>{_status_label(layer2.get('status', 'unknown'))}</span></h3>"
        "<p class='mini'>涨跌停/整手/冲击成本/next_open vs close</p>"
        "<ul>"
        f"<li>阻塞日占比：{_fmt_ratio_pct(layer2.get('blocked_day_ratio'))}</li>"
        f"<li>部分成交日占比：{_fmt_ratio_pct(layer2.get('partial_day_ratio'))}</li>"
        f"<li>额外执行成本：{_fmt_num(layer2.get('avg_extra_exec_cost_bps'), 3)} bps/日</li>"
        f"<li>模式敏感度(年化差绝对值)：{_fmt_pct(l2_mode.get('abs_annual_gap_pctpt'))}</li>"
        "</ul>"
        "</article>"
        "<article class='three-card'>"
        f"<h3>Layer 3 稳定性 <span class='tag {_status_class(layer3.get('status', 'unknown'))}'>{_status_label(layer3.get('status', 'unknown'))}</span></h3>"
        "<p class='mini'>月度分布 + Walk-forward</p>"
        "<ul>"
        f"<li>样本月数：{int(_safe_float(layer3.get('month_count'), 0))}</li>"
        f"<li>正收益月份占比：{_fmt_ratio_pct(layer3.get('positive_month_ratio'))}</li>"
        f"<li>月收益均值/波动：{_fmt_pct(layer3.get('mean_monthly_return_pct'))} / {_fmt_pct(layer3.get('std_monthly_return_pct'))}</li>"
        f"<li>WF 全KPI通过率：{_fmt_ratio_pct(l3_wf.get('kpi_all_ok_ratio'))}</li>"
        "</ul>"
        "</article>"
        "</div>"
        "</section>"
    )


def build_html(
    df: pd.DataFrame,
    start: str | None,
    end: str | None,
    three_layer: dict[str, Any] | None = None,
) -> str:
    summary = _build_summary(df)
    panels = _build_symbol_panels(df)
    three_layer_section = _build_three_layer_section(three_layer)

    if not df.empty:
        min_dt = _fmt_date(df["entry_date"].min())
        max_dt = _fmt_date(df["exit_date"].max())
    else:
        min_dt = "N/A"
        max_dt = "N/A"

    start_txt = start or min_dt
    end_txt = end or max_dt

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>交易日志看板</title>
  <style>
    :root {{
      --bg: #f3f6fb;
      --card: #ffffff;
      --ink: #122338;
      --sub: #5d6e86;
      --line: #d8e1ef;
      --accent: #0f6fff;
      --pos: #0f8a4b;
      --neg: #c4372f;
      --flat: #6a768a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: radial-gradient(circle at top right, #e8f0ff, var(--bg) 55%);
      color: var(--ink);
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 20px; }}
    .head {{
      background: linear-gradient(135deg, #0f6fff, #1c86ff);
      color: #fff;
      border-radius: 14px;
      padding: 18px 20px;
      box-shadow: 0 10px 24px rgba(25, 80, 180, 0.18);
      margin-bottom: 16px;
    }}
    .head h1 {{ margin: 0 0 6px; font-size: 24px; }}
    .meta {{ font-size: 14px; opacity: 0.95; line-height: 1.6; }}
    .note {{
      margin-top: 10px;
      background: rgba(255, 255, 255, 0.16);
      border: 1px solid rgba(255, 255, 255, 0.25);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 13px;
      line-height: 1.7;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
      margin: 14px 0 18px;
    }}
    .stat-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }}
    .three-layer {{
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 14px;
    }}
    .three-head {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .three-title {{ font-size: 16px; font-weight: 700; }}
    .three-meta {{ color: var(--sub); font-size: 13px; }}
    .overall {{
      margin-left: auto;
      font-size: 12px;
      font-weight: 700;
      padding: 4px 8px;
      border-radius: 999px;
    }}
    .three-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
    }}
    .three-card {{
      border: 1px solid #e7eef9;
      border-radius: 10px;
      padding: 10px;
      background: #fbfdff;
    }}
    .three-card h3 {{
      margin: 0 0 6px;
      font-size: 14px;
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .mini {{ margin: 0 0 6px; color: var(--sub); font-size: 12px; }}
    .three-card ul {{ margin: 0; padding-left: 18px; }}
    .three-card li {{ margin: 4px 0; font-size: 13px; }}
    .tag {{
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 999px;
      border: 1px solid transparent;
    }}
    .st-ok {{ color: #0f8a4b; background: #ebf9f1; border-color: #c6ecd8; }}
    .st-warn {{ color: #a56a00; background: #fff5e6; border-color: #ffe0ad; }}
    .st-risk {{ color: #b53730; background: #fff0ef; border-color: #ffc9c5; }}
    .st-unknown {{ color: #51637b; background: #eef3fb; border-color: #d4dfef; }}
    .k {{ font-size: 13px; color: var(--sub); margin-bottom: 4px; }}
    .v {{ font-size: 22px; font-weight: 700; }}
    .toolbar {{
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    .toolbar input {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      min-width: 220px;
      background: #fff;
    }}
    .symbol-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      margin-bottom: 10px;
      overflow: hidden;
    }}
    .symbol-card summary {{
      list-style: none;
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      padding: 12px 14px;
      background: #f8fbff;
      border-bottom: 1px solid var(--line);
    }}
    .symbol-card summary::-webkit-details-marker {{ display: none; }}
    .sym {{ font-weight: 700; font-size: 16px; }}
    .stat {{ color: var(--sub); font-size: 13px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 1320px; }}
    th, td {{
      border-bottom: 1px solid #edf2fa;
      padding: 8px 10px;
      font-size: 13px;
      text-align: left;
      white-space: nowrap;
      vertical-align: top;
    }}
    th {{ background: #fbfdff; color: #31445f; position: sticky; top: 0; z-index: 1; }}
    .pnl-pos {{ color: var(--pos); font-weight: 700; }}
    .pnl-neg {{ color: var(--neg); font-weight: 700; }}
    .pnl-flat {{ color: var(--flat); font-weight: 700; }}
    .empty {{
      background: #fff;
      border: 1px dashed var(--line);
      border-radius: 12px;
      padding: 20px;
      color: var(--sub);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <h1>交易日志看板（按股票分组）</h1>
      <div class="meta">区间：{start_txt} ~ {end_txt}（日频回测）</div>
      <div class="note">
        口径说明：<br/>
        1) 买入价/卖出价为回测字段 entry_price/exit_price，单位是“元/股”。<br/>
        2) 当前项目价格口径为 HFQ（后复权），不是实时行情裸价。<br/>
        3) 新增“参考价(不复权近似)”列，使用同日 amount/volume 估算，并做100股单位自适应换算，仅用于对照展示。<br/>
        4) 本页面不包含“买了多少股/成交金额”，仓位(%)表示组合权重。
      </div>
    </div>

    <section class="stats">
      <div class="stat-card"><div class="k">交易笔数</div><div class="v">{summary['n_trades']}</div></div>
      <div class="stat-card"><div class="k">涉及股票</div><div class="v">{summary['n_symbols']}</div></div>
      <div class="stat-card"><div class="k">胜率</div><div class="v">{summary['win_rate']:.1f}%</div></div>
      <div class="stat-card"><div class="k">平均单笔</div><div class="v">{summary['avg_pnl']:+.2f}%</div></div>
      <div class="stat-card"><div class="k">累计收益(简单相加)</div><div class="v">{summary['sum_pnl']:+.2f}%</div></div>
    </section>

    {three_layer_section}

    <div class="toolbar">
      <input id="q" placeholder="筛选股票代码，例如 600111 或 002028" />
    </div>

    <div id="panels">
      {panels}
    </div>
  </div>

  <script>
    const q = document.getElementById("q");
    q.addEventListener("input", () => {{
      const kw = q.value.trim().toUpperCase();
      document.querySelectorAll(".symbol-card").forEach((card) => {{
        const sym = card.querySelector(".sym").innerText.toUpperCase();
        card.style.display = (!kw || sym.includes(kw)) ? "" : "none";
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate grouped trade journal dashboard (HTML).")
    ap.add_argument("--base-dir", default=".", help="project root")
    ap.add_argument("--start-date", default=None, help="filter start date, e.g. 2026-01-01")
    ap.add_argument("--end-date", default=None, help="filter end date, e.g. 2026-02-10")
    ap.add_argument(
        "--trades-file",
        default="data/backtests/backtest_strategy_v3_trades.csv",
        help="trades csv path (relative to base-dir)",
    )
    ap.add_argument(
        "--three-layer-file",
        default="data/backtests/backtest_strategy_v3_three_layer_report.json",
        help="three-layer report json path (relative to base-dir)",
    )
    ap.add_argument("--output", default="out/trade_journal_dashboard.html", help="output html path")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    trades_path = base_dir / args.trades_file
    three_layer_path = base_dir / args.three_layer_file
    output_path = base_dir / args.output

    df = _load_backtest_trades(trades_path)
    df = _filter_date_range(df, args.start_date, args.end_date)
    df = _attach_reference_prices(df, base_dir=base_dir)
    three_layer = _load_three_layer_report(three_layer_path)
    html = build_html(df, args.start_date, args.end_date, three_layer=three_layer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    if args.start_date or args.end_date:
        s = args.start_date or "min"
        e = args.end_date or "max"
        dated = output_path.with_name(f"{output_path.stem}_{s}_to_{e}{output_path.suffix}")
        dated.write_text(html, encoding="utf-8")
        print(f"saved: {output_path}")
        print(f"saved: {dated}")
    else:
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
