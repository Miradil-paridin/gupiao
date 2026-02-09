from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def _fmt_pct(x: float) -> str:
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "n/a"


def _action_badge(action: str) -> str:
    # Markdown-friendly labels
    if action == "INVEST_MORE":
        return "✅ INVEST_MORE"
    if action == "REDUCE":
        return "🟡 REDUCE"
    if action == "WITHDRAW":
        return "🛑 WITHDRAW"
    if action == "LEAST":
        return "⚠️ LEAST"
    return "⏸ HOLD"


def generate_daily_report_md(bundle_path: Path, out_dir: Path) -> Path:
    """
    Generate a Markdown report from ALL.json.
    """
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    as_of = bundle["as_of"]
    items = bundle["universe"]

    # Build table data
    rows = []
    for b in items:
        snap = b["snapshot"]
        flags = b["flags"]
        rows.append({
            "rank": b["rank"],
            "symbol": b["symbol"],
            "action": b["action"],
            "score": b["score"],
            "close": snap["close"],
            "ma_dist_20": snap["ma_dist_20"],
            "ret_20d": snap["ret_20d"],
            "ret_60d": snap["ret_60d"],
            "vol_20d": snap["vol_20d"],
            "atr_pct": snap["atr_pct"],
            "vol_ratio_20": snap["vol_ratio_20"],
            "trend_up": flags["trend_up"],
            "mom_bad": flags["mom_bad"],
            "risk_high": flags["risk_high"],
        })

    df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)

    # Buckets
    invest_more = df[df["action"] == "INVEST_MORE"]
    withdraw = df[df["action"] == "WITHDRAW"]
    least = df[df["action"] == "LEAST"]
    hold = df[df["action"] == "HOLD"]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"daily_report_{as_of}.md"

    def md_table(d: pd.DataFrame) -> str:
        if d.empty:
            return "_None_\n"
        lines = []
        lines.append("| Rank | Symbol | Action | Score | Close | Trend(MA20) | 20D | 60D | Vol20 | ATR% | VolRatio | Flags |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, r in d.iterrows():
            flags = []
            if int(r["trend_up"]) == 1:
                flags.append("trend_up")
            if int(r["mom_bad"]) == 1:
                flags.append("mom_bad")
            if int(r["risk_high"]) == 1:
                flags.append("risk_high")
            flag_txt = ",".join(flags) if flags else "-"
            lines.append(
                f'| {int(r["rank"])} | {r["symbol"]} | {_action_badge(r["action"])} | {r["score"]:.3f} | {r["close"]:.2f} | {_fmt_pct(r["ma_dist_20"])} | '
                f'{_fmt_pct(r["ret_20d"])} | {_fmt_pct(r["ret_60d"])} | {r["vol_20d"]:.3f} | {_fmt_pct(r["atr_pct"])} | {r["vol_ratio_20"]:.2f} | {flag_txt} |'
            )
        return "\n".join(lines) + "\n"

    # Write report
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Daily Quant Memo — {as_of}\n\n")
        f.write("This memo is **quant-only** (price/volume features). Fundamentals/news are not included yet.\n\n")

        f.write("## Summary Actions\n\n")
        f.write(f"- INVEST_MORE: {', '.join(invest_more['symbol'].tolist()) if not invest_more.empty else 'None'}\n")
        f.write(f"- WITHDRAW: {', '.join(withdraw['symbol'].tolist()) if not withdraw.empty else 'None'}\n")
        f.write(f"- LEAST: {', '.join(least['symbol'].tolist()) if not least.empty else 'None'}\n")
        f.write(f"- HOLD: {', '.join(hold['symbol'].tolist()) if not hold.empty else 'None'}\n\n")

        f.write("## Ranking Table\n\n")
        f.write(md_table(df))
        f.write("\n")

        f.write("## Notes on Top & Bottom Names\n\n")
        top_n = df.head(3)
        bot_n = df.tail(3)

        f.write("### Top 3 (by score)\n\n")
        for _, r in top_n.iterrows():
            brief = next(b for b in items if b["symbol"] == r["symbol"])
            f.write(f"**{r['symbol']}** — {_action_badge(r['action'])}\n\n")
            for line in brief["reasons"]:
                f.write(f"- {line}\n")
            f.write("\n")

        f.write("### Bottom 3 (by score)\n\n")
        for _, r in bot_n.iterrows():
            brief = next(b for b in items if b["symbol"] == r["symbol"])
            f.write(f"**{r['symbol']}** — {_action_badge(r['action'])}\n\n")
            for line in brief["reasons"]:
                f.write(f"- {line}\n")
            f.write("\n")

        f.write("## Next Improvements\n\n")
        f.write("- Add fundamentals lane (PE/PB/ROE, earnings, cash flow).\n")
        f.write("- Add event/news lane (CNINFO + major finance headlines) to explain *why* signals changed.\n")
        f.write("- Replace hard `risk_high => WITHDRAW` with a softer rule (e.g., REDUCE), and add position sizing.\n")

    return out_path
