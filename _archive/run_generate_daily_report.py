from __future__ import annotations

"""
日度 Markdown 报告生成器（data/reports）。

你在“当天开盘前 / 周末 / 假期”运行时：
- 行情（日线）只能拿到“上一交易日收盘”的数据，所以 as_of 可能停在上一交易日；
- 新闻是实时抓取的，所以日期可能是今天。

这会导致你看到 reports 里都是 2026-01-30，以为 2026-02-02 没生成。

本脚本做两件事：
1) 仍然以信号日期生成：daily_report_<signal_as_of>.md
2) 如果发现最新新闻日期 news_as_of != signal_as_of，则额外复制一份“别名文件”：
   daily_report_<news_as_of>.md，并在标题下插入日期说明。

可选：
- 设置环境变量 REPORT_DATE=YYYY-MM-DD 强制别名文件的日期命名。
"""

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from quant.reporting import generate_daily_report_md

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _latest_dated_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"{root} not found")
    dirs = [p for p in root.iterdir() if p.is_dir() and DATE_RE.match(p.name)]
    if not dirs:
        raise FileNotFoundError(f"No YYYY-MM-DD folders in {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def _latest_news_date(base_dir: Path) -> str | None:
    news_root = base_dir / "data" / "news"
    if not news_root.exists():
        return None
    dirs = [p for p in news_root.iterdir() if p.is_dir() and DATE_RE.match(p.name)]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.name)[-1].name


def _inject_note(md_path: Path, signal_as_of: str, news_as_of: str | None, alias_date: str) -> None:
    """在 Markdown 主标题后插入一段说明（不破坏原内容结构）。"""
    txt = md_path.read_text(encoding="utf-8")
    lines = txt.splitlines()

    note_lines = [f"> **信号日期**：{signal_as_of}（基于上一交易日收盘数据）"]
    if news_as_of:
        note_lines.append(f"> **新闻抓取截至**：{news_as_of}")
    if alias_date != signal_as_of:
        note_lines.append(f"> **文件命名日期**：{alias_date}（仅用于命名，不代表收盘数据日期）")
    note_lines.append(f"> **生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    note = "\n".join(note_lines)

    if lines and lines[0].startswith("#"):
        new_txt = "\n".join([lines[0], "", note, ""] + lines[1:]) + "\n"
    else:
        new_txt = note + "\n\n" + txt

    md_path.write_text(new_txt, encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    briefs_root = base_dir / "data" / "briefs"
    if not briefs_root.exists():
        raise FileNotFoundError("data/briefs not found. Run run_build_ai_briefs.py first.")

    latest_dir = _latest_dated_dir(briefs_root)

    bundle_path = latest_dir / "ALL.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"{bundle_path} not found. Run run_build_ai_briefs.py first.")

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    signal_as_of = str(bundle.get("as_of") or latest_dir.name)
    news_as_of = _latest_news_date(base_dir)

    forced = os.getenv("REPORT_DATE", "").strip()
    alias_date = forced if (forced and DATE_RE.match(forced)) else (news_as_of or signal_as_of)

    out_dir = base_dir / "data" / "reports"
    report_path = Path(generate_daily_report_md(bundle_path, out_dir))

    print("\nReport generated:")
    print(report_path)
    print(f"Signal as_of: {signal_as_of}")
    if news_as_of:
        print(f"News   as_of: {news_as_of}")

    if alias_date and alias_date != signal_as_of:
        alias_path = report_path.with_name(f"daily_report_{alias_date}.md")
        shutil.copy2(report_path, alias_path)
        _inject_note(alias_path, signal_as_of=signal_as_of, news_as_of=news_as_of, alias_date=alias_date)
        print(f"Alias generated: {alias_path}")


if __name__ == "__main__":
    main()
