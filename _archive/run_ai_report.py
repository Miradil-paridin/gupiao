"""
AI 研究报告生成器 (多模型版)

支持模型：
- DeepSeek (deepseek-reasoner, deepseek-chat)
- 小米 MiMo (mimo-v2-flash)

通过 LLM_PROVIDER 环境变量切换

调教方向：
- 保守但不错失机会
- 目标：月收益 2-4%（适合小资金稳健增长）
- 强调风险控制和仓位管理
- 避免追高，偏好回调买入
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# 关键改动：使用 llm_multi 替代 llm_deepseek
from llm_multi import chat_complete, get_provider


# ----------------------------
# 配置：根据模型读取不同的环境变量
# ----------------------------
def get_token_config() -> dict:
    """
    根据当前 LLM 提供商获取 token 配置
    """
    provider = get_provider()

    if provider == "mimo":
        # MiMo 的 token 配置
        return {
            "stock_max_tokens_1": int(os.getenv("MIMO_STOCK_MAX_TOKENS", "4000")),
            "stock_max_tokens_2": int(os.getenv("MIMO_STOCK_MAX_TOKENS_RETRY", "8000")),
            "summary_max_tokens_1": int(os.getenv("MIMO_SUMMARY_MAX_TOKENS", "4000")),
            "summary_max_tokens_2": int(os.getenv("MIMO_SUMMARY_MAX_TOKENS_RETRY", "8000")),
        }
    else:
        # DeepSeek 的 token 配置（默认）
        return {
            "stock_max_tokens_1": int(os.getenv("DEEPSEEK_STOCK_MAX_TOKENS", "4000")),
            "stock_max_tokens_2": int(os.getenv("DEEPSEEK_STOCK_MAX_TOKENS_RETRY", "8000")),
            "summary_max_tokens_1": int(os.getenv("DEEPSEEK_SUMMARY_MAX_TOKENS", "4000")),
            "summary_max_tokens_2": int(os.getenv("DEEPSEEK_SUMMARY_MAX_TOKENS_RETRY", "8000")),
        }


# ----------------------------
# Utils
# ----------------------------
def _safe_json_load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sanitize_json_text(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text or "")


def _try_parse_json(text: str) -> dict | None:
    cleaned = _sanitize_json_text(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start: end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def _extract_stock_items(bundle: dict) -> list[dict]:
    for key in ["items", "briefs", "stocks", "universe"]:
        v = bundle.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    for key in ["data", "payload", "bundle"]:
        v = bundle.get(key)
        if isinstance(v, dict):
            for k2 in ["items", "briefs", "stocks", "universe"]:
                vv = v.get(k2)
                if isinstance(vv, list) and vv and isinstance(vv[0], dict):
                    return vv
    return []


def _get_symbol(x: dict) -> str:
    for k in ["symbol", "ticker", "secid", "code", "code6"]:
        v = x.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in ["code", "code6"]:
        v = x.get(k)
        if isinstance(v, int):
            return f"{v:06d}"
    return ""


def _render_decision_md(report: dict, provider: str) -> str:
    as_of = report.get("as_of", "")
    title = report.get("title", f"AI Quant Report — {as_of}")

    def block_list(items):
        if not items:
            return "_None_\n"
        out = []
        for x in items:
            sym = x.get("symbol", "")
            headline = x.get("headline", "")
            bullets = x.get("bullets", []) or []
            risks = x.get("risks", []) or []
            news_view = x.get("news_view", "")
            sizing = x.get("position_sizing", "")
            entry_condition = x.get("entry_condition", "")

            out.append(f"**{sym}** — {headline}".strip())
            if bullets:
                for b in bullets:
                    out.append(f"- {b}")
            if entry_condition:
                out.append(f"- 📍 入场条件: {entry_condition}")
            if risks:
                out.append(f"- ⚠️ 风险: " + "；".join(risks))
            if news_view:
                out.append(f"- 📰 新闻观点: {news_view}")
            if sizing:
                out.append(f"- 💰 仓位建议: {sizing}")
            out.append("")
        return "\n".join(out) + "\n"

    md = []
    md.append(f"# {title}\n")
    md.append(f"> 🤖 Powered by **{provider.upper()}**\n")
    md.append(report.get("disclaimer", "本报告为研究自动化输出，不构成投资建议。请结合自身风险承受能力决策。") + "\n")

    # 投资目标提醒
    md.append("> 💡 **投资目标**: 月收益 2-4%，稳健为主，宁可错过不可做错\n")

    md.append("## 📊 概要\n")
    md.append(report.get("summary", "") + "\n")

    market_outlook = report.get("market_outlook", "")
    if market_outlook:
        md.append("## 🌐 市场展望\n")
        md.append(market_outlook + "\n")

    # 仓位建议总览
    total_position = report.get("total_position_suggestion", "")
    if total_position:
        md.append("## 💼 总仓位建议\n")
        md.append(total_position + "\n")

    md.append("## ✅ 建议加仓（高确定性）\n")
    md.append(block_list(report.get("invest_more", [])))

    md.append("## ⏳ 观望等待（等更好价格）\n")
    md.append(block_list(report.get("wait_better_price", [])))

    md.append("## ⚠️ 建议减仓（风险上升）\n")
    md.append(block_list(report.get("reduce", [])))

    md.append("## 🚫 回避（不建议操作）\n")
    md.append(block_list(report.get("withdraw", [])))

    md.append("## 📋 下个交易日关注\n")
    wl = report.get("watchlist", [])
    if wl:
        for w in wl:
            md.append(f"- {w}")
        md.append("")
    else:
        md.append("_None_\n")

    # 风险提醒
    md.append("## ⚠️ 风险提醒\n")
    risk_warnings = report.get("risk_warnings", [])
    if risk_warnings:
        for w in risk_warnings:
            md.append(f"- {w}")
    else:
        md.append("- 控制单只股票仓位不超过总资金的 20%")
        md.append("- 设置止损位，亏损超过 5% 考虑减仓")
    md.append("")

    return "\n".join(md)


# ----------------------------
# LLM Calls (使用 llm_multi)
# ----------------------------
def _call_with_length_retry(
        messages: list[dict],
        response_format: dict | None,
        max_tokens_first: int,
        max_tokens_second: int,
        retries: int = 3,
        backoff_sec: float = 3.0,
):
    """带长度截断重试的 LLM 调用"""
    resp = chat_complete(
        messages=messages,
        response_format=response_format,
        max_tokens=max_tokens_first,
        retries=retries,
        backoff_sec=backoff_sec,
    )

    if getattr(resp, "finish_reason", None) == "length":
        resp2 = chat_complete(
            messages=messages,
            response_format=response_format,
            max_tokens=max_tokens_second,
            retries=retries,
            backoff_sec=backoff_sec,
        )
        return resp2

    return resp


def main() -> None:
    load_dotenv()

    # 获取当前使用的模型提供商
    provider = get_provider()
    print(f"\n🤖 Using LLM Provider: {provider.upper()}")

    # 获取 token 配置
    token_cfg = get_token_config()
    print(f"   Stock tokens: {token_cfg['stock_max_tokens_1']} / {token_cfg['stock_max_tokens_2']}")
    print(f"   Summary tokens: {token_cfg['summary_max_tokens_1']} / {token_cfg['summary_max_tokens_2']}")

    base_dir = Path(__file__).resolve().parent

    briefs_root = base_dir / "data" / "briefs"
    if not briefs_root.exists():
        raise FileNotFoundError("data/briefs not found. Run run_build_ai_briefs.py first.")

    date_dirs = [p for p in briefs_root.iterdir() if p.is_dir()]
    if not date_dirs:
        raise FileNotFoundError("No date folders in data/briefs.")

    latest_dir = sorted(date_dirs, key=lambda p: p.name)[-1]
    bundle_path = latest_dir / "ALL.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"{bundle_path} not found.")

    bundle = _safe_json_load(bundle_path)
    as_of = bundle.get("as_of", latest_dir.name)
    has_news = bool(bundle.get("has_news_data", False))

    items = _extract_stock_items(bundle)

    # Output folders - 文件名包含提供商标识
    reports_dir = base_dir / "data" / "reports"
    _ensure_dir(reports_dir)

    out_json = reports_dir / f"{provider}_report_{as_of}.json"
    out_md = reports_dir / f"{provider}_report_{as_of}.md"
    detail_root = reports_dir / f"{provider}_details_{as_of}"
    detail_stocks_dir = detail_root / "stocks"
    debug_dir = detail_root / "_debug"

    _ensure_dir(detail_stocks_dir)
    _ensure_dir(debug_dir)

    # ----------------------------
    # System Prompt - 单只股票分析
    # ----------------------------
    system_stock = {
        "role": "system",
        "content": """你是一位经验丰富的A股买方研究员，服务于一位小资金（5-10万RMB）的个人投资者。

## 投资者画像
- 目标：月收益 2-4%（约 1000-4000 RMB）
- 风格：稳健保守，但不想错过明显的机会
- 原则：宁可少赚，不可大亏

## 你的分析原则
1. **安全边际优先**：只有在估值合理或低估时才建议买入
2. **趋势确认**：不抄底、不猜顶，等趋势明朗再行动
3. **仓位控制**：单只股票建议仓位通常 5-15%，极度看好也不超过 20%
4. **止损纪律**：每只股票必须给出明确的止损位（通常 -5% 到 -8%）

## 量化信号解读标准
- ma_dist_20 > 5%：短期超买，不宜追高
- ma_dist_20 < -5%：可能超卖，关注反弹机会
- vol_20d > 40%：波动过大，降低仓位
- trend_up = False + mom_bad = True：趋势走坏，回避
- risk_high = True：风险较高，仓位减半

## 输出格式（必须严格遵守）
ACTION: <INVEST_MORE|WAIT_BETTER_PRICE|REDUCE|WITHDRAW>
CONFIDENCE: <LOW|MED|HIGH>
POSITION: <建议仓位，如 5-10%>
STOP_LOSS: <止损位，如 -5% 或具体价格>
ENTRY_CONDITION: <入场条件，如"回调至MA20附近">

## 结论（一句话总结）

## 量化信号（引用具体数值）

## 技术面分析（趋势、支撑、压力）

## 新闻与事件（如有新闻，分析影响；无则写"无重大新闻"）

## 风险清单（至少列出 2 个风险）

## 操作建议（具体、可执行的建议）
""",
    }

    per_stock_meta = []

    for idx, x in enumerate(items):
        symbol = _get_symbol(x) or f"STOCK_{idx + 1}"
        hint_action = x.get("action") or x.get("recommended_action") or ""

        user_stock = {
            "role": "user",
            "content": (
                f"请分析这只股票并给出操作建议。\n"
                f"日期: {as_of}\n"
                f"数据包提供的action提示: {hint_action}\n\n"
                f"重要提醒：\n"
                f"- 投资者是小资金散户，风险承受能力有限\n"
                f"- 如果不确定，宁可建议 WAIT_BETTER_PRICE\n"
                f"- 必须给出具体的止损位和入场条件\n\n"
                f"STOCK_JSON:\n"
                f"{json.dumps(x, ensure_ascii=False)}"
            ),
        }

        print(f"\n📈 Analyzing {symbol} ({idx + 1}/{len(items)})...")

        resp_stock = _call_with_length_retry(
            messages=[system_stock, user_stock],
            response_format=None,
            max_tokens_first=token_cfg["stock_max_tokens_1"],
            max_tokens_second=token_cfg["stock_max_tokens_2"],
            retries=4,
            backoff_sec=3.0,
        )

        stock_md = resp_stock.content or ""
        stock_path = detail_stocks_dir / f"{symbol}.md"
        with open(stock_path, "w", encoding="utf-8") as f:
            f.write(stock_md)

        if resp_stock.finish_reason == "length":
            (debug_dir / f"{symbol}_truncated.txt").write_text(
                f"FINISH_REASON=length\nCONTENT_LEN={len(stock_md)}\nUSAGE={resp_stock.usage}\n",
                encoding="utf-8",
            )

        head = "\n".join(stock_md.splitlines()[:12])
        per_stock_meta.append({
            "symbol": symbol,
            "action_hint": hint_action,
            "head": head,
            "file": str(stock_path),
        })

    # ----------------------------
    # System Prompt - 汇总决策
    # ----------------------------
    system_sum = {
        "role": "system",
        "content": """你是一位谨慎的投资顾问，为小资金个人投资者提供每日操作建议。

## 核心原则
1. **保守为主**：不确定就不推荐，宁可错过不可做错
2. **分散风险**：同时持有不超过 3-5 只股票
3. **总仓位控制**：
   - 市场情绪好：总仓位可到 60-80%
   - 市场情绪一般：总仓位 40-60%
   - 市场情绪差：总仓位 20-40%，甚至空仓观望
4. **机会识别**：当出现明显低估+趋势向上的机会，要敢于建仓

## 分类标准
- **INVEST_MORE**（建议加仓）：趋势向上 + 回调到支撑 + 风险可控，确定性高
- **WAIT_BETTER_PRICE**（等待更好价格）：看好但当前价格偏高，等回调
- **REDUCE**（建议减仓）：趋势走弱或风险上升
- **WITHDRAW**（回避）：趋势明确向下或风险过高

你只输出 JSON，不要有其他内容。
""",
    }

    news_instruction = ""
    if has_news:
        news_instruction = (
            "- 新闻分析：在 news_view 中说明新闻对股价的可能影响（利好/利空/中性）\n"
            "- 若新闻标题为无意义内容，标注'新闻质量较低'\n"
        )

    user_sum = {
        "role": "user",
        "content": (
            f"请基于以下信息输出最终投资决策 JSON（日期 {as_of}）。\n\n"
            "输出结构（严格遵守）：\n"
            "{\n"
            '  "as_of": "YYYY-MM-DD",\n'
            '  "title": "每日投资建议 - YYYY-MM-DD",\n'
            '  "summary": "今日市场概述和操作建议（100字内）",\n'
            '  "total_position_suggestion": "建议总仓位：XX%，原因：...",\n'
            '  "invest_more": [{"symbol":"","headline":"一句话理由","bullets":["要点1","要点2"],"risks":["风险1"],"entry_condition":"入场条件","news_view":"","position_sizing":"5-10%"}],\n'
            '  "wait_better_price": [{"symbol":"","headline":"","bullets":[],"risks":[],"entry_condition":"等待条件","news_view":"","position_sizing":""}],\n'
            '  "reduce": [{"symbol":"","headline":"","bullets":[],"risks":[],"news_view":"","position_sizing":""}],\n'
            '  "withdraw": [{"symbol":"","headline":"","bullets":[],"risks":[],"news_view":"","position_sizing":""}],\n'
            '  "market_outlook": "对A股短期走势的判断（50字内）",\n'
            '  "watchlist": ["明日关注点1","明日关注点2"],\n'
            '  "risk_warnings": ["风险提醒1","风险提醒2"],\n'
            '  "disclaimer": "本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。"\n'
            "}\n\n"
            "规则：\n"
            "- invest_more 最多推荐 2 只，必须是高确定性机会\n"
            "- 如果没有好机会，invest_more 可以为空数组\n"
            "- 每只股票必须有 entry_condition（入场条件）和 position_sizing（仓位建议）\n"
            "- position_sizing 要保守，单只不超过 15%\n"
            f"{news_instruction}"
            "\n"
            "BUNDLE_META:\n"
            f"{json.dumps({'as_of': as_of, 'has_news_data': has_news, 'universe_size': bundle.get('universe_size', None)}, ensure_ascii=False)}\n\n"
            "PER_STOCK_NOTE_HEADS:\n"
            f"{json.dumps(per_stock_meta, ensure_ascii=False)}\n\n"
            "DATA_BUNDLE_FULL:\n"
            f"{json.dumps(bundle, ensure_ascii=False)}"
        ),
    }

    print(f"\n📊 Generating summary report...")

    resp_sum = _call_with_length_retry(
        messages=[system_sum, user_sum],
        response_format={"type": "json_object"},
        max_tokens_first=token_cfg["summary_max_tokens_1"],
        max_tokens_second=token_cfg["summary_max_tokens_2"],
        retries=5,
        backoff_sec=3.0,
    )

    report_obj = _try_parse_json(resp_sum.content)

    if report_obj is None:
        (debug_dir / f"summary_raw_{as_of}.txt").write_text(resp_sum.content or "", encoding="utf-8")
        report_obj = {
            "as_of": as_of,
            "title": f"每日投资建议 — {as_of}",
            "summary": "模型输出解析失败，请查看 _debug 文件夹。",
            "total_position_suggestion": "",
            "invest_more": [],
            "wait_better_price": [],
            "reduce": [],
            "withdraw": [],
            "market_outlook": "",
            "watchlist": [],
            "risk_warnings": ["请检查原始输出"],
            "disclaimer": "Not financial advice.",
        }

    report_obj["_meta"] = {
        "provider": provider,
        "detail_folder": str(detail_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary_finish_reason": resp_sum.finish_reason,
        "summary_usage": resp_sum.usage,
    }

    # Write JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)

    # Write Markdown
    md_parts = []
    md_parts.append(_render_decision_md(report_obj, provider))
    md_parts.append("\n---\n")
    md_parts.append("# 📑 个股详细分析\n")

    if items:
        for x in items:
            symbol = _get_symbol(x)
            if not symbol:
                continue
            p = detail_stocks_dir / f"{symbol}.md"
            if p.exists():
                md_parts.append(f"## {symbol}\n")
                md_parts.append(p.read_text(encoding="utf-8"))
                md_parts.append("\n")

    md = "\n".join(md_parts)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    # Optional: save CoT (Chain of Thought)
    save_cot = os.getenv("DEEPSEEK_SAVE_COT", "0").strip() == "1"
    if save_cot and resp_sum.reasoning_content:
        with open(detail_root / f"summary_cot_{as_of}.txt", "w", encoding="utf-8") as f:
            f.write(resp_sum.reasoning_content)

    print("\n" + "=" * 50)
    print(f"🤖 {provider.upper()} 报告生成完成")
    print("=" * 50)
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")
    print(f"详情: {detail_root}")
    print(f"\nFINISH_REASON: {resp_sum.finish_reason}")
    print(f"CONTENT_LEN: {len(resp_sum.content or '')}")


if __name__ == "__main__":
    main()