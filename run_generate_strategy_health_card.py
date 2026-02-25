from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _infer_as_of(base_dir: Path, arg_as_of: str) -> str:
    if str(arg_as_of).strip():
        return str(arg_as_of).strip()
    rank_path = base_dir / "data" / "signals" / "latest_daily_rank.csv"
    if rank_path.exists():
        try:
            df = pd.read_csv(rank_path, nrows=1)
            if not df.empty and "date" in df.columns:
                d = str(df["date"].iloc[0]).strip()
                if d:
                    return d
        except Exception:
            pass
    return date.today().isoformat()


def _load_signal_snapshot(base_dir: Path) -> dict[str, Any]:
    p = base_dir / "data" / "signals" / "latest_daily_rank.csv"
    if not p.exists():
        return {
            "total": 0,
            "invest_more": 0,
            "withdraw": 0,
            "reduce": 0,
            "least": 0,
            "avg_score_invest": 0.0,
            "avg_strength_invest": 0.0,
            "market_regime": "N/A",
        }
    try:
        df = pd.read_csv(p)
    except Exception:
        return {
            "total": 0,
            "invest_more": 0,
            "withdraw": 0,
            "reduce": 0,
            "least": 0,
            "avg_score_invest": 0.0,
            "avg_strength_invest": 0.0,
            "market_regime": "N/A",
        }

    if df.empty:
        return {
            "total": 0,
            "invest_more": 0,
            "withdraw": 0,
            "reduce": 0,
            "least": 0,
            "avg_score_invest": 0.0,
            "avg_strength_invest": 0.0,
            "market_regime": "N/A",
        }

    invest = df[df.get("action", "").astype(str) == "INVEST_MORE"] if "action" in df.columns else pd.DataFrame()
    return {
        "total": int(len(df)),
        "invest_more": int((df.get("action", pd.Series([], dtype=object)) == "INVEST_MORE").sum()) if "action" in df.columns else 0,
        "withdraw": int((df.get("action", pd.Series([], dtype=object)) == "WITHDRAW").sum()) if "action" in df.columns else 0,
        "reduce": int((df.get("action", pd.Series([], dtype=object)) == "REDUCE").sum()) if "action" in df.columns else 0,
        "least": int((df.get("action", pd.Series([], dtype=object)) == "LEAST").sum()) if "action" in df.columns else 0,
        "avg_score_invest": float(pd.to_numeric(invest.get("score", pd.Series([], dtype=float)), errors="coerce").mean()) if not invest.empty else 0.0,
        "avg_strength_invest": float(pd.to_numeric(invest.get("_signal_strength", pd.Series([], dtype=float)), errors="coerce").mean()) if not invest.empty else 0.0,
        "market_regime": str(df["market_regime"].iloc[0]) if "market_regime" in df.columns else "N/A",
    }


def _load_paper_snapshot(base_dir: Path) -> dict[str, Any]:
    state = _load_json(base_dir / "data" / "paper" / "state.json")
    if not state:
        return {
            "enabled": False,
            "last_signal_date": None,
            "positions": 0,
            "pending_orders": 0,
            "cash": 0.0,
            "initial_cash": 0.0,
            "cash_ratio": 0.0,
        }

    positions = state.get("positions", {}) or {}
    pos_count = sum(1 for p in positions.values() if _safe_float((p or {}).get("shares", 0.0), 0.0) > 0)
    cash = _safe_float(state.get("cash", 0.0), 0.0)
    initial_cash = _safe_float(state.get("initial_cash", 0.0), 0.0)
    cash_ratio = cash / initial_cash if initial_cash > 0 else 0.0
    return {
        "enabled": True,
        "last_signal_date": state.get("last_signal_date"),
        "positions": int(pos_count),
        "pending_orders": int(len(state.get("pending_orders", []) or [])),
        "cash": float(cash),
        "initial_cash": float(initial_cash),
        "cash_ratio": float(cash_ratio),
    }


def _derive_overall_status(card: dict[str, Any]) -> str:
    gate = card.get("gate", {})
    perf = card.get("performance", {})
    signal = card.get("signal", {})
    paper = card.get("paper", {})

    trade_action = str(gate.get("trade_action", "hold")).lower()
    quality_status = str(gate.get("quality_status", "unknown")).lower()
    three_layer_status = str(perf.get("three_layer_status", "unknown")).lower()

    if trade_action == "stop" or quality_status == "fail":
        return "critical"
    if trade_action == "reduce":
        return "warning"
    if three_layer_status in {"warning", "risk"}:
        return "warning"
    if _safe_int(signal.get("invest_more", 0), 0) == 0:
        return "warning"
    if _safe_int(paper.get("pending_orders", 0), 0) > 0:
        return "watch"
    return "healthy"


def _build_recommendations(card: dict[str, Any]) -> list[str]:
    rec: list[str] = []
    gate = card.get("gate", {})
    perf = card.get("performance", {})
    signal = card.get("signal", {})
    paper = card.get("paper", {})

    action = str(gate.get("trade_action", "hold")).lower()
    cap = gate.get("trade_position_cap")
    if action == "stop":
        rec.append("交易闸门为 stop：建议暂停新增仓位，仅做复盘与回归检查。")
    elif action == "reduce":
        cap_text = f"{_safe_float(cap, 0.0):.2f}" if cap is not None else "N/A"
        rec.append(f"交易闸门为 reduce：建议按仓位上限 {cap_text} 执行降仓。")
    else:
        rec.append("交易闸门为 normal：可按策略正常执行，但继续观察失效监控窗口。")

    blocked_ratio = _safe_float(perf.get("blocked_day_ratio", 0.0), 0.0)
    if blocked_ratio >= 0.40:
        rec.append("执行层阻塞天占比较高：建议降低换手并复核参与率/滑点参数。")

    if _safe_int(signal.get("invest_more", 0), 0) == 0:
        rec.append("当前无 INVEST_MORE 标的：建议以防守为主，等待下一轮信号。")

    if _safe_int(paper.get("pending_orders", 0), 0) > 0:
        rec.append("Paper 存在未完成挂单：建议优先检查流动性与价格限制影响。")

    return rec


def _build_recovery_conditions(summary: dict[str, Any]) -> list[str]:
    cond: list[str] = []
    quality_gate = summary.get("quality_gate", {}) if isinstance(summary, dict) else {}
    checks = quality_gate.get("checks", []) if isinstance(quality_gate, dict) else []
    failed_checks = [c for c in checks if isinstance(c, dict) and not bool(c.get("passed", False))]

    if failed_checks:
        for c in failed_checks[:5]:
            cond.append(f"{c.get('name', 'unknown')} 需满足规则：{c.get('rule', '')}")
    else:
        monitor = (summary.get("failure_monitor", {}) if isinstance(summary, dict) else {}) or {}
        short_win = ((monitor.get("windows", {}) or {}).get("short", {}) if isinstance(monitor, dict) else {}) or {}
        long_win = ((monitor.get("windows", {}) or {}).get("long", {}) if isinstance(monitor, dict) else {}) or {}
        if short_win:
            cond.append(
                f"短窗口恢复：Sharpe 回到基线附近（当前 { _safe_float(short_win.get('recent_sharpe'), 0.0):.2f}）。"
            )
        if long_win:
            cond.append(
                f"长窗口恢复：年化回报改善（当前 { _safe_float(long_win.get('recent_annual_return_pct'), 0.0):.2f}%）。"
            )
    if not cond:
        cond.append("暂无硬性失败项，维持现有风控并持续跟踪。")
    return cond


def _render_md(card: dict[str, Any]) -> str:
    as_of = card.get("as_of", "N/A")
    status = card.get("overall_status", "unknown")
    gate = card.get("gate", {})
    perf = card.get("performance", {})
    signal = card.get("signal", {})
    paper = card.get("paper", {})

    rec_lines = "\n".join([f"- {x}" for x in card.get("recommendations", [])]) or "- 无"
    recov_lines = "\n".join([f"- {x}" for x in card.get("recovery_conditions", [])]) or "- 无"

    return f"""# 策略健康诊断卡 - {as_of}

- 生成时间：{card.get("generated_at", "N/A")}
- 总体状态：**{status}**

## 1. 闸门状态
- quality_gate：`{gate.get("quality_status", "unknown")}`
- trade_gate：`{gate.get("trade_status", "unknown")}` / action=`{gate.get("trade_action", "hold")}`
- 建议仓位上限：`{_safe_float(gate.get("trade_position_cap", 0.0), 0.0):.2f}`
- failure_monitor：`{gate.get("failure_monitor_status", "unknown")}` / action=`{gate.get("failure_monitor_action", "hold")}`

## 2. 回测与执行
- 年化：`{_safe_float(perf.get("annual_return_pct", 0.0), 0.0):+.2f}%`
- 最大回撤：`{_safe_float(perf.get("max_drawdown_pct", 0.0), 0.0):.2f}%`
- Sharpe：`{_safe_float(perf.get("sharpe", 0.0), 0.0):.2f}`
- 三层评估状态：`{perf.get("three_layer_status", "unknown")}`
- 执行阻塞天占比：`{_safe_float(perf.get("blocked_day_ratio", 0.0), 0.0):.2%}`
- 执行部分成交天占比：`{_safe_float(perf.get("partial_day_ratio", 0.0), 0.0):.2%}`

## 3. 信号与持仓
- 市场状态：`{signal.get("market_regime", "N/A")}`
- 信号总数：`{_safe_int(signal.get("total", 0), 0)}`
- INVEST_MORE：`{_safe_int(signal.get("invest_more", 0), 0)}`
- WITHDRAW：`{_safe_int(signal.get("withdraw", 0), 0)}`
- REDUCE：`{_safe_int(signal.get("reduce", 0), 0)}`
- 平均入选分数：`{_safe_float(signal.get("avg_score_invest", 0.0), 0.0):.3f}`
- 平均入选强度：`{_safe_float(signal.get("avg_strength_invest", 0.0), 0.0):.3f}`

## 4. Paper 运行状态
- 已启用：`{paper.get("enabled", False)}`
- 持仓数：`{_safe_int(paper.get("positions", 0), 0)}`
- 未完成挂单：`{_safe_int(paper.get("pending_orders", 0), 0)}`
- 最新信号日期：`{paper.get("last_signal_date", "N/A")}`
- 现金占比：`{_safe_float(paper.get("cash_ratio", 0.0), 0.0):.2%}`

## 5. 建议动作
{rec_lines}

## 6. 恢复条件
{recov_lines}
"""


def build_health_card(base_dir: Path, as_of: str) -> dict[str, Any]:
    strategy_summary = _load_json(base_dir / "data" / "backtests" / "strategy_process_summary.json")
    bt_stats = _load_json(base_dir / "data" / "backtests" / "backtest_strategy_v3_stats.json")
    signal = _load_signal_snapshot(base_dir)
    paper = _load_paper_snapshot(base_dir)

    quality_gate = strategy_summary.get("quality_gate", {}) if isinstance(strategy_summary, dict) else {}
    trade_gate = strategy_summary.get("trade_gate", {}) if isinstance(strategy_summary, dict) else {}
    failure_monitor = strategy_summary.get("failure_monitor", {}) if isinstance(strategy_summary, dict) else {}

    three_layer = bt_stats.get("three_layer_evaluation", {}) if isinstance(bt_stats, dict) else {}
    layer2 = three_layer.get("layer_2_execution", {}) if isinstance(three_layer, dict) else {}

    card: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": as_of,
        "gate": {
            "quality_status": str(quality_gate.get("status", "unknown")),
            "trade_status": str(trade_gate.get("status", "unknown")),
            "trade_action": str(trade_gate.get("action", "hold")),
            "trade_position_cap": _safe_float(trade_gate.get("position_cap", 0.0), 0.0),
            "failure_monitor_status": str(failure_monitor.get("status", "unknown")),
            "failure_monitor_action": str(failure_monitor.get("action", "hold")),
            "failure_monitor_cap": _safe_float(failure_monitor.get("suggested_position_cap", 0.0), 0.0),
        },
        "performance": {
            "annual_return_pct": _safe_float(bt_stats.get("annual_return_pct", 0.0), 0.0),
            "max_drawdown_pct": _safe_float(bt_stats.get("max_drawdown_pct", 0.0), 0.0),
            "sharpe": _safe_float(bt_stats.get("sharpe", 0.0), 0.0),
            "three_layer_status": str(three_layer.get("overall_status", "unknown")),
            "blocked_day_ratio": _safe_float(layer2.get("blocked_day_ratio", 0.0), 0.0),
            "partial_day_ratio": _safe_float(layer2.get("partial_day_ratio", 0.0), 0.0),
            "avg_total_cost_bps": _safe_float(layer2.get("avg_total_cost_bps", 0.0), 0.0),
        },
        "signal": signal,
        "paper": paper,
    }
    card["overall_status"] = _derive_overall_status(card)
    card["recommendations"] = _build_recommendations(card)
    card["recovery_conditions"] = _build_recovery_conditions(strategy_summary)
    return card


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate strategy health card from latest pipeline artifacts.")
    ap.add_argument("--base-dir", default=".", help="project root")
    ap.add_argument("--as-of", default="", help="as-of date (YYYY-MM-DD), default infer from latest signal")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    as_of = _infer_as_of(base_dir, args.as_of)
    card = build_health_card(base_dir, as_of=as_of)

    bt_dir = base_dir / "data" / "backtests"
    report_dir = base_dir / "data" / "reports"
    bt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = bt_dir / "strategy_health_card.json"
    md_path = report_dir / f"strategy_health_card_{as_of}.md"
    latest_md_path = report_dir / "strategy_health_card_latest.md"

    json_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    md_text = _render_md(card)
    md_path.write_text(md_text, encoding="utf-8")
    latest_md_path.write_text(md_text, encoding="utf-8")

    print("Strategy health card generated")
    print(f"json: {json_path}")
    print(f"md  : {md_path}")
    print(f"latest: {latest_md_path}")
    print(
        f"status={card.get('overall_status', 'unknown')} | "
        f"gate={card.get('gate', {}).get('trade_action', 'hold')} | "
        f"invest_more={card.get('signal', {}).get('invest_more', 0)}"
    )


if __name__ == "__main__":
    main()
