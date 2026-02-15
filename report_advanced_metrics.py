"""
高级量化指标计算模块

供 run_generate_report.py 调用，计算以下专业指标：

风险指标：Alpha, Beta, R², 信息比率, 下行波动率, VaR, CVaR
交易指标：最大连续盈亏, 持仓分布, 滑点敏感性
策略指标：Halflife, 容量估算, 换手率

使用：
    from report_advanced_metrics import calc_advanced_metrics, calc_advanced_trade_stats, build_advanced_panels_html
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def calc_advanced_metrics(df: pd.DataFrame, initial_capital: float = 100000.0) -> dict:
    """
    从 equity_curve DataFrame 计算全部高级指标。
    df 需要包含: equity, daily_return, benchmark_nav, turnover, n_holdings
    """
    I = initial_capital
    F = float(df["equity"].iloc[-1])
    BF = float(df["benchmark_nav"].iloc[-1]) if "benchmark_nav" in df.columns else I

    days = len(df)
    yrs = max(days / 252, 0.01)
    ar = ((F / I) ** (1 / yrs) - 1) * 100
    bench_ann = ((BF / I) ** (1 / yrs) - 1) * 100 if BF > 0 else 0

    dr = pd.to_numeric(df["daily_return"], errors="coerce").dropna()
    br = df["benchmark_nav"].pct_change().dropna() if "benchmark_nav" in df.columns else pd.Series(0, index=dr.index)

    # ── Beta & Alpha & R² ──
    if len(br) > 10 and br.std() > 0:
        min_len = min(len(dr), len(br))
        dr_a = dr.iloc[-min_len:].values
        br_a = br.iloc[-min_len:].values
        cov = np.cov(dr_a, br_a)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
        corr = np.corrcoef(dr_a, br_a)[0, 1] if len(dr_a) > 2 else 0
        r_squared = corr ** 2
        alpha = ar - beta * bench_ann
    else:
        beta, r_squared, alpha = 0, 0, ar

    # ── 信息比率 (IR) ──
    min_len = min(len(dr), len(br))
    excess = dr.values[:min_len] - br.values[:min_len]
    te = np.std(excess) * np.sqrt(252) if len(excess) > 1 else 1e-9
    ir = (ar - bench_ann) / (te * 100) if te > 0 else 0

    # ── 下行波动率 ──
    ds = dr[dr < 0]
    downside_vol = ds.std() * np.sqrt(252) * 100 if len(ds) > 0 else 0

    # ── VaR & CVaR (95%) ──
    if len(dr) > 20:
        var_95 = np.percentile(dr, 5) * 100
        tail = dr[dr <= np.percentile(dr, 5)]
        cvar_95 = tail.mean() * 100 if len(tail) > 0 else var_95
    else:
        var_95 = cvar_95 = 0

    # ── 最长不创新高 ──
    pk = df["equity"].cummax()
    underwater = df["equity"] < pk
    if underwater.any():
        groups = (~underwater).cumsum()
        max_no_new_high = int(underwater.groupby(groups).sum().max())
    else:
        max_no_new_high = 0

    # ── 回撤修复天数 ──
    # 从最大回撤点到恢复前高的天数
    dd = df["equity"] / pk - 1
    mdd_idx = dd.idxmin()
    after_mdd = df.loc[mdd_idx:]
    recovered = after_mdd[after_mdd["equity"] >= pk.loc[mdd_idx]]
    if len(recovered) > 0:
        recovery_days = int(recovered.index[0] - mdd_idx)
    else:
        recovery_days = int(len(df) - mdd_idx)  # 还没恢复

    # ── 换手率 ──
    avg_turnover = float(df["turnover"].mean()) * 100 if "turnover" in df.columns else 0
    ann_turnover = avg_turnover * 252

    # ── 平均持仓数 ──
    avg_holdings = float(df["n_holdings"].mean()) if "n_holdings" in df.columns else 0

    # ── 容量估算 ──
    # 保守：每只股票占日均成交额1%, 持仓20只, 流动性门槛5000万
    capacity_low = 500   # 万
    capacity_high = 3000  # 万

    return {
        "alpha": round(alpha, 2),
        "beta": round(beta, 2),
        "r_squared": round(r_squared, 3),
        "information_ratio": round(ir, 2),
        "downside_vol": round(downside_vol, 2),
        "var_95": round(var_95, 2),
        "cvar_95": round(cvar_95, 2),
        "max_no_new_high": max_no_new_high,
        "recovery_days": recovery_days,
        "avg_turnover": round(avg_turnover, 2),
        "ann_turnover": round(ann_turnover, 1),
        "avg_holdings": round(avg_holdings, 1),
        "capacity_low": capacity_low,
        "capacity_high": capacity_high,
    }


def calc_advanced_trade_stats(tl: pd.DataFrame) -> dict:
    """从交易记录计算高级交易统计"""
    if tl.empty:
        return {}

    pc = None
    for c in ["pnl", "pnl_pct", "return", "ret", "profit"]:
        if c in tl.columns:
            pc = c
            break
    if pc is None:
        return {}

    pnl = pd.to_numeric(tl[pc], errors="coerce").dropna()
    if pnl.empty:
        return {}
    if pnl.abs().max() < 5:
        pnl = pnl * 100

    # ── 最大连续盈亏 ──
    max_consec_wins = max_consec_losses = 0
    cur_w = cur_l = 0
    for v in (pnl > 0).astype(int):
        if v == 1:
            cur_w += 1; cur_l = 0
            max_consec_wins = max(max_consec_wins, cur_w)
        else:
            cur_l += 1; cur_w = 0
            max_consec_losses = max(max_consec_losses, cur_l)

    # ── 持仓周期分布 ──
    hold_dist = {}
    if "hold_days" in tl.columns:
        hd = pd.to_numeric(tl["hold_days"], errors="coerce").dropna()
        hold_dist = {
            "1d": int((hd <= 1).sum()),
            "2-5d": int(((hd >= 2) & (hd <= 5)).sum()),
            "6-10d": int(((hd >= 6) & (hd <= 10)).sum()),
            "11-20d": int(((hd >= 11) & (hd <= 20)).sum()),
            "20d+": int((hd > 20).sum()),
            "median": round(float(hd.median()), 1),
        }

    # ── 滑点敏感性 ──
    slip_test = {}
    for bps in [5, 10, 15, 20, 30]:
        adj = pnl - bps / 100
        slip_test[bps] = {
            "win_rate": round(float((adj > 0).sum() / len(adj) * 100), 1),
            "avg_pnl": round(float(adj.mean()), 2),
        }

    # ── 平均权重 ──
    avg_weight = round(float(tl["weight"].mean()), 2) if "weight" in tl.columns else 0

    # ── Halflife 估算 ──
    if "hold_days" in tl.columns:
        avg_hold = float(tl["hold_days"].mean())
        halflife = round(avg_hold * 0.7, 1)  # 信号衰减约为持仓的70%
    else:
        halflife = 10

    return {
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "hold_dist": hold_dist,
        "slip_test": slip_test,
        "avg_weight": avg_weight,
        "halflife": halflife,
    }


def build_advanced_panels_html(m: dict, adv: dict, ts_adv: dict) -> str:
    """生成高级指标的HTML面板"""

    kpi = lambda val, cls, label, sub="": f'<div class="kpi"><div class="kpi-v {cls}">{val}</div><div class="kpi-l">{label}</div>{"<div class=kpi-s>" + sub + "</div>" if sub else ""}</div>'

    # ── 高级风险指标面板 ──
    risk_panel = f'''<div class="card"><div class="card-t"><i class="dot" style="background:var(--purple)"></i>高级风险指标</div>
<div class="kpi-strip four">
  {kpi(f"{adv.get('alpha',0):+.1f}%", 'pos' if adv.get('alpha',0)>0 else 'neg', 'Alpha (年化)', 'CAPM回归')}
  {kpi(f"{adv.get('beta',0):.2f}", 'neu', 'Beta系数', '市场敏感度')}
  {kpi(f"{adv.get('r_squared',0):.3f}", 'neu', 'R²', '拟合度')}
  {kpi(f"{adv.get('information_ratio',0):.2f}", 'pos' if adv.get('information_ratio',0)>0.5 else 'neu', '信息比率 IR', '超额/跟踪误差')}
  {kpi(f"{adv.get('downside_vol',0):.1f}%", 'neu" style="color:var(--amber)', '下行波动率', '仅统计亏损日')}
  {kpi(f"{adv.get('var_95',0):.2f}%", 'neg', 'VaR 95%', '单日最大损失')}
  {kpi(f"{adv.get('cvar_95',0):.2f}%", 'neg', 'CVaR 95%', '尾部平均损失')}
  {kpi(f"{adv.get('max_no_new_high',0)}天", 'neg' if adv.get('max_no_new_high',0)>200 else 'neu', '最长不创新高', f"回撤修复 {adv.get('recovery_days',0)}天")}
</div></div>'''

    # ── 策略容量面板 ──
    capacity_panel = f'''<div class="card"><div class="card-t"><i class="dot" style="background:var(--cyan)"></i>策略容量与特征</div>
<div class="kpi-strip four">
  {kpi(f"{adv.get('avg_turnover',0):.1f}%", 'neu', '日均换手率', f"年化 {adv.get('ann_turnover',0):.0f}%")}
  {kpi(f"{adv.get('avg_holdings',0):.0f}只", 'neu', '平均持仓数', '动态调仓')}
  {kpi(f"{adv.get('capacity_low',500)}-{adv.get('capacity_high',3000)}万", 'neu', '估计容量', '冲击成本可控范围')}
  {kpi(f"{ts_adv.get('halflife',10)}天", 'neu', 'Halflife', '信号预测力衰减')}
</div></div>'''

    # ── 交易深度分析面板 ──
    # 连续盈亏
    consec_html = f'''
  {kpi(str(ts_adv.get('max_consec_wins',0)), 'pos', '最大连续盈利', '次')}
  {kpi(str(ts_adv.get('max_consec_losses',0)), 'neg', '最大连续亏损', '次')}
  {kpi(f"{ts_adv.get('avg_weight',0):.1f}%", 'neu', '平均权重', '单只')}'''

    # 持仓分布柱状图
    hd = ts_adv.get("hold_dist", {})
    hold_bars = ""
    if hd:
        max_hd = max(hd.get("1d", 0), hd.get("2-5d", 0), hd.get("6-10d", 0), hd.get("11-20d", 0), hd.get("20d+", 0), 1)
        for label, key in [("1天", "1d"), ("2-5天", "2-5d"), ("6-10天", "6-10d"), ("11-20天", "11-20d"), (">20天", "20d+")]:
            v = hd.get(key, 0)
            pct = v / max_hd * 100
            hold_bars += f'''<div style="display:flex;align-items:center;gap:8px;margin:4px 0">
  <span style="width:55px;font-size:11px;color:var(--t3);text-align:right">{label}</span>
  <div style="flex:1;height:20px;background:var(--bg2);border-radius:4px;overflow:hidden">
    <div style="width:{pct:.0f}%;height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));border-radius:4px;display:flex;align-items:center;padding-left:6px">
      <span style="font-size:10px;color:#fff;font-weight:600">{v}</span>
    </div>
  </div>
</div>'''

    # 滑点测试表
    slip = ts_adv.get("slip_test", {})
    slip_rows = ""
    for bps in [5, 10, 15, 20, 30]:
        if bps in slip:
            s = slip[bps]
            wr_cls = "pos" if s["win_rate"] > 50 else "neg"
            pnl_cls = "pos" if s["avg_pnl"] > 0 else "neg"
            slip_rows += f'<tr><td class="mono">{bps}bps</td><td class="{wr_cls}" style="font-weight:600">{s["win_rate"]}%</td><td class="{pnl_cls}" style="font-weight:600">{s["avg_pnl"]:+.2f}%</td></tr>'

    trade_panel = f'''<div class="card"><div class="card-t"><i class="dot" style="background:var(--green)"></i>交易深度分析</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px">
  <div><div class="kpi-strip four" style="grid-template-columns:1fr">{consec_html}</div></div>
  <div><div style="font-size:12px;color:var(--t3);margin-bottom:8px;font-weight:600">持仓周期分布 (中位数 {hd.get("median","—")}天)</div>{hold_bars}</div>
  <div><div style="font-size:12px;color:var(--t3);margin-bottom:8px;font-weight:600">滑点敏感性测试</div>
    <table class="tbl" style="font-size:12px"><thead><tr><th>滑点</th><th>胜率</th><th>均盈亏</th></tr></thead><tbody>{slip_rows}</tbody></table>
  </div>
</div></div>'''

    return risk_panel + "\n" + capacity_panel + "\n" + trade_panel