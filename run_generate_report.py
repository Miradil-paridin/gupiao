"""
📊 一站式回测报告生成器

自动检测配置、回测数据、信号数据，生成高品质交互式 HTML 报告。

使用:  python run_generate_report.py
输出:  out/backtest_report.html
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import json, datetime, sys

# ── 配置 ──────────────────────────────────────────────────

def load_project_config(base_dir: Path) -> dict:
    for name in ["config_v31.yaml", "config_optimized.yaml", "config.yaml"]:
        p = base_dir / name
        if p.exists():
            try:
                import yaml
                with open(p, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                print(f"  📂 配置: {name}")
                return cfg
            except Exception as e:
                print(f"  ⚠ 读取 {name} 失败: {e}")
    print("  ⚠ 未找到配置文件，使用默认值")
    return {}

def get_start_date(config: dict) -> str:
    return str(
        config.get("backtest", {}).get("start_date")
        or config.get("market_data", {}).get("start_date")
        or "2020-01-01"
    )

# ── 数据加载 ──────────────────────────────────────────────

def load_backtest_equity(base_dir: Path):
    candidates = [
        ("backtest_strategy_v3_equity.csv", "V3 涨停回调+多重风控"),
        ("backtest_strategy_v2_equity.csv", "V2 仓位控制"),
        ("backtest_strategy_equity.csv",    "V1 基础因子"),
    ]
    bt_dir = base_dir / "data" / "backtests"
    for fn, label in candidates:
        p = bt_dir / fn
        if p.exists():
            try:
                df = pd.read_csv(p)
                if "date" in df.columns and "equity" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    print(f"  ✓ 回测: {fn} ({label})")
                    print(f"    {df['date'].min().date()} → {df['date'].max().date()}, {len(df)} 行")
                    return df, label
            except Exception as e:
                print(f"  ⚠ {fn}: {e}")
    return None, ""

def load_trade_log(base_dir: Path) -> pd.DataFrame:
    for fn in ["backtest_strategy_v3_trades.csv","backtest_strategy_v2_trades.csv","backtest_strategy_trades.csv"]:
        p = base_dir / "data" / "backtests" / fn
        if p.exists():
            try:
                df = pd.read_csv(p); print(f"  ✓ 交易记录: {fn} ({len(df)}条)"); return df
            except: pass
    print("  ○ 无交易记录"); return pd.DataFrame()

def load_latest_signal(base_dir: Path) -> pd.DataFrame:
    p = base_dir / "data" / "signals" / "latest_daily_rank.csv"
    if p.exists():
        try:
            df = pd.read_csv(p); print(f"  ✓ 最新信号: {len(df)}只"); return df
        except: pass
    print("  ○ 无最新信号"); return pd.DataFrame()

def fetch_benchmark(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0": return pd.DataFrame()
        try:
            rs = bs.query_history_k_data_plus("sh.000300","date,close",start_date,end_date,"d","3")
            rows = []
            while rs.next(): rows.append(rs.get_row_data())
            if not rows: return pd.DataFrame()
            df = pd.DataFrame(rows, columns=rs.fields)
            df["date"] = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            print(f"  ✓ 沪深300: {len(df)}个交易日")
            return df
        finally: bs.logout()
    except Exception as e:
        print(f"  ⚠ 获取沪深300失败: {e}"); return pd.DataFrame()

# ── 数据处理 ──────────────────────────────────────────────

def filter_and_normalize(df, start_date, initial=100000):
    df = df.copy()
    before = len(df)
    df = df[df["date"] >= pd.to_datetime(start_date)].reset_index(drop=True)
    if len(df) == 0: raise ValueError(f"过滤后无数据! start={start_date}")
    if len(df) < before: print(f"  ✓ 过滤: 剔除 {before-len(df)} 行 ({start_date} 之前)")
    first = df["equity"].iloc[0]
    if abs(first - initial) > 1:
        df["equity"] = df["equity"] / first * initial
        print(f"  ✓ 权益归一化到 ¥{initial:,.0f}")
    return df

def merge_benchmark(df, benchmark, initial=100000):
    df = df.copy()
    if not benchmark.empty:
        bm = benchmark.sort_values("date").copy()
        bm["benchmark_nav"] = bm["close"] / bm["close"].iloc[0] * initial
        df = df.merge(bm[["date","benchmark_nav"]], on="date", how="left")
        df["benchmark_nav"] = df["benchmark_nav"].ffill().bfill()
        if df["benchmark_nav"].isna().all(): df["benchmark_nav"] = initial
    else:
        df["benchmark_nav"] = initial
    return df

# ── 指标 ──────────────────────────────────────────────────

def _sf(v, d=0.0):
    try: r=float(v); return r if np.isfinite(r) else d
    except: return d

def calc_metrics(df):
    I=100000.0; F=_sf(df["equity"].iloc[-1],I); BF=_sf(df["benchmark_nav"].iloc[-1],I)
    tr=(F/I-1)*100; br=(BF/I-1)*100
    days=len(df); yrs=max(days/252,.01)
    ar=((F/I)**(1/yrs)-1)*100
    pk=df["equity"].cummax(); dd=(df["equity"]/pk-1)*100; mdd=dd.min()
    bpk=df["benchmark_nav"].cummax(); bdd=(df["benchmark_nav"]/bpk-1)*100; bmdd=bdd.min()
    uw=dd< -0.01
    if uw.any():
        g=(~uw).cumsum(); mdd_d=int(uw.groupby(g).sum().max())
    else: mdd_d=0
    dr=pd.to_numeric(df["daily_return"],errors="coerce").dropna() if "daily_return" in df.columns else df["equity"].pct_change().dropna()
    sh=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    ds=dr[dr<0]; so=dr.mean()/ds.std()*np.sqrt(252) if len(ds)>0 and ds.std()>0 else 0
    ca=ar/abs(mdd) if mdd!=0 else 0
    wr=(dr>0).sum()/len(dr)*100 if len(dr)>0 else 0
    aw=dr[dr>0].mean() if (dr>0).any() else 0
    al=abs(dr[dr<0].mean()) if (dr<0).any() else 1e-9
    return dict(total_return=round(tr,2),ann_return=round(ar,2),bench_return=round(br,2),
        excess_return=round(tr-br,2),max_dd=round(mdd,2),bench_max_dd=round(bmdd,2),
        max_dd_days=mdd_d,sharpe=round(sh,2),sortino=round(so,2),calmar=round(ca,2),
        win_rate=round(wr,1),pl_ratio=round(aw/al,2),
        max_gain=round(dr.max()*100,2) if len(dr)>0 else 0,
        max_loss=round(dr.min()*100,2) if len(dr)>0 else 0,
        ann_vol=round(dr.std()*np.sqrt(252)*100,2),
        trading_days=days,years=round(yrs,1),final_equity=round(F,0))

def calc_monthly(df):
    t=df.copy(); t["year"]=t["date"].dt.year; t["month"]=t["date"].dt.month
    m=t.groupby(["year","month"]).agg(s=("equity","first"),e=("equity","last"))
    m["ret"]=((m["e"]/m["s"])-1)*100
    return m.reset_index()[["year","month","ret"]].round(2).to_dict("records")

def calc_yearly(df, col="equity"):
    t=df.copy(); t["year"]=t["date"].dt.year
    y=t.groupby("year").agg(s=(col,"first"),e=(col,"last")); y["ret"]=((y["e"]/y["s"])-1)*100
    return [{"year":int(i),"ret":round(r["ret"],2)} for i,r in y.iterrows()]

def calc_trade_stats(tl):
    if tl.empty: return None
    n=len(tl); pc=None
    for c in ["pnl","pnl_pct","return","ret","profit"]:
        if c in tl.columns: pc=c; break
    if pc is None: return {"total":n}
    pnl=pd.to_numeric(tl[pc],errors="coerce").dropna()
    if pnl.empty: return {"total":n}
    if pnl.abs().max()<5: pnl=pnl*100
    w=(pnl>0).sum()
    return dict(total=n,wins=int(w),win_rate=round(w/n*100,1),
        avg_pnl=round(pnl.mean(),2),best=round(pnl.max(),2),worst=round(pnl.min(),2))

# ── HTML 模板 ─────────────────────────────────────────────

CSS = """
:root{
  --bg:#070b14;--bg2:#0c1121;--card:#111827;--card2:#16203a;
  --bdr:#1c2744;--bdr2:#263356;
  --t1:#e5e9f0;--t2:#8694b2;--t3:#4a577a;
  --blue:#3b82f6;--green:#22c55e;--red:#ef4444;--amber:#f59e0b;--purple:#a855f7;--cyan:#06b6d4;
  --green-g:rgba(34,197,94,.12);--red-g:rgba(239,68,68,.12);--amber-g:rgba(245,158,11,.12);
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Noto Sans SC',system-ui,sans-serif;background:var(--bg);color:var(--t1);min-height:100vh;line-height:1.6;-webkit-font-smoothing:antialiased}
.mono{font-family:'JetBrains Mono',monospace}
.hdr{padding:44px 0 36px;background:linear-gradient(180deg,#0e1529,var(--bg));border-bottom:1px solid var(--bdr);position:relative;overflow:hidden}
.hdr::after{content:'';position:absolute;top:-80px;right:5%;width:360px;height:360px;background:radial-gradient(circle,rgba(59,130,246,.12),transparent 70%);pointer-events:none}
.wrap{max-width:1400px;margin:0 auto;padding:0 36px;position:relative;z-index:1}
.hdr h1{font-size:30px;font-weight:900;letter-spacing:-.5px}
.hdr h1 em{font-style:normal;color:var(--blue)}
.hdr-row{display:flex;align-items:center;gap:14px;margin-top:10px;flex-wrap:wrap}
.pill{display:inline-flex;align-items:center;gap:5px;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600}
.pill-v{background:linear-gradient(135deg,var(--blue),var(--purple));color:#fff}
.pill-d{background:var(--card);border:1px solid var(--bdr);color:var(--t2)}
.pill-cfg{background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.2);color:var(--green);font-size:11px}
.kpi-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:14px;margin:28px 0}
.kpi-strip.four{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.kpi{background:var(--card);border:1px solid var(--bdr);border-radius:14px;padding:20px;transition:border-color .2s,transform .15s}
.kpi:hover{border-color:var(--bdr2);transform:translateY(-2px)}
.kpi-v{font-size:28px;font-weight:700;font-family:'JetBrains Mono',monospace;line-height:1.15}
.kpi-l{font-size:11px;color:var(--t3);text-transform:uppercase;letter-spacing:.5px;margin-top:4px}
.kpi-s{font-size:11px;color:var(--t3);margin-top:2px}
.pos{color:var(--green)}.neg{color:var(--red)}.neu{color:var(--blue)}
.card{background:var(--card);border:1px solid var(--bdr);border-radius:16px;padding:26px;margin-bottom:22px}
.card-t{font-size:15px;font-weight:600;margin-bottom:18px;display:flex;align-items:center;gap:9px}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:22px}
@media(max-width:900px){.row2{grid-template-columns:1fr}}
.lg{display:flex;gap:18px;font-size:11px;color:var(--t2);margin-bottom:10px}
.lg b{display:inline-block;width:16px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
.hm{display:grid;grid-template-columns:56px repeat(12,1fr);gap:3px;font-size:11px;font-family:'JetBrains Mono',monospace}
.hm-h{text-align:center;color:var(--t3);padding:5px 0;font-weight:500}
.hm-y{display:flex;align-items:center;justify-content:center;color:var(--t2);font-weight:600}
.hm-c{text-align:center;padding:7px 2px;border-radius:6px;font-weight:500;transition:transform .12s}
.hm-c:hover{transform:scale(1.15);z-index:2}
.tbl{width:100%;border-collapse:separate;border-spacing:0;font-size:12px}
.tbl th{text-align:left;padding:9px 12px;background:var(--bg2);color:var(--t3);font-weight:500;font-size:10px;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid var(--bdr)}
.tbl th:first-child{border-radius:8px 0 0 0}.tbl th:last-child{border-radius:0 8px 0 0}
.tbl td{padding:9px 12px;border-bottom:1px solid var(--bdr)}
.tbl tr:hover td{background:var(--card2)}
.act{display:inline-block;padding:2px 10px;border-radius:10px;font-size:10px;font-weight:600}
.tag-buy{background:var(--green-g);color:var(--green)}
.tag-sell{background:var(--red-g);color:var(--red)}
.tag-hold{background:rgba(255,255,255,.05);color:var(--t3)}
.disc{margin-top:28px;padding:14px 18px;background:var(--red-g);border:1px solid rgba(239,68,68,.18);border-radius:10px;font-size:11px;color:#e87171;line-height:1.7}
.foot{text-align:center;padding:28px;color:var(--t3);font-size:11px;border-top:1px solid var(--bdr)}
"""

CHART_JS = """
const D=window._D,E=window._E,B=window._B,DD=window._DD,BDD=window._BDD,EX=window._EX,
      YL=window._YL,YS=window._YS,YB=window._YB,MO=window._MO;
Chart.defaults.color='#4a577a';Chart.defaults.borderColor='rgba(255,255,255,.03)';
Chart.defaults.font.family="'Noto Sans SC',sans-serif";
const tA={type:'time',time:{unit:'year',displayFormats:{year:'yyyy'}},ticks:{maxRotation:0}};
const tip={backgroundColor:'#111827',borderColor:'#1c2744',borderWidth:1,titleColor:'#8694b2',bodyColor:'#e5e9f0',padding:12};
function mk(id,type,ds,yFmt){
  new Chart(document.getElementById(id),{type,data:{labels:D,datasets:ds},
    options:{responsive:true,interaction:{intersect:false,mode:'index'},
      plugins:{legend:{display:false},tooltip:{...tip,callbacks:{label:yFmt?c=>c.dataset.label+': '+yFmt(c.parsed.y):undefined}}},
      scales:{x:tA,y:{ticks:{callback:yFmt||undefined}}}}});
}
const ln=(c,b,f,d)=>({label:c,data:b,borderColor:f,backgroundColor:d||'transparent',fill:!!d,tension:.15,pointRadius:0,borderWidth:2.2});

mk('cEquity','line',[ln('策略',E,'#3b82f6','rgba(59,130,246,.07)'),{...ln('沪深300',B,'#f59e0b'),borderWidth:1.6,borderDash:[6,4]}],v=>'¥'+(v/1000).toFixed(0)+'k');
mk('cExcess','line',[{...ln('超额',EX,'#22c55e'),backgroundColor:ctx=>{const g=ctx.chart.ctx.createLinearGradient(0,0,0,ctx.chart.height);g.addColorStop(0,'rgba(34,197,94,.22)');g.addColorStop(1,'rgba(34,197,94,0)');return g},fill:true}],v=>v.toFixed(1)+'%');
mk('cDD','line',[ln('策略',DD,'#ef4444','rgba(239,68,68,.12)'),{...ln('基准',BDD,'#f59e0b'),borderWidth:1.2,borderDash:[5,4]}],v=>v.toFixed(1)+'%');

new Chart(document.getElementById('cYearly'),{type:'bar',
  data:{labels:YL,datasets:[
    {label:'策略',data:YS,backgroundColor:YS.map(v=>v>=0?'#3b82f6':'rgba(239,68,68,.65)'),borderRadius:5,barPercentage:.45,categoryPercentage:.7},
    {label:'沪深300',data:YB,backgroundColor:YB.map(v=>v>=0?'rgba(245,158,11,.6)':'rgba(239,68,68,.3)'),borderRadius:5,barPercentage:.45,categoryPercentage:.7}]},
  options:{responsive:true,plugins:{legend:{labels:{color:'#8694b2',boxWidth:14,boxHeight:14,borderRadius:3,useBorderRadius:true}}},scales:{y:{ticks:{callback:v=>v+'%'}}}}});

(()=>{const g=document.getElementById('heatmap');
  const ms=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'];
  g.innerHTML='<div class="hm-h"></div>';ms.forEach(m=>g.innerHTML+='<div class="hm-h">'+m+'</div>');
  const lk={};MO.forEach(d=>lk[d.year+'-'+d.month]=d.ret);
  const yrs=[...new Set(MO.map(d=>d.year))].sort();
  yrs.forEach(y=>{g.innerHTML+='<div class="hm-y">'+y+'</div>';
    for(let m=1;m<=12;m++){const v=lk[y+'-'+m];
      if(v===undefined){g.innerHTML+='<div class="hm-c" style="color:var(--t3)">—</div>';continue}
      const i=Math.min(Math.abs(v)/8,1);let bg,fg;
      if(v>=0){bg='rgba(34,197,94,'+(0.1+i*.55)+')';fg=i>.4?'#fff':'#22c55e'}
      else{bg='rgba(239,68,68,'+(0.1+i*.55)+')';fg=i>.4?'#fff':'#ef4444'}
      g.innerHTML+='<div class="hm-c" style="background:'+bg+';color:'+fg+'">'+v.toFixed(1)+'</div>';}});})();
"""


def build_html(version, start_date, end_date, m, dates, equity, bench_nav,
               dd_list, bench_dd_list, excess_list, yearly_strat, yearly_bench,
               monthly, trade_stats, signal_df, config):

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    yr_labels = [d["year"] for d in yearly_strat]
    yr_strat = [d["ret"] for d in yearly_strat]
    blk = {d["year"]: d["ret"] for d in yearly_bench}
    yr_bench = [blk.get(y, 0) for y in yr_labels]
    win_years = sum(1 for r in yr_strat if r > 0)

    # Signal rows
    sig_rows = ""
    if not signal_df.empty and "action" in signal_df.columns:
        inv = signal_df[signal_df["action"]=="INVEST_MORE"]
        if inv.empty: inv = signal_df.head(10)
        for _,r in inv.head(15).iterrows():
            act=r.get("action","HOLD")
            cls="tag-buy" if act=="INVEST_MORE" else "tag-sell" if "WITHDRAW" in str(act) else "tag-hold"
            sig_rows += f'<tr><td class="mono">{r.get("symbol","")}</td><td>{r.get("industry","—")}</td><td class="mono">{_sf(r.get("score",0)):.2f}</td><td>#{int(_sf(r.get("rank",0)))}</td><td><span class="act {cls}">{act}</span></td><td class="mono">{_sf(r.get("target_weight",0))*100:.1f}%</td><td class="mono">{_sf(r.get("tdx_score",0)):.1f}</td></tr>'

    # Trade stats
    th = ""
    if trade_stats and trade_stats.get("total",0) > 0:
        ts = trade_stats
        th = f'''<div class="card"><div class="card-t"><i class="dot" style="background:var(--amber)"></i>交易统计</div>
        <div class="kpi-strip four">
        <div class="kpi"><div class="kpi-v neu">{ts["total"]}</div><div class="kpi-l">总交易</div></div>
        <div class="kpi"><div class="kpi-v {"pos" if ts.get("win_rate",0)>50 else "neg"}">{ts.get("win_rate","—")}%</div><div class="kpi-l">胜率</div></div>
        <div class="kpi"><div class="kpi-v pos">{ts.get("best","—"):+.2f}%</div><div class="kpi-l">最佳单笔</div></div>
        <div class="kpi"><div class="kpi-v neg">{ts.get("worst","—"):+.2f}%</div><div class="kpi-l">最差单笔</div></div>
        </div></div>'''

    # Config summary
    st=config.get("strategy",{}); rk=config.get("risk_control",{})
    parts=[]
    if st.get("entry_mode"): parts.append(f'入场: {st["entry_mode"]}')
    if st.get("top_k"): parts.append(f'选股: {st["top_k"]}只')
    if rk.get("stop_loss_pct"): parts.append(f'止损: {_sf(rk["stop_loss_pct"])*100:.0f}%')
    if rk.get("trailing_stop_pct"): parts.append(f'止盈: {_sf(rk["trailing_stop_pct"])*100:.0f}%')
    if st.get("industry_diversification",{}).get("enabled"): parts.append("行业分散")
    if rk.get("use_tdx_protection"): parts.append("TDX保护")
    cfg_s = " · ".join(parts) if parts else "默认配置"

    # Data JS
    data_js = f"""
window._D={json.dumps(dates)};window._E={json.dumps(equity)};window._B={json.dumps(bench_nav)};
window._DD={json.dumps(dd_list)};window._BDD={json.dumps(bench_dd_list)};window._EX={json.dumps(excess_list)};
window._YL={json.dumps(yr_labels)};window._YS={json.dumps(yr_strat)};window._YB={json.dumps(yr_bench)};
window._MO={json.dumps(monthly)};"""

    # Signal section
    sig_sec = ""
    if sig_rows:
        sig_sec = f'''<div class="card"><div class="card-t"><i class="dot" style="background:var(--green)"></i>最新信号 ({end_date})</div>
        <table class="tbl"><thead><tr><th>代码</th><th>行业</th><th>得分</th><th>排名</th><th>动作</th><th>仓位</th><th>TDX</th></tr></thead>
        <tbody>{sig_rows}</tbody></table></div>'''

    kpi_html = lambda val, cls, label, sub="": f'<div class="kpi"><div class="kpi-v {cls}">{val}</div><div class="kpi-l">{label}</div>{f"<div class=kpi-s>{sub}</div>" if sub else ""}</div>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>量化回测报告 | {version}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;600;700;900&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>

<div class="hdr"><div class="wrap">
<h1>📊 量化策略<em>回测报告</em></h1>
<div class="hdr-row">
  <span class="pill pill-v">{version}</span>
  <span class="pill pill-d">📅 {start_date} → {end_date}</span>
  <span class="pill pill-d">📈 {m["trading_days"]} 交易日 · {m["years"]} 年</span>
</div>
<div class="hdr-row"><span class="pill pill-cfg">⚙️ {cfg_s}</span></div>
</div></div>

<div class="wrap">
<div class="kpi-strip">
  {kpi_html(f"{'+' if m['total_return']>0 else ''}{m['total_return']}%", 'pos' if m['total_return']>0 else 'neg', '策略总收益', f"年化 {'+' if m['ann_return']>0 else ''}{m['ann_return']}%")}
  {kpi_html(f"{'+' if m['excess_return']>0 else ''}{m['excess_return']}%", 'pos' if m['excess_return']>0 else 'neg', '超额收益', f"沪深300 {'+' if m['bench_return']>0 else ''}{m['bench_return']}%")}
  {kpi_html(f"{m['max_dd']}%", 'neg', '最大回撤', f"持续{m['max_dd_days']}天 · 基准{m['bench_max_dd']}%")}
  {kpi_html(str(m['sharpe']), 'neu', '夏普比率', f"索提诺{m['sortino']} · 卡尔玛{m['calmar']}")}
  {kpi_html(f"{m['win_rate']}%", 'neu', '日胜率', f"盈亏比 {m['pl_ratio']}")}
  {kpi_html(f"{m['ann_vol']}%", 'neu" style="color:var(--amber)', '年化波动', f"盈利年 {win_years}/{len(yr_labels)}")}
  {kpi_html(f"{m['max_gain']:+.2f}%", 'pos', '最大日涨')}
  {kpi_html(f"{m['max_loss']:+.2f}%", 'neg', '最大日跌')}
</div>

<div class="card"><div class="card-t"><i class="dot" style="background:var(--blue)"></i>权益曲线 vs 沪深300</div>
<div class="lg"><span><b style="background:var(--blue)"></b>策略净值</span><span><b style="background:var(--amber)"></b>沪深300</span></div>
<canvas id="cEquity" height="380"></canvas></div>

<div class="row2">
<div class="card"><div class="card-t"><i class="dot" style="background:var(--green)"></i>超额收益</div><canvas id="cExcess" height="260"></canvas></div>
<div class="card"><div class="card-t"><i class="dot" style="background:var(--red)"></i>回撤对比</div>
<div class="lg"><span><b style="background:var(--red)"></b>策略</span><span><b style="background:var(--amber)"></b>基准</span></div>
<canvas id="cDD" height="240"></canvas></div>
</div>

<div class="card"><div class="card-t"><i class="dot" style="background:var(--purple)"></i>年度收益对比</div><canvas id="cYearly" height="280"></canvas></div>
<div class="card"><div class="card-t"><i class="dot" style="background:var(--cyan)"></i>月度收益热力图 (%)</div><div id="heatmap" class="hm"></div></div>

{th}{sig_sec}

<div class="disc">⚠️ <strong>风险提示：</strong>本报告基于历史数据回测，不保证未来收益。股市有风险，投资需谨慎。仅供学习研究参考，不构成投资建议。</div>
</div>
<div class="foot">A股智能量化交易系统 · {version} · {now_str}</div>

<script>{data_js}</script>
<script>{CHART_JS}</script>
</body></html>"""


# ── 主函数 ────────────────────────────────────────────────

def main():
    base_dir = Path(__file__).resolve().parent
    print("="*60); print("📊 一站式回测报告生成器"); print("="*60)

    print("\n[0] 加载配置...")
    config = load_project_config(base_dir)
    start_date = get_start_date(config)
    print(f"  📅 起始日期: {start_date}")

    print("\n[1] 加载回测数据...")
    df, version = load_backtest_equity(base_dir)
    if df is None:
        print("\n❌ 未找到回测数据！请先运行回测脚本。"); sys.exit(1)
    df = filter_and_normalize(df, start_date)
    sd = df["date"].min().strftime("%Y-%m-%d")
    ed = df["date"].max().strftime("%Y-%m-%d")

    print("\n[2] 获取沪深300...")
    bm = fetch_benchmark(sd, ed)
    df = merge_benchmark(df, bm)

    print("\n[3] 计算指标...")
    m = calc_metrics(df)
    mo = calc_monthly(df)
    ys = calc_yearly(df, "equity")
    yb = calc_yearly(df, "benchmark_nav") if "benchmark_nav" in df.columns else []
    print(f"  收益: {m['total_return']:+.2f}% | 年化: {m['ann_return']:+.2f}% | 超额: {m['excess_return']:+.2f}% | 夏普: {m['sharpe']}")

    print("\n[4] 辅助数据...")
    tl = load_trade_log(base_dir)
    ts = calc_trade_stats(tl)
    sig = load_latest_signal(base_dir)

    print("\n[5] 生成HTML...")
    pk=df["equity"].cummax(); bpk=df["benchmark_nav"].cummax()
    html = build_html(
        version, sd, ed, m,
        df["date"].dt.strftime("%Y-%m-%d").tolist(),
        df["equity"].round(0).tolist(),
        df["benchmark_nav"].round(0).tolist(),
        ((df["equity"]/pk-1)*100).round(2).tolist(),
        ((df["benchmark_nav"]/bpk-1)*100).round(2).tolist(),
        ((df["equity"]/df["benchmark_nav"]-1)*100).round(2).tolist(),
        ys, yb, mo, ts, sig, config)

    out = base_dir / "out"; out.mkdir(exist_ok=True)
    op = out / "backtest_report.html"
    op.write_text(html, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"✅ 报告已生成!")
    print(f"   📂 {op}")
    print(f"   📏 {op.stat().st_size/1024:.1f} KB")
    print(f"   📅 {sd} → {ed}")
    print(f"   📈 {m['total_return']:+.2f}% (年化 {m['ann_return']:+.2f}%)")
    print(f"{'='*60}")
    print(f"\n💡 浏览器打开: {op.resolve()}")

if __name__ == "__main__":
    main()