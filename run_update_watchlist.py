"""
📊 智能股票池更新器 V2（Smart Beta + TDX叠加）

全市场扫描 → 多层过滤 → 横截面评分 → 输出 watchlist_cache.yaml

运行频率: 每周五收盘后
用时预估: 3-10分钟

筛选架构（老师指导的 Smart Beta 框架）:
  The Pool（可投资性）:
    - 剔除 ST / 北交所 / B股
    - 流通市值 80亿~5000亿（机构可容纳）
    - 20日均成交额 > 8000万
    - 20日均换手率 > 0.3%

  风控剔除:
    - 妖股：30日涨停≥3次 → 直接剔除
    - 极端波动：年化波动率>100% → 直接剔除
    - 极端异动：近30天|ret|>7%超过4次 → 直接剔除

  综合评分（横截面rank标准化）:
    - 趋势/动量 30%: 高30创新高 + 120日动量 + 60日回撤
    - 低波动 30%: 20日/60日波动率（越低越好）
    - 资金流 15%: 主力控盘 + 相对成交量
    - 市值偏好 25%: 偏好中大盘

使用: python run_update_watchlist.py [--min-pool 300] [--max-pool 500]
"""
from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def get_all_a_shares() -> list[str]:
    """获取全部A股代码"""
    symbols = []

    # 方法1: BaoStock
    try:
        import baostock as bs
        lg = bs.login()
        try:
            # 尝试今天，如果是非交易日则往前找
            for delta in range(0, 10):
                day = (datetime.date.today() - datetime.timedelta(days=delta)).isoformat()
                rs = bs.query_all_stock(day=day)
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if len(rows) > 100:
                    print(f"  ✓ BaoStock查询日期: {day}, 返回 {len(rows)} 条")
                    break

            for row in rows:
                code = row[0]  # "sh.600000" 或 "sz.000001" 格式
                if not isinstance(code, str) or "." not in code:
                    continue
                prefix, pure = code.split(".", 1)
                # 过滤: 只要主板/创业板/科创板的A股
                if prefix == "sh" and (pure.startswith("6") or pure.startswith("688")):
                    symbols.append(pure)
                elif prefix == "sz" and (pure.startswith("0") or pure.startswith("3")):
                    symbols.append(pure)
        finally:
            bs.logout()

        if len(symbols) > 100:
            print(f"  ✓ BaoStock获取 {len(symbols)} 只A股")
            return symbols
        else:
            print(f"  ⚠ BaoStock只获取到 {len(symbols)} 只，尝试AKShare...")
    except Exception as e:
        print(f"  ⚠ BaoStock失败: {e}")

    # 方法2: AKShare
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        for code in df["代码"]:
            code = str(code).zfill(6)
            if code.startswith(("6", "0", "3")):
                symbols.append(code)
        if len(symbols) > 100:
            print(f"  ✓ AKShare获取 {len(symbols)} 只A股")
            return symbols
    except Exception as e:
        print(f"  ⚠ AKShare失败: {e}")

    # 方法3: 硬编码获取 ── 如果前两个都失败，用BaoStock的stock_industry
    try:
        import baostock as bs
        lg = bs.login()
        try:
            rs = bs.query_stock_industry()
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            for row in rows:
                code = str(row[1]) if len(row) > 1 else ""  # 第2列是代码
                if "." in code:
                    prefix, pure = code.split(".", 1)
                    if prefix == "sh" and pure.startswith("6"):
                        symbols.append(pure)
                    elif prefix == "sz" and (pure.startswith("0") or pure.startswith("3")):
                        symbols.append(pure)
        finally:
            bs.logout()
        symbols = list(set(symbols))
        if len(symbols) > 100:
            print(f"  ✓ BaoStock行业接口获取 {len(symbols)} 只A股")
            return symbols
    except Exception as e:
        print(f"  ⚠ BaoStock行业接口也失败: {e}")

    if symbols:
        print(f"  ⚠ 只获取到 {len(symbols)} 只，继续使用")
        return symbols

    raise RuntimeError("无法获取A股列表，请检查网络连接")


def fetch_daily_data_batch(symbols: list[str], days: int = 90) -> pd.DataFrame:
    """批量获取日线数据（只取最近N天，够算指标即可）"""
    import baostock as bs

    end_date = datetime.date.today().isoformat()
    start_date = (datetime.date.today() - datetime.timedelta(days=days * 1.6)).isoformat()

    lg = bs.login()
    try:
        all_data = []
        total = len(symbols)

        for i, sym in enumerate(symbols):
            # BaoStock格式
            if sym.startswith(("6", "688")):
                bs_code = f"sh.{sym}"
            else:
                bs_code = f"sz.{sym}"

            try:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,code,close,high,low,volume,amount,turn",
                    start_date, end_date, "d", "2"  # 前复权
                )
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())

                if rows:
                    df = pd.DataFrame(rows, columns=rs.fields)
                    df["symbol"] = sym
                    all_data.append(df)
            except Exception:
                pass

            if (i + 1) % 100 == 0:
                print(f"    进度: {i+1}/{total} ({(i+1)/total*100:.0f}%)")

        print(f"  ✓ 获取到 {len(all_data)} 只股票的日线数据")
    finally:
        bs.logout()

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    for col in ["close", "high", "low", "volume", "amount", "turn"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    result["date"] = pd.to_datetime(result["date"])
    return result


def calc_tdx_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算选股指标（V2 — Smart Beta + TDX叠加）

    变更（第1步优化）：
    - 涨停从"加分"改为"妖股惩罚"
    - 新增：20日/60日波动率
    - 新增：近120日动量（价格相对强度）
    - 新增：均线结构（MA20/MA60趋势）
    - 新增：极端波动频次（|ret|>7%的天数）
    - 新增：涨停后暴跌检测
    """
    results = []

    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < 30:
            continue

        close = g["close"].values
        high = g["high"].values
        low = g["low"].values
        volume = g["volume"].values
        amount = g["amount"].values
        turn = g["turn"].values

        n = len(close)

        # ── 高30（趋势向上）──
        x1 = (close + low + high) / 1.5
        x2 = pd.Series(x1).ewm(span=3, adjust=False).mean().values
        high30 = pd.Series(x2).rolling(30, min_periods=10).max().values
        high30_prev = np.roll(high30, 1)
        high30_prev[0] = np.nan
        high30_ok = 1 if (not np.isnan(high30[-1]) and not np.isnan(high30_prev[-1])
                          and high30[-1] > high30_prev[-1]) else 0

        # ── 主力控盘 ──
        gu1 = (close * 2 + high + low) / 4.0
        ema9 = pd.Series(close).ewm(span=9, adjust=False).mean().values
        ema9_9 = pd.Series(ema9).ewm(span=9, adjust=False).mean().values
        qibao_prev = np.roll(ema9_9, 1)
        qibao_prev[0] = np.nan
        main_force_pct = (gu1[-1] - qibao_prev[-1]) / qibao_prev[-1] if qibao_prev[-1] > 0 else 0
        main_force_ok = 1 if main_force_pct > 0.005 else 0

        # ── 涨停统计（不再作为入场条件，改为妖股检测）──
        daily_ret = np.zeros(n)
        daily_ret[1:] = close[1:] / close[:-1] - 1
        is_limit_up = (daily_ret > 0.095).astype(float)
        limit_up_count_30d = int(pd.Series(is_limit_up).rolling(30, min_periods=1).sum().values[-1])

        # ── 妖股检测 ──
        # 涨停后次日大跌（跌>3%）的次数
        limit_up_then_drop = 0
        for j in range(1, n - 1):
            if is_limit_up[j] == 1 and daily_ret[j + 1] < -0.03:
                limit_up_then_drop += 1

        # ── 波动率（核心新增）──
        ret_series = pd.Series(daily_ret)
        vol_20d = ret_series.rolling(20, min_periods=10).std().values[-1] * np.sqrt(252)
        vol_60d = ret_series.rolling(min(60, n), min_periods=20).std().values[-1] * np.sqrt(252) if n >= 20 else vol_20d

        # 极端波动频次：近30天 |ret| > 7% 的天数
        extreme_days_30d = int((np.abs(daily_ret[-30:]) > 0.07).sum()) if n >= 30 else 0

        # ── 动量/趋势（替代涨停作为趋势信号）──
        # 近120日收益率（如果数据不够就用全部）
        lookback = min(120, n - 1)
        mom_120d = (close[-1] / close[-1 - lookback] - 1) if lookback > 0 and close[-1 - lookback] > 0 else 0

        # 均线结构
        ma20 = pd.Series(close).rolling(20, min_periods=10).mean().values
        ma60 = pd.Series(close).rolling(min(60, n), min_periods=20).mean().values
        above_ma60 = 1 if (not np.isnan(ma60[-1]) and close[-1] > ma60[-1]) else 0
        # MA20斜率（最近5天MA20的变化）
        ma20_slope = 0
        if n >= 25 and not np.isnan(ma20[-1]) and not np.isnan(ma20[-5]) and ma20[-5] > 0:
            ma20_slope = (ma20[-1] / ma20[-5] - 1)

        # 近60日最大回撤
        if n >= 60:
            recent_close = close[-60:]
        else:
            recent_close = close
        peak = np.maximum.accumulate(recent_close)
        dd = (recent_close / peak - 1)
        max_dd_60d = float(dd.min())

        # ── 流动性 ──
        avg_amount_20d = pd.Series(amount).rolling(20, min_periods=5).mean().values[-1]
        avg_turn_20d = pd.Series(turn).rolling(20, min_periods=5).mean().values[-1]

        # ── 相对成交量 RVOL ──
        vol_ma20 = pd.Series(volume).rolling(20, min_periods=5).mean().values[-1]
        rvol = volume[-1] / vol_ma20 if vol_ma20 > 0 else 1.0

        # ── 估算流通市值 ──
        last_turn = turn[-1] if turn[-1] > 0 else np.nan
        float_mkt_cap = amount[-1] / (last_turn / 100.0) if last_turn and last_turn > 0 else 0

        # ── 综合评分（V2: 趋势+主力-妖股惩罚-高波动）──
        # 趋势/动量分（25%权重）
        trend_score = high30_ok * 1.0 + above_ma60 * 0.5 + (1.0 if ma20_slope > 0.005 else 0)

        # 主力资金分（10%权重 — 降权）
        flow_score = main_force_ok * 1.0 + (0.5 if rvol > 1.2 else 0)

        # 妖股惩罚（关键改动！）
        yaogu_penalty = 0
        if limit_up_count_30d >= 3:
            yaogu_penalty += 2.0  # 重度妖股
        elif limit_up_count_30d >= 2:
            yaogu_penalty += 1.0  # 中度
        if limit_up_then_drop >= 1:
            yaogu_penalty += 1.0  # 涨停后暴跌
        if extreme_days_30d >= 3:
            yaogu_penalty += 1.0  # 极端波动频繁

        # 高波动惩罚
        vol_penalty = 0
        if vol_20d > 0.8:  # 年化波动率>80%
            vol_penalty = 1.5
        elif vol_20d > 0.6:  # >60%
            vol_penalty = 0.5

        tdx_score = trend_score + flow_score - yaogu_penalty - vol_penalty

        results.append({
            "symbol": sym,
            # 趋势
            "high30_ok": high30_ok,
            "above_ma60": above_ma60,
            "ma20_slope": round(ma20_slope * 100, 2),
            "mom_120d": round(mom_120d * 100, 2),
            "max_dd_60d": round(max_dd_60d * 100, 2),
            # 主力
            "main_force_pct": round(main_force_pct * 100, 2),
            "main_force_ok": main_force_ok,
            "rvol": round(rvol, 2),
            # 妖股检测
            "limit_up_count_30d": limit_up_count_30d,
            "limit_up_then_drop": limit_up_then_drop,
            "extreme_days_30d": extreme_days_30d,
            "yaogu_penalty": round(yaogu_penalty, 1),
            # 波动率
            "vol_20d": round(vol_20d * 100, 1),
            "vol_60d": round(vol_60d * 100, 1),
            "vol_penalty": round(vol_penalty, 1),
            # 综合
            "trend_score": round(trend_score, 2),
            "flow_score": round(flow_score, 2),
            "tdx_score": round(tdx_score, 2),
            # 基础
            "avg_amount_20d": round(avg_amount_20d, 0),
            "avg_turn_20d": round(avg_turn_20d, 3),
            "float_mkt_cap": round(float_mkt_cap, 0),
            "last_close": round(close[-1], 2),
            "last_date": str(g["date"].iloc[-1].date()),
        })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════
# Shield 层：财务质量过滤（BaoStock季频数据）
# ══════════════════════════════════════════════════════════════

def fetch_financial_data(symbols: list[str]) -> pd.DataFrame:
    """
    从BaoStock获取财务数据用于质量过滤（Shield层）

    严格PIT对齐：只用 pubDate <= 今天 的数据，防止未来函数
    获取：ROE、净利润同比、CFO/净利润比率
    """
    import baostock as bs

    print(f"  获取财务数据: {len(symbols)} 只...")
    lg = bs.login()

    # ── 确定要查询的年份和季度（最近3年的所有季度）──
    now = datetime.date.today()
    query_periods = []
    for y in range(now.year - 2, now.year + 1):
        for q in [1, 2, 3, 4]:
            # 跳过未来的季度
            quarter_end_month = q * 3
            if y == now.year and quarter_end_month > now.month:
                break
            query_periods.append((y, q))
    # 按时间倒序（最新的在前）
    query_periods.reverse()
    print(f"    查询周期: {len(query_periods)} 个季度 ({query_periods[-1][0]}Q{query_periods[-1][1]} ~ {query_periods[0][0]}Q{query_periods[0][1]})")

    # ── 先获取行业信息，区分金融/非金融 ──
    finance_symbols = set()
    try:
        rs_ind = bs.query_stock_industry()
        ind_rows = []
        while rs_ind.next():
            ind_rows.append(rs_ind.get_row_data())
        if ind_rows:
            ind_df = pd.DataFrame(ind_rows, columns=rs_ind.fields)
            finance_keywords = ["银行", "保险", "证券", "多元金融", "金融"]
            for _, row in ind_df.iterrows():
                industry = str(row.get("industry", "")) + str(row.get("industryClassification", ""))
                if any(kw in industry for kw in finance_keywords):
                    code_raw = str(row.get("code", ""))
                    if "." in code_raw:
                        _, pure = code_raw.split(".", 1)
                        finance_symbols.add(pure)
            print(f"    识别金融股: {len(finance_symbols)} 只")
    except Exception as e:
        print(f"    ⚠ 获取行业信息失败: {e}")

    results = []
    failed = 0
    pit_now = pd.Timestamp.now() - pd.Timedelta(days=1)  # PIT延迟1天

    for i, sym in enumerate(symbols):
        code6 = sym.split(".")[0] if "." in sym else sym
        bs_code = f"sh.{code6}" if code6.startswith(("6", "5")) else f"sz.{code6}"

        roe_latest = None
        roe_list = []
        net_profit_positive = False
        yoyni = None
        cfo_to_np = None

        try:
            # ── 逐季度查询，找到最新可用数据 ──
            for year, quarter in query_periods:
                if roe_latest is not None and yoyni is not None:
                    # 已找到最新的ROE和YOYNI，但还要继续收集ROE历史（算稳定性）
                    if len(roe_list) >= 8:
                        break

                # 盈利数据
                if roe_latest is None or len(roe_list) < 12:
                    rs_p = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                    rows_p = []
                    while rs_p.next():
                        rows_p.append(rs_p.get_row_data())
                    if rows_p:
                        pf = pd.DataFrame(rows_p, columns=rs_p.fields)
                        pf["pubDate"] = pd.to_datetime(pf["pubDate"], errors="coerce")
                        pf = pf[pf["pubDate"] <= pit_now]
                        if not pf.empty:
                            roe_col = "roeAvg" if "roeAvg" in pf.columns else ("dupontROE" if "dupontROE" in pf.columns else None)
                            if roe_col:
                                val = pd.to_numeric(pf[roe_col].iloc[0], errors="coerce")
                                if pd.notna(val):
                                    if roe_latest is None:
                                        roe_latest = float(val)
                                    roe_list.append(float(val))
                            if "netProfit" in pf.columns and not net_profit_positive:
                                np_val = pd.to_numeric(pf["netProfit"].iloc[0], errors="coerce")
                                if pd.notna(np_val):
                                    net_profit_positive = float(np_val) > 0

                # 成长数据（只需最新一期）
                if yoyni is None:
                    rs_g = bs.query_growth_data(code=bs_code, year=year, quarter=quarter)
                    rows_g = []
                    while rs_g.next():
                        rows_g.append(rs_g.get_row_data())
                    if rows_g:
                        gf = pd.DataFrame(rows_g, columns=rs_g.fields)
                        gf["pubDate"] = pd.to_datetime(gf["pubDate"], errors="coerce")
                        gf = gf[gf["pubDate"] <= pit_now]
                        if not gf.empty:
                            for c in ["YOYNI", "YOYNetProfit"]:
                                if c in gf.columns:
                                    val = pd.to_numeric(gf[c].iloc[0], errors="coerce")
                                    if pd.notna(val):
                                        yoyni = float(val)
                                        break

                # 现金流数据（只需最新一期）
                if cfo_to_np is None:
                    rs_c = bs.query_cash_flow_data(code=bs_code, year=year, quarter=quarter)
                    rows_c = []
                    while rs_c.next():
                        rows_c.append(rs_c.get_row_data())
                    if rows_c:
                        cf = pd.DataFrame(rows_c, columns=rs_c.fields)
                        cf["pubDate"] = pd.to_datetime(cf["pubDate"], errors="coerce")
                        cf = cf[cf["pubDate"] <= pit_now]
                        if not cf.empty:
                            for c in ["CFOToNP", "CAToAsset"]:
                                if c in cf.columns:
                                    val = pd.to_numeric(cf[c].iloc[0], errors="coerce")
                                    if pd.notna(val):
                                        cfo_to_np = float(val)
                                        break

            # ROE稳定性
            roe_std = float(np.std(roe_list)) if len(roe_list) >= 4 else None

            # 金融股判断
            is_finance = code6 in finance_symbols

            results.append({
                "symbol": sym,
                "roe_latest": roe_latest,
                "roe_std": roe_std,
                "yoyni": yoyni,
                "cfo_to_np": cfo_to_np,
                "net_profit_positive": net_profit_positive,
                "is_finance": is_finance,
                "has_financial_data": roe_latest is not None,
            })

        except Exception as e:
            failed += 1
            results.append({
                "symbol": sym,
                "roe_latest": None, "roe_std": None, "yoyni": None,
                "cfo_to_np": None, "net_profit_positive": False,
                "is_finance": False, "has_financial_data": False,
            })

        if (i + 1) % 100 == 0:
            has_so_far = sum(1 for r in results if r["has_financial_data"])
            print(f"    财务进度: {i+1}/{len(symbols)} ({(i+1)/len(symbols)*100:.0f}%) | 有数据: {has_so_far}")

    bs.logout()

    df = pd.DataFrame(results)
    has_data = df["has_financial_data"].sum()
    n_fin = df["is_finance"].sum()
    print(f"  ✓ 财务数据: {has_data}/{len(df)} 只有数据, {n_fin} 只金融股 ({failed} 只失败)")

    return df


def apply_shield(indicators: pd.DataFrame, financials: pd.DataFrame) -> pd.DataFrame:
    """
    Shield层：质量防暴雷过滤

    规则（非金融）：
      - ROE > 8%
      - 净利润同比(YOYNI) > 0
      - CFO/NP in [0.8, 5]（区间钳制，防噪声）
      - ROE稳定性：剔除全市场最差20%
      - 缺失财务数据 → 剔除

    规则（金融）：
      - ROE > 8%
      - YOYNI > 0
      - 跳过CFO过滤
    """
    df = indicators.copy()
    n_before = len(df)

    # 合并财务数据
    if financials.empty:
        print(f"  Shield: 无财务数据，跳过质量过滤")
        return df

    # symbol格式统一
    if "symbol" in financials.columns:
        fin = financials.copy()
        # 确保格式一致
        df_syms = set(df["symbol"].astype(str))
        fin_syms = set(fin["symbol"].astype(str))
        # 如果indicators用的是纯代码（无后缀），financials也用纯代码
        # 匹配方式：直接merge
        df = df.merge(fin, on="symbol", how="left", suffixes=("", "_fin"))

    # ── 非金融股过滤 ──
    is_fin = df.get("is_finance", pd.Series(False, index=df.index)).fillna(False)
    non_fin = df[~is_fin].copy()
    fin_stocks = df[is_fin].copy()

    # 非金融过滤条件
    if not non_fin.empty:
        n_nf_before = len(non_fin)

        # 1. 必须有财务数据
        has_data = non_fin["has_financial_data"].fillna(False)
        non_fin = non_fin[has_data]

        # 2. ROE > 8%（BaoStock返回小数，0.08 = 8%）
        if "roe_latest" in non_fin.columns:
            non_fin = non_fin[non_fin["roe_latest"].fillna(0) > 0.08]

        # 3. 净利润同比 > 0（BaoStock返回小数，0 = 0%）
        if "yoyni" in non_fin.columns:
            non_fin = non_fin[non_fin["yoyni"].fillna(-1) > 0]

        # 4. CFO/NP >= 0.5 且 <= 5（放宽到0.5，茅台0.70也能过）
        if "cfo_to_np" in non_fin.columns:
            cfo = non_fin["cfo_to_np"].fillna(-999)
            non_fin = non_fin[(cfo >= 0.5) & (cfo <= 5)]

        # 5. ROE稳定性：剔除最差20%
        if "roe_std" in non_fin.columns and len(non_fin) > 10:
            roe_std = non_fin["roe_std"].fillna(999)
            threshold = roe_std.quantile(0.8)  # 最差20%的阈值
            non_fin = non_fin[roe_std <= threshold]

        n_nf_after = len(non_fin)
        print(f"  Shield(非金融): {n_nf_before} → {n_nf_after} (剔除 {n_nf_before - n_nf_after} 只)")

    # ── 金融股过滤（宽松）──
    if not fin_stocks.empty:
        n_f_before = len(fin_stocks)

        if "roe_latest" in fin_stocks.columns:
            fin_stocks = fin_stocks[fin_stocks["roe_latest"].fillna(0) > 0.08]
        if "yoyni" in fin_stocks.columns:
            fin_stocks = fin_stocks[fin_stocks["yoyni"].fillna(-1) > 0]

        n_f_after = len(fin_stocks)
        print(f"  Shield(金融): {n_f_before} → {n_f_after} (剔除 {n_f_before - n_f_after} 只)")

    # ── 合并 ──
    df = pd.concat([non_fin, fin_stocks], ignore_index=True)

    # 清理合并产生的多余列
    drop_cols = [c for c in df.columns if c.endswith("_fin")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    n_after = len(df)
    print(f"  Shield总计: {n_before} → {n_after} (剔除 {n_before - n_after} 只质量不达标)")

    return df


def filter_watchlist(indicators: pd.DataFrame, min_pool: int = 300, max_pool: int = 500) -> pd.DataFrame:
    """
    按条件筛选股票池（V2 — Smart Beta优化）

    变更：
    - 市值门槛从30亿提升到80亿（老师建议）
    - 成交额门槛从5000万提升到8000万
    - 新增：妖股剔除（30日涨停≥3次）
    - 新增：极端高波动剔除（年化波动>100%）
    - 评分改为横截面rank标准化
    """
    df = indicators.copy()
    n0 = len(df)

    # ══ 第1层：The Pool（可投资性筛选）══

    # 1a. 流动性：20日均成交额 > 8000万（从5000万提升）
    df = df[df["avg_amount_20d"] >= 8e7]
    n1 = len(df)

    # 1b. 市值：流通市值 80亿 ~ 5000亿（从30亿提升到80亿）
    df = df[(df["float_mkt_cap"] >= 8e9) & (df["float_mkt_cap"] <= 5e11)]
    n2 = len(df)

    # 1c. 换手率 > 0.3%
    df = df[df["avg_turn_20d"] >= 0.3]
    n3 = len(df)

    print(f"  The Pool: 全市场{n0} → 流动性≥8000万={n1} → 市值80-5000亿={n2} → 换手率={n3}")

    # ══ 第2层：妖股剔除 + 极端波动剔除 ══

    before_filter = len(df)

    # 2a. 妖股剔除：30日涨停≥3次直接踢出
    if "limit_up_count_30d" in df.columns:
        df = df[df["limit_up_count_30d"] < 3]

    # 2b. 极端高波动剔除：年化波动率>100%
    if "vol_20d" in df.columns:
        df = df[df["vol_20d"] <= 100]  # 已经是百分比

    # 2c. 极端波动频次：近30天 |ret|>7% 超过4次
    if "extreme_days_30d" in df.columns:
        df = df[df["extreme_days_30d"] <= 4]

    n4 = len(df)
    removed = before_filter - n4
    print(f"  风控过滤: 剔除妖股/极端波动 {removed} 只 → 剩余 {n4}")

    # ══ 第3层：横截面排名评分 ══

    if len(df) < 10:
        print(f"  ⚠️ 剩余股票太少({len(df)}只)，放宽条件")
        return df

    # 用rank做横截面标准化（0-1），避免牛熊切换时阈值失效
    df = df.copy()

    # ── 质量因子分（45%）── Shield层已过滤掉烂票，这里对好票排名
    df["rank_quality"] = 0.5  # 默认中位数
    has_quality = "roe_latest" in df.columns and df["roe_latest"].notna().sum() > 5
    if has_quality:
        # ROE越高越好
        df["rank_quality"] = df["roe_latest"].fillna(0).rank(pct=True).fillna(0.5) * 0.40
        # YOYNI越高越好（成长性）
        if "yoyni" in df.columns:
            df["rank_quality"] += df["yoyni"].fillna(0).clip(-100, 500).rank(pct=True).fillna(0.5) * 0.25
        # CFO/NP越高越好（现金流质量，钳制在合理区间）
        if "cfo_to_np" in df.columns:
            df["rank_quality"] += df["cfo_to_np"].fillna(0).clip(0, 5).rank(pct=True).fillna(0.5) * 0.20
        # ROE稳定性越好越好（std越小越好）
        if "roe_std" in df.columns:
            df["rank_quality"] += (1 - df["roe_std"].fillna(99).rank(pct=True).fillna(0.5)) * 0.15

    # ── 趋势/动量分（25%）──
    df["rank_trend"] = 0.0
    if "trend_score" in df.columns:
        df["rank_trend"] += df["trend_score"].rank(pct=True).fillna(0.5) * 0.5
    if "mom_120d" in df.columns:
        df["rank_trend"] += df["mom_120d"].rank(pct=True).fillna(0.5) * 0.3
    if "max_dd_60d" in df.columns:
        # 回撤越小越好（值越大越好，因为是负数）
        df["rank_trend"] += df["max_dd_60d"].rank(pct=True).fillna(0.5) * 0.2

    # ── 低波动分（20%）── 波动率越低越好
    df["rank_lowvol"] = 0.0
    if "vol_20d" in df.columns:
        df["rank_lowvol"] += (1 - df["vol_20d"].rank(pct=True).fillna(0.5)) * 0.6
    if "vol_60d" in df.columns:
        df["rank_lowvol"] += (1 - df["vol_60d"].rank(pct=True).fillna(0.5)) * 0.4

    # ── 资金流/主力分（10%）──
    df["rank_flow"] = 0.0
    if "main_force_pct" in df.columns:
        df["rank_flow"] += df["main_force_pct"].rank(pct=True).fillna(0.5) * 0.7
    if "rvol" in df.columns:
        df["rank_flow"] += df["rvol"].clip(0, 3).rank(pct=True).fillna(0.5) * 0.3

    # ── 综合评分（老师建议的权重）──
    # Quality 45% + Trend 25% + LowVol 20% + Flow 10%
    if has_quality:
        df["composite_score"] = (
            df["rank_quality"] * 0.45
            + df["rank_trend"] * 0.25
            + df["rank_lowvol"] * 0.20
            + df["rank_flow"] * 0.10
        )
    else:
        # 没有质量数据时，退回Step1的权重
        df["composite_score"] = (
            df["rank_trend"] * 0.30
            + df["rank_lowvol"] * 0.30
            + df["rank_flow"] * 0.15
            + df.get("float_mkt_cap", pd.Series(0, index=df.index)).rank(pct=True).fillna(0.5) * 0.25
        )

    # ══ 第4层：选Top N ══

    # 必须有趋势或主力信号（至少一个）
    has_signal = (df["high30_ok"] == 1) | (df["main_force_ok"] == 1) | (df["above_ma60"] == 1)
    signaled = df[has_signal].copy()
    unsignaled = df[~has_signal].copy()

    # 有信号的按综合评分排序
    signaled = signaled.sort_values("composite_score", ascending=False)

    print(f"  有趋势/主力信号: {len(signaled)} 只 | 无信号: {len(unsignaled)} 只")

    # 优先选有信号的
    selected = signaled.head(max_pool).copy()

    # 不够则从无信号中按综合评分补充
    if len(selected) < min_pool:
        unsignaled = unsignaled.sort_values("composite_score", ascending=False)
        need = min_pool - len(selected)
        selected = pd.concat([selected, unsignaled.head(need)], ignore_index=True)

    # 截断到max_pool
    if len(selected) > max_pool:
        selected = selected.head(max_pool)

    # 向后兼容：确保有tdx_pass_count列
    if "tdx_pass_count" not in selected.columns:
        selected["tdx_pass_count"] = (
            selected.get("high30_ok", 0).astype(int)
            + selected.get("main_force_ok", 0).astype(int)
            + (selected.get("limit_up_count_30d", 0) >= 1).astype(int)
        )

    print(f"  最终股票池: {len(selected)} 只")
    print(f"    综合评分: {selected['composite_score'].mean():.3f} (均值) | {selected['composite_score'].max():.3f} (最高)")
    if "vol_20d" in selected.columns:
        print(f"    平均波动率: {selected['vol_20d'].mean():.1f}% | 妖股惩罚: {(selected['yaogu_penalty'] > 0).sum()} 只被扣分")

    return selected


def save_watchlist_cache(selected: pd.DataFrame, base_dir: Path):
    """保存到 watchlist_cache.yaml"""
    symbols = selected["symbol"].tolist()

    cache = {
        "watchlist": symbols,
        "meta": {
            "updated": datetime.datetime.now().isoformat(),
            "total": len(symbols),
            "tdx_3of3": int((selected["tdx_pass_count"] == 3).sum()) if "tdx_pass_count" in selected.columns else 0,
            "tdx_2of3": int((selected["tdx_pass_count"] == 2).sum()) if "tdx_pass_count" in selected.columns else 0,
        },
    }

    cache_path = base_dir / "watchlist_cache.yaml"
    with open(cache_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cache, f, sort_keys=False, allow_unicode=True)

    print(f"  ✓ 已保存: {cache_path}")
    print(f"    股票数: {len(symbols)}")

    # 同时保存详细报告
    report_path = base_dir / "data" / "watchlist_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ 详细报告: {report_path}")

    return cache_path


def main():
    ap = argparse.ArgumentParser(description="智能股票池更新器")
    ap.add_argument("--min-pool", type=int, default=300, help="最小股票池数量")
    ap.add_argument("--max-pool", type=int, default=500, help="最大股票池数量")
    ap.add_argument("--days", type=int, default=90, help="获取多少天的日线数据")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    total_start = time.time()

    print("=" * 60)
    print("📊 智能股票池更新器 V2 (Smart Beta + Shield)")
    print("=" * 60)

    print(f"\n[1/5] 获取全市场A股列表...")
    all_symbols = get_all_a_shares()

    # 排除ST、退市、北交所等
    exclude_prefixes = ("900", "8", "4")  # B股 + 北交所
    all_symbols = [s for s in all_symbols if not s.startswith(tuple(exclude_prefixes))]
    print(f"  待扫描: {len(all_symbols)} 只（已排除B股/北交所）")

    print(f"\n[2/5] 获取日线数据（最近{args.days}天）...")
    data = fetch_daily_data_batch(all_symbols, days=args.days)
    if data.empty:
        print("❌ 获取数据失败！")
        return

    print(f"\n[3/5] 计算技术指标...")
    indicators = calc_tdx_indicators(data)
    print(f"  ✓ 计算完成: {len(indicators)} 只有效股票")

    # ── 先做技术面预过滤（The Pool + 风控），减少财务查询量 ──
    print(f"\n[3.5/5] 技术面预过滤...")
    pre = indicators.copy()
    n_pre = len(pre)
    pre = pre[pre["avg_amount_20d"] >= 8e7]
    pre = pre[(pre["float_mkt_cap"] >= 8e9) & (pre["float_mkt_cap"] <= 5e11)]
    pre = pre[pre["avg_turn_20d"] >= 0.3]
    if "limit_up_count_30d" in pre.columns:
        pre = pre[pre["limit_up_count_30d"] < 3]
    if "vol_20d" in pre.columns:
        pre = pre[pre["vol_20d"] <= 100]
    print(f"  技术面预过滤: {n_pre} → {len(pre)} 只（只查这些的财务数据）")

    print(f"\n[4/5] Shield层：财务质量过滤...")
    fin_symbols = pre["symbol"].tolist()
    if fin_symbols:
        financials = fetch_financial_data(fin_symbols)
        # 只对预过滤后的股票应用Shield
        pre = apply_shield(pre, financials)
    else:
        print(f"  ⚠️ 预过滤后0只股票，跳过Shield")

    print(f"\n[5/5] 综合评分 + 选股...")
    selected = filter_watchlist(pre, min_pool=args.min_pool, max_pool=args.max_pool)
    cache_path = save_watchlist_cache(selected, base_dir)

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"✅ 股票池更新完成! (耗时: {elapsed:.0f}秒)")
    print(f"  📂 {cache_path}")
    print(f"  📊 {len(selected)} 只股票")
    print(f"  📅 下次更新: 一周后")
    print(f"{'=' * 60}")
    print(f"\n💡 后续步骤:")
    print(f"   1. python run_fetch_daily.py        # 抓取这些股票的历史数据")
    print(f"   2. python run_build_market_daily_all.py")
    print(f"   3. python run_build_features_daily.py")
    print(f"   4. python run_backtest_strategy_v3.py  # 回测")


if __name__ == "__main__":
    main()