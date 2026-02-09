"""
特征构建脚本 (升级版)

功能：
1. 构建基础特征（均线、波动率、RSI、ATR）
2. 构建涨停相关特征（已集成在 features.py 中）
3. 更新沪深300指数数据
"""
from __future__ import annotations

from pathlib import Path
from quant.features import load_market_daily_all, build_features_daily, save_features


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    # Step 1: 加载原始行情数据
    print("=" * 50)
    print("Step 1: 加载行情数据")
    print("=" * 50)
    mkt = load_market_daily_all(base_dir)
    print(f"  行数: {len(mkt)}")
    print(f"  股票数: {mkt['symbol'].nunique()}")

    # Step 2: 构建特征（含涨停特征 + 通达信指标）
    print("\n" + "=" * 50)
    print("Step 2: 构建特征（含涨停 + 通达信指标）")
    print("=" * 50)
    # include_limit_up=True 会自动计算涨停相关特征
    # include_tdx=True 会自动计算通达信指标
    feats = build_features_daily(mkt, include_limit_up=True, include_tdx=True)
    print(f"  特征数: {len(feats.columns)}")

    # Step 3: 更新沪深300指数数据（可选）
    print("\n" + "=" * 50)
    print("Step 3: 更新沪深300指数数据")
    print("=" * 50)
    try:
        from quant.market_regime import MarketRegime
        regime = MarketRegime(base_dir)
        index_df = regime.update_index_data()
        print(f"  沪深300数据已更新")

        # 显示最近的市场环境
        if not index_df.empty:
            latest = index_df.iloc[-1]
            print(f"  最新日期: {latest['date']}")
            print(f"  市场环境: {latest['regime']}")
            print(f"  允许开仓: {'是' if latest['can_open'] else '否'}")
    except ImportError:
        print("  ⚠️ market_regime 模块不存在，跳过沪深300更新")
    except Exception as e:
        print(f"  ⚠️ 沪深300更新失败: {e}")
        print(f"  继续运行，但市场环境过滤将不可用")

    # Step 4: 保存特征
    print("\n" + "=" * 50)
    print("Step 4: 保存特征")
    print("=" * 50)
    out_path = save_features(base_dir, feats)

    # 输出统计
    print("\n" + "=" * 50)
    print("Done ✅")
    print("=" * 50)
    print(f"Saved: {out_path}")
    print(f"Rows: {len(feats)}")
    print(f"Date range: {feats['date'].min()} -> {feats['date'].max()}")
    print(f"Symbols: {sorted(feats['symbol'].unique().tolist())}")

    # 涨停统计（如果有涨停特征）
    if "limit_up_flag" in feats.columns:
        lu_count = int(feats["limit_up_flag"].sum())
        lu_stocks = feats[feats["limit_up_flag"] == 1]["symbol"].nunique()
        print(f"\n涨停统计:")
        print(f"  总涨停次数: {lu_count}")
        print(f"  涉及股票数: {lu_stocks}")
    else:
        print("\n提示: 涨停特征未生成")

    # 通达信指标统计（新增）
    if "tdx_score" in feats.columns:
        tdx_eligible = int(feats["tdx_eligible"].sum())
        high30_count = int(feats["high30_breakout"].sum()) if "high30_breakout" in feats.columns else 0
        main_force_count = int(feats["main_force_strong"].sum()) if "main_force_strong" in feats.columns else 0
        print(f"\n通达信指标统计:")
        print(f"  高30突破次数: {high30_count}")
        print(f"  主力强控盘次数: {main_force_count}")
        print(f"  符合TDX入场条件: {tdx_eligible}")
    else:
        print("\n提示: 通达信指标未生成")


if __name__ == "__main__":
    main()