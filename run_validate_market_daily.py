from __future__ import annotations

from pathlib import Path
from quant.validate import validate_market_daily

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    summary = validate_market_daily(base_dir)

    print("\n=== Validation Summary ===")
    print(summary.to_string(index=False))

    # quick “red flags”
    bad = summary[
        (summary["dup_rows"] > 0) |
        (summary["invalid_rows"] > 0) |
        (summary["missing_trade_days"].notna() & (summary["missing_trade_days"] > 3))
    ]
    if not bad.empty:
        print("\n[WARN] Symbols with issues:")
        print(bad.to_string(index=False))
    else:
        print("\nAll symbols look clean enough for feature generation.")

if __name__ == "__main__":
    main()
