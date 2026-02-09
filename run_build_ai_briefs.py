from __future__ import annotations

from pathlib import Path

from quant.briefs import build_ai_briefs
from quant.logger import setup_logger


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    
    # Setup logging
    log_file = base_dir / "data" / "logs" / "build_briefs.log"
    logger = setup_logger("quant", log_file=log_file)
    
    ranking_path = base_dir / "data" / "signals" / "latest_daily_rank.csv"
    if not ranking_path.exists():
        raise FileNotFoundError("latest_daily_rank.csv not found. Run: python run_make_daily_rank.py")

    out_dir = base_dir / "data" / "briefs"
    
    # Build briefs with news integration
    all_path = build_ai_briefs(
        ranking_csv=ranking_path,
        out_dir=out_dir,
        base_dir=base_dir,  # Enable news integration
        include_news=True,
        include_all_bundle=True,
    )
    
    logger.info(f"AI briefs generated: {all_path}")
    print("\nAI briefs generated.")
    print("Bundle:", all_path)
    print("Folder:", all_path.parent)
    print("\nNote: News data will be included if available (run run_fetch_news.py first)")


if __name__ == "__main__":
    main()
