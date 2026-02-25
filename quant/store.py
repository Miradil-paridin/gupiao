from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pandas as pd

from .data_backend import get_data_backend, is_duckdb_enabled, resolve_duckdb_path
from .data_manager import DataManager


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def symbol_paths(base_dir: Path, symbol: str) -> tuple[Path, Path]:
    raw_dir = base_dir / "data" / "raw" / "market_daily"
    clean_dir = base_dir / "data" / "clean" / "market_daily"
    ensure_dir(raw_dir)
    ensure_dir(clean_dir)
    raw_csv = raw_dir / f"{symbol}.csv"
    clean_parquet = clean_dir / f"{symbol}.parquet"
    return raw_csv, clean_parquet


@lru_cache(maxsize=4)
def _cached_manager(db_path: str) -> DataManager:
    return DataManager(db_path=db_path)


def _get_manager(base_dir: Path) -> DataManager:
    db_path = resolve_duckdb_path(base_dir)
    ensure_dir(db_path.parent)
    return _cached_manager(str(db_path))


def read_existing_market_daily(
    base_dir: Path,
    symbol: str,
    parquet_path: Path | None = None,
) -> pd.DataFrame:
    backend = get_data_backend()
    sym = str(symbol).strip().upper()

    if backend in {"duckdb", "hybrid"}:
        try:
            mgr = _get_manager(base_dir)
            df_db = mgr.load_market_daily(symbols=[sym])
            if df_db is not None and not df_db.empty:
                return df_db
            if backend == "duckdb":
                return pd.DataFrame()
        except ModuleNotFoundError:
            if backend == "duckdb":
                raise

    if parquet_path is None:
        _, parquet_path = symbol_paths(base_dir, sym)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


def read_existing_parquet(path: Path) -> pd.DataFrame:
    """
    Backward-compatible wrapper.
    """
    backend = get_data_backend()
    if backend == "parquet":
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    # Try inferring base_dir/symbol from .../data/clean/market_daily/<symbol>.parquet
    try:
        symbol = path.stem
        base_dir = path.parents[3]
        return read_existing_market_daily(base_dir=base_dir, symbol=symbol, parquet_path=path)
    except Exception:
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()


def write_outputs(
    raw_csv: Path,
    clean_parquet: Path,
    df: pd.DataFrame,
    base_dir: Path | None = None,
    symbol: str | None = None,
) -> None:
    # Raw CSV (easy debugging)
    df.to_csv(raw_csv, index=False, encoding="utf-8-sig")

    backend = get_data_backend()
    if backend in {"parquet", "hybrid"}:
        # Clean parquet (fast analytics / backward compatibility)
        df.to_parquet(clean_parquet, index=False)

    if backend in {"duckdb", "hybrid"}:
        resolved_base = base_dir
        if resolved_base is None:
            try:
                resolved_base = clean_parquet.parents[3]
            except Exception:
                resolved_base = None
        if resolved_base is None:
            raise ValueError("base_dir is required for duckdb backend")

        df_to_save = df.copy()
        if "symbol" not in df_to_save.columns and symbol:
            df_to_save.insert(0, "symbol", str(symbol).upper())
        try:
            _get_manager(resolved_base).upsert_market_daily(df_to_save)
        except ModuleNotFoundError:
            if backend == "duckdb":
                raise


def load_market_daily_all_from_backend(base_dir: Path) -> pd.DataFrame:
    backend = get_data_backend()
    if backend in {"duckdb", "hybrid"} and is_duckdb_enabled():
        db_path = resolve_duckdb_path(base_dir)
        if db_path.exists():
            try:
                df = _get_manager(base_dir).load_market_daily()
                if df is not None and not df.empty:
                    return df
            except ModuleNotFoundError:
                if backend == "duckdb":
                    raise

    clean_all = base_dir / "data" / "clean" / "market_daily_all.parquet"
    if clean_all.exists():
        return pd.read_parquet(clean_all)
    return pd.DataFrame()


def duckdb_status(base_dir: Path) -> dict[str, str | bool]:
    db_path = resolve_duckdb_path(base_dir)
    return {
        "backend": get_data_backend(),
        "duckdb_enabled": is_duckdb_enabled(),
        "duckdb_path": str(db_path),
        "duckdb_exists": db_path.exists(),
    }
