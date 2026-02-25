from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import duckdb
except Exception:  # pragma: no cover - optional dependency
    duckdb = None


MARKET_DAILY_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "amplitude",
    "pct_chg",
    "chg",
]


class DataManager:
    def __init__(self, db_path: str | Path):
        if duckdb is None:
            raise ModuleNotFoundError(
                "duckdb is not installed. Install with: pip install duckdb>=1.0.0"
            )
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_daily (
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                amount DOUBLE,
                turnover DOUBLE,
                amplitude DOUBLE,
                pct_chg DOUBLE,
                chg DOUBLE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_daily_date ON market_daily(date);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_daily_symbol_date ON market_daily(symbol, date);"
        )

    @staticmethod
    def _normalize_market_daily(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=MARKET_DAILY_COLUMNS)

        out = df.copy()
        for col in MARKET_DAILY_COLUMNS:
            if col not in out.columns:
                out[col] = None
        out = out[MARKET_DAILY_COLUMNS].copy()
        out["symbol"] = out["symbol"].astype(str).str.upper()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        out = out.dropna(subset=["symbol", "date"]).drop_duplicates(
            ["symbol", "date"], keep="last"
        )
        return out.reset_index(drop=True)

    def upsert_market_daily(self, df: pd.DataFrame) -> int:
        norm = self._normalize_market_daily(df)
        if norm.empty:
            return 0

        self.conn.register("_market_daily_input", norm)
        try:
            self.conn.execute("DROP TABLE IF EXISTS _market_daily_stage")
            self.conn.execute(
                """
                CREATE TEMP TABLE _market_daily_stage AS
                SELECT
                    UPPER(CAST(symbol AS TEXT)) AS symbol,
                    CAST(date AS DATE) AS date,
                    CAST(open AS DOUBLE) AS open,
                    CAST(high AS DOUBLE) AS high,
                    CAST(low AS DOUBLE) AS low,
                    CAST(close AS DOUBLE) AS close,
                    CAST(volume AS DOUBLE) AS volume,
                    CAST(amount AS DOUBLE) AS amount,
                    CAST(turnover AS DOUBLE) AS turnover,
                    CAST(amplitude AS DOUBLE) AS amplitude,
                    CAST(pct_chg AS DOUBLE) AS pct_chg,
                    CAST(chg AS DOUBLE) AS chg
                FROM _market_daily_input
                """
            )
            self.conn.execute(
                """
                DELETE FROM market_daily
                WHERE EXISTS (
                    SELECT 1
                    FROM _market_daily_stage s
                    WHERE s.symbol = market_daily.symbol
                      AND s.date = market_daily.date
                )
                """
            )
            self.conn.execute(
                """
                INSERT INTO market_daily (
                    symbol, date, open, high, low, close, volume, amount,
                    turnover, amplitude, pct_chg, chg, updated_at
                )
                SELECT
                    symbol, date, open, high, low, close, volume, amount,
                    turnover, amplitude, pct_chg, chg, CURRENT_TIMESTAMP
                FROM _market_daily_stage
                """
            )
            cnt = int(len(norm))
            return cnt
        finally:
            try:
                self.conn.execute("DROP TABLE IF EXISTS _market_daily_stage")
            except Exception:
                pass
            try:
                self.conn.unregister("_market_daily_input")
            except Exception:
                pass

    def load_market_daily(
        self,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        cols = MARKET_DAILY_COLUMNS if not columns else [str(c) for c in columns]
        select_cols = [c for c in cols if c in set(MARKET_DAILY_COLUMNS)]
        if not select_cols:
            select_cols = MARKET_DAILY_COLUMNS.copy()

        sql = f"SELECT {', '.join(select_cols)} FROM market_daily"
        where_parts: list[str] = []
        params: list[Any] = []

        if symbols:
            vals = [str(s).upper() for s in symbols if str(s).strip()]
            if vals:
                placeholders = ", ".join(["?"] * len(vals))
                where_parts.append(f"symbol IN ({placeholders})")
                params.extend(vals)
        if start_date:
            where_parts.append("date >= CAST(? AS DATE)")
            params.append(str(start_date))
        if end_date:
            where_parts.append("date <= CAST(? AS DATE)")
            params.append(str(end_date))

        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
        sql += " ORDER BY symbol, date"
        return self.conn.execute(sql, params).df()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

