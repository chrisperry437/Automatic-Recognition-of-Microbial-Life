from pathlib import Path
import sqlite3

import pandas as pd


DB_PATH = Path("./results/realtime/predictions.sqlite")


def query_predictions(sql: str, params: tuple = ()) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def load_recent_predictions(minutes: int = 30, limit: int = 50_000) -> pd.DataFrame:
    return query_predictions(
        """
        SELECT *
        FROM predictions
        WHERE timestamp >= (
            SELECT datetime(MAX(timestamp), ?)
            FROM predictions
        )
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (f"-{minutes} minutes", limit),
    )


def load_recent_raw_files(limit: int = 10) -> pd.DataFrame:
    return query_predictions(
        """
        SELECT
            raw_file,
            COUNT(*) AS particle_count,
            MIN(timestamp) AS first_timestamp,
            MAX(timestamp) AS latest_timestamp,
            AVG(prediction_confidence) AS mean_confidence
        FROM predictions
        GROUP BY raw_file
        ORDER BY latest_timestamp DESC
        LIMIT ?
        """,
        (limit,),
    )