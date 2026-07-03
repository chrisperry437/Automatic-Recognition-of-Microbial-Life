"""
Prediction storage utilities for the real-time Rapid-E pipeline.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

StorageBackend = Literal["csv", "sqlite", "both"]
MAX_SQLITE_PREDICTION_ROWS = 50_000


@dataclass(frozen=True)
class PredictionStoreConfig:
    output_dir: Path = Path("results/realtime")
    sqlite_path: Path | None = None
    backend: StorageBackend = "both"
    append_csv: bool = True

    predictions_csv: str = "predictions.csv"
    file_summary_csv: str = "file_summary.csv"
    label_summary_csv: str = "label_summary.csv"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())

    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def make_dataframe_storage_safe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    for col in out.columns:
        out[col] = out[col].map(make_json_safe)

    return out


def append_or_write_csv(
    df: pd.DataFrame,
    path: Path,
    append: bool = True,
) -> None:
    ensure_dir(path.parent)
    safe_df = make_dataframe_storage_safe(df)

    if append and path.exists():
        safe_df.to_csv(path, mode="a", index=False, header=False)
    else:
        safe_df.to_csv(path, index=False)


def summarize_file(
    predictions: pd.DataFrame,
    source_file: str | None = None,
    batch_id: str | None = None,
) -> pd.DataFrame:
    processed_at = pd.Timestamp.utcnow().isoformat()

    if predictions.empty:
        return pd.DataFrame(
            [
                {
                    "batch_id": batch_id,
                    "source_file": source_file,
                    "processed_at": processed_at,
                    "n_predictions": 0,
                    "n_unknown": 0,
                    "unknown_fraction": None,
                    "mean_confidence": None,
                    "top_label": None,
                    "top_label_count": 0,
                    "top_label_proportion": 0.0,
                }
            ]
        )

    counts = predictions["predicted_label"].value_counts(dropna=False)
    total = int(len(predictions))

    n_unknown = (
        int(predictions["is_unknown"].sum())
        if "is_unknown" in predictions.columns
        else None
    )

    mean_confidence = (
        float(predictions["prediction_confidence"].mean())
        if "prediction_confidence" in predictions.columns
        else None
    )

    return pd.DataFrame(
        [
            {
                "batch_id": batch_id,
                "source_file": source_file,
                "processed_at": processed_at,
                "n_predictions": total,
                "n_unknown": n_unknown,
                "unknown_fraction": n_unknown / total
                if n_unknown is not None and total
                else None,
                "mean_confidence": mean_confidence,
                "top_label": str(counts.index[0]),
                "top_label_count": int(counts.iloc[0]),
                "top_label_proportion": int(counts.iloc[0]) / total,
            }
        ]
    )


def summarize_labels(
    predictions: pd.DataFrame,
    source_file: str | None = None,
    batch_id: str | None = None,
) -> pd.DataFrame:
    processed_at = pd.Timestamp.utcnow().isoformat()

    if predictions.empty or "predicted_label" not in predictions.columns:
        return pd.DataFrame(
            columns=[
                "batch_id",
                "source_file",
                "processed_at",
                "predicted_label",
                "count",
                "proportion",
                "mean_confidence",
            ]
        )

    total = len(predictions)

    if "prediction_confidence" in predictions.columns:
        summary = (
            predictions.groupby("predicted_label", dropna=False)
            .agg(
                count=("predicted_label", "size"),
                mean_confidence=("prediction_confidence", "mean"),
            )
            .reset_index()
        )
    else:
        summary = (
            predictions.groupby("predicted_label", dropna=False)
            .agg(count=("predicted_label", "size"))
            .reset_index()
        )
        summary["mean_confidence"] = None

    summary["proportion"] = summary["count"] / total
    summary["batch_id"] = batch_id
    summary["source_file"] = source_file
    summary["processed_at"] = processed_at

    return summary[
        [
            "batch_id",
            "source_file",
            "processed_at",
            "predicted_label",
            "count",
            "proportion",
            "mean_confidence",
        ]
    ].sort_values("count", ascending=False)


class PredictionStore:
    def __init__(self, config: PredictionStoreConfig | None = None) -> None:
        self.config = config or PredictionStoreConfig()
        ensure_dir(self.config.output_dir)

        self.sqlite_path = (
            self.config.sqlite_path
            if self.config.sqlite_path is not None
            else self.config.output_dir / "predictions.sqlite"
        )

        if self.config.backend in ("sqlite", "both"):
            self.initialize_sqlite()

    def connect(self) -> sqlite3.Connection:
        ensure_dir(self.sqlite_path.parent)
        return sqlite3.connect(self.sqlite_path)

    def initialize_sqlite(self) -> None:
        """
        Initialize SQLite file only.

        Tables are created dynamically by pandas.to_sql so the schema matches
        the prediction dataframe produced by the current model/parser.
        """
        with self.connect() as conn:
            conn.commit()

    def save_predictions_csv(self, predictions: pd.DataFrame) -> None:
        append_or_write_csv(
            predictions,
            self.config.output_dir / self.config.predictions_csv,
            append=self.config.append_csv,
        )

    def save_file_summary_csv(self, file_summary: pd.DataFrame) -> None:
        append_or_write_csv(
            file_summary,
            self.config.output_dir / self.config.file_summary_csv,
            append=self.config.append_csv,
        )

    def save_label_summary_csv(self, label_summary: pd.DataFrame) -> None:
        append_or_write_csv(
            label_summary,
            self.config.output_dir / self.config.label_summary_csv,
            append=self.config.append_csv,
        )

    def table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        result = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name=?
            """,
            (table_name,),
        ).fetchone()
        return result is not None

    def add_missing_columns(
        self,
        conn: sqlite3.Connection,
        df: pd.DataFrame,
        table_name: str,
    ) -> None:
        existing_cols = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }

        for col in df.columns:
            if col not in existing_cols:
                conn.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" TEXT')

    def create_indexes_if_possible(
        self,
        conn: sqlite3.Connection,
        table_name: str,
    ) -> None:
        if table_name != "predictions":
            return

        cols = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }

        index_candidates = {
            "timestamp": "idx_predictions_timestamp",
            "raw_file": "idx_predictions_raw_file",
            "predicted_label": "idx_predictions_label",
            "stored_at": "idx_predictions_stored_at",
            "source_file": "idx_predictions_source_file",
            "batch_id": "idx_predictions_batch_id",
        }

        for col, index_name in index_candidates.items():
            if col in cols:
                conn.execute(
                    f'CREATE INDEX IF NOT EXISTS "{index_name}" '
                    f'ON "{table_name}" ("{col}")'
                )

    def save_dataframe_sqlite(
        self,
        df: pd.DataFrame,
        table_name: str,
    ) -> None:
        if df.empty:
            return

        safe_df = make_dataframe_storage_safe(df)

        with self.connect() as conn:
            if self.table_exists(conn, table_name):
                self.add_missing_columns(conn, safe_df, table_name)

            safe_df.to_sql(
                table_name,
                conn,
                if_exists="append",
                index=False,
            )

            self.create_indexes_if_possible(conn, table_name)
            conn.commit()

    def save_batch(
        self,
        predictions: pd.DataFrame,
        source_file: str | None = None,
        batch_id: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        predictions = predictions.copy()
        processed_at = pd.Timestamp.utcnow().isoformat()

        if "model_name" not in predictions.columns:
            predictions["model_name"] = None

        if "batch_id" not in predictions.columns:
            predictions["batch_id"] = batch_id

        if "source_file" not in predictions.columns:
            predictions["source_file"] = source_file

        if "stored_at" not in predictions.columns:
            predictions["stored_at"] = processed_at

        file_summary = summarize_file(
            predictions=predictions,
            source_file=source_file,
            batch_id=batch_id,
        )

        label_summary = summarize_labels(
            predictions=predictions,
            source_file=source_file,
            batch_id=batch_id,
        )

        if self.config.backend in ("csv", "both"):
            self.save_predictions_csv(predictions)
            self.save_file_summary_csv(file_summary)
            self.save_label_summary_csv(label_summary)

        if self.config.backend in ("sqlite", "both"):
            self.save_dataframe_sqlite(predictions, "predictions")
            self.save_dataframe_sqlite(file_summary, "file_summary")
            self.save_dataframe_sqlite(label_summary, "label_summary")
            self.prune_sqlite_predictions()

        logger.info(
            "Stored prediction batch: batch_id=%s source_file=%s n=%d",
            batch_id,
            source_file,
            len(predictions),
        )

        return {
            "predictions": predictions,
            "file_summary": file_summary,
            "label_summary": label_summary,
        }

    def read_predictions(self, limit: int | None = 1000) -> pd.DataFrame:
        query = """
        SELECT *
        FROM predictions
        ORDER BY stored_at DESC
        """

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with self.connect() as conn:
            return pd.read_sql_query(query, conn)

    def read_file_summary(self, limit: int | None = 1000) -> pd.DataFrame:
        query = "SELECT * FROM file_summary"

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with self.connect() as conn:
            return pd.read_sql_query(query, conn)

    def read_label_summary(self, limit: int | None = 1000) -> pd.DataFrame:
        query = "SELECT * FROM label_summary"

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with self.connect() as conn:
            return pd.read_sql_query(query, conn)

    def prune_sqlite_predictions(self) -> None:
        with self.connect() as conn:
            if not self.table_exists(conn, "predictions"):
                return

            conn.execute(
                """
                DELETE FROM predictions
                WHERE rowid NOT IN (
                    SELECT rowid
                    FROM predictions
                    ORDER BY rowid DESC
                    LIMIT ?
                )
                """,
                (MAX_SQLITE_PREDICTION_ROWS,),
            )
            conn.commit()


def create_prediction_store(
    output_dir: str | Path = "results/realtime",
    backend: StorageBackend = "sqlite",
) -> PredictionStore:
    return PredictionStore(
        PredictionStoreConfig(
            output_dir=Path(output_dir),
            backend=backend,
        )
    )


__all__ = [
    "PredictionStoreConfig",
    "PredictionStore",
    "create_prediction_store",
    "summarize_file",
    "summarize_labels",
]