## Receives procesing particles and returns predictions

"""
Inference engine for real-time Rapid-E bioaerosol classification.

This module provides a higher-level interface around the particle predictor.
It is intended to be used by:

    - real-time stream processing
    - FastAPI endpoints
    - dashboard backend logic
    - offline replay of Rapid-E files

The engine handles:
    - model loading
    - particle-level prediction
    - rejected low-fluorescence tracking
    - prediction summaries
    - rolling mixture-style composition summaries
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from lif_thesis.data.schemas import RAPIDE_DIMS

import pandas as pd


CONFIG_PATH = Path("configs/active_model.json")
MODELS_DIR = Path("models/trained")


def get_active_model_dir() -> Path:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Active model config not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    active_model = config["active_model"]
    model_dir = MODELS_DIR / active_model

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Active model directory not found: {model_dir}"
        )

    return model_dir

from lif_thesis.data.preprocessing import (
    PreprocessingConfig,
    filter_by_fluorescence_threshold,
)
from lif_thesis.models.predict import (
    ParticlePredictor,
    PredictionConfig,
    summarize_predictions,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceEngineConfig:
    """Configuration for the real-time inference engine."""

    model_path: str | Path
    model_type: str = "sklearn"
    task: str = "species"

    label_mapping_path: str | Path | None = None
    feature_columns_path: str | Path | None = None

    confidence_threshold: float = 0.60
    unknown_label: str = "unknown"

    fluorescence_threshold: float = 2000.0
    print("\nQC DEBUG")
    print(f"Loaded particles: {len(df)}")
    print(f"Accepted > threshold: {len(accepted)}")
    print(f"Rejected <= threshold: {len(rejected)}")

    if "peak_fluorescence" in df.columns:
        print(df["peak_fluorescence"].describe())
    scattering_target_acquisitions: int = RAPIDE_DIMS.SCATTERING_TARGET_ACQUISITIONS
    scattering_normalize: bool = True

    rolling_windows: tuple[int, ...] = (1, 5, 10)


@dataclass
class InferenceOutput:
    """Structured inference result for one batch/file of particles."""

    predictions: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, Any]


class InferenceEngine:
    """
    Real-time inference engine for Rapid-E particle batches.

    The expected input is a particle-level dataframe with columns compatible with
    the preprocessing pipeline, typically including:

        - fluorescence_spectra
        - fluorescence_lifetime
        - scattering

    If a dataframe contains low-fluorescence particles, they are counted in
    metadata but not passed to the model.
    """

    def __init__(self, config: InferenceEngineConfig) -> None:
        self.config = config

        preprocessing_config = PreprocessingConfig(
            fluorescence_threshold=config.fluorescence_threshold,
            scattering_target_acquisitions=config.scattering_target_acquisitions,
            scattering_normalize=config.scattering_normalize,
        )

        prediction_config = PredictionConfig(
            model_path=config.model_path,
            model_type=config.model_type,  # type: ignore[arg-type]
            task=config.task,  # type: ignore[arg-type]
            label_mapping_path=config.label_mapping_path,
            feature_columns_path=config.feature_columns_path,
            confidence_threshold=config.confidence_threshold,
            unknown_label=config.unknown_label,
            preprocessing=preprocessing_config,
        )

        self.predictor = ParticlePredictor(prediction_config)

        logger.info("Inference engine initialized")
        logger.info("Model path: %s", config.model_path)
        logger.info("Model type: %s", config.model_type)

    def split_eligible_particles(
        self,
        particles: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split particles into model-eligible and rejected particles.

        A particle is model-eligible if it passes the fluorescence threshold.
        """
        accepted, rejected = filter_by_fluorescence_threshold(
            particles,
            threshold=self.config.fluorescence_threshold,
            keep_rejected=True,
        )

        return accepted, rejected

    def predict_batch(
        self,
        particles: pd.DataFrame,
        batch_id: str | None = None,
        source_file: str | None = None,
    ) -> InferenceOutput:
        """
        Predict labels for one batch of particles.

        Parameters
        ----------
        particles:
            Raw particle-level dataframe.
        batch_id:
            Optional batch identifier.
        source_file:
            Optional source file name.

        Returns
        -------
        InferenceOutput
            Predictions, summary, and metadata.
        """
        processed_at = pd.Timestamp.utcnow().isoformat()

        if particles.empty:
            metadata = {
                "batch_id": batch_id,
                "source_file": source_file,
                "processed_at": processed_at,
                "n_particles_total": 0,
                "n_particles_eligible": 0,
                "n_particles_rejected": 0,
                "fluorescence_threshold": self.config.fluorescence_threshold,
            }

            return InferenceOutput(
                predictions=pd.DataFrame(),
                summary=pd.DataFrame(),
                metadata=metadata,
            )

        accepted, rejected = self.split_eligible_particles(particles)

        metadata = {
            "batch_id": batch_id,
            "source_file": source_file,
            "processed_at": processed_at,
            "n_particles_total": int(len(particles)),
            "n_particles_eligible": int(len(accepted)),
            "n_particles_rejected": int(len(rejected)),
            "rejected_fraction": float(len(rejected) / len(particles))
            if len(particles)
            else 0.0,
            "fluorescence_threshold": self.config.fluorescence_threshold,
        }

        if accepted.empty:
            return InferenceOutput(
                predictions=pd.DataFrame(),
                summary=pd.DataFrame(),
                metadata=metadata,
            )

        predictions = self.predictor.predict_particles(accepted)

        predictions["batch_id"] = batch_id
        predictions["source_file"] = source_file
        predictions["processed_at"] = processed_at

        summary = summarize_predictions(predictions)

        summary["batch_id"] = batch_id
        summary["source_file"] = source_file
        summary["processed_at"] = processed_at
        summary["n_particles_total"] = metadata["n_particles_total"]
        summary["n_particles_eligible"] = metadata["n_particles_eligible"]
        summary["n_particles_rejected"] = metadata["n_particles_rejected"]

        return InferenceOutput(
            predictions=predictions,
            summary=summary,
            metadata=metadata,
        )

    def predict_one(
        self,
        particle: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Predict a single particle represented as a dictionary.

        Returns a JSON-serializable dictionary.
        """
        df = pd.DataFrame([particle])
        output = self.predict_batch(df)

        if output.predictions.empty:
            return {
                "predicted_label": self.config.unknown_label,
                "prediction_confidence": 0.0,
                "is_unknown": True,
                "reason": "Particle did not pass fluorescence threshold or could not be processed.",
                "metadata": output.metadata,
            }

        row = output.predictions.iloc[0]

        result = {
            "predicted_label": row.get("predicted_label"),
            "prediction_confidence": float(row.get("prediction_confidence", 0.0)),
            "is_unknown": bool(row.get("is_unknown", True)),
            "metadata": output.metadata,
        }

        probability_cols = [
            col for col in output.predictions.columns if col.startswith("proba_")
        ]

        result["probabilities"] = {
            col.replace("proba_", ""): float(row[col])
            for col in probability_cols
        }

        return result

    def estimate_composition(
        self,
        predictions: pd.DataFrame,
        label_col: str = "predicted_label",
        exclude_unknown: bool = False,
    ) -> pd.DataFrame:
        """
        Estimate batch-level composition from particle-level predictions.

        Parameters
        ----------
        predictions:
            Particle-level prediction dataframe.
        label_col:
            Column containing predicted labels.
        exclude_unknown:
            Whether to remove unknown predictions before computing proportions.

        Returns
        -------
        pd.DataFrame
            Label counts and proportions.
        """
        if predictions.empty or label_col not in predictions.columns:
            return pd.DataFrame(
                columns=["predicted_label", "count", "proportion"]
            )

        df = predictions.copy()

        if exclude_unknown:
            df = df[df[label_col] != self.config.unknown_label]

        if df.empty:
            return pd.DataFrame(
                columns=["predicted_label", "count", "proportion"]
            )

        counts = (
            df[label_col]
            .value_counts(dropna=False)
            .rename_axis("predicted_label")
            .reset_index(name="count")
        )

        counts["proportion"] = counts["count"] / counts["count"].sum()

        return counts

    def rolling_composition(
        self,
        predictions: pd.DataFrame,
        time_col: str = "processed_at",
        label_col: str = "predicted_label",
        window_minutes: int = 5,
    ) -> pd.DataFrame:
        """
        Compute rolling composition estimates over time.

        This is useful for dashboard views such as:
            - last 1 minute
            - last 5 minutes
            - last 10 minutes
        """
        if predictions.empty:
            return pd.DataFrame(
                columns=[
                    "window_minutes",
                    "predicted_label",
                    "count",
                    "proportion",
                ]
            )

        if time_col not in predictions.columns:
            return self.estimate_composition(predictions, label_col=label_col)

        df = predictions.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)

        latest_time = df[time_col].max()

        if pd.isna(latest_time):
            return self.estimate_composition(predictions, label_col=label_col)

        cutoff = latest_time - pd.Timedelta(minutes=window_minutes)

        recent = df[df[time_col] >= cutoff]

        composition = self.estimate_composition(recent, label_col=label_col)
        composition["window_minutes"] = window_minutes

        return composition

    def dashboard_summary(
        self,
        predictions: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Create a compact summary dictionary for dashboards or APIs.
        """
        if predictions.empty:
            return {
                "n_predictions": 0,
                "top_label": None,
                "top_label_proportion": 0.0,
                "mean_confidence": None,
                "unknown_fraction": None,
                "composition": [],
            }

        composition = self.estimate_composition(predictions)

        top = composition.iloc[0] if not composition.empty else None

        unknown_fraction = None
        if "is_unknown" in predictions.columns:
            unknown_fraction = float(predictions["is_unknown"].mean())

        mean_confidence = None
        if "prediction_confidence" in predictions.columns:
            mean_confidence = float(predictions["prediction_confidence"].mean())

        return {
            "n_predictions": int(len(predictions)),
            "top_label": str(top["predicted_label"]) if top is not None else None,
            "top_label_proportion": float(top["proportion"])
            if top is not None
            else 0.0,
            "mean_confidence": mean_confidence,
            "unknown_fraction": unknown_fraction,
            "composition": composition.to_dict(orient="records"),
        }


def create_inference_engine(
    model_path: str | Path,
    model_type: str = "sklearn",
    label_mapping_path: str | Path | None = None,
    feature_columns_path: str | Path | None = None,
    confidence_threshold: float = 0.60,
) -> InferenceEngine:
    """
    Convenience factory for creating an inference engine.
    """
    return InferenceEngine(
        InferenceEngineConfig(
            model_path=model_path,
            model_type=model_type,
            label_mapping_path=label_mapping_path,
            feature_columns_path=feature_columns_path,
            confidence_threshold=confidence_threshold,
        )
    )


__all__ = [
    "InferenceEngineConfig",
    "InferenceOutput",
    "InferenceEngine",
    "create_inference_engine",
]