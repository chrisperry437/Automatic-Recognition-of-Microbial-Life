"""
paper_rf.py

Reusable paper-style Random Forest utilities for Exp01.

This model keeps the original paper-style feature-combination search:
- spectra
- lifetime
- scattering
- spectra + lifetime
- spectra + scattering
- lifetime + scattering
- spectra + lifetime + scattering

It is intended for grouped raw-file evaluation in Exp01.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

from lif_thesis.data.splits import make_group_split
from lif_thesis.models.checkpoint import save_model_bundle


@dataclass(slots=True)
class PaperRFConfig:
    feature_sets: tuple[str, ...] = (
        "spectra",
        "lifetime",
        "scattering",
        "spectra+lifetime",
        "spectra+scattering",
        "lifetime+scattering",
        "spectra+lifetime+scattering",
    )
    n_estimators_grid: tuple[int, ...] = (100, 300, 500)
    max_depth_grid: tuple[int | None, ...] = (None, 20, 40)

    class_weight: str = "balanced"
    max_features: str = "sqrt"
    n_jobs: int = -1
    random_state: int = 42

    fluorescence_threshold: float = 2000.0
    scattering_target_acquisitions: int = 60
    n_scattering_angles: int = 24

    train_size: float = 0.60
    val_size: float = 0.20
    test_size: float = 0.20


def to_array(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.asarray(x)
    return np.asarray(x)


def peak_fluorescence(spectrum) -> float:
    arr = to_array(spectrum)
    return float(np.max(arr))


def stack_vector_column(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise ValueError(f"{col} not found in dataframe.")

    if df[col].isna().any():
        raise ValueError(f"{df[col].isna().sum()} rows have missing {col} values.")

    return np.stack(df[col].apply(to_array).values)


def crop_pad_scattering(
    scattering,
    n_acquisitions: int = 60,
    n_angles: int = 24,
) -> np.ndarray:
    target_len = n_acquisitions * n_angles

    arr = to_array(scattering).astype(float).flatten()

    if len(arr) >= target_len:
        arr = arr[:target_len]
    else:
        arr = np.pad(arr, (0, target_len - len(arr)), mode="constant")

    max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val

    return arr


def build_paper_feature_matrix(
    df: pd.DataFrame,
    feature_set: str,
    spectrometer_col: str = "spectrometer",
    lifetime_col: str = "lifetime",
    scattering_col: str = "scattering_image",
    config: PaperRFConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    config = config or PaperRFConfig()

    parts = []
    names = []

    if "spectra" in feature_set:
        X_spec = stack_vector_column(df, spectrometer_col)
        parts.append(X_spec)
        names.extend([f"{spectrometer_col}_{i}" for i in range(X_spec.shape[1])])

    if "lifetime" in feature_set:
        X_life = stack_vector_column(df, lifetime_col)
        parts.append(X_life)
        names.extend([f"{lifetime_col}_{i}" for i in range(X_life.shape[1])])

    if "scattering" in feature_set:
        if scattering_col not in df.columns:
            raise ValueError(f"{scattering_col} not found in dataframe.")

        if df[scattering_col].isna().any():
            raise ValueError(
                f"{df[scattering_col].isna().sum()} rows have missing "
                f"{scattering_col} values."
            )

        X_scat = np.stack(
            df[scattering_col]
            .apply(
                lambda x: crop_pad_scattering(
                    x,
                    n_acquisitions=config.scattering_target_acquisitions,
                    n_angles=config.n_scattering_angles,
                )
            )
            .values
        )
        parts.append(X_scat)
        names.extend([f"scattering_{i}" for i in range(X_scat.shape[1])])

    if not parts:
        raise ValueError(f"No valid features selected for {feature_set}")

    X = np.concatenate(parts, axis=1)

    if not np.isfinite(X).all():
        raise ValueError(f"Feature matrix for {feature_set} contains NaN/Inf.")

    return X, names


def make_paper_rf(
    n_estimators: int = 100,
    max_depth: int | None = None,
    config: PaperRFConfig | None = None,
) -> RandomForestClassifier:
    config = config or PaperRFConfig()

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=config.class_weight,
        max_features=config.max_features,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )


def evaluate_classifier(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    split_name: str,
) -> dict:
    y_pred = model.predict(X)

    labels = np.arange(len(label_encoder.classes_))
    target_names = label_encoder.classes_.astype(str)

    return {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def prepare_paper_dataframe(
    df: pd.DataFrame,
    *,
    label_col: str = "species",
    group_col: str = "raw_file",
    spectrometer_col: str = "spectrometer",
    lifetime_col: str = "lifetime",
    scattering_col: str = "scattering_image",
    config: PaperRFConfig | None = None,
) -> pd.DataFrame:
    config = config or PaperRFConfig()

    required_cols = [
        label_col,
        group_col,
        spectrometer_col,
        lifetime_col,
        scattering_col,
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[
        df[label_col].notna()
        & df[group_col].notna()
        & df[spectrometer_col].notna()
        & df[lifetime_col].notna()
        & df[scattering_col].notna()
    ].copy()

    df = df.reset_index(drop=True)

    df["peak_fluorescence"] = df[spectrometer_col].apply(peak_fluorescence)
    df = df[df["peak_fluorescence"] > config.fluorescence_threshold].reset_index(
        drop=True
    )

    if df.empty:
        raise ValueError("No particles remain after fluorescence thresholding.")

    return df


def run_paper_rf_grouped_experiment(
    df: pd.DataFrame,
    *,
    model_id: str,
    display_name: str,
    label_col: str = "species",
    group_col: str = "raw_file",
    spectrometer_col: str = "spectrometer",
    lifetime_col: str = "lifetime",
    scattering_col: str = "scattering_image",
    output_dir: str | Path = "results/paper_rf_grouped",
    deploy_dir: str | Path = "models/trained",
    config: PaperRFConfig | None = None,
):
    config = config or PaperRFConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_paper_dataframe(
        df,
        label_col=label_col,
        group_col=group_col,
        spectrometer_col=spectrometer_col,
        lifetime_col=lifetime_col,
        scattering_col=scattering_col,
        config=config,
    )

    print(f"Particles after fluorescence > {config.fluorescence_threshold}: {len(df)}")
    print("\nClass counts after filtering:")
    print(df[label_col].value_counts())

    print("\nRaw files per class after filtering:")
    print(df.groupby(label_col)[group_col].nunique())

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col].astype(str))

    train_idx, val_idx, test_idx = make_group_split(
        df,
        label_col=label_col,
        group_col=group_col,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        stratify=True,
        random_state=config.random_state,
        verbose=True,
    )

    tuning_results = []
    best_score = -np.inf
    best_model = None
    best_config = None
    best_feature_names = None
    best_X = None

    print("\nStarting grouped paper-style RF tuning...")

    for feature_set, n_estimators, max_depth in product(
        config.feature_sets,
        config.n_estimators_grid,
        config.max_depth_grid,
    ):
        print(
            f"\nTraining RF | features={feature_set} | "
            f"n_estimators={n_estimators} | max_depth={max_depth}"
        )

        X, feature_names = build_paper_feature_matrix(
            df,
            feature_set=feature_set,
            spectrometer_col=spectrometer_col,
            lifetime_col=lifetime_col,
            scattering_col=scattering_col,
            config=config,
        )

        model = make_paper_rf(
            n_estimators=n_estimators,
            max_depth=max_depth,
            config=config,
        )

        model.fit(X[train_idx], y[train_idx])

        y_val_pred = model.predict(X[val_idx])

        val_accuracy = accuracy_score(y[val_idx], y_val_pred)
        val_balanced_accuracy = balanced_accuracy_score(y[val_idx], y_val_pred)
        val_macro_f1 = f1_score(
            y[val_idx],
            y_val_pred,
            average="macro",
            zero_division=0,
        )

        result = {
            "feature_set": feature_set,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "val_accuracy": val_accuracy,
            "val_balanced_accuracy": val_balanced_accuracy,
            "val_macro_f1": val_macro_f1,
            "n_features": X.shape[1],
        }

        tuning_results.append(result)

        print(
            f"Val accuracy={val_accuracy:.4f} | "
            f"Val balanced accuracy={val_balanced_accuracy:.4f} | "
            f"Val macro F1={val_macro_f1:.4f}"
        )

        if val_balanced_accuracy > best_score:
            best_score = val_balanced_accuracy
            best_model = model
            best_config = result
            best_feature_names = feature_names
            best_X = X

    if (
        best_model is None
        or best_config is None
        or best_feature_names is None
        or best_X is None
    ):
        raise RuntimeError("No best paper RF model was selected.")

    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(output_dir / "tuning_results.csv", index=False)

    metrics = {
        "experiment": {
            "model_id": model_id,
            "display_name": display_name,
            "model_type": "sklearn_random_forest",
            "model_module": "lif_thesis.models.paper_rf",
            "label_col": label_col,
            "group_col": group_col,
            "fluorescence_threshold": config.fluorescence_threshold,
            "split_protocol": "grouped_raw_file_60_train_20_val_20_test",
            "note": (
                "Uses paper-style filtering and feature combinations, "
                "but applies a grouped raw-file split."
            ),
        },
        "config": asdict(config),
        "best_config": best_config,
        "train": evaluate_classifier(
            best_model,
            best_X[train_idx],
            y[train_idx],
            label_encoder,
            "train",
        ),
        "val_tuning": evaluate_classifier(
            best_model,
            best_X[val_idx],
            y[val_idx],
            label_encoder,
            "val_tuning",
        ),
        "test_final": evaluate_classifier(
            best_model,
            best_X[test_idx],
            y[test_idx],
            label_encoder,
            "test_final",
        ),
    }

    label_mapping = {
        int(i): str(label)
        for i, label in enumerate(label_encoder.classes_)
    }

    checkpoint = {
        "model_id": model_id,
        "model_type": "sklearn_random_forest",
        "framework": "sklearn",
        "model_module": "lif_thesis.models.paper_rf",
        "class_names": label_encoder.classes_.astype(str).tolist(),
        "feature_set": best_config["feature_set"],
        "feature_names": best_feature_names,
        "best_config": best_config,
        "config": asdict(config),
    }

    metadata = {
        "model_id": model_id,
        "display_name": display_name,
        "model_type": "sklearn_random_forest",
        "framework": "sklearn",
        "model_file": "model.joblib",
        "labels_file": "labels.json",
        "label_encoder_file": "label_encoder.joblib",
        "feature_names_file": "feature_names.json",
        "preprocessing": {
            "fluorescence_threshold": config.fluorescence_threshold,
            "scattering_target_acquisitions": config.scattering_target_acquisitions,
            "n_scattering_angles": config.n_scattering_angles,
            "scattering_normalize": True,
            "feature_set": best_config["feature_set"],
        },
        "performance": {
            "train_balanced_accuracy": metrics["train"]["balanced_accuracy"],
            "val_balanced_accuracy": metrics["val_tuning"]["balanced_accuracy"],
            "test_balanced_accuracy": metrics["test_final"]["balanced_accuracy"],
            "test_macro_f1": metrics["test_final"]["macro_f1"],
        },
    }

    save_model_bundle(
        model=best_model,
        model_id=model_id,
        deploy_dir=Path(deploy_dir),
        checkpoint=checkpoint,
        metadata=metadata,
        label_mapping=label_mapping,
        label_encoder=label_encoder,
        framework="sklearn",
        extra_artifacts={
            "feature_names.json": best_feature_names,
        },
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(best_feature_names, f, indent=4)

    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "val_tuning_idx.npy", val_idx)
    np.save(output_dir / "test_final_idx.npy", test_idx)

    print(f"\nSaved experiment outputs to: {output_dir}")
    print(f"Saved model bundle to: {Path(deploy_dir) / model_id}")

    return best_model, label_encoder, metrics, (train_idx, val_idx, test_idx)


__all__ = [
    "PaperRFConfig",
    "to_array",
    "peak_fluorescence",
    "crop_pad_scattering",
    "build_paper_feature_matrix",
    "make_paper_rf",
    "evaluate_classifier",
    "prepare_paper_dataframe",
    "run_paper_rf_grouped_experiment",
]