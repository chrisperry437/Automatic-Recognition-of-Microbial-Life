"""
Experiment 00: Paper-style Random Forest reproduction.

This script follows the original paper protocol as closely as possible:

1. Particle-level classification
2. Filter particles with peak fluorescence intensity > 2000 a.u.
3. Use particle-level train/test/validation split:
   - 60% training
   - 20% test for hyperparameter tuning
   - 20% validation for final evaluation
4. Tune Random Forest over:
   - n_estimators
   - max_depth
   - feature combinations
5. Evaluate using balanced accuracy and related metrics

Important:
This is intentionally NOT leakage-safe by raw_file.
It is designed to reproduce the paper-style methodology.

"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from lif_thesis.models.baseline_rf import make_baseline_rf
from lif_thesis.models.checkpoint import save_model_bundle


DATA_PATH = Path("data/processed/bacterial_samples.parquet")
OUTPUT_DIR = Path("results/exp00_paper_reproduction")
MODEL_ID = "exp00_paper_rf_v1"


def to_array(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.asarray(x)
    return np.asarray(x)


def peak_fluorescence(spectrum) -> float:
    return float(np.max(to_array(spectrum)))


def stack_vector_column(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack(df[col].apply(to_array).values)


def crop_pad_scattering(scattering, n_acquisitions: int = 60, n_angles: int = 24) -> np.ndarray:
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


def build_feature_matrix(df: pd.DataFrame, feature_set: str) -> tuple[np.ndarray, list[str]]:
    parts = []
    names = []

    if "spectra" in feature_set:
        X_spec = stack_vector_column(df, "spectrometer")
        parts.append(X_spec)
        names.extend([f"spectrometer_{i}" for i in range(X_spec.shape[1])])

    if "lifetime" in feature_set:
        X_life = stack_vector_column(df, "lifetime")
        parts.append(X_life)
        names.extend([f"lifetime_{i}" for i in range(X_life.shape[1])])

    if "scattering" in feature_set:
        X_scat = np.stack(df["scattering_image"].apply(crop_pad_scattering).values)
        parts.append(X_scat)
        names.extend([f"scattering_{i}" for i in range(X_scat.shape[1])])

    if not parts:
        raise ValueError(f"No valid features selected for {feature_set}")

    X = np.concatenate(parts, axis=1)

    if not np.isfinite(X).all():
        raise ValueError(f"Feature matrix for {feature_set} contains NaN/Inf.")

    return X, names


def make_paper_particle_split(df: pd.DataFrame, label_col: str, random_state: int = 42):
    indices = np.arange(len(df))
    y = df[label_col].values

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=0.60,
        stratify=y,
        random_state=random_state,
    )

    test_idx, val_idx = train_test_split(
        temp_idx,
        train_size=0.50,
        stratify=y[temp_idx],
        random_state=random_state,
    )

    return train_idx, test_idx, val_idx


def evaluate_model(model, X: np.ndarray, y: np.ndarray, label_encoder: LabelEncoder, split_name: str) -> dict:
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_col = "species"
    fluorescence_threshold = 2000.0
    random_state = 42

    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    required_cols = [
        label_col,
        "spectrometer",
        "lifetime",
        "scattering_image",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[
        df[label_col].notna()
        & df["spectrometer"].notna()
        & df["lifetime"].notna()
        & df["scattering_image"].notna()
    ].reset_index(drop=True)

    print(f"Particles before fluorescence threshold: {len(df)}")

    df["peak_fluorescence"] = df["spectrometer"].apply(peak_fluorescence)
    df = df[df["peak_fluorescence"] > fluorescence_threshold].reset_index(drop=True)

    print(f"Particles after fluorescence > {fluorescence_threshold}: {len(df)}")
    print(df[label_col].value_counts())

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col].astype(str))

    train_idx, test_idx, val_idx = make_paper_particle_split(
        df,
        label_col=label_col,
        random_state=random_state,
    )

    feature_sets = [
        "spectra",
        "lifetime",
        "scattering",
        "spectra+lifetime",
        "spectra+scattering",
        "lifetime+scattering",
        "spectra+lifetime+scattering",
    ]

    n_estimators_grid = [100, 300, 500]
    max_depth_grid = [None, 20, 40]

    tuning_results = []
    best_score = -np.inf
    best_model = None
    best_config = None
    best_feature_names = None

    print("\nStarting paper-style RF tuning...")

    for feature_set, n_estimators, max_depth in product(
        feature_sets,
        n_estimators_grid,
        max_depth_grid,
    ):
        print(
            f"\nTraining RF | features={feature_set} | "
            f"n_estimators={n_estimators} | max_depth={max_depth}"
        )

        X, feature_names = build_feature_matrix(df, feature_set)

        model = make_baseline_rf(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

        model.fit(X[train_idx], y[train_idx])

        y_test_pred = model.predict(X[test_idx])

        test_accuracy = accuracy_score(y[test_idx], y_test_pred)
        test_balanced_accuracy = balanced_accuracy_score(y[test_idx], y_test_pred)
        test_macro_f1 = f1_score(
            y[test_idx],
            y_test_pred,
            average="macro",
            zero_division=0,
        )

        result = {
            "feature_set": feature_set,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "test_accuracy": test_accuracy,
            "test_balanced_accuracy": test_balanced_accuracy,
            "test_macro_f1": test_macro_f1,
            "n_features": X.shape[1],
        }

        tuning_results.append(result)

        print(
            f"Test accuracy={test_accuracy:.4f} | "
            f"Test balanced accuracy={test_balanced_accuracy:.4f} | "
            f"Test macro F1={test_macro_f1:.4f}"
        )

        if test_balanced_accuracy > best_score:
            best_score = test_balanced_accuracy
            best_model = model
            best_config = result
            best_feature_names = feature_names

    if best_model is None or best_config is None or best_feature_names is None:
        raise RuntimeError("No model was selected during tuning.")

    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)

    best_feature_set = best_config["feature_set"]
    X_best, _ = build_feature_matrix(df, best_feature_set)

    metrics = {
        "experiment": {
            "name": "exp00_paper_reproduction",
            "model_id": MODEL_ID,
            "model_type": "RandomForestClassifier",
            "model_module": "lif_thesis.models.baseline_rf",
            "label_col": label_col,
            "fluorescence_threshold": fluorescence_threshold,
            "split_protocol": "particle_level_60_train_20_test_tuning_20_validation_final",
            "note": "Paper-style particle-level split; not grouped by raw_file.",
        },
        "best_config": best_config,
        "train": evaluate_model(
            best_model,
            X_best[train_idx],
            y[train_idx],
            label_encoder,
            "train",
        ),
        "test_tuning": evaluate_model(
            best_model,
            X_best[test_idx],
            y[test_idx],
            label_encoder,
            "test_tuning",
        ),
        "validation_final": evaluate_model(
            best_model,
            X_best[val_idx],
            y[val_idx],
            label_encoder,
            "validation_final",
        ),
    }

    label_mapping = {
        int(i): str(label)
        for i, label in enumerate(label_encoder.classes_)
    }

    checkpoint = {
        "model_id": MODEL_ID,
        "model_type": "sklearn_random_forest",
        "model_module": "lif_thesis.models.baseline_rf",
        "class_names": label_encoder.classes_.astype(str).tolist(),
        "feature_set": best_config["feature_set"],
        "feature_names": best_feature_names,
        "fluorescence_threshold": fluorescence_threshold,
        "best_config": best_config,
        "validation_accuracy": metrics["validation_final"]["accuracy"],
        "validation_balanced_accuracy": metrics["validation_final"]["balanced_accuracy"],
        "validation_macro_f1": metrics["validation_final"]["macro_f1"],
    }

    metadata = {
        "model_id": MODEL_ID,
        "display_name": "Exp00 Paper RF Baseline",
        "model_type": "sklearn_random_forest",
        "model_file": "model.joblib",
        "labels_file": "labels.json",
        "label_encoder_file": "label_encoder.joblib",
        "feature_names_file": "feature_names.json",
        "framework": "sklearn",
        "task": "multiclass_species_classification",
        "feature_set": best_config["feature_set"],
        "preprocessing": {
            "fluorescence_threshold": fluorescence_threshold,
            "scattering_target_acquisitions": 60,
            "n_scattering_angles": 24,
            "scattering_normalize": True,
        },
        "performance": {
            "validation_accuracy": metrics["validation_final"]["accuracy"],
            "validation_balanced_accuracy": metrics["validation_final"]["balanced_accuracy"],
            "validation_macro_f1": metrics["validation_final"]["macro_f1"],
        },
        "notes": "Paper-style particle-level split; not grouped by raw_file.",
    }

    save_model_bundle(
        model=best_model,
        model_id=MODEL_ID,
        deploy_dir=Path("models/trained"),
        checkpoint=checkpoint,
        metadata=metadata,
        label_mapping=label_mapping,
        label_encoder=label_encoder,
        framework="sklearn",
        extra_artifacts={
            "feature_names.json": best_feature_names,
        },
    )

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(OUTPUT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(best_feature_names, f, indent=4)

    np.save(OUTPUT_DIR / "train_idx.npy", train_idx)
    np.save(OUTPUT_DIR / "test_tuning_idx.npy", test_idx)
    np.save(OUTPUT_DIR / "validation_final_idx.npy", val_idx)

    print(f"\nSaved experiment outputs to: {OUTPUT_DIR}")
    print(f"Saved model bundle to: models/trained/{MODEL_ID}")

    print("\nFinal validation performance:")
    print(
        json.dumps(
            {
                "accuracy": metrics["validation_final"]["accuracy"],
                "balanced_accuracy": metrics["validation_final"]["balanced_accuracy"],
                "macro_f1": metrics["validation_final"]["macro_f1"],
            },
            indent=4,
        )
    )


if __name__ == "__main__":
    main()