"""
Experiment 03: Hyperparameter search contribution.

Compares:
1. Simple RF with fixed parameters
2. Tuned RF using grouped cross-validation

Uses the selected improved feature set:
- spectrometer
- lifetime
- size
- time_asymmetry
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dataclasses import asdict, dataclass

from lif_thesis.data.splits import make_group_split
from lif_thesis.models.baseline_rf import make_baseline_rf
from lif_thesis.models.checkpoint import save_model_bundle
from lif_thesis.models.updated_rf import (
    UpdatedRFConfig,
    build_multimodal_feature_matrix,
    evaluate_classifier,
    train_updated_rf,
)


DATA_PATH = Path("data/processed/bacterial_samples.parquet")
OUTPUT_DIR = Path("results/exp03_hyperparameter_search")
MODEL_ID = "exp03_tuned_rf_v1"


def to_array(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.asarray(x)
    return np.asarray(x)


def peak_fluorescence(spectrum) -> float:
    return float(np.max(to_array(spectrum)))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_col = "species"
    group_col = "raw_file"
    scalar_cols = ["size", "time_asymmetry"]
    fluorescence_threshold = 2000.0
    random_state = 42

    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    required_cols = [
        label_col,
        group_col,
        "spectrometer",
        "lifetime",
        *scalar_cols,
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[
        df[label_col].notna()
        & df[group_col].notna()
        & df["spectrometer"].notna()
        & df["lifetime"].notna()
        & df["size"].notna()
        & df["time_asymmetry"].notna()
    ].reset_index(drop=True)

    print(f"Particles before fluorescence threshold: {len(df)}")

    df["peak_fluorescence"] = df["spectrometer"].apply(peak_fluorescence)
    df = df[df["peak_fluorescence"] > fluorescence_threshold].reset_index(drop=True)

    print(f"Particles after fluorescence > {fluorescence_threshold}: {len(df)}")
    print("\nClass counts:")
    print(df[label_col].value_counts())

    print("\nRaw files per class:")
    print(df.groupby(label_col)[group_col].nunique())

    X, feature_names = build_multimodal_feature_matrix(
        df,
        spectra_col="spectrometer",
        lifetime_col="lifetime",
        scalar_cols=scalar_cols,
        include_spectrometer=True,
        include_lifetime=True,
        include_scalars=True,
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col].astype(str))

    train_idx, val_idx, test_idx = make_group_split(
        df,
        label_col=label_col,
        group_col=group_col,
        train_size=0.60,
        val_size=0.20,
        test_size=0.20,
        stratify=True,
        random_state=random_state,
        verbose=True,
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    groups_train = df.iloc[train_idx][group_col].values

    print("\nTraining simple baseline RF...")
    simple_model = make_baseline_rf(
        n_estimators=500,
        max_depth=None,
        random_state=random_state,
    )
    simple_model.fit(X_train, y_train)

    print("\nTraining tuned RF with grouped CV...")
    tuning_config = UpdatedRFConfig(
        n_estimators_grid=(300, 500, 800),
        max_depth_grid=(None, 20, 40, 60),
        min_samples_split_grid=(2, 5, 10),
        min_samples_leaf_grid=(1, 2, 4),
        max_features_grid=("sqrt", "log2"),
        random_state=random_state,
        cv_splits=3,
        scoring="balanced_accuracy",
    )

    tuned_model, tuning_summary = train_updated_rf(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        config=tuning_config,
    )

    simple_metrics = {
        "train": evaluate_classifier(simple_model, X_train, y_train, label_encoder, "train"),
        "val": evaluate_classifier(simple_model, X_val, y_val, label_encoder, "val"),
        "test": evaluate_classifier(simple_model, X_test, y_test, label_encoder, "test"),
    }

    tuned_metrics = {
        "train": evaluate_classifier(tuned_model, X_train, y_train, label_encoder, "train"),
        "val": evaluate_classifier(tuned_model, X_val, y_val, label_encoder, "val"),
        "test": evaluate_classifier(tuned_model, X_test, y_test, label_encoder, "test"),
    }

    comparison = {
        "experiment": {
            "name": "exp03_hyperparameter_search",
            "model_id": MODEL_ID,
            "label_col": label_col,
            "group_col": group_col,
            "features": ["spectrometer", "lifetime", *scalar_cols],
            "fluorescence_threshold": fluorescence_threshold,
            "split_protocol": "grouped_raw_file_60_train_20_val_20_test",
            "tuning_protocol": "StratifiedGroupKFold on training set",
            "primary_question": "How much gain comes from hyperparameter tuning?",
        },
        "feature_names": feature_names,
        "tuning_config": asdict(tuning_config),
        "tuning_summary": tuning_summary,
        "simple_rf": simple_metrics,
        "tuned_rf": tuned_metrics,
        "test_improvement": {
            "accuracy_delta": tuned_metrics["test"]["accuracy"]
            - simple_metrics["test"]["accuracy"],
            "balanced_accuracy_delta": tuned_metrics["test"]["balanced_accuracy"]
            - simple_metrics["test"]["balanced_accuracy"],
            "macro_f1_delta": tuned_metrics["test"]["macro_f1"]
            - simple_metrics["test"]["macro_f1"],
        },
    }

    label_mapping = {
        int(i): str(label)
        for i, label in enumerate(label_encoder.classes_)
    }

    checkpoint = {
        "model_id": MODEL_ID,
        "model_type": "sklearn_random_forest",
        "framework": "sklearn",
        "model_module": "lif_thesis.models.updated_rf",
        "class_names": label_encoder.classes_.astype(str).tolist(),
        "feature_names": feature_names,
        "feature_columns": {
            "spectra_col": "spectrometer",
            "lifetime_col": "lifetime",
            "scalar_cols": scalar_cols,
        },
        "fluorescence_threshold": fluorescence_threshold,
        "tuning_config": asdict(tuning_config),
        "tuning_summary": tuning_summary,
    }

    metadata = {
        "model_id": MODEL_ID,
        "display_name": "Exp03 Tuned RF",
        "model_type": "sklearn_random_forest",
        "framework": "sklearn",
        "model_file": "model.joblib",
        "labels_file": "labels.json",
        "label_encoder_file": "label_encoder.joblib",
        "feature_names_file": "feature_names.json",
        "preprocessing": {
            "fluorescence_threshold": fluorescence_threshold,
            "uses_spectrometer": True,
            "uses_lifetime": True,
            "uses_scalars": True,
            "scalar_cols": scalar_cols,
        },
        "performance": {
            "test_accuracy": tuned_metrics["test"]["accuracy"],
            "test_balanced_accuracy": tuned_metrics["test"]["balanced_accuracy"],
            "test_macro_f1": tuned_metrics["test"]["macro_f1"],
            "simple_test_balanced_accuracy": simple_metrics["test"]["balanced_accuracy"],
            "balanced_accuracy_delta": comparison["test_improvement"][
                "balanced_accuracy_delta"
            ],
        },
    }

    save_model_bundle(
        model=tuned_model,
        model_id=MODEL_ID,
        deploy_dir=Path("models/trained"),
        checkpoint=checkpoint,
        metadata=metadata,
        label_mapping=label_mapping,
        label_encoder=label_encoder,
        framework="sklearn",
        extra_artifacts={
            "feature_names.json": feature_names,
        },
    )

    with open(OUTPUT_DIR / "metrics_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)

    with open(OUTPUT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=4)

    np.save(OUTPUT_DIR / "train_idx.npy", train_idx)
    np.save(OUTPUT_DIR / "val_idx.npy", val_idx)
    np.save(OUTPUT_DIR / "test_idx.npy", test_idx)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")
    print(f"Saved tuned model bundle to: models/trained/{MODEL_ID}")

    print("\nSimple RF test performance:")
    print(
        json.dumps(
            {
                "accuracy": simple_metrics["test"]["accuracy"],
                "balanced_accuracy": simple_metrics["test"]["balanced_accuracy"],
                "macro_f1": simple_metrics["test"]["macro_f1"],
            },
            indent=4,
        )
    )

    print("\nTuned RF test performance:")
    print(
        json.dumps(
            {
                "accuracy": tuned_metrics["test"]["accuracy"],
                "balanced_accuracy": tuned_metrics["test"]["balanced_accuracy"],
                "macro_f1": tuned_metrics["test"]["macro_f1"],
            },
            indent=4,
        )
    )

    print("\nTuning improvement:")
    print(json.dumps(comparison["test_improvement"], indent=4))


if __name__ == "__main__":
    main()