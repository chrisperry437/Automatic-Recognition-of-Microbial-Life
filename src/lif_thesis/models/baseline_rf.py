"""
Baseline Random Forest classifier utilities for LIF thesis experiments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
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


@dataclass(slots=True)
class BaselineRFConfig:
    n_estimators: int = 500
    max_depth: int | None = None
    criterion: str = "gini"
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | int | float = "sqrt"
    class_weight: str | dict | None = "balanced"
    n_jobs: int = -1
    random_state: int = 42


def make_baseline_rf(
    n_estimators: int = 500,
    max_depth: int | None = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    config = BaselineRFConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    return build_baseline_rf(config)


def build_baseline_rf(config: BaselineRFConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        criterion=config.criterion,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        class_weight=config.class_weight,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )


def _to_array(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.asarray(x)
    return np.asarray(x)


def build_spectra_matrix(
    df: pd.DataFrame,
    spectra_col: str = "spectrometer",
) -> np.ndarray:
    if spectra_col not in df.columns:
        raise ValueError(f"{spectra_col} not found in dataframe.")

    valid_mask = df[spectra_col].notna()
    if not valid_mask.all():
        raise ValueError(
            f"{(~valid_mask).sum()} rows have missing {spectra_col} values."
        )

    X = np.stack(df[spectra_col].apply(_to_array).values)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {X.shape}")

    return X


def train_baseline_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: BaselineRFConfig | None = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    config = config or BaselineRFConfig(random_state=random_state)

    model = build_baseline_rf(config)
    model.fit(X_train, y_train)

    return model


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


def run_baseline_rf_experiment(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "raw_file",
    spectra_col: str = "spectrometer",
    output_dir: str | Path = "results/baseline_rf",
    config: BaselineRFConfig | None = None,
    random_state: int = 42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = config or BaselineRFConfig(random_state=random_state)

    required_cols = [label_col, group_col, spectra_col]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    df = df[
        df[label_col].notna()
        & df[group_col].notna()
        & df[spectra_col].notna()
    ].reset_index(drop=True)

    print(f"Using {len(df)} valid particle rows.")
    print(f"Number of labels: {df[label_col].nunique()}")
    print(f"Number of groups: {df[group_col].nunique()}")

    X = build_spectra_matrix(df, spectra_col=spectra_col)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col].astype(str))

    train_idx, val_idx, test_idx = make_group_split(
        df,
        label_col=label_col,
        group_col=group_col,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        stratify=True,
        random_state=config.random_state,
        verbose=True,
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = train_baseline_rf(
        X_train,
        y_train,
        config=config,
    )

    metrics = {
        "config": asdict(config),
        "train": evaluate_classifier(model, X_train, y_train, label_encoder, "train"),
        "val": evaluate_classifier(model, X_val, y_val, label_encoder, "val"),
        "test": evaluate_classifier(model, X_test, y_test, label_encoder, "test"),
    }

    joblib.dump(model, output_dir / "baseline_rf.joblib")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "val_idx.npy", val_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    print(f"\nSaved outputs to: {output_dir}")

    return model, label_encoder, metrics, (train_idx, val_idx, test_idx)


__all__ = [
    "BaselineRFConfig",
    "make_baseline_rf",
    "build_baseline_rf",
    "build_spectra_matrix",
    "train_baseline_rf",
    "evaluate_classifier",
    "run_baseline_rf_experiment",
]