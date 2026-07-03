"""
Updated Random Forest utilities for Exp01 and Exp02.

Supports:
- multimodal RF feature construction
- grouped-CV RF tuning
- model evaluation
- deployable model bundle saving via checkpoint.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from lif_thesis.data.splits import make_group_split
from lif_thesis.models.checkpoint import save_model_bundle


@dataclass(slots=True)
class UpdatedRFConfig:
    n_estimators_grid: tuple[int, ...] = (300, 500)
    max_depth_grid: tuple[int | None, ...] = (None, 20, 40)
    min_samples_split_grid: tuple[int, ...] = (2, 5)
    min_samples_leaf_grid: tuple[int, ...] = (1, 2, 4)
    max_features_grid: tuple[str, ...] = ("sqrt", "log2")

    class_weight: str = "balanced"
    criterion: str = "gini"
    n_jobs: int = -1
    random_state: int = 42
    cv_splits: int = 3
    scoring: str = "balanced_accuracy"


def make_updated_rf(
    n_estimators: int = 500,
    max_depth: int | None = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion="gini",
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )


def _to_array(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.asarray(x)
    return np.asarray(x)


def _safe_stack_vector_column(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise ValueError(f"{col} not found in dataframe.")

    if df[col].isna().any():
        raise ValueError(f"{df[col].isna().sum()} rows have missing {col} values.")

    X = np.stack(df[col].apply(_to_array).values)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix for {col}, got shape {X.shape}")

    return X


def build_spectra_matrix(
    df: pd.DataFrame,
    spectra_col: str = "spectrometer",
) -> np.ndarray:
    return _safe_stack_vector_column(df, spectra_col)


def build_multimodal_feature_matrix(
    df: pd.DataFrame,
    spectra_col: str = "spectrometer",
    lifetime_col: str = "lifetime",
    scalar_cols: list[str] | None = None,
    include_spectrometer: bool = True,
    include_lifetime: bool = True,
    include_scalars: bool = True,
) -> tuple[np.ndarray, list[str]]:
    if scalar_cols is None:
        scalar_cols = ["size", "time_asymmetry"]

    feature_parts = []
    feature_names = []

    if include_spectrometer:
        X_spec = _safe_stack_vector_column(df, spectra_col)
        feature_parts.append(X_spec)
        feature_names.extend([f"{spectra_col}_{i}" for i in range(X_spec.shape[1])])

    if include_lifetime:
        X_life = _safe_stack_vector_column(df, lifetime_col)
        feature_parts.append(X_life)
        feature_names.extend([f"{lifetime_col}_{i}" for i in range(X_life.shape[1])])

    if include_scalars:
        missing_scalars = [col for col in scalar_cols if col not in df.columns]
        if missing_scalars:
            raise ValueError(f"Missing scalar columns: {missing_scalars}")

        X_scalar = df[scalar_cols].copy()

        for col in scalar_cols:
            X_scalar[col] = pd.to_numeric(X_scalar[col], errors="coerce")

        if X_scalar.isna().any().any():
            missing_summary = X_scalar.isna().sum()
            raise ValueError(
                "Missing scalar values found:\n"
                f"{missing_summary[missing_summary > 0]}"
            )

        feature_parts.append(X_scalar.to_numpy(dtype=float))
        feature_names.extend(scalar_cols)

    if not feature_parts:
        raise ValueError("No feature groups selected.")

    X = np.concatenate(feature_parts, axis=1)

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")

    return X, feature_names


def train_updated_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    config: UpdatedRFConfig | None = None,
) -> tuple[RandomForestClassifier, dict]:
    config = config or UpdatedRFConfig()

    base_model = RandomForestClassifier(
        criterion=config.criterion,
        class_weight=config.class_weight,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )

    param_grid = {
        "n_estimators": list(config.n_estimators_grid),
        "max_depth": list(config.max_depth_grid),
        "min_samples_split": list(config.min_samples_split_grid),
        "min_samples_leaf": list(config.min_samples_leaf_grid),
        "max_features": list(config.max_features_grid),
    }

    cv = StratifiedGroupKFold(
        n_splits=config.cv_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=config.scoring,
        cv=cv,
        n_jobs=config.n_jobs,
        verbose=2,
        refit=True,
    )

    grid_search.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    tuning_summary = {
        "best_params": grid_search.best_params_,
        "best_cv_score": float(grid_search.best_score_),
        "scoring": config.scoring,
        "cv_splits": config.cv_splits,
    }

    return grid_search.best_estimator_, tuning_summary


def _safe_roc_auc(y: np.ndarray, y_proba: np.ndarray) -> float | None:
    try:
        if y_proba.shape[1] == 2:
            return float(roc_auc_score(y, y_proba[:, 1]))

        return float(
            roc_auc_score(
                y,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        return None


def _safe_pr_auc(y: np.ndarray, y_proba: np.ndarray) -> float | None:
    try:
        n_classes = y_proba.shape[1]

        if n_classes == 2:
            return float(average_precision_score(y, y_proba[:, 1]))

        y_onehot = np.zeros_like(y_proba)

        for cls in range(n_classes):
            y_onehot[:, cls] = (y == cls).astype(int)

        return float(
            average_precision_score(
                y_onehot,
                y_proba,
                average="macro",
            )
        )
    except Exception:
        return None


def evaluate_classifier(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    split_name: str,
) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    labels = np.arange(len(label_encoder.classes_))
    target_names = label_encoder.classes_.astype(str)

    return {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "roc_auc": _safe_roc_auc(y, y_proba),
        "pr_auc": _safe_pr_auc(y, y_proba),
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


def run_updated_rf_experiment(
    df: pd.DataFrame,
    *,
    model_id: str,
    display_name: str,
    label_col: str = "species",
    group_col: str = "raw_file",
    spectra_col: str = "spectrometer",
    lifetime_col: str = "lifetime",
    scalar_cols: list[str] | None = None,
    output_dir: str | Path = "results/updated_rf",
    deploy_dir: str | Path = "models/trained",
    config: UpdatedRFConfig | None = None,
    random_state: int = 42,
):
    if scalar_cols is None:
        scalar_cols = ["size", "time_asymmetry"]

    config = config or UpdatedRFConfig(random_state=random_state)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = [
        label_col,
        group_col,
        spectra_col,
        lifetime_col,
        *scalar_cols,
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    usable_mask = (
        df[label_col].notna()
        & df[group_col].notna()
        & df[spectra_col].notna()
        & df[lifetime_col].notna()
    )

    for col in scalar_cols:
        usable_mask &= df[col].notna()

    df = df[usable_mask].reset_index(drop=True)

    X, feature_names = build_multimodal_feature_matrix(
        df,
        spectra_col=spectra_col,
        lifetime_col=lifetime_col,
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

    groups_train = df.iloc[train_idx][group_col].values

    model, tuning_summary = train_updated_rf(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        config=config,
    )

    metrics = {
        "experiment": {
            "model_id": model_id,
            "display_name": display_name,
            "model_type": "sklearn_random_forest",
            "label_col": label_col,
            "group_col": group_col,
            "feature_columns": {
                "spectra_col": spectra_col,
                "lifetime_col": lifetime_col,
                "scalar_cols": scalar_cols,
            },
            "split_protocol": "grouped_raw_file_70_train_15_val_15_test",
        },
        "config": asdict(config),
        "tuning": tuning_summary,
        "train": evaluate_classifier(model, X_train, y_train, label_encoder, "train"),
        "val": evaluate_classifier(model, X_val, y_val, label_encoder, "val"),
        "test": evaluate_classifier(model, X_test, y_test, label_encoder, "test"),
    }

    label_mapping = {
        int(i): str(label)
        for i, label in enumerate(label_encoder.classes_)
    }

    checkpoint = {
        "model_id": model_id,
        "model_type": "sklearn_random_forest",
        "framework": "sklearn",
        "model_module": "lif_thesis.models.updated_rf",
        "class_names": label_encoder.classes_.astype(str).tolist(),
        "feature_names": feature_names,
        "feature_columns": {
            "spectra_col": spectra_col,
            "lifetime_col": lifetime_col,
            "scalar_cols": scalar_cols,
        },
        "tuning": tuning_summary,
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
            "uses_spectrometer": True,
            "uses_lifetime": True,
            "uses_scalars": True,
            "scalar_cols": scalar_cols,
        },
        "performance": {
            "val_balanced_accuracy": metrics["val"]["balanced_accuracy"],
            "val_macro_f1": metrics["val"]["macro_f1"],
            "test_balanced_accuracy": metrics["test"]["balanced_accuracy"],
            "test_macro_f1": metrics["test"]["macro_f1"],
        },
    }

    save_model_bundle(
        model=model,
        model_id=model_id,
        deploy_dir=Path(deploy_dir),
        checkpoint=checkpoint,
        metadata=metadata,
        label_mapping=label_mapping,
        label_encoder=label_encoder,
        framework="sklearn",
        extra_artifacts={
            "feature_names.json": feature_names,
        },
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=4)

    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "val_idx.npy", val_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    return model, label_encoder, metrics, (train_idx, val_idx, test_idx)


__all__ = [
    "UpdatedRFConfig",
    "make_updated_rf",
    "build_spectra_matrix",
    "build_multimodal_feature_matrix",
    "train_updated_rf",
    "evaluate_classifier",
    "run_updated_rf_experiment",
]