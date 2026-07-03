"""
Experiment 02: Grouped split + engineered scattering summary features.

Tests whether adding size and time_asymmetry improves grouped RF performance.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from lif_thesis.data.splits import make_group_split
from lif_thesis.models.checkpoint import save_model_bundle
from lif_thesis.models.updated_rf import (
    evaluate_classifier,
    make_updated_rf,
)


DATA_PATH = Path("data/processed/bacterial_samples.parquet")
OUTPUT_DIR = Path("results/exp02_grouped_engineered_features")
MODEL_ID = "exp02_grouped_engineered_rf_v1"


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


def build_feature_matrix(
    df: pd.DataFrame,
    feature_set: str,
    scalar_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if scalar_cols is None:
        scalar_cols = ["size", "time_asymmetry"]

    parts = []
    names = []
    tokens = feature_set.split("+")

    if "spectra" in tokens:
        X_spec = stack_vector_column(df, "spectrometer")
        parts.append(X_spec)
        names.extend([f"spectrometer_{i}" for i in range(X_spec.shape[1])])

    if "lifetime" in tokens:
        X_life = stack_vector_column(df, "lifetime")
        parts.append(X_life)
        names.extend([f"lifetime_{i}" for i in range(X_life.shape[1])])

    if "scattering" in tokens:
        X_scat = np.stack(df["scattering_image"].apply(crop_pad_scattering).values)
        parts.append(X_scat)
        names.extend([f"scattering_{i}" for i in range(X_scat.shape[1])])

    if "scalar" in tokens:
        missing_scalars = [col for col in scalar_cols if col not in df.columns]
        if missing_scalars:
            raise ValueError(f"Missing scalar columns: {missing_scalars}")

        X_scalar = df[scalar_cols].copy()
        for col in scalar_cols:
            X_scalar[col] = pd.to_numeric(X_scalar[col], errors="coerce")

        if X_scalar.isna().any().any():
            raise ValueError(f"Missing scalar values:\n{X_scalar.isna().sum()}")

        parts.append(X_scalar.to_numpy(dtype=float))
        names.extend(scalar_cols)

    if not parts:
        raise ValueError(f"No valid features selected for {feature_set}")

    X = np.concatenate(parts, axis=1)

    if not np.isfinite(X).all():
        raise ValueError(f"Feature matrix for {feature_set} contains NaN/Inf.")

    return X, names


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
        "scattering_image",
        *scalar_cols,
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[
        df[label_col].notna()
        & df[group_col].notna()
        & df["spectrometer"].notna()
        & df["lifetime"].notna()
        & df["scattering_image"].notna()
        & df["size"].notna()
        & df["time_asymmetry"].notna()
    ].reset_index(drop=True)

    print(f"Particles before fluorescence threshold: {len(df)}")

    df["peak_fluorescence"] = df["spectrometer"].apply(peak_fluorescence)
    df = df[df["peak_fluorescence"] > fluorescence_threshold].reset_index(drop=True)

    print(f"Particles after fluorescence > {fluorescence_threshold}: {len(df)}")
    print("\nClass counts:")
    print(df[label_col].value_counts())

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

    feature_sets = [
        "spectra",
        "lifetime",
        "scattering",
        "spectra+lifetime",
        "spectra+scattering",
        "lifetime+scattering",
        "spectra+lifetime+scattering",
        "spectra+scalar",
        "lifetime+scalar",
        "scattering+scalar",
        "spectra+lifetime+scalar",
        "spectra+scattering+scalar",
        "lifetime+scattering+scalar",
        "spectra+lifetime+scattering+scalar",
    ]

    n_estimators_grid = [100, 300, 500]
    max_depth_grid = [None, 20, 40]

    tuning_results = []
    best_score = -np.inf
    best_model = None
    best_config = None
    best_feature_names = None
    best_X = None

    for feature_set, n_estimators, max_depth in product(
        feature_sets,
        n_estimators_grid,
        max_depth_grid,
    ):
        print(
            f"\nTraining RF | features={feature_set} | "
            f"n_estimators={n_estimators} | max_depth={max_depth}"
        )

        X, feature_names = build_feature_matrix(
            df,
            feature_set=feature_set,
            scalar_cols=scalar_cols,
        )

        model = make_updated_rf(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

        model.fit(X[train_idx], y[train_idx])

        y_val_pred = model.predict(X[val_idx])

        result = {
            "feature_set": feature_set,
            "uses_engineered_scalars": "scalar" in feature_set.split("+"),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "val_accuracy": accuracy_score(y[val_idx], y_val_pred),
            "val_balanced_accuracy": balanced_accuracy_score(y[val_idx], y_val_pred),
            "val_macro_f1": f1_score(
                y[val_idx],
                y_val_pred,
                average="macro",
                zero_division=0,
            ),
            "n_features": X.shape[1],
        }

        tuning_results.append(result)

        print(
            f"Val balanced accuracy={result['val_balanced_accuracy']:.4f} | "
            f"Val macro F1={result['val_macro_f1']:.4f}"
        )

        if result["val_balanced_accuracy"] > best_score:
            best_score = result["val_balanced_accuracy"]
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
        raise RuntimeError("No model selected.")

    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)

    feature_summary = (
        tuning_df.sort_values("val_balanced_accuracy", ascending=False)
        .groupby("feature_set", as_index=False)
        .first()
        .sort_values("val_balanced_accuracy", ascending=False)
    )
    feature_summary.to_csv(OUTPUT_DIR / "feature_set_summary.csv", index=False)

    metrics = {
        "experiment": {
            "name": "exp02_grouped_engineered_features",
            "model_id": MODEL_ID,
            "label_col": label_col,
            "group_col": group_col,
            "fluorescence_threshold": fluorescence_threshold,
            "split_protocol": "grouped_raw_file_60_train_20_val_20_test",
            "scalar_features": scalar_cols,
            "note": "Compares paper-style feature sets with and without engineered scattering summary features.",
        },
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
        "model_id": MODEL_ID,
        "model_type": "sklearn_random_forest",
        "framework": "sklearn",
        "model_module": "lif_thesis.models.updated_rf",
        "class_names": label_encoder.classes_.astype(str).tolist(),
        "feature_set": best_config["feature_set"],
        "feature_names": best_feature_names,
        "best_config": best_config,
        "fluorescence_threshold": fluorescence_threshold,
    }

    metadata = {
        "model_id": MODEL_ID,
        "display_name": "Exp02 Grouped Engineered RF",
        "model_type": "sklearn_random_forest",
        "framework": "sklearn",
        "model_file": "model.joblib",
        "labels_file": "labels.json",
        "label_encoder_file": "label_encoder.joblib",
        "feature_names_file": "feature_names.json",
        "preprocessing": {
            "fluorescence_threshold": fluorescence_threshold,
            "scattering_target_acquisitions": 60,
            "n_scattering_angles": 24,
            "scattering_normalize": True,
            "feature_set": best_config["feature_set"],
            "scalar_features": scalar_cols,
        },
        "performance": {
            "test_accuracy": metrics["test_final"]["accuracy"],
            "test_balanced_accuracy": metrics["test_final"]["balanced_accuracy"],
            "test_macro_f1": metrics["test_final"]["macro_f1"],
        },
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
    np.save(OUTPUT_DIR / "val_tuning_idx.npy", val_idx)
    np.save(OUTPUT_DIR / "test_final_idx.npy", test_idx)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")
    print(f"Saved model bundle to: models/trained/{MODEL_ID}")

    print("\nFinal test performance:")
    print(
        json.dumps(
            {
                "accuracy": metrics["test_final"]["accuracy"],
                "balanced_accuracy": metrics["test_final"]["balanced_accuracy"],
                "macro_f1": metrics["test_final"]["macro_f1"],
                "best_config": best_config,
            },
            indent=4,
        )
    )


if __name__ == "__main__":
    main()