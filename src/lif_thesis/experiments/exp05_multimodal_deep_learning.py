"""
Experiment 05: Multimodal Deep Learning Model

Inputs:
- spectrometer vector
- lifetime vector
- scattering image vector, cropped/padded/normalized
- scalar features: size, time_asymmetry

Protocol:
- peak fluorescence > 2000 a.u.
- grouped raw-file train/val/test split
- particle-level multiclass species prediction
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from lif_thesis.data.schemas import RAPIDE_DIMS
from lif_thesis.data.splits import make_group_split
from lif_thesis.models.multimodal_cnn import MultimodalDeepClassifier


DATA_PATH = Path("data/processed/bacterial_samples.parquet")
OUTPUT_DIR = Path("results/exp05_multimodal_deep_learning")

MODEL_ID = "exp05_multimodal_species_v1"
DEPLOY_DIR = Path("models/trained") / MODEL_ID


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
    n_acquisitions: int = RAPIDE_DIMS.SCATTERING_TARGET_ACQUISITIONS,
    n_angles: int = RAPIDE_DIMS.N_SCATTERING_ANGLES,
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


def build_inputs(df: pd.DataFrame):
    X_spec = stack_vector_column(df, "spectrometer")
    X_life = stack_vector_column(df, "lifetime")
    X_scat = np.stack(df["scattering_image"].apply(crop_pad_scattering).values)

    X_scalar = df[["size", "time_asymmetry"]].copy()
    X_scalar["size"] = pd.to_numeric(X_scalar["size"], errors="coerce")
    X_scalar["time_asymmetry"] = pd.to_numeric(
        X_scalar["time_asymmetry"],
        errors="coerce",
    )

    if X_scalar.isna().any().any():
        raise ValueError(f"Missing scalar values:\n{X_scalar.isna().sum()}")

    return (
        X_spec.astype(np.float32),
        X_life.astype(np.float32),
        X_scat.astype(np.float32),
        X_scalar.to_numpy(dtype=np.float32),
    )


class MultimodalParticleDataset(Dataset):
    def __init__(
        self,
        X_spec: np.ndarray,
        X_life: np.ndarray,
        X_scat: np.ndarray,
        X_scalar: np.ndarray,
        y: np.ndarray,
    ):
        self.X_spec = torch.tensor(X_spec, dtype=torch.float32)
        self.X_life = torch.tensor(X_life, dtype=torch.float32)
        self.X_scat = torch.tensor(X_scat, dtype=torch.float32)
        self.X_scalar = torch.tensor(X_scalar, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "spectrometer": self.X_spec[idx].unsqueeze(0),
            "lifetime": self.X_life[idx].unsqueeze(0),
            "scattering": self.X_scat[idx].unsqueeze(0),
            "scalar": self.X_scalar[idx],
            "label": self.y[idx],
        }


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        y = batch["label"]

        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1)

        total_loss += loss.item() * len(y)
        total_correct += (pred == y).sum().item()
        total_n += len(y)

    return {
        "loss": total_loss / total_n,
        "accuracy": total_correct / total_n,
    }


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_n = 0

    all_y = []
    all_pred = []
    all_proba = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        y = batch["label"]

        logits = model(batch)
        loss = criterion(logits, y)

        proba = torch.softmax(logits, dim=1)
        pred = proba.argmax(dim=1)

        total_loss += loss.item() * len(y)
        total_n += len(y)

        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_proba.append(proba.cpu().numpy())

    return {
        "loss": total_loss / total_n,
        "y_true": np.concatenate(all_y),
        "y_pred": np.concatenate(all_pred),
        "y_proba": np.concatenate(all_proba),
    }


def compute_metrics(y_true, y_pred, label_encoder):
    labels = np.arange(len(label_encoder.classes_))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_encoder.classes_.astype(str),
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    label_col = "species"
    group_col = "raw_file"
    fluorescence_threshold = 2000.0
    random_state = 42

    batch_size = 128
    max_epochs = 40
    patience = 8
    learning_rate = 5e-4
    weight_decay = 1e-4

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    required_cols = [
        label_col,
        group_col,
        "spectrometer",
        "lifetime",
        "scattering_image",
        "size",
        "time_asymmetry",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[
        df[label_col].notna()
        & df[group_col].notna()
        & df["spectrometer"].notna()
        & df["lifetime"].notna()
        & df["scattering_image"].notna()
        & df["size"].notna()
        & df["time_asymmetry"].notna()
    ].reset_index(drop=True)

    print(f"Particles before threshold: {len(df)}")

    df["peak_fluorescence"] = df["spectrometer"].apply(peak_fluorescence)
    df = df[df["peak_fluorescence"] > fluorescence_threshold].reset_index(drop=True)

    print(f"Particles after fluorescence > {fluorescence_threshold}: {len(df)}")
    print("\nClass counts:")
    print(df[label_col].value_counts())

    X_spec, X_life, X_scat, X_scalar = build_inputs(df)

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

    spec_scaler = StandardScaler()
    life_scaler = StandardScaler()
    scat_scaler = StandardScaler()
    scalar_scaler = StandardScaler()

    X_spec_train = spec_scaler.fit_transform(X_spec[train_idx])
    X_spec_val = spec_scaler.transform(X_spec[val_idx])
    X_spec_test = spec_scaler.transform(X_spec[test_idx])

    X_life_train = life_scaler.fit_transform(X_life[train_idx])
    X_life_val = life_scaler.transform(X_life[val_idx])
    X_life_test = life_scaler.transform(X_life[test_idx])

    X_scat_train = scat_scaler.fit_transform(X_scat[train_idx])
    X_scat_val = scat_scaler.transform(X_scat[val_idx])
    X_scat_test = scat_scaler.transform(X_scat[test_idx])

    X_scalar_train = scalar_scaler.fit_transform(X_scalar[train_idx])
    X_scalar_val = scalar_scaler.transform(X_scalar[val_idx])
    X_scalar_test = scalar_scaler.transform(X_scalar[test_idx])

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    train_loader = DataLoader(
        MultimodalParticleDataset(
            X_spec_train,
            X_life_train,
            X_scat_train,
            X_scalar_train,
            y_train,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        MultimodalParticleDataset(
            X_spec_val,
            X_life_val,
            X_scat_val,
            X_scalar_val,
            y_val,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        MultimodalParticleDataset(
            X_spec_test,
            X_life_test,
            X_scat_test,
            X_scalar_test,
            y_test,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = MultimodalDeepClassifier(
        n_classes=len(label_encoder.classes_)
    ).to(device)

    class_counts = np.bincount(y_train)
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_eval = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
        )

        val_metrics = compute_metrics(
            val_eval["y_true"],
            val_eval["y_pred"],
            label_encoder,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_eval["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }

        history.append(record)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={record['train_loss']:.4f} | "
            f"val_loss={record['val_loss']:.4f} | "
            f"val_bal_acc={record['val_balanced_accuracy']:.4f} | "
            f"val_macro_f1={record['val_macro_f1']:.4f}"
        )

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    model.load_state_dict(
        torch.load(OUTPUT_DIR / "best_model.pt", map_location=device)
    )

    train_eval = evaluate_model(model, train_loader, criterion, device)
    val_eval = evaluate_model(model, val_loader, criterion, device)
    test_eval = evaluate_model(model, test_loader, criterion, device)

    metrics = {
        "experiment": {
            "name": "exp05_multimodal_deep_learning",
            "model_id": MODEL_ID,
            "label_col": label_col,
            "group_col": group_col,
            "input_features": [
                "spectrometer",
                "lifetime",
                "scattering_image",
                "size",
                "time_asymmetry",
            ],
            "fluorescence_threshold": fluorescence_threshold,
            "split_protocol": "grouped_raw_file_60_train_20_val_20_test",
            "model": "MultimodalDeepClassifier",
        },
        "best_epoch": best_epoch,
        "train": compute_metrics(train_eval["y_true"], train_eval["y_pred"], label_encoder),
        "val": compute_metrics(val_eval["y_true"], val_eval["y_pred"], label_encoder),
        "test": compute_metrics(test_eval["y_true"], test_eval["y_pred"], label_encoder),
    }

    deployment_checkpoint = {
        "model_state_dict": model.state_dict(),
        "n_classes": len(label_encoder.classes_),
        "class_names": label_encoder.classes_.astype(str).tolist(),
        "model_id": MODEL_ID,
        "architecture": "MultimodalDeepClassifier",
        "model_module": "lif_thesis.models.multimodal_cnn",
        "input_features": [
            "spectrometer",
            "lifetime",
            "scattering_image",
            "size",
            "time_asymmetry",
        ],
        "fluorescence_threshold": fluorescence_threshold,
        "scattering_target_acquisitions": RAPIDE_DIMS.SCATTERING_TARGET_ACQUISITIONS,
        "n_scattering_angles": RAPIDE_DIMS.N_SCATTERING_ANGLES,
        "best_epoch": best_epoch,
        "test_accuracy": metrics["test"]["accuracy"],
        "test_balanced_accuracy": metrics["test"]["balanced_accuracy"],
        "test_macro_f1": metrics["test"]["macro_f1"],
    }

    torch.save(deployment_checkpoint, DEPLOY_DIR / "model.pt")

    label_mapping = {
        int(i): str(label)
        for i, label in enumerate(label_encoder.classes_)
    }

    metadata = {
        "model_id": MODEL_ID,
        "display_name": "Exp05 Multimodal CNN",
        "model_type": "torch_multimodal",
        "model_file": "model.pt",
        "labels_file": "labels.json",
        "architecture": "MultimodalDeepClassifier",
        "model_module": "lif_thesis.models.multimodal_cnn",
        "preprocessing": {
            "fluorescence_threshold": fluorescence_threshold,
            "scattering_target_acquisitions": RAPIDE_DIMS.SCATTERING_TARGET_ACQUISITIONS,
            "n_scattering_angles": RAPIDE_DIMS.N_SCATTERING_ANGLES,
            "scattering_normalize": True,
            "spectrometer_scaler": "spectrometer_scaler.joblib",
            "lifetime_scaler": "lifetime_scaler.joblib",
            "scattering_scaler": "scattering_scaler.joblib",
            "scalar_scaler": "scalar_scaler.joblib",
        },
        "performance": {
            "best_epoch": best_epoch,
            "test_accuracy": metrics["test"]["accuracy"],
            "test_balanced_accuracy": metrics["test"]["balanced_accuracy"],
            "test_macro_f1": metrics["test"]["macro_f1"],
        },
        "uses_rejection": False,
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(DEPLOY_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(DEPLOY_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=4)

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False,
    )

    joblib.dump(label_encoder, OUTPUT_DIR / "label_encoder.joblib")
    joblib.dump(spec_scaler, OUTPUT_DIR / "spectrometer_scaler.joblib")
    joblib.dump(life_scaler, OUTPUT_DIR / "lifetime_scaler.joblib")
    joblib.dump(scat_scaler, OUTPUT_DIR / "scattering_scaler.joblib")
    joblib.dump(scalar_scaler, OUTPUT_DIR / "scalar_scaler.joblib")

    joblib.dump(label_encoder, DEPLOY_DIR / "label_encoder.joblib")
    joblib.dump(spec_scaler, DEPLOY_DIR / "spectrometer_scaler.joblib")
    joblib.dump(life_scaler, DEPLOY_DIR / "lifetime_scaler.joblib")
    joblib.dump(scat_scaler, DEPLOY_DIR / "scattering_scaler.joblib")
    joblib.dump(scalar_scaler, DEPLOY_DIR / "scalar_scaler.joblib")

    np.save(OUTPUT_DIR / "train_idx.npy", train_idx)
    np.save(OUTPUT_DIR / "val_idx.npy", val_idx)
    np.save(OUTPUT_DIR / "test_idx.npy", test_idx)

    print(f"\nSaved experiment outputs to: {OUTPUT_DIR}")
    print(f"Saved deployable model bundle to: {DEPLOY_DIR}")

    print("\nFinal test performance:")
    print(
        json.dumps(
            {
                "accuracy": metrics["test"]["accuracy"],
                "balanced_accuracy": metrics["test"]["balanced_accuracy"],
                "macro_f1": metrics["test"]["macro_f1"],
                "best_epoch": best_epoch,
            },
            indent=4,
        )
    )


if __name__ == "__main__":
    main()