"""
cnn1d.py

Reusable baseline 1D CNN for direct fluorescence spectra analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import LabelEncoder, StandardScaler

from lif_thesis.data.splits import make_group_split
from lif_thesis.evaluation.metrics import compute_classification_metrics
from lif_thesis.models.checkpoint import save_model_bundle
from dataclasses import dataclass


@dataclass(slots=True)
class BaselineCNNPreprocessingConfig:
    """
    Preprocessing configuration for the baseline CNN.
    """

    fluorescence_threshold: float = 2000.0

    label_col: str = "species"
    group_col: str = "raw_file"
    spectra_col: str = "spectrometer"

    train_size: float = 0.60
    val_size: float = 0.20
    test_size: float = 0.20


@dataclass(slots=True)
class BaselineCNNTrainingConfig:
    """
    Training hyperparameters.
    """

    batch_size: int = 128
    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    random_state: int = 42

class SpectraDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]


class BaselineCNN1D(nn.Module):
    """
    Simple 1D CNN for fluorescence spectra.

    Input shape:
        batch_size x 1 x n_spectral_bins
    """

    def __init__(self, input_length: int, n_classes: int):
        super().__init__()

        self.input_length = input_length
        self.n_classes = n_classes

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def to_array(x) -> np.ndarray:
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

    if df[spectra_col].isna().any():
        raise ValueError(
            f"{df[spectra_col].isna().sum()} rows have missing "
            f"{spectra_col} values."
        )

    X = np.stack(df[spectra_col].apply(to_array).values)

    if X.ndim != 2:
        raise ValueError(
            f"Expected spectra matrix with shape (n, bins), got {X.shape}"
        )

    if not np.isfinite(X).all():
        raise ValueError("Spectra matrix contains NaN or infinite values.")

    return X.astype(np.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_samples += len(X_batch)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_y = []
    all_pred = []
    all_proba = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        proba = torch.softmax(logits, dim=1)
        pred = proba.argmax(dim=1)

        total_loss += loss.item() * len(X_batch)
        total_samples += len(X_batch)

        all_y.append(y_batch.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_proba.append(proba.cpu().numpy())

    return {
        "loss": total_loss / total_samples,
        "y_true": np.concatenate(all_y),
        "y_pred": np.concatenate(all_pred),
        "y_proba": np.concatenate(all_proba),
    }


def run_baseline_cnn_experiment(
    df: pd.DataFrame,
    *,
    model_id: str,
    display_name: str,
    output_dir: str | Path = "results/baseline_cnn1d",
    deploy_dir: str | Path = "models/trained",
    preprocessing: BaselineCNNPreprocessingConfig | None = None,
    training: BaselineCNNTrainingConfig | None = None,
):
    preprocessing = preprocessing or BaselineCNNPreprocessingConfig()
    training = training or BaselineCNNTrainingConfig()

    torch.manual_seed(training.random_state)
    np.random.seed(training.random_state)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_col = preprocessing.label_col
    group_col = preprocessing.group_col
    spectra_col = preprocessing.spectra_col

    df = df.copy()

    required_cols = [label_col, group_col, spectra_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[
        df[label_col].notna()
        & df[group_col].notna()
        & df[spectra_col].notna()
    ].reset_index(drop=True)

    df["peak_fluorescence"] = df[spectra_col].apply(
        lambda x: float(np.max(to_array(x)))
    )

    df = df[
        df["peak_fluorescence"] > preprocessing.fluorescence_threshold
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows remain after fluorescence thresholding.")

    print(f"Using {len(df)} rows after threshold.")
    print(f"Labels: {df[label_col].value_counts().to_dict()}")
    print(f"Groups: {df[group_col].nunique()}")

    X = build_spectra_matrix(df, spectra_col=spectra_col)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col].astype(str))

    train_idx, val_idx, test_idx = make_group_split(
        df,
        label_col=label_col,
        group_col=group_col,
        train_size=preprocessing.train_size,
        val_size=preprocessing.val_size,
        test_size=preprocessing.test_size,
        stratify=True,
        random_state=training.random_state,
        verbose=True,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    train_loader = DataLoader(
        SpectraDataset(X_train, y_train),
        batch_size=training.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        SpectraDataset(X_val, y_val),
        batch_size=training.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        SpectraDataset(X_test, y_test),
        batch_size=training.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BaselineCNN1D(
        input_length=X.shape[1],
        n_classes=len(label_encoder.classes_),
    ).to(device)

    class_counts = np.bincount(y_train)
    class_weights = class_counts.sum() / (
        len(class_counts) * np.maximum(class_counts, 1)
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )

    best_val_loss = np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, training.max_epochs + 1):
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

        val_metrics = compute_classification_metrics(
            y_true=val_eval["y_true"],
            y_pred=val_eval["y_pred"],
            y_proba=val_eval["y_proba"],
            class_names=label_encoder.classes_.tolist(),
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_eval["loss"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }

        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"val_loss={val_eval['loss']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= training.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", map_location=device)
    )

    train_eval = evaluate_model(model, train_loader, criterion, device)
    val_eval = evaluate_model(model, val_loader, criterion, device)
    test_eval = evaluate_model(model, test_loader, criterion, device)

    metrics = {
        "experiment": {
            "model_id": model_id,
            "display_name": display_name,
            "model_type": "torch_cnn1d",
            "model_module": "lif_thesis.models.cnn1d",
            "label_col": label_col,
            "group_col": group_col,
            "spectra_col": spectra_col,
            "fluorescence_threshold": preprocessing.fluorescence_threshold,
            "split_protocol": "grouped_raw_file_split",
            "train_size": preprocessing.train_size,
            "val_size": preprocessing.val_size,
            "test_size": preprocessing.test_size,
        },
        "train": compute_classification_metrics(
            train_eval["y_true"],
            train_eval["y_pred"],
            train_eval["y_proba"],
            class_names=label_encoder.classes_.tolist(),
        ),
        "val": compute_classification_metrics(
            val_eval["y_true"],
            val_eval["y_pred"],
            val_eval["y_proba"],
            class_names=label_encoder.classes_.tolist(),
        ),
        "test": compute_classification_metrics(
            test_eval["y_true"],
            test_eval["y_pred"],
            test_eval["y_proba"],
            class_names=label_encoder.classes_.tolist(),
        ),
        "best_epoch": best_epoch,
        "label_classes": label_encoder.classes_.tolist(),
    }

    label_mapping = {
        int(i): str(label)
        for i, label in enumerate(label_encoder.classes_)
    }

    checkpoint = {
        "n_classes": len(label_encoder.classes_),
        "class_names": label_encoder.classes_.astype(str).tolist(),
        "model_id": model_id,
        "architecture": "BaselineCNN1D",
        "model_module": "lif_thesis.models.cnn1d",
        "model_type": "torch_cnn1d",
        "input_length": int(X.shape[1]),
        "spectra_col": spectra_col,
        "fluorescence_threshold": preprocessing.fluorescence_threshold,
        "best_epoch": best_epoch,
        "scaler": "spectra_scaler.joblib",
    }

    metadata = {
        "model_id": model_id,
        "display_name": display_name,
        "model_type": "torch_cnn1d",
        "framework": "torch",
        "model_file": "model.pt",
        "labels_file": "labels.json",
        "label_encoder_file": "label_encoder.joblib",
        "architecture": "BaselineCNN1D",
        "model_module": "lif_thesis.models.cnn1d",
        "preprocessing": {
            "spectra_col": spectra_col,
            "fluorescence_threshold": preprocessing.fluorescence_threshold,
            "spectra_scaler": "spectra_scaler.joblib",
            "uses_spectrometer": True,
        },
        "training": {
            "batch_size": training.batch_size,
            "max_epochs": training.max_epochs,
            "learning_rate": training.learning_rate,
            "weight_decay": training.weight_decay,
            "patience": training.patience,
            "random_state": training.random_state,
        },
        "performance": {
            "best_epoch": best_epoch,
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
        scalers={
            "spectra": scaler,
        },
        framework="torch",
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "val_idx.npy", val_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    print(f"\nSaved CNN experiment outputs to: {output_dir}")
    print(f"Saved model bundle to: {Path(deploy_dir) / model_id}")

    return model, label_encoder, scaler, metrics


__all__ = [
    "SpectraDataset",
    "BaselineCNN1D",
    "build_spectra_matrix",
    "train_one_epoch",
    "evaluate_model",
    "run_baseline_cnn_experiment",
]