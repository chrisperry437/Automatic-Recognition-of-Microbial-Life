"""
Experiment 06: Robust Multimodal CNN for Rapid-E Live / Failed-Control Conditions

Purpose
-------
This experiment extends exp05_multimodal_deep_learning.py with safeguards for
real-world / failed experimental conditions such as the June 17 saturation run.

Key changes vs exp05
--------------------
1. Optional inclusion of control / ringer particles as a real training class.
2. Optional inclusion of external failed-control files as extra negative examples.
3. Two-stage inference logic:
   - Stage A: closed-set species prediction from the CNN.
   - Stage B: confidence / distribution-shift rejection to UNKNOWN.
4. Modality dropout during training to reduce dependence on corrupted modalities.
5. Optional branch masking at evaluation/inference for ablation testing.
6. Per-file QC outputs: prediction entropy, unknown fraction, particle counts,
   fluorescence statistics, and dominant predicted class.
7. Saves deployable checkpoint, scalers, label maps, and rejection config.

Recommended use
---------------
Train first with your clean training data plus any June 17 failed-control data:

python -m src.lif_thesis.experiments.exp06_robust_cnn \
    --data-path data/processed/bacterial_samples.parquet \
    --failed-control-path data/processed/june17_failed_control.parquet \
    --include-control \
    --output-dir results/exp06_robust_cnn

If your processed data does not include a control class, the script still trains,
but UNKNOWN rejection becomes much more important.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from lif_thesis.data.schemas import RAPIDE_DIMS
from lif_thesis.data.splits import make_group_split
from lif_thesis.models.checkpoint import save_model_bundle

from lif_thesis.models.robust_multimodal_cnn import (
    RobustMultimodalClassifier,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_DATA_PATH = Path("data/processed/bacterial_samples.parquet")
DEFAULT_OUTPUT_DIR = Path("results/exp06_robust_cnn")
DEFAULT_MODEL_NAME = "robust_multimodal_species_v1"

UNKNOWN_LABEL = "unknown"
FAILED_CONTROL_LABEL = "failed_control"
CONTROL_ALIASES = {
    "control",
    "ringer",
    "ringer_control",
    "negative_control",
    "blank",
    "media_control",
}


@dataclass
class RejectionConfig:
    softmax_threshold: float = 0.80
    margin_threshold: float = 0.15
    entropy_threshold: float | None = None
    use_train_distance_rejection: bool = True
    train_distance_quantile: float = 0.995


@dataclass
class TrainingConfig:
    fluorescence_threshold: float = 2000.0
    train_size: float = 0.60
    val_size: float = 0.20
    test_size: float = 0.20
    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 10
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    random_state: int = 42
    modality_dropout_p: float = 0.15
    label_col: str = "species"
    group_col: str = "raw_file"


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

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
    """
    Paper-style scattering processing:
    - crop to 30 us equivalent
    - zero-pad shorter signals
    - normalize each particle to [0, 1]
    """
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


def normalize_label(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def is_control_label(value: object) -> bool:
    return normalize_label(value) in CONTROL_ALIASES


def load_parquet_or_csv(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def build_inputs(df: pd.DataFrame):
    X_spec = stack_vector_column(df, "spectrometer")
    X_life = stack_vector_column(df, "lifetime")
    X_scat = np.stack(df["scattering_image"].apply(crop_pad_scattering).values)

    X_scalar = df[["size", "time_asymmetry"]].copy()
    X_scalar["size"] = pd.to_numeric(X_scalar["size"], errors="coerce")
    X_scalar["time_asymmetry"] = pd.to_numeric(
        X_scalar["time_asymmetry"], errors="coerce"
    )

    if X_scalar.isna().any().any():
        raise ValueError(f"Missing scalar values:\n{X_scalar.isna().sum()}")

    return (
        X_spec.astype(np.float32),
        X_life.astype(np.float32),
        X_scat.astype(np.float32),
        X_scalar.to_numpy(dtype=np.float32),
    )


def prepare_dataframe(
    data_path: Path,
    failed_control_path: Path | None,
    config: TrainingConfig,
    include_control: bool,
) -> pd.DataFrame:
    required_cols = [
        config.label_col,
        config.group_col,
        "spectrometer",
        "lifetime",
        "scattering_image",
        "size",
        "time_asymmetry",
    ]

    df = load_parquet_or_csv(data_path)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns from main data: {missing}")

    df = df[required_cols].copy()

    if not include_control:
        df = df[~df[config.label_col].apply(is_control_label)].copy()

    if failed_control_path is not None:
        fc = load_parquet_or_csv(failed_control_path)
        missing_fc = [col for col in required_cols if col not in fc.columns]
        if missing_fc:
            raise ValueError(
                f"Missing required columns from failed-control data: {missing_fc}"
            )
        fc = fc[required_cols].copy()
        fc[config.label_col] = FAILED_CONTROL_LABEL
        df = pd.concat([df, fc], ignore_index=True)

    df = df[
        df[config.label_col].notna()
        & df[config.group_col].notna()
        & df["spectrometer"].notna()
        & df["lifetime"].notna()
        & df["scattering_image"].notna()
        & df["size"].notna()
        & df["time_asymmetry"].notna()
    ].reset_index(drop=True)

    df["peak_fluorescence"] = df["spectrometer"].apply(peak_fluorescence)
    df = df[df["peak_fluorescence"] > config.fluorescence_threshold].reset_index(
        drop=True
    )

    if len(df) == 0:
        raise ValueError("No particles remain after filtering.")

    return df


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class RobustParticleDataset(Dataset):
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


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------

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

    return {"loss": total_loss / total_n, "accuracy": total_correct / total_n}


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, enabled_modalities=None):
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_y, all_pred, all_proba = [], [], []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        y = batch["label"]
        logits = model(batch, enabled_modalities=enabled_modalities)
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


def entropy_from_proba(proba: np.ndarray) -> np.ndarray:
    safe = np.clip(proba, 1e-12, 1.0)
    return -(safe * np.log(safe)).sum(axis=1)


def prediction_margin(proba: np.ndarray) -> np.ndarray:
    if proba.shape[1] == 1:
        return np.ones(proba.shape[0])
    sorted_p = np.sort(proba, axis=1)
    return sorted_p[:, -1] - sorted_p[:, -2]


def fit_train_distance_reference(X_scalar_train_scaled: np.ndarray) -> dict:
    """
    Lightweight OOD reference using scaled scalar features.

    This intentionally uses scalar features only because live failures often show up
    in size / time_asymmetry shifts, and this avoids storing massive spectral arrays.
    """
    center = X_scalar_train_scaled.mean(axis=0)
    scale = X_scalar_train_scaled.std(axis=0) + 1e-8
    z = (X_scalar_train_scaled - center) / scale
    dist = np.linalg.norm(z, axis=1)
    return {"center": center, "scale": scale, "train_distances": dist}


def apply_rejection(
    proba: np.ndarray,
    X_scalar_scaled: np.ndarray,
    rejection_config: RejectionConfig,
    distance_reference: dict | None,
) -> tuple[np.ndarray, dict]:
    max_proba = proba.max(axis=1)
    margin = prediction_margin(proba)
    entropy = entropy_from_proba(proba)

    rejected = max_proba < rejection_config.softmax_threshold
    rejected |= margin < rejection_config.margin_threshold

    if rejection_config.entropy_threshold is not None:
        rejected |= entropy > rejection_config.entropy_threshold

    scalar_distance = None
    distance_threshold = None
    if rejection_config.use_train_distance_rejection and distance_reference is not None:
        center = distance_reference["center"]
        scale = distance_reference["scale"]
        train_distances = distance_reference["train_distances"]
        scalar_distance = np.linalg.norm((X_scalar_scaled - center) / scale, axis=1)
        distance_threshold = float(
            np.quantile(train_distances, rejection_config.train_distance_quantile)
        )
        rejected |= scalar_distance > distance_threshold

    diagnostics = {
        "max_proba": max_proba,
        "margin": margin,
        "entropy": entropy,
        "scalar_distance": scalar_distance,
        "distance_threshold": distance_threshold,
        "rejected": rejected,
    }
    return rejected, diagnostics


def summarize_with_unknown(
    y_pred: np.ndarray,
    proba: np.ndarray,
    label_encoder: LabelEncoder,
    rejected: np.ndarray,
) -> pd.DataFrame:
    labels = label_encoder.inverse_transform(y_pred).astype(str)
    labels = labels.copy()
    labels[rejected] = UNKNOWN_LABEL
    return pd.DataFrame(
        {
            "predicted_label_raw": label_encoder.inverse_transform(y_pred).astype(str),
            "predicted_label_robust": labels,
            "max_proba": proba.max(axis=1),
            "margin": prediction_margin(proba),
            "entropy": entropy_from_proba(proba),
            "is_unknown": rejected,
        }
    )


def make_file_qc_summary(df: pd.DataFrame, pred_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    tmp = pd.concat([df[[group_col, "peak_fluorescence", "size"]].reset_index(drop=True), pred_df], axis=1)

    rows = []
    for raw_file, g in tmp.groupby(group_col):
        counts = g["predicted_label_robust"].value_counts(dropna=False)
        dominant_label = str(counts.index[0]) if len(counts) else None
        dominant_fraction = float(counts.iloc[0] / len(g)) if len(g) else np.nan
        rows.append(
            {
                group_col: raw_file,
                "n_particles_after_threshold": int(len(g)),
                "unknown_fraction": float(g["is_unknown"].mean()),
                "mean_entropy": float(g["entropy"].mean()),
                "median_max_proba": float(g["max_proba"].median()),
                "median_peak_fluorescence": float(g["peak_fluorescence"].median()),
                "q95_peak_fluorescence": float(g["peak_fluorescence"].quantile(0.95)),
                "median_size": float(g["size"].median()),
                "dominant_label": dominant_label,
                "dominant_fraction": dominant_fraction,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--failed-control-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--include-control", action="store_true")
    parser.add_argument("--fluorescence-threshold", type=float, default=2000.0)
    parser.add_argument("--softmax-threshold", type=float, default=0.80)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    parser.add_argument("--entropy-threshold", type=float, default=None)
    parser.add_argument("--no-distance-rejection", action="store_true")
    parser.add_argument("--modality-dropout-p", type=float, default=0.15)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    deploy_model_dir = Path("models/trained")
    deploy_config_dir = Path("models/configs")
    deploy_label_dir = Path("models/label_maps")
    deploy_model_dir.mkdir(parents=True, exist_ok=True)
    deploy_config_dir.mkdir(parents=True, exist_ok=True)
    deploy_label_dir.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        fluorescence_threshold=args.fluorescence_threshold,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        modality_dropout_p=args.modality_dropout_p,
    )
    rejection_config = RejectionConfig(
        softmax_threshold=args.softmax_threshold,
        margin_threshold=args.margin_threshold,
        entropy_threshold=args.entropy_threshold,
        use_train_distance_rejection=not args.no_distance_rejection,
    )

    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)

    print("Loading and preparing data...")
    df = prepare_dataframe(
        data_path=args.data_path,
        failed_control_path=args.failed_control_path,
        config=config,
        include_control=args.include_control,
    )

    print(f"Particles after fluorescence > {config.fluorescence_threshold}: {len(df)}")
    print("\nClass counts:")
    print(df[config.label_col].value_counts())

    X_spec, X_life, X_scat, X_scalar = build_inputs(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[config.label_col].astype(str))

    train_idx, val_idx, test_idx = make_group_split(
        df,
        label_col=config.label_col,
        group_col=config.group_col,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        stratify=True,
        random_state=config.random_state,
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

    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    train_loader = DataLoader(
        RobustParticleDataset(X_spec_train, X_life_train, X_scat_train, X_scalar_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        RobustParticleDataset(X_spec_val, X_life_val, X_scat_val, X_scalar_val, y_val),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        RobustParticleDataset(X_spec_test, X_life_test, X_scat_test, X_scalar_test, y_test),
        batch_size=config.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = RobustMultimodalClassifier(
        n_classes=len(label_encoder.classes_),
        modality_dropout_p=config.modality_dropout_p,
    ).to(device)

    class_counts = np.bincount(y_train)
    class_weights = class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_eval = evaluate_model(model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_eval["y_true"], val_eval["y_pred"], label_encoder)

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
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))

    train_eval = evaluate_model(model, train_loader, criterion, device)
    val_eval = evaluate_model(model, val_loader, criterion, device)
    test_eval = evaluate_model(model, test_loader, criterion, device)

    distance_reference = fit_train_distance_reference(X_scalar_train)

    rejected_test, rejection_diag_test = apply_rejection(
        test_eval["y_proba"],
        X_scalar_test,
        rejection_config,
        distance_reference,
    )
    test_pred_df = summarize_with_unknown(
        test_eval["y_pred"],
        test_eval["y_proba"],
        label_encoder,
        rejected_test,
    )

    test_df = df.iloc[test_idx].reset_index(drop=True)
    file_qc = make_file_qc_summary(test_df, test_pred_df, config.group_col)

    ablation_results = {}
    ablations = {
        "full": ["spectrometer", "lifetime", "scattering", "scalar"],
        "lifetime_scalar": ["lifetime", "scalar"],
        "no_scattering": ["spectrometer", "lifetime", "scalar"],
        "lifetime_only": ["lifetime"],
    }
    for name, modalities in ablations.items():
        ev = evaluate_model(model, test_loader, criterion, device, enabled_modalities=modalities)
        ablation_results[name] = compute_metrics(ev["y_true"], ev["y_pred"], label_encoder)

    metrics = {
        "experiment": {
            "name": "exp06_robust_cnn",
            "model_name": args.model_name,
            "input_features": [
                "spectrometer",
                "lifetime",
                "scattering_image",
                "size",
                "time_asymmetry",
            ],
            "robustness_changes": [
                "control_or_failed_control_class_support",
                "softmax_margin_entropy_rejection",
                "scalar_distribution_shift_rejection",
                "modality_dropout",
                "modality_ablation_evaluation",
                "file_level_qc_summary",
            ],
        },
        "training_config": asdict(config),
        "rejection_config": asdict(rejection_config),
        "best_epoch": best_epoch,
        "classes": label_encoder.classes_.astype(str).tolist(),
        "train": compute_metrics(train_eval["y_true"], train_eval["y_pred"], label_encoder),
        "val": compute_metrics(val_eval["y_true"], val_eval["y_pred"], label_encoder),
        "test_closed_set": compute_metrics(test_eval["y_true"], test_eval["y_pred"], label_encoder),
        "test_rejection_summary": {
            "unknown_fraction": float(rejected_test.mean()),
            "median_max_proba": float(np.median(test_pred_df["max_proba"])),
            "median_margin": float(np.median(test_pred_df["margin"])),
            "median_entropy": float(np.median(test_pred_df["entropy"])),
            "distance_threshold": rejection_diag_test["distance_threshold"],
        },
        "ablation_results": ablation_results,
    }

    deployment_checkpoint = {
        "model_state_dict": model.state_dict(),
        "n_classes": len(label_encoder.classes_),
        "class_names": label_encoder.classes_.astype(str).tolist(),
        "model_name": args.model_name,
        "architecture": "RobustMultimodalClassifier",
        "model_module": "lif_thesis.models.robust_multimodal_cnn",
        "input_features": [
            "spectrometer",
            "lifetime",
            "scattering_image",
            "size",
            "time_asymmetry",
        ],
        "fluorescence_threshold": config.fluorescence_threshold,
        "scattering_target_acquisitions": RAPIDE_DIMS.SCATTERING_TARGET_ACQUISITIONS,
        "n_scattering_angles": RAPIDE_DIMS.N_SCATTERING_ANGLES,
        "rejection_config": asdict(rejection_config),
        "distance_reference": {
            "center": distance_reference["center"].tolist(),
            "scale": distance_reference["scale"].tolist(),
            "train_distance_quantile": rejection_config.train_distance_quantile,
            "distance_threshold": rejection_diag_test["distance_threshold"],
        },
    }

    label_mapping = {
        int(i): str(label)
        for i, label in enumerate(label_encoder.classes_)
    }

    metadata = {
        "model_id": "exp06_robust_multimodal_species_v1",
        "display_name": "Exp06 Robust Multimodal CNN",
        "model_type": "torch_multimodal_robust",
        "model_file": "model.pt",
        "labels_file": "labels.json",
        "architecture": "RobustMultimodalClassifier",
        "model_module": "lif_thesis.models.robust_multimodal_cnn",
        "preprocessing": {
            "fluorescence_threshold": config.fluorescence_threshold,
            "scattering_target_acquisitions": RAPIDE_DIMS.SCATTERING_TARGET_ACQUISITIONS,
            "n_scattering_angles": RAPIDE_DIMS.N_SCATTERING_ANGLES,
            "scattering_normalize": True,
            "spectrometer_scaler": "spectrometer_scaler.joblib",
            "lifetime_scaler": "lifetime_scaler.joblib",
            "scattering_scaler": "scattering_scaler.joblib",
            "scalar_scaler": "scalar_scaler.joblib",
        },
        "uses_rejection": True,
        "rejection": asdict(rejection_config),
        "performance": {
            "best_epoch": best_epoch,
            "closed_set_accuracy": metrics["test_closed_set"]["accuracy"],
            "closed_set_balanced_accuracy": metrics["test_closed_set"]["balanced_accuracy"],
            "closed_set_macro_f1": metrics["test_closed_set"]["macro_f1"],
            "unknown_fraction": metrics["test_rejection_summary"]["unknown_fraction"],
        },
    }

    save_model_bundle(
        model=model,
        model_id="exp06_robust_multimodal_species_v1",
        deploy_dir=Path("models/trained"),
        checkpoint=deployment_checkpoint,
        metadata=metadata,
        label_mapping=label_mapping,
        label_encoder=label_encoder,
        scalers={
            "spectrometer": spec_scaler,
            "lifetime": life_scaler,
            "scattering": scat_scaler,
            "scalar": scalar_scaler,
        },
        extra_artifacts={
            "scalar_distance_reference": distance_reference,
        },
    )


    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    test_pred_df.to_csv(output_dir / "test_predictions_with_unknown.csv", index=False)
    file_qc.to_csv(output_dir / "test_file_qc_summary.csv", index=False)

    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "val_idx.npy", val_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    print(f"\nSaved outputs to: {output_dir}")
    print("\nFinal test performance:")
    print(
        json.dumps(
            {
                "closed_set_accuracy": metrics["test_closed_set"]["accuracy"],
                "closed_set_balanced_accuracy": metrics["test_closed_set"]["balanced_accuracy"],
                "closed_set_macro_f1": metrics["test_closed_set"]["macro_f1"],
                "unknown_fraction_after_rejection": metrics["test_rejection_summary"]["unknown_fraction"],
                "best_epoch": best_epoch,
            },
            indent=4,
        )
    )


if __name__ == "__main__":
    main()
