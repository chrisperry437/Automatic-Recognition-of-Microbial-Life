from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import torch


# in src/lif_thesis/models/checkpoint.py

def save_model_bundle(
    *,
    model,
    model_id: str,
    deploy_dir: Path,
    checkpoint: dict[str, Any],
    metadata: dict[str, Any],
    label_mapping: dict[int, str],
    label_encoder=None,
    scalers: dict[str, Any] | None = None,
    extra_artifacts: dict[str, Any] | None = None,
    framework: str = "torch",
) -> None:
    deploy_dir = Path(deploy_dir) / model_id
    deploy_dir.mkdir(parents=True, exist_ok=True)

    if framework == "torch":
        checkpoint = dict(checkpoint)
        checkpoint["model_state_dict"] = model.state_dict()
        torch.save(checkpoint, deploy_dir / "model.pt")
    elif framework == "sklearn":
        joblib.dump(model, deploy_dir / "model.joblib")
        with open(deploy_dir / "checkpoint.json", "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=4)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    with open(deploy_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(deploy_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=4)

    if label_encoder is not None:
        joblib.dump(label_encoder, deploy_dir / "label_encoder.joblib")

    for name, artifact in (scalers or {}).items():
        joblib.dump(artifact, deploy_dir / f"{name}_scaler.joblib")

    for name, artifact in (extra_artifacts or {}).items():
        if name.endswith(".json"):
            with open(deploy_dir / name, "w", encoding="utf-8") as f:
                json.dump(artifact, f, indent=4)
        else:
            joblib.dump(artifact, deploy_dir / f"{name}.joblib")