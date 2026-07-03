"""
Experiment 04: Baseline 1D CNN on fluorescence spectra.

Uses:
- fluorescence threshold > 2000 a.u.
- grouped raw-file train/validation/test split
- spectrometer vector only
- deployable checkpoint bundle via cnn1d.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lif_thesis.models.cnn1d import (
    BaselineCNNPreprocessingConfig,
    BaselineCNNTrainingConfig,
    run_baseline_cnn_experiment,
)


DATA_PATH = Path("data/processed/bacterial_samples.parquet")


def main() -> None:
    print("Loading processed bacterial dataset...")

    df = pd.read_parquet(DATA_PATH)

    run_baseline_cnn_experiment(
        df=df,
        model_id="exp04_baseline_cnn_v1",
        display_name="Experiment 04 Baseline CNN",
        output_dir="results/exp04_baseline_cnn",
        deploy_dir="models/trained",
        preprocessing=BaselineCNNPreprocessingConfig(
            fluorescence_threshold=2000.0,
            label_col="species",
            group_col="raw_file",
            spectra_col="spectrometer",
            train_size=0.70,
            val_size=0.15,
            test_size=0.15,
        ),
        training=BaselineCNNTrainingConfig(
            batch_size=128,
            max_epochs=50,
            learning_rate=1e-3,
            weight_decay=1e-4,
            patience=8,
            random_state=42,
        ),
    )


if __name__ == "__main__":
    main()