"""
Experiment 01

Paper-style Random Forest with grouped train/validation/test split.

This experiment reproduces the paper feature engineering while replacing
the particle-level split with a grouped raw-file split to evaluate
generalization to unseen aerosol samples.
"""

from pathlib import Path

import pandas as pd

from lif_thesis.models.paper_rf import (
    PaperRFConfig,
    run_paper_rf_grouped_experiment,
)


DATA_PATH = Path("data/processed/bacterial_samples.parquet")


def main():

    print("Loading processed bacterial dataset...")

    df = pd.read_parquet(DATA_PATH)

    config = PaperRFConfig(
        random_state=42,
    )

    run_paper_rf_grouped_experiment(
        df=df,
        model_id="exp01_grouped_paper_rf_v1",
        display_name="Experiment 01 Grouped Paper RF",
        label_col="species",
        group_col="raw_file",
        spectrometer_col="spectrometer",
        lifetime_col="lifetime",
        scattering_col="scattering_image",
        output_dir="results/exp01_grouped_paper_rf",
        deploy_dir="models/trained",
        config=config,
    )


if __name__ == "__main__":
    main()