##Reproduction of originla Random forest Baseline

from pathlib import Path

import pandas as pd

from lif_thesis.models.updated_rf import run_baseline_rf_experiment


def main():
    data_path = Path("data/processed/bacterial_samples.parquet")
    df = pd.read_parquet(data_path)

    run_baseline_rf_experiment(
        df=df,
        label_col="species",
        group_col="raw_file",
        spectra_col="spectrometer",
        lifetime_col="lifetime",
        scalar_cols=["size", "time_asymmetry"],
        output_dir="results/baseline_rf_multimodal",
        random_state=42,
    )


if __name__ == "__main__":
    main()