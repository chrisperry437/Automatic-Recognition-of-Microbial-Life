from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from lif_thesis.data.parsers import parse_raw_file
from lif_thesis.data.preprocessing import (
    PreprocessingConfig,
    preprocess_particles,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

INPUT_DIR = Path(
    r"C:\Users\chris\OneDrive\Documents\Universitat de Barcelona\lif_thesis"
    r"\data\live_rapid_e\archive\sorted_by_experiment\2026-06-19"
    r"\bacillus25_micrococcus75_1426_1442"
)

OUTPUT_DIR = Path(
    r"C:\Users\chris\OneDrive\Documents\Universitat de Barcelona\lif_thesis"
    r"\data\live_rapid_e\parsed"
)

OUTPUT_FILE = OUTPUT_DIR / "june19_bacillus25_micrococcus75_1426_1442_preprocessed.parquet"


def decode_raw_file(raw_path: Path) -> pd.DataFrame:
    """
    Parse one Rapid-E RAW file using the canonical parsers.py parser
    and standardize column names for preprocessing.py.
    """
    parsed = parse_raw_file(
        raw_path,
        keep_thresholds=False,
        extra_params=True,
    )

    df = parsed.particle_data.copy()

    df["raw_file"] = raw_path.name
    df["raw_path"] = str(raw_path)
    df["source_file"] = raw_path.name
    df["source_path"] = str(raw_path)

    aliases = {
        "spectrometer": "fluorescence_spectra",
        "lifetime": "fluorescence_lifetime",
        "scattering_image": "scattering",
    }

    for source_col, target_col in aliases.items():
        if source_col in df.columns and target_col not in df.columns:
            df[target_col] = df[source_col]

    return df


def parse_raw_folder(input_dir: Path) -> pd.DataFrame:
    raw_files = sorted(input_dir.rglob("*.raw"))

    if not raw_files:
        raise FileNotFoundError(f"No .raw files found in: {input_dir}")

    logging.info("Found %d RAW files", len(raw_files))

    frames: list[pd.DataFrame] = []

    for i, raw_path in enumerate(raw_files, start=1):
        logging.info("[%d/%d] Parsing %s", i, len(raw_files), raw_path.name)

        try:
            df = decode_raw_file(raw_path)
            frames.append(df)
            logging.info("Parsed %d particles", len(df))

        except Exception as exc:
            logging.exception("Failed to parse %s: %s", raw_path.name, exc)

    if not frames:
        raise RuntimeError("No RAW files were successfully parsed.")

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    particles = parse_raw_folder(INPUT_DIR)

    logging.info("Total parsed particles before preprocessing: %d", len(particles))

    config = PreprocessingConfig(
        fluorescence_threshold=2000.0,
        scattering_target_acquisitions=60,
        scattering_normalize=True,
        scattering_fill_value=0.0,
    )

    processed = preprocess_particles(
        particles,
        config=config,
        spectra_col="fluorescence_spectra",
        lifetime_col="fluorescence_lifetime",
        scattering_col="scattering",
        flatten=True,
        keep_rejected=False,
    )

    logging.info("Particles after fluorescence threshold: %d", len(processed))

    processed["experiment_date"] = "2026-06-19"
    processed["experiment_name"] = "bacillus25_micrococcus75_1426_1442"
    processed["expected_composition"] = "B_cereus_25_M_luteus_75"

    # Parquet cannot reliably store object columns containing 2D numpy arrays.
    # Keep the flattened fs_*, lt_*, and si_* model-ready columns instead.
    array_object_cols = [
        "fluorescence_spectra",
        "fluorescence_lifetime",
        "scattering",
        "scattering_processed",
        "spectrometer",
        "lifetime",
        "scattering_image",
    ]

    cols_to_drop = [col for col in array_object_cols if col in processed.columns]

    processed_for_parquet = processed.drop(columns=cols_to_drop)

    processed_for_parquet.to_parquet(OUTPUT_FILE, index=False)


    logging.info("Saved preprocessed parquet to: %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()