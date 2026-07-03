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

BASE_DIR = Path(
    r"C:\Users\chris\OneDrive\Documents\Universitat de Barcelona\lif_thesis"
)

INPUT_DIRS = {
    "bacillus_cereus_possible": BASE_DIR
    / r"data\live_rapid_e\archive\old_june17_separated\bacillus_cereus_possible",
    "micrococcus": BASE_DIR
    / r"data\live_rapid_e\archive\old_june17_separated\micrococcus",
}

OUTPUT_DIR = BASE_DIR / r"data\live_rapid_e\parsed"

COMBINED_OUTPUT_FILE = OUTPUT_DIR / "june17_bacillus_cereus_possible_micrococcus_preprocessed.parquet"
SUMMARY_OUTPUT_FILE = OUTPUT_DIR / "june17_bacillus_cereus_possible_micrococcus_parse_summary.csv"


EXPERIMENT_METADATA = {
    "bacillus_cereus_possible": {
        "experiment_date": "2026-06-17",
        "experiment_name": "bacillus_cereus_possible",
        "expected_sample": "B_cereus",
        "expected_composition": "B_cereus_possible",
    },
    "micrococcus": {
        "experiment_date": "2026-06-17",
        "experiment_name": "micrococcus",
        "expected_sample": "M_luteus",
        "expected_composition": "M_luteus",
    },
}


def decode_raw_file(raw_path: Path, experiment_name: str) -> pd.DataFrame:
    """
    Parse one Rapid-E RAW file and standardize column names for preprocessing.
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
    df["experiment_name"] = experiment_name

    aliases = {
        "spectrometer": "fluorescence_spectra",
        "lifetime": "fluorescence_lifetime",
        "scattering_image": "scattering",
    }

    for source_col, target_col in aliases.items():
        if source_col in df.columns and target_col not in df.columns:
            df[target_col] = df[source_col]

    return df


def parse_raw_folder(input_dir: Path, experiment_name: str) -> pd.DataFrame:
    raw_files = sorted(input_dir.rglob("*.raw"))

    if not raw_files:
        raise FileNotFoundError(f"No .raw files found in: {input_dir}")

    logging.info("[%s] Found %d RAW files", experiment_name, len(raw_files))

    frames: list[pd.DataFrame] = []

    for i, raw_path in enumerate(raw_files, start=1):
        logging.info(
            "[%s] [%d/%d] Parsing %s",
            experiment_name,
            i,
            len(raw_files),
            raw_path.name,
        )

        try:
            df = decode_raw_file(raw_path, experiment_name)
            frames.append(df)
            logging.info("[%s] Parsed %d particles", experiment_name, len(df))

        except Exception as exc:
            logging.exception(
                "[%s] Failed to parse %s: %s",
                experiment_name,
                raw_path.name,
                exc,
            )

    if not frames:
        raise RuntimeError(f"No RAW files were successfully parsed for {experiment_name}.")

    return pd.concat(frames, ignore_index=True)


def add_peak_fluorescence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add peak_fluorescence before preprocessing if it is not already present.
    Useful for summary diagnostics.
    """
    if "peak_fluorescence" in df.columns:
        return df

    spectra_col = "fluorescence_spectra"
    if spectra_col not in df.columns:
        return df

    df = df.copy()
    df["peak_fluorescence"] = df[spectra_col].apply(lambda x: float(pd.Series(x).explode().max()))
    return df


def preprocess_experiment(experiment_name: str, input_dir: Path) -> pd.DataFrame:
    particles = parse_raw_folder(input_dir, experiment_name)
    particles = add_peak_fluorescence(particles)

    logging.info(
        "[%s] Total parsed particles before preprocessing: %d",
        experiment_name,
        len(particles),
    )

    if "peak_fluorescence" in particles.columns:
        logging.info(
            "[%s] Particles above 2000 a.u. before preprocessing: %d",
            experiment_name,
            int((particles["peak_fluorescence"] > 2000.0).sum()),
        )

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

    metadata = EXPERIMENT_METADATA[experiment_name]
    for key, value in metadata.items():
        processed[key] = value

    logging.info(
        "[%s] Particles after fluorescence threshold: %d",
        experiment_name,
        len(processed),
    )

    return processed


def drop_array_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parquet cannot reliably store object columns containing numpy arrays/lists.
    Keep flattened fs_*, lt_*, and si_* model-ready columns instead.
    """
    array_object_cols = [
        "fluorescence_spectra",
        "fluorescence_lifetime",
        "scattering",
        "scattering_processed",
        "spectrometer",
        "lifetime",
        "scattering_image",
    ]

    cols_to_drop = [col for col in array_object_cols if col in df.columns]
    return df.drop(columns=cols_to_drop)


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["experiment_date", "experiment_name", "expected_sample"])
        .size()
        .reset_index(name="n_particles_after_threshold")
    )

    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_frames: list[pd.DataFrame] = []

    for experiment_name, input_dir in INPUT_DIRS.items():
        processed = preprocess_experiment(experiment_name, input_dir)
        processed_for_parquet = drop_array_object_columns(processed)

        single_output_file = OUTPUT_DIR / f"june17_{experiment_name}_preprocessed.parquet"
        processed_for_parquet.to_parquet(single_output_file, index=False)

        logging.info("Saved %s", single_output_file)

        processed_frames.append(processed_for_parquet)

    combined = pd.concat(processed_frames, ignore_index=True)
    combined.to_parquet(COMBINED_OUTPUT_FILE, index=False)

    summary = make_summary(combined)
    summary.to_csv(SUMMARY_OUTPUT_FILE, index=False)

    logging.info("Saved combined parquet to: %s", COMBINED_OUTPUT_FILE)
    logging.info("Saved parse summary to: %s", SUMMARY_OUTPUT_FILE)
    logging.info("Done. Total processed particles: %d", len(combined))


if __name__ == "__main__":
    main()