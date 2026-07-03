from __future__ import annotations

import ast
import json
import math
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path("data/live_rapid_e/experiments")
RESULTS_ROOT = Path("results/live_rapid_e/exp07_model_evaluation")
LABEL_MAP_PATH = Path("models/label_maps/multimodal_species_v1_labels.json")

TARGET_DATES = {"2026-06-17", "2026-06-19"}

MODELS = {
    "exp05_multimodal": {
        "model_path": Path("models/trained/multimodal_species_v1.pt"),
        "robust": False,
    },
    "exp06_robust": {
        "model_path": Path("models/trained/robust_multimodal_species_v1.pt"),
        "robust": True,
    },
}

def prediction_table_exists(output_dir: Path) -> bool:
    sqlite_path = output_dir / "predictions.sqlite"

    if not sqlite_path.exists() or sqlite_path.stat().st_size == 0:
        return False

    try:
        with sqlite3.connect(sqlite_path) as conn:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table'",
                conn,
            )["name"].tolist()

        return "predictions" in tables

    except Exception:
        return False


def count_prediction_rows(output_dir: Path) -> int:
    csv_path = output_dir / "predictions.csv"
    sqlite_path = output_dir / "predictions.sqlite"

    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            return len(pd.read_csv(csv_path))
        except Exception:
            pass

    if sqlite_path.exists() and sqlite_path.stat().st_size > 0:
        try:
            with sqlite3.connect(sqlite_path) as conn:
                tables = pd.read_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                    conn,
                )["name"].tolist()

                if "predictions" not in tables:
                    return 0

                out = pd.read_sql("SELECT COUNT(*) AS n FROM predictions", conn)
                return int(out["n"].iloc[0])

        except Exception:
            return 0

    return 0

def run_stream_processor_batch(
    input_dir: Path,
    output_dir: Path,
    model_path: Path,
    max_runtime_seconds: int = 3600,
    stable_seconds: int = 90,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(input_dir.rglob("*.raw"))
    n_raw_files = len(raw_files)

    if n_raw_files == 0:
        raise RuntimeError(f"No RAW files found in {input_dir}")

    cmd = [
        sys.executable,
        "-m",
        "src.lif_thesis.realtime.stream_processor",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--model-path",
        str(model_path),
        "--model-type",
        "torch",
        "--label-mapping-path",
        str(LABEL_MAP_PATH),
        "--include-existing",
    ]

    print("\nRunning:")
    print(" ".join(cmd))
    print(f"Waiting for {n_raw_files} RAW files to finish processing.")

    process = subprocess.Popen(cmd)

    start = time.time()
    last_row_count = 0
    last_change_time = time.time()

    try:
        while True:
            if process.poll() is not None:
                print("stream_processor exited normally.")
                return

            row_count = count_prediction_rows(output_dir)

            if row_count != last_row_count:
                print(f"Prediction rows so far: {row_count}")
                last_row_count = row_count
                last_change_time = time.time()

            has_valid_table = prediction_table_exists(output_dir)
            has_predictions = row_count > 0
            stable_long_enough = time.time() - last_change_time >= stable_seconds

            if has_valid_table and has_predictions and stable_long_enough:
                print(
                    f"Prediction row count stable for {stable_seconds}s. "
                    "Stopping stream_processor."
                )

                process.terminate()

                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

                return

            if time.time() - start > max_runtime_seconds:
                process.terminate()
                process.kill()

                raise TimeoutError(
                    f"stream_processor exceeded {max_runtime_seconds}s for {input_dir}. "
                    f"Rows collected: {row_count}"
                )

            time.sleep(10)

    except KeyboardInterrupt:
        process.terminate()
        raise


def read_predictions(output_dir: Path) -> pd.DataFrame:
    csv_path = output_dir / "predictions.csv"
    sqlite_path = output_dir / "predictions.sqlite"

    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                print(f"Reading predictions from: {csv_path}")
                return df
        except Exception as e:
            print(f"Could not read CSV predictions: {e}")

    if sqlite_path.exists() and sqlite_path.stat().st_size > 0:
        with sqlite3.connect(sqlite_path) as conn:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table'",
                conn,
            )["name"].tolist()

            if "predictions" not in tables:
                raise RuntimeError(
                    f"SQLite file exists but has no predictions table: {sqlite_path}"
                )

            df = pd.read_sql("SELECT * FROM predictions", conn)

        if df.empty:
            raise RuntimeError(f"Predictions table exists but is empty: {sqlite_path}")

        print(f"Reading predictions from: {sqlite_path}")
        return df

    raise FileNotFoundError(f"No usable predictions found under {output_dir}")


def parse_array(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)

    if pd.isna(value):
        return np.asarray([], dtype=float)

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            try:
                parsed = ast.literal_eval(value)
            except Exception:
                return np.asarray([], dtype=float)

        try:
            return np.asarray(parsed, dtype=float)
        except Exception:
            return np.asarray([], dtype=float)

    return np.asarray([], dtype=float)


def compute_peak_fluorescence(row: pd.Series) -> float:
    for col in ["peak_fluorescence", "fluorescence_peak"]:
        if col in row and pd.notna(row[col]):
            return float(row[col])

    for col in ["spectrometer", "fluorescence_spectra"]:
        if col in row:
            arr = parse_array(row[col])
            if arr.size > 0:
                return float(np.nanmax(arr))

    return math.nan


def compute_entropy(row: pd.Series) -> float:
    if "entropy" in row and pd.notna(row["entropy"]):
        return float(row["entropy"])

    prob_cols = [c for c in row.index if c.startswith("prob_")]

    if not prob_cols:
        return math.nan

    probs = row[prob_cols].astype(float).to_numpy()
    probs = probs[np.isfinite(probs)]

    if probs.size == 0 or probs.sum() <= 0:
        return math.nan

    probs = probs / probs.sum()
    probs = probs[probs > 0]

    return float(-np.sum(probs * np.log(probs)))


def standardize_predictions(
    df: pd.DataFrame,
    model_name: str,
    experiment_date: str,
    experiment_id: str,
    experiment_block: str,
    robust_model: bool,
) -> pd.DataFrame:
    out = df.copy()

    if "predicted_label" not in out.columns and "closed_set_prediction" in out.columns:
        out["predicted_label"] = out["closed_set_prediction"]

    if "prediction_confidence" not in out.columns:
        prob_cols = [c for c in out.columns if c.startswith("prob_")]
        out["prediction_confidence"] = out[prob_cols].max(axis=1) if prob_cols else np.nan

    if "robust_prediction" not in out.columns:
        out["robust_prediction"] = out["predicted_label"]

    if not robust_model:
        out["robust_prediction"] = out["predicted_label"]
        out["unknown_flag"] = False
    else:
        out["unknown_flag"] = (
            out["robust_prediction"]
            .astype(str)
            .str.lower()
            .isin(["unknown", "rejected", "ood"])
        )

    out["entropy"] = out.apply(compute_entropy, axis=1)
    out["peak_fluorescence"] = out.apply(compute_peak_fluorescence, axis=1)

    if "size" not in out.columns:
        out["size"] = np.nan

    if "time_asymmetry" not in out.columns:
        out["time_asymmetry"] = np.nan

    out["model_name"] = model_name
    out["experiment_date"] = experiment_date
    out["experiment_id"] = experiment_id
    out["experiment_block"] = experiment_block

    keep_cols = [
        "model_name",
        "experiment_date",
        "experiment_id",
        "experiment_block",
        "raw_file",
        "source_file",
        "particle_index",
        "predicted_label",
        "prediction_confidence",
        "entropy",
        "robust_prediction",
        "unknown_flag",
        "peak_fluorescence",
        "size",
        "time_asymmetry",
    ]

    prob_cols = [c for c in out.columns if c.startswith("prob_")]
    keep_cols += prob_cols

    return out[[c for c in keep_cols if c in out.columns]]


def summarize_experiment(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)

    summary = {
        "model_name": df["model_name"].iloc[0],
        "experiment_date": df["experiment_date"].iloc[0],
        "experiment_id": df["experiment_id"].iloc[0],
        "experiment_block": df["experiment_block"].iloc[0],
        "n_particles": total,
        "unknown_fraction": float(df["unknown_flag"].mean()) if total else np.nan,
        "median_confidence": float(df["prediction_confidence"].median()) if total else np.nan,
        "median_entropy": float(df["entropy"].median()) if total else np.nan,
        "median_peak_fluorescence": float(df["peak_fluorescence"].median()) if total else np.nan,
        "fraction_above_2000": float((df["peak_fluorescence"] > 2000).mean()) if total else np.nan,
        "median_size": float(df["size"].median()) if total else np.nan,
        "median_time_asymmetry": float(df["time_asymmetry"].median()) if total else np.nan,
    }

    pred_fracs = df["robust_prediction"].astype(str).value_counts(normalize=True).to_dict()

    for label, frac in pred_fracs.items():
        safe = (
            label.replace(" ", "_")
            .replace(".", "")
            .replace("/", "_")
            .replace("-", "_")
        )
        summary[f"frac_{safe}"] = float(frac)

    return pd.DataFrame([summary])


def discover_experiments() -> list[Path]:
    experiment_dirs = []

    for date_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not date_dir.is_dir():
            continue

        if date_dir.name not in TARGET_DATES:
            continue

        for exp_dir in sorted(date_dir.iterdir()):
            if exp_dir.is_dir() and list(exp_dir.rglob("*.raw")):
                experiment_dirs.append(exp_dir)

    return experiment_dirs


def parse_experiment_dir(exp_dir: Path):
    experiment_date = exp_dir.parent.name

    if exp_dir.name.startswith("EXP"):
        parts = exp_dir.name.split("_", 1)
        experiment_id = parts[0]
        experiment_block = parts[1] if len(parts) > 1 else exp_dir.name
        output_name = exp_dir.name
    else:
        experiment_id = exp_dir.name
        experiment_block = exp_dir.name
        output_name = exp_dir.name

    return experiment_date, experiment_id, experiment_block, output_name


def main():
    experiment_dirs = discover_experiments()

    if not experiment_dirs:
        raise RuntimeError(f"No June 17 or June 19 experiment folders found in {EXPERIMENT_ROOT}")

    if RESULTS_ROOT.exists():
        shutil.rmtree(RESULTS_ROOT)

    all_particle_outputs = []
    all_summaries = []

    for exp_dir in experiment_dirs:
        experiment_date, experiment_id, experiment_block, output_name = parse_experiment_dir(exp_dir)

        print("\n" + "=" * 80)
        print(f"Experiment: {experiment_date} / {experiment_id} / {experiment_block}")
        print(f"Input: {exp_dir}")
        print(f"RAW files: {len(list(exp_dir.rglob('*.raw')))}")

        for model_name, cfg in MODELS.items():
            model_path = cfg["model_path"]

            if not model_path.exists():
                raise FileNotFoundError(f"Missing model: {model_path}")

            output_dir = RESULTS_ROOT / model_name / experiment_date / output_name

            run_stream_processor_batch(
                input_dir=exp_dir,
                output_dir=output_dir,
                model_path=model_path,
            )

            raw_predictions = read_predictions(output_dir)

            standardized = standardize_predictions(
                raw_predictions,
                model_name=model_name,
                experiment_date=experiment_date,
                experiment_id=experiment_id,
                experiment_block=experiment_block,
                robust_model=cfg["robust"],
            )

            particle_parquet = output_dir / "particle_predictions.parquet"
            particle_csv = output_dir / "particle_predictions.csv"
            summary_csv = output_dir / "experiment_summary.csv"

            standardized.to_parquet(particle_parquet, index=False)
            standardized.to_csv(particle_csv, index=False)

            summary = summarize_experiment(standardized)
            summary.to_csv(summary_csv, index=False)

            all_particle_outputs.append(standardized)
            all_summaries.append(summary)

            print(f"Saved: {particle_parquet}")
            print(f"Saved: {summary_csv}")

    combined_particles = pd.concat(all_particle_outputs, ignore_index=True)
    combined_summary = pd.concat(all_summaries, ignore_index=True)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    combined_particles.to_parquet(
        RESULTS_ROOT / "all_particle_predictions.parquet",
        index=False,
    )

    combined_particles.to_csv(
        RESULTS_ROOT / "all_particle_predictions.csv",
        index=False,
    )

    combined_summary.to_csv(
        RESULTS_ROOT / "all_experiment_summary.csv",
        index=False,
    )

    print("\nDONE")
    print(f"Combined particle predictions: {RESULTS_ROOT / 'all_particle_predictions.parquet'}")
    print(f"Combined summary: {RESULTS_ROOT / 'all_experiment_summary.csv'}")


if __name__ == "__main__":
    main()