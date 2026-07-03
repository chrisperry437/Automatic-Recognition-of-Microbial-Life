"""
Preprocessing utilities for Rapid-E particle data.

This module contains reusable preprocessing steps used by both offline
experiments and the real-time inference pipeline.

Core operations:
    - fluorescence thresholding
    - fluorescence spectra flattening
    - fluorescence lifetime flattening
    - scattering crop / pad / normalize
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from lif_thesis.data.schemas import RAPIDE_DIMS

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for Rapid-E particle preprocessing."""

    fluorescence_threshold: float = 2000.0
    scattering_target_acquisitions: int = RAPIDE_DIMS.SCATTERING_TARGET_ACQUISITIONS
    scattering_normalize: bool = True
    scattering_fill_value: float = 0.0


def to_numpy_array(value: Any, dtype: type = float) -> np.ndarray:
    """
    Convert a stored feature value to a NumPy array.

    Handles values that are already arrays, Python lists, tuples, or stringified
    lists from CSV/parquet round trips.

    Parameters
    ----------
    value:
        Input value to convert.
    dtype:
        Desired NumPy dtype.

    Returns
    -------
    np.ndarray
        Converted array.
    """
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)

    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=dtype)

    if pd.isna(value):
        return np.asarray([], dtype=dtype)

    if isinstance(value, str):
        text = value.strip()

        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]

        if not text:
            return np.asarray([], dtype=dtype)

        # Handles strings like "1, 2, 3" or "1 2 3"
        separator = "," if "," in text else None
        return np.fromstring(text, sep=separator or " ", dtype=dtype)

    return np.asarray(value, dtype=dtype)


def max_fluorescence_intensity(
    spectra: Any | None = None,
    lifetime: Any | None = None,
) -> float:
    """
    Compute the maximum fluorescence intensity for one particle.

    Parameters
    ----------
    spectra:
        Fluorescence spectra array-like value.
    lifetime:
        Fluorescence lifetime array-like value.

    Returns
    -------
    float
        Maximum intensity across available fluorescence inputs.
    """
    maxima: list[float] = []

    if spectra is not None:
        spectra_arr = to_numpy_array(spectra)
        if spectra_arr.size:
            maxima.append(float(np.nanmax(spectra_arr)))

    if lifetime is not None:
        lifetime_arr = to_numpy_array(lifetime)
        if lifetime_arr.size:
            maxima.append(float(np.nanmax(lifetime_arr)))

    if not maxima:
        return np.nan

    return max(maxima)


def add_peak_fluorescence_column(
    df: pd.DataFrame,
    spectra_col: str = "fluorescence_spectra",
    lifetime_col: str = "fluorescence_lifetime",
    output_col: str = "peak_fluorescence",
) -> pd.DataFrame:
    """
    Add a peak fluorescence intensity column.

    Parameters
    ----------
    df:
        Particle-level dataframe.
    spectra_col:
        Column containing fluorescence spectra arrays.
    lifetime_col:
        Column containing fluorescence lifetime arrays.
    output_col:
        Name of output column.

    Returns
    -------
    pd.DataFrame
        Copy of dataframe with peak fluorescence column.
    """
    df = df.copy()

    has_spectra = spectra_col in df.columns
    has_lifetime = lifetime_col in df.columns

    if not has_spectra and not has_lifetime:
        raise KeyError(
            f"Neither '{spectra_col}' nor '{lifetime_col}' exists in dataframe."
        )

    df[output_col] = df.apply(
        lambda row: max_fluorescence_intensity(
            spectra=row[spectra_col] if has_spectra else None,
            lifetime=row[lifetime_col] if has_lifetime else None,
        ),
        axis=1,
    )

    return df


def filter_by_fluorescence_threshold(
    df: pd.DataFrame,
    threshold: float = 2000.0,
    peak_col: str = "peak_fluorescence",
    spectra_col: str = "fluorescence_spectra",
    lifetime_col: str = "fluorescence_lifetime",
    keep_rejected: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter particles using a peak fluorescence threshold.

    If the peak fluorescence column does not exist, it is computed from spectra
    and/or lifetime data.

    Parameters
    ----------
    df:
        Particle-level dataframe.
    threshold:
        Minimum peak fluorescence intensity required.
    peak_col:
        Name of peak fluorescence column.
    spectra_col:
        Column containing fluorescence spectra arrays.
    lifetime_col:
        Column containing fluorescence lifetime arrays.
    keep_rejected:
        If True, return both accepted and rejected particles.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]
        Accepted particles, or accepted and rejected particles.
    """
    if peak_col not in df.columns:
        df = add_peak_fluorescence_column(
            df,
            spectra_col=spectra_col,
            lifetime_col=lifetime_col,
            output_col=peak_col,
        )
    else:
        df = df.copy()

    mask = df[peak_col] >= threshold

    accepted = df.loc[mask].copy()
    rejected = df.loc[~mask].copy()

    if keep_rejected:
        return accepted, rejected

    return accepted


def flatten_array_feature(value: Any, prefix: str) -> dict[str, float]:
    """
    Flatten one array-like feature into a dictionary of scalar columns.

    Parameters
    ----------
    value:
        Array-like input.
    prefix:
        Prefix for generated column names.

    Returns
    -------
    dict[str, float]
        Flattened feature dictionary.
    """
    arr = to_numpy_array(value).ravel()

    return {f"{prefix}_{i:04d}": float(v) for i, v in enumerate(arr)}


def flatten_feature_column(
    df: pd.DataFrame,
    column: str,
    prefix: str,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Flatten an array-valued dataframe column into scalar feature columns.

    Parameters
    ----------
    df:
        Input dataframe.
    column:
        Name of array-valued column.
    prefix:
        Prefix for output feature columns.
    drop_original:
        Whether to remove the original array-valued column.

    Returns
    -------
    pd.DataFrame
        Dataframe with flattened feature columns.
    """
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    df = df.copy()

    feature_rows = [flatten_array_feature(value, prefix) for value in df[column]]

    features = pd.DataFrame(feature_rows, index=df.index)

    output = pd.concat([df, features], axis=1)

    if drop_original:
        output = output.drop(columns=[column])

    return output


def flatten_spectra(
    df: pd.DataFrame,
    spectra_col: str = "fluorescence_spectra",
    prefix: str = "fs",
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Flatten fluorescence spectra into scalar feature columns.

    Parameters
    ----------
    df:
        Particle-level dataframe.
    spectra_col:
        Column containing spectra arrays.
    prefix:
        Prefix for output feature columns.
    drop_original:
        Whether to remove the original spectra column.

    Returns
    -------
    pd.DataFrame
        Dataframe with flattened spectra features.
    """
    return flatten_feature_column(
        df=df,
        column=spectra_col,
        prefix=prefix,
        drop_original=drop_original,
    )


def flatten_lifetime(
    df: pd.DataFrame,
    lifetime_col: str = "fluorescence_lifetime",
    prefix: str = "lt",
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Flatten fluorescence lifetime data into scalar feature columns.

    Parameters
    ----------
    df:
        Particle-level dataframe.
    lifetime_col:
        Column containing lifetime arrays.
    prefix:
        Prefix for output feature columns.
    drop_original:
        Whether to remove the original lifetime column.

    Returns
    -------
    pd.DataFrame
        Dataframe with flattened lifetime features.
    """
    return flatten_feature_column(
        df=df,
        column=lifetime_col,
        prefix=prefix,
        drop_original=drop_original,
    )


def crop_or_pad_scattering(
    scattering: Any,
    target_acquisitions: int = 60,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Crop or pad a scattering array to a fixed acquisition length.

    Expected input shape is usually:
        acquisitions x angles

    If a 1D vector is provided, it is treated as:
        acquisitions x 1

    Parameters
    ----------
    scattering:
        Scattering image / signal array.
    target_acquisitions:
        Target number of acquisition time steps.
    fill_value:
        Value used for padding shorter signals.

    Returns
    -------
    np.ndarray
        Array with shape (target_acquisitions, n_angles).
    """
    arr = to_numpy_array(scattering)

    if arr.size == 0:
        return np.full((target_acquisitions, 1), fill_value, dtype=float)

    if arr.ndim == 1:
        n_angles = RAPIDE_DIMS.N_SCATTERING_ANGLES
        remainder = arr.size % n_angles

        if remainder != 0:
            pad_width = n_angles - remainder
            arr = np.pad(
                arr,
                (0, pad_width),
                mode="constant",
                constant_values=fill_value,
            )

        arr = arr.reshape(-1, n_angles)

    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    n_acquisitions, n_angles = arr.shape

    if n_acquisitions >= target_acquisitions:
        return arr[:target_acquisitions, :].astype(float, copy=False)

    padded = np.full(
        (target_acquisitions, n_angles),
        fill_value,
        dtype=float,
    )
    padded[:n_acquisitions, :] = arr

    return padded


def normalize_scattering(
    scattering: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize scattering values to the [0, 1] range per particle.

    Parameters
    ----------
    scattering:
        Cropped/padded scattering array.
    eps:
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalized scattering array.
    """
    arr = np.asarray(scattering, dtype=float)

    min_value = np.nanmin(arr)
    max_value = np.nanmax(arr)

    value_range = max_value - min_value

    if value_range < eps:
        return np.zeros_like(arr, dtype=float)

    return (arr - min_value) / value_range


def preprocess_scattering_array(
    scattering: Any,
    target_acquisitions: int = 60,
    normalize: bool = True,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Crop, pad, and optionally normalize one scattering array.

    Parameters
    ----------
    scattering:
        Raw scattering signal.
    target_acquisitions:
        Target number of acquisitions.
    normalize:
        Whether to normalize values to [0, 1].
    fill_value:
        Padding value.

    Returns
    -------
    np.ndarray
        Preprocessed scattering array.
    """
    arr = crop_or_pad_scattering(
        scattering=scattering,
        target_acquisitions=target_acquisitions,
        fill_value=fill_value,
    )

    if normalize:
        arr = normalize_scattering(arr)

    return arr


def preprocess_scattering_column(
    df: pd.DataFrame,
    scattering_col: str = "scattering",
    output_col: str = "scattering_processed",
    target_acquisitions: int = 60,
    normalize: bool = True,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Apply crop / pad / normalize to a scattering column.

    Parameters
    ----------
    df:
        Input dataframe.
    scattering_col:
        Column containing raw scattering arrays.
    output_col:
        Column where processed scattering arrays are stored.
    target_acquisitions:
        Target number of acquisitions.
    normalize:
        Whether to normalize scattering arrays.
    fill_value:
        Padding value.

    Returns
    -------
    pd.DataFrame
        Dataframe with processed scattering column.
    """
    if scattering_col not in df.columns:
        raise KeyError(f"Column not found: {scattering_col}")

    df = df.copy()

    df[output_col] = df[scattering_col].apply(
        lambda x: preprocess_scattering_array(
            scattering=x,
            target_acquisitions=target_acquisitions,
            normalize=normalize,
            fill_value=fill_value,
        )
    )

    return df


def flatten_scattering(
    df: pd.DataFrame,
    scattering_col: str = "scattering_processed",
    prefix: str = "si",
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Flatten processed scattering arrays into scalar feature columns.

    Parameters
    ----------
    df:
        Input dataframe.
    scattering_col:
        Column containing processed scattering arrays.
    prefix:
        Prefix for output feature columns.
    drop_original:
        Whether to remove original processed scattering column.

    Returns
    -------
    pd.DataFrame
        Dataframe with flattened scattering features.
    """
    return flatten_feature_column(
        df=df,
        column=scattering_col,
        prefix=prefix,
        drop_original=drop_original,
    )


def build_feature_matrix(
    df: pd.DataFrame,
    feature_prefixes: tuple[str, ...] = ("fs_", "lt_", "si_"),
) -> pd.DataFrame:
    """
    Extract flattened feature columns for model input.

    Parameters
    ----------
    df:
        Dataframe containing flattened features.
    feature_prefixes:
        Prefixes identifying model feature columns.

    Returns
    -------
    pd.DataFrame
        Feature matrix.
    """
    feature_cols = [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in feature_prefixes)
    ]

    if not feature_cols:
        raise ValueError(
            f"No feature columns found with prefixes: {feature_prefixes}"
        )

    return df[feature_cols].copy()


def preprocess_particles(
    df: pd.DataFrame,
    config: PreprocessingConfig | None = None,
    spectra_col: str = "fluorescence_spectra",
    lifetime_col: str = "fluorescence_lifetime",
    scattering_col: str = "scattering",
    flatten: bool = True,
    keep_rejected: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline for particle-level Rapid-E data.

    Steps:
        1. Add peak fluorescence intensity
        2. Filter by fluorescence threshold
        3. Flatten spectra
        4. Flatten lifetime
        5. Crop / pad / normalize scattering
        6. Flatten scattering

    Parameters
    ----------
    df:
        Input particle dataframe.
    config:
        Preprocessing configuration.
    spectra_col:
        Column containing fluorescence spectra.
    lifetime_col:
        Column containing fluorescence lifetime data.
    scattering_col:
        Column containing scattering data.
    flatten:
        Whether to flatten feature arrays into scalar columns.
    keep_rejected:
        Whether to return rejected low-fluorescence particles.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]
        Preprocessed accepted particles, optionally with rejected particles.
    """
    config = config or PreprocessingConfig()

    filtered = filter_by_fluorescence_threshold(
        df=df,
        threshold=config.fluorescence_threshold,
        spectra_col=spectra_col,
        lifetime_col=lifetime_col,
        keep_rejected=keep_rejected,
    )

    if keep_rejected:
        accepted, rejected = filtered
    else:
        accepted = filtered
        rejected = None

    if spectra_col in accepted.columns and flatten:
        accepted = flatten_spectra(
            accepted,
            spectra_col=spectra_col,
            prefix="fs",
            drop_original=False,
        )

    if lifetime_col in accepted.columns and flatten:
        accepted = flatten_lifetime(
            accepted,
            lifetime_col=lifetime_col,
            prefix="lt",
            drop_original=False,
        )

    if scattering_col in accepted.columns:
        accepted = preprocess_scattering_column(
            accepted,
            scattering_col=scattering_col,
            output_col="scattering_processed",
            target_acquisitions=config.scattering_target_acquisitions,
            normalize=config.scattering_normalize,
            fill_value=config.scattering_fill_value,
        )

        if flatten:
            accepted = flatten_scattering(
                accepted,
                scattering_col="scattering_processed",
                prefix="si",
                drop_original=False,
            )

    if keep_rejected:
        return accepted, rejected

    return accepted