# src/lif_thesis/streaming/rapid_e_raw_parser.py

from __future__ import annotations

import csv
import re
import struct
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
from lif_thesis.data.parsers import parse_raw_file

# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_DIR = Path("data/live_rapid_e/raw")
OUTPUT_DIR = Path("data/live_rapid_e/parsed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "raw_file_summary.csv"
RECORDS_CSV = OUTPUT_DIR / "particle_record_summary.csv"

MAGIC_BYTES = bytes.fromhex("22d5ebe7")

TIMESTAMP_PATTERN = re.compile(r"_(\d{12})\.raw$")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class RawFileSummary:
    filename: str
    path: str
    timestamp: str
    size_bytes: int
    size_mb: float
    magic_at_start: bool
    magic_count: int
    particle_record_count: int
    min_record_length: int
    max_record_length: int
    mean_record_length: float


@dataclass
class ParticleRecordSummary:
    filename: str
    record_index: int
    offset_start: int
    offset_end: int
    record_length: int
    timestamp: str

    uint32_be_0: int
    uint32_be_1: int
    uint32_be_2: int
    uint32_be_3: int
    uint32_be_4: int
    uint32_be_5: int
    uint32_be_6: int
    uint32_be_7: int
    uint32_be_8: int
    uint32_be_9: int
    uint32_be_10: int
    uint32_be_11: int

    first_64_hex: str


# =============================================================================
# HELPERS
# =============================================================================

def decode_raw_file(path: str | Path) -> pd.DataFrame:
    """
    Decode one Rapid-E RAW file into the particle dataframe expected by the
    preprocessing and prediction pipeline.
    """
    path = Path(path)

    parsed = parse_raw_file(
        path,
        keep_thresholds=False,
        extra_params=True,
    )

    df = parsed.particle_data.copy()

    # File metadata
    df["raw_file"] = path.name
    df["raw_path"] = str(path)
    df["filename"] = path.name

    # Standardize column names expected by preprocessing
    column_aliases = {
        "spectrometer": "fluorescence_spectra",
        "lifetime": "fluorescence_lifetime",
        "scattering_image": "scattering",
    }

    for source, target in column_aliases.items():
        if source in df.columns and target not in df.columns:
            df[target] = df[source]

    return df



def parse_timestamp_from_name(filename: str) -> Optional[datetime]:
    match = TIMESTAMP_PATTERN.search(filename)

    if not match:
        return None

    return datetime.strptime(match.group(1), "%Y%m%d%H%M")


def read_file_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


def find_magic_offsets(data: bytes) -> list[int]:
    offsets: list[int] = []
    start = 0

    while True:
        idx = data.find(MAGIC_BYTES, start)

        if idx == -1:
            break

        offsets.append(idx)
        start = idx + len(MAGIC_BYTES)

    return offsets


def split_into_records(data: bytes) -> list[tuple[int, int, bytes]]:
    """
    Splits a RAW file into candidate particle records.

    Each record starts with the Rapid-E magic bytes:
        22 d5 eb e7

    Record boundaries are estimated as:
        magic_offset[i] -> magic_offset[i + 1]

    The final record ends at EOF.
    """

    offsets = find_magic_offsets(data)

    records: list[tuple[int, int, bytes]] = []

    for i, start in enumerate(offsets):
        if i + 1 < len(offsets):
            end = offsets[i + 1]
        else:
            end = len(data)

        records.append(
            (
                start,
                end,
                data[start:end],
            )
        )

    return records


def unpack_uint32_be(chunk: bytes, count: int = 12) -> list[int]:
    values = []

    for i in range(count):
        start = i * 4
        end = start + 4

        if end > len(chunk):
            values.append(-1)
        else:
            values.append(
                struct.unpack(">I", chunk[start:end])[0]
            )

    return values


def inspect_record(
    filename: str,
    timestamp: str,
    record_index: int,
    offset_start: int,
    offset_end: int,
    record: bytes,
) -> ParticleRecordSummary:
    values = unpack_uint32_be(record, count=12)

    return ParticleRecordSummary(
        filename=filename,
        record_index=record_index,
        offset_start=offset_start,
        offset_end=offset_end,
        record_length=len(record),
        timestamp=timestamp,

        uint32_be_0=values[0],
        uint32_be_1=values[1],
        uint32_be_2=values[2],
        uint32_be_3=values[3],
        uint32_be_4=values[4],
        uint32_be_5=values[5],
        uint32_be_6=values[6],
        uint32_be_7=values[7],
        uint32_be_8=values[8],
        uint32_be_9=values[9],
        uint32_be_10=values[10],
        uint32_be_11=values[11],

        first_64_hex=record[:64].hex(),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"No rows to write for {path}")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# PARSING
# =============================================================================

def parse_raw_file(
    path: Path,
) -> tuple[RawFileSummary, list[ParticleRecordSummary]]:
    data = read_file_bytes(path)

    timestamp_dt = parse_timestamp_from_name(path.name)
    timestamp = timestamp_dt.isoformat() if timestamp_dt else ""

    records = split_into_records(data)

    record_lengths = [
        len(record)
        for _, _, record in records
    ]

    particle_summaries = [
        inspect_record(
            filename=path.name,
            timestamp=timestamp,
            record_index=i,
            offset_start=start,
            offset_end=end,
            record=record,
        )
        for i, (start, end, record) in enumerate(records)
    ]

    summary = RawFileSummary(
        filename=path.name,
        path=str(path),
        timestamp=timestamp,
        size_bytes=len(data),
        size_mb=round(len(data) / 1024 / 1024, 3),
        magic_at_start=data.startswith(MAGIC_BYTES),
        magic_count=len(records),
        particle_record_count=len(records),
        min_record_length=min(record_lengths) if record_lengths else 0,
        max_record_length=max(record_lengths) if record_lengths else 0,
        mean_record_length=(
            round(sum(record_lengths) / len(record_lengths), 2)
            if record_lengths
            else 0
        ),
    )

    return summary, particle_summaries


def main() -> None:
    raw_files = sorted(RAW_DIR.glob("*.raw"))

    if not raw_files:
        raise FileNotFoundError(
            f"No RAW files found in {RAW_DIR}"
        )

    all_file_summaries: list[RawFileSummary] = []
    all_particle_summaries: list[ParticleRecordSummary] = []

    print(f"Parsing RAW files from: {RAW_DIR}")
    print(f"Found {len(raw_files)} files.")

    for raw_file in raw_files:
        print(f"\nParsing: {raw_file.name}")

        file_summary, particle_summaries = parse_raw_file(raw_file)

        all_file_summaries.append(file_summary)
        all_particle_summaries.extend(particle_summaries)

        print(f"Size: {file_summary.size_mb} MB")
        print(f"Magic at start: {file_summary.magic_at_start}")
        print(f"Particle records: {file_summary.particle_record_count}")
        print(f"Min record length: {file_summary.min_record_length}")
        print(f"Max record length: {file_summary.max_record_length}")
        print(f"Mean record length: {file_summary.mean_record_length}")

    write_csv(
        SUMMARY_CSV,
        [asdict(row) for row in all_file_summaries],
    )

    write_csv(
        RECORDS_CSV,
        [asdict(row) for row in all_particle_summaries],
    )

    print("\nDone.")
    print(f"File summary saved to: {SUMMARY_CSV}")
    print(f"Particle record summary saved to: {RECORDS_CSV}")


if __name__ == "__main__":
    main()