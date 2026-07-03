from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path


def find_raw_files(source_dir: Path) -> list[Path]:
    return sorted(source_dir.rglob("*.raw"))


def copy_mock_raw_files(
    source_dir: Path,
    output_dir: Path,
    count: int,
    interval_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = find_raw_files(source_dir)

    if not raw_files:
        raise FileNotFoundError(f"No .raw files found in {source_dir}")

    print(f"Found {len(raw_files)} source raw files.")
    print(f"Writing mock files to: {output_dir}")

    for i in range(count):
        source_file = raw_files[i % len(raw_files)]

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_name = f"MOCK_{i + 1:04d}_{timestamp}__{source_file.name}"
        output_path = output_dir / output_name

        shutil.copy2(source_file, output_path)

        print(f"[{i + 1}/{count}] Created: {output_path}")

        if i < count - 1:
            time.sleep(interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mock Rapid-E .raw files for realtime pipeline testing."
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("src/lif_thesis/streaming/rapid_e_downloads"),
        help="Directory containing existing .raw files to copy.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/realtime_mock"),
        help="Directory where mock .raw files should be created.",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of mock raw files to generate.",
    )

    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=2.0,
        help="Seconds to wait between creating files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    copy_mock_raw_files(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        count=args.count,
        interval_seconds=args.interval_seconds,
    )


if __name__ == "__main__":
    main()