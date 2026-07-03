from pathlib import Path
import re
import shutil
from datetime import datetime

INPUT_DIR = Path("data/live_rapid_e/experiment")
OUTPUT_DIR = Path("data/live_rapid_e/june17_separated")

WINDOWS = [
    ("ringer", "202606171003", "202606171047"),
    ("distilled_water_1107_1223", "202606171107", "202606171223"),
    ("bacillus_cereus_possible", "202606171224", "202606171246"),
    ("distilled_water_1248_1320", "202606171248", "202606171320"),
    ("micrococcus", "202606171323", "202606171345"),
    ("distilled_water_1346_1420", "202606171346", "202606171420"),
]

TIMESTAMP_RE = re.compile(r"(\d{12})")

def extract_timestamp(path: Path) -> datetime | None:
    match = TIMESTAMP_RE.search(path.stem)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d%H%M")

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    windows = [
        (
            label,
            datetime.strptime(start, "%Y%m%d%H%M"),
            datetime.strptime(end, "%Y%m%d%H%M"),
        )
        for label, start, end in WINDOWS
    ]

    files = sorted(INPUT_DIR.rglob("*.raw"))
    print(f"Found {len(files)} .raw files")

    counts = {label: 0 for label, _, _ in windows}
    unmatched = 0

    for file in files:
        ts = extract_timestamp(file)

        if ts is None:
            unmatched += 1
            continue

        matched = False

        for label, start, end in windows:
            if start <= ts <= end:
                out_dir = OUTPUT_DIR / label
                out_dir.mkdir(parents=True, exist_ok=True)

                shutil.copy2(file, out_dir / file.name)
                counts[label] += 1
                matched = True
                break

        if not matched:
            unmatched += 1

    print("\nCopied files:")
    for label, count in counts.items():
        print(f"{label}: {count}")

    print(f"\nUnmatched files: {unmatched}")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()