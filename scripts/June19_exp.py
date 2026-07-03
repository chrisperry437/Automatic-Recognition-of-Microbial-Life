from pathlib import Path
import shutil
import re
from datetime import datetime

SOURCE_DIR = Path("data/live_rapid_e/June_19")
OUTPUT_DIR = Path("data/live_rapid_e/sorted_by_experiment")

DRY_RUN = False  # set to False when the printed moves look correct

# June 19 experiment sequence blocks based on your observed ordering
JUNE19_BLOCKS = [
    ("distilled_water_1216_1240_75pct_35A", 10000, 10023),
    ("distilled_water_1240_1246_80pct_36A", 10024, 10029),
    ("bacillus75_micrococcus25_1250_1312", 10030, 10051),
    ("distilled_water_1313_1346_80pct_36A", 10052, 10084),
    ("bacillus50_micrococcus50_1347_1402", 10085, 10099),
    ("distilled_water_1404_1424_80pct_36A", 10100, 10119),
    ("bacillus25_micrococcus75_1426_1442", 10120, 10135),
    ("distilled_water_after_1442_80pct_36A", 10136, 10201),
]

FOLDER_PATTERN = re.compile(r"^D_(\d+)_(\d{12})$")


def parse_folder(folder: Path):
    match = FOLDER_PATTERN.match(folder.name)
    if not match:
        return None

    seq = int(match.group(1))
    timestamp_text = match.group(2)

    try:
        timestamp = datetime.strptime(timestamp_text, "%Y%m%d%H%M")
    except ValueError:
        return None

    return seq, timestamp


def june19_block_for_sequence(seq: int):
    for block_name, start_seq, end_seq in JUNE19_BLOCKS:
        if start_seq <= seq <= end_seq:
            return block_name
    return None


def destination_for(folder: Path):
    parsed = parse_folder(folder)

    if parsed is None:
        return None

    seq, timestamp = parsed

    block_name = june19_block_for_sequence(seq)

    if block_name is not None:
        return OUTPUT_DIR / "2026-06-19" / block_name / folder.name

    date_folder = timestamp.strftime("%Y-%m-%d")
    return OUTPUT_DIR / date_folder / "raw_date_group" / folder.name


def safe_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    if DRY_RUN:
        print(f"[DRY RUN] {src} -> {dst}")
    else:
        shutil.move(str(src), str(dst))
        print(f"Moved: {src.name} -> {dst.parent}")


def main():
    folders = [p for p in SOURCE_DIR.rglob("D_*") if p.is_dir() and parse_folder(p)]

    print(f"Found {len(folders)} data folders")

    for folder in sorted(folders, key=lambda p: parse_folder(p)[0]):
        dst = destination_for(folder)

        if dst is None:
            print(f"Skipped: {folder}")
            continue

        safe_move(folder, dst)

    print("Done.")

    if DRY_RUN:
        print("\nDRY_RUN is True. Set DRY_RUN = False to actually move folders.")


if __name__ == "__main__":
    main()