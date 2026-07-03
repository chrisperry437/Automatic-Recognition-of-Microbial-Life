from pathlib import Path
import shutil
import re

ROOT = Path("data/live_rapid_e")
DRY_RUN = False

ARCHIVE_DIR = ROOT / "archive"
INCOMING_ZIPS = ROOT / "incoming_zips"
RAW_EXTRACTED = ROOT / "raw_extracted"
EXPERIMENTS = ROOT / "experiments"
PROCESSED = ROOT / "processed"
METADATA = ROOT / "metadata"

JUNE19_BLOCKS = {
    "distilled_water_1216_1240_75pct_35A": range(10000, 10024),
    "distilled_water_1240_1246_80pct_36A": range(10024, 10030),
    "bacillus75_micrococcus25_1250_1312": range(10030, 10052),
    "distilled_water_1313_1346_80pct_36A": range(10052, 10085),
    "bacillus50_micrococcus50_1347_1402": range(10085, 10100),
    "distilled_water_1404_1424_80pct_36A": range(10100, 10120),
    "bacillus25_micrococcus75_1426_1442": range(10120, 10136),
    "distilled_water_after_1442_80pct_36A": range(10136, 10202),
}

ARCHIVE_TOP_LEVEL = [
    "flat_raw",
    "raw",
    "extracted",
    "sorted_by_experiment",
    "ringer_control_20260617_1003_1045",
]

MOVE_DIRS_TO_PROCESSED = {
    "parsed": "parsed",
    "inspection": "inspection",
}

MOVE_FILES_TO_METADATA = {
    ROOT / "logs" / "processed_files.csv": "processed_files.csv",
}


def move_path(src: Path, dst: Path):
    if not src.exists():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        print(f"SKIP exists: {dst}")
        return

    if DRY_RUN:
        print(f"[DRY RUN] {src} -> {dst}")
    else:
        shutil.move(str(src), str(dst))
        print(f"MOVED {src} -> {dst}")


def copy_or_move_raw_to_experiment(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if DRY_RUN:
        print(f"[DRY RUN] {src} -> {dst}")
    else:
        shutil.copy2(src, dst)
        print(f"COPIED {src.name} -> {dst.parent}")


def organize_june17_separated():
    src_root = ROOT / "june17_separated"
    if not src_root.exists():
        return

    mapping = {
        "ringer": "ringer_1003_1047",
        "distilled_water_1107_1223": "distilled_water_1107_1223",
        "bacillus_cereus_possible": "bacillus_cereus_possible_1224_1246",
        "distilled_water_1248_1320": "distilled_water_1248_1320",
        "micrococcus": "micrococcus_1323_1345",
        "distilled_water_1346_1420": "distilled_water_1346_1420",
    }

    for old_name, new_name in mapping.items():
        src = src_root / old_name
        if not src.exists():
            continue

        for raw_file in src.glob("*.raw"):
            dst = EXPERIMENTS / "2026-06-17" / new_name / raw_file.name
            copy_or_move_raw_to_experiment(raw_file, dst)


def organize_june19_blocks():
    src_root = ROOT / "June_19"
    if not src_root.exists():
        return

    # Move already-created experiment folders into clean experiments/ path.
    for block in JUNE19_BLOCKS:
        candidates = [
            src_root / block,
            src_root / block.replace("_75pct_35A", ""),
            src_root / block.replace("_80pct_36A", ""),
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                for raw_file in candidate.rglob("*.raw"):
                    dst = EXPERIMENTS / "2026-06-19" / block / raw_file.name
                    copy_or_move_raw_to_experiment(raw_file, dst)

    # Also sort folders by sequence number if extracted D_* folders remain.
    pattern = re.compile(r"^D_(\d+)_(\d{12})$")

    for folder in src_root.iterdir():
        if not folder.is_dir():
            continue

        match = pattern.match(folder.name)
        if not match:
            continue

        seq = int(match.group(1))

        for block, seq_range in JUNE19_BLOCKS.items():
            if seq in seq_range:
                for raw_file in folder.rglob("*.raw"):
                    dst = EXPERIMENTS / "2026-06-19" / block / raw_file.name
                    copy_or_move_raw_to_experiment(raw_file, dst)


def organize_zips():
    for zip_file in ROOT.rglob("*.zip"):
        if "archive" in zip_file.parts or "incoming_zips" in zip_file.parts:
            continue

        name = zip_file.name

        if "20260617" in name:
            dst = INCOMING_ZIPS / "2026-06-17" / name
        elif zip_file.parent.name == "June_19":
            dst = INCOMING_ZIPS / "2026-06-19" / name
        elif "20260619" in name:
            dst = INCOMING_ZIPS / "2026-06-19" / name
        else:
            dst = INCOMING_ZIPS / "unknown_date" / name

        move_path(zip_file, dst)


def archive_old_top_level_folders():
    for folder_name in ARCHIVE_TOP_LEVEL:
        src = ROOT / folder_name
        if src.exists():
            dst = ARCHIVE_DIR / folder_name
            move_path(src, dst)


def move_processed_and_metadata():
    for old_name, new_name in MOVE_DIRS_TO_PROCESSED.items():
        src = ROOT / old_name
        if src.exists():
            dst = PROCESSED / new_name
            move_path(src, dst)

    for src, new_name in MOVE_FILES_TO_METADATA.items():
        if src.exists():
            dst = METADATA / new_name
            move_path(src, dst)

    logs_dir = ROOT / "logs"
    if logs_dir.exists():
        dst = ARCHIVE_DIR / "logs"
        move_path(logs_dir, dst)


def create_experiment_log_template():
    METADATA.mkdir(parents=True, exist_ok=True)

    log_path = METADATA / "experiment_log.csv"

    if log_path.exists():
        return

    content = """experiment_date,experiment_block,start_time,end_time,sample,laser_pct,current_A,notes
2026-06-17,ringer_1003_1047,10:03,10:47,Ringer 1:4,75,35,Control
2026-06-17,distilled_water_1107_1223,11:07,12:23,Distilled Water,75,35,Control
2026-06-17,bacillus_cereus_possible_1224_1246,12:24,12:46,Bacillus cereus possible,75,35,Experimental sample
2026-06-17,distilled_water_1248_1320,12:48,13:20,Distilled Water,75,35,Control
2026-06-17,micrococcus_1323_1345,13:23,13:45,Micrococcus,75,35,Experimental sample
2026-06-17,distilled_water_1346_1420,13:46,14:20,Distilled Water,75,35,Control
2026-06-19,distilled_water_1216_1240_75pct_35A,12:16,12:40,Distilled Water,75,35,Control
2026-06-19,distilled_water_1240_1246_80pct_36A,12:40,12:46,Distilled Water,80,36,Control
2026-06-19,bacillus75_micrococcus25_1250_1312,12:50,13:12,75 Bacillus / 25 Micrococcus,80,36,Mixture
2026-06-19,distilled_water_1313_1346_80pct_36A,13:13,13:46,Distilled Water,80,36,Control
2026-06-19,bacillus50_micrococcus50_1347_1402,13:47,14:02,50 Bacillus / 50 Micrococcus,80,36,Mixture
2026-06-19,distilled_water_1404_1424_80pct_36A,14:04,14:24,Distilled Water,80,36,Control
2026-06-19,bacillus25_micrococcus75_1426_1442,14:26,14:42,25 Bacillus / 75 Micrococcus,80,36,Mixture
2026-06-19,distilled_water_after_1442_80pct_36A,14:42,,Distilled Water,80,36,Control
"""

    if DRY_RUN:
        print(f"[DRY RUN] create {log_path}")
    else:
        log_path.write_text(content, encoding="utf-8")
        print(f"CREATED {log_path}")


def main():
    print(f"DRY_RUN = {DRY_RUN}")

    organize_june17_separated()
    organize_june19_blocks()
    organize_zips()
    move_processed_and_metadata()
    create_experiment_log_template()

    # Archive duplicated folders last, after useful files have been copied.
    archive_old_top_level_folders()

    print("Done.")


if __name__ == "__main__":
    main()