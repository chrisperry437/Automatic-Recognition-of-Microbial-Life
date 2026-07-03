from pathlib import Path
import shutil
from datetime import datetime

ROOT = Path("data/live_rapid_e")

DRY_RUN = False

NEW_DIRS = [
    ROOT / "experiments",
    ROOT / "incoming_zips",
    ROOT / "metadata",
    ROOT / "processed",
    ROOT / "archive",
]

ARCHIVE_MAP = {
    "experiment": "archive/old_experiment",
    "June_19": "archive/old_june19_working",
    "june17_separated": "archive/old_june17_separated",
    "window_1323_1340": "archive/old_temp_windows",
}

ZIP_FOLDERS = [
    "zips",
]

REPORT = []


def log(msg):
    print(msg)
    REPORT.append(msg)


def move_path(src: Path, dst: Path):
    if not src.exists():
        return

    if dst.exists():
        log(f"SKIP EXISTS: {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if DRY_RUN:
        log(f"[DRY RUN] MOVE {src} -> {dst}")
    else:
        shutil.move(str(src), str(dst))
        log(f"MOVED {src} -> {dst}")


def create_structure():
    for d in NEW_DIRS:
        if DRY_RUN:
            log(f"[DRY RUN] MKDIR {d}")
        else:
            d.mkdir(parents=True, exist_ok=True)
            log(f"CREATED {d}")


def archive_old_folders():
    for old_name, new_relative in ARCHIVE_MAP.items():

        src = ROOT / old_name
        dst = ROOT / new_relative

        if src.exists():
            move_path(src, dst)


def consolidate_zip_folders():

    incoming = ROOT / "incoming_zips"
    incoming.mkdir(exist_ok=True)

    for folder_name in ZIP_FOLDERS:

        src = ROOT / folder_name

        if not src.exists():
            continue

        for item in src.iterdir():

            dst = incoming / item.name

            if dst.exists():
                log(f"ZIP ALREADY EXISTS: {dst}")
                continue

            if DRY_RUN:
                log(f"[DRY RUN] MOVE {item} -> {dst}")
            else:
                shutil.move(str(item), str(dst))
                log(f"MOVED {item} -> {dst}")

        if not DRY_RUN:
            try:
                src.rmdir()
                log(f"REMOVED EMPTY {src}")
            except OSError:
                pass


def create_metadata_templates():

    metadata_dir = ROOT / "metadata"

    experiment_log = metadata_dir / "experiment_log.csv"

    evaluation_manifest = metadata_dir / "evaluation_manifest.csv"

    if DRY_RUN:
        log(f"[DRY RUN] CREATE {experiment_log}")
        log(f"[DRY RUN] CREATE {evaluation_manifest}")
        return

    if not experiment_log.exists():

        experiment_log.write_text(
            "experiment_date,experiment_id,experiment_block,start_time,end_time,sample,laser_pct,current_A,notes\n",
            encoding="utf-8",
        )

    if not evaluation_manifest.exists():

        evaluation_manifest.write_text(
            "experiment_id,experiment_path\n",
            encoding="utf-8",
        )

    log("CREATED METADATA TEMPLATES")


def write_report():

    report_file = ROOT / f"migration_report_{datetime.now():%Y%m%d_%H%M%S}.txt"

    if DRY_RUN:
        log(f"[DRY RUN] REPORT -> {report_file}")
        return

    report_file.write_text("\n".join(REPORT), encoding="utf-8")


def print_final_structure():

    print("\nTARGET STRUCTURE:\n")

    print(
        """
data/live_rapid_e/
├── experiments/
├── incoming_zips/
├── metadata/
│   ├── experiment_log.csv
│   └── evaluation_manifest.csv
├── processed/
└── archive/
    ├── old_experiment/
    ├── old_june19_working/
    ├── old_june17_separated/
    └── old_temp_windows/
"""
    )


def main():

    log("=== RECONFIGURE LIVE RAPID-E ===")

    create_structure()

    archive_old_folders()

    consolidate_zip_folders()

    create_metadata_templates()

    write_report()

    print_final_structure()

    log("COMPLETE")


if __name__ == "__main__":
    main()