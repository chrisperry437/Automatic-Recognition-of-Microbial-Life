import re
from pathlib import Path
from zipfile import ZipFile
from datetime import datetime, date, timedelta

import paramiko


HOST = "192.168.1.103"
PORT = 22
USERNAME = "Rapid-E-user"
PASSWORD = "QEYKvnnw"

REMOTE_DIR = "/DATA/D_00001"
LOCAL_DIR = Path("data/live_rapid_e/experiment")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# Automatically extract all files from yesterday
TARGET_DATE = date.today() - timedelta(days=2)

TIMESTAMP_PATTERN = re.compile(r"D_\d+_(\d{12})\.zip$")


def extract_timestamp(filename: str) -> datetime:
    match = TIMESTAMP_PATTERN.match(filename)
    if not match:
        return datetime.min
    return datetime.strptime(match.group(1), "%Y%m%d%H%M")


def connect_sftp():
    transport = paramiko.Transport((HOST, PORT))
    transport.connect(username=USERNAME, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return transport, sftp


def download_and_extract_zip(sftp, zip_name: str) -> Path:
    remote_zip = f"{REMOTE_DIR}/{zip_name}"
    local_zip = LOCAL_DIR / zip_name

    if not local_zip.exists():
        print(f"\nDownloading {remote_zip}...")
        sftp.get(remote_zip, str(local_zip))
        print("Download complete.")
    else:
        print(f"\nAlready downloaded: {local_zip}")

    extract_dir = LOCAL_DIR / zip_name.replace(".zip", "")
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting to {extract_dir}...")
    with ZipFile(local_zip, "r") as z:
        z.extractall(extract_dir)

    return extract_dir


def inspect_raw_files(extract_dir: Path) -> None:
    raw_files = list(extract_dir.rglob("*.raw"))
    print(f"Found {len(raw_files)} RAW files in {extract_dir}")


def main():
    transport, sftp = connect_sftp()

    try:
        files = sftp.listdir(REMOTE_DIR)

        zip_files = sorted(
            [f for f in files if f.lower().endswith(".zip")],
            key=extract_timestamp,
        )

        yesterday_zips = [
            f for f in zip_files
            if extract_timestamp(f).date() == TARGET_DATE
        ]

        print(f"\nFound {len(zip_files)} total ZIP files in {REMOTE_DIR}.")
        print(f"Target date: {TARGET_DATE}")
        print(f"Found {len(yesterday_zips)} ZIP files from yesterday.")

        if not yesterday_zips:
            print("No ZIP files found for yesterday.")
            return

        for zip_name in yesterday_zips:
            ts = extract_timestamp(zip_name)
            print("\n================================")
            print(f"Processing ZIP: {zip_name}")
            print(f"Parsed timestamp: {ts}")

            extract_dir = download_and_extract_zip(sftp, zip_name)
            inspect_raw_files(extract_dir)

        print("\nDone extracting yesterday's Rapid-E data.")

    finally:
        sftp.close()
        transport.close()


if __name__ == "__main__":
    main()