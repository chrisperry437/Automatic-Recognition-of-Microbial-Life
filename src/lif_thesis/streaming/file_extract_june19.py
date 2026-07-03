import re
import time
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from datetime import datetime

import paramiko


HOST = "192.168.1.103"
PORT = 22
USERNAME = "Rapid-E-user"
PASSWORD = "QEYKvnnw"

REMOTE_DIR = "/DATA/D_00001"
LOCAL_DIR = Path("data/live_rapid_e/June_19")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

POLL_SECONDS = 10
STABLE_SECONDS = 5

TIMESTAMP_PATTERN = re.compile(r"D_\d+_(\d{12})\.zip$")


def extract_timestamp(filename: str) -> datetime:
    match = TIMESTAMP_PATTERN.match(filename)
    if not match:
        return datetime.min
    return datetime.strptime(match.group(1), "%Y%m%d%H%M")


def connect_sftp():
    transport = paramiko.Transport((HOST, PORT))
    transport.connect(username=USERNAME, password=PASSWORD)
    return transport, paramiko.SFTPClient.from_transport(transport)


def remote_file_is_stable(sftp, zip_name: str) -> bool:
    remote_path = f"{REMOTE_DIR}/{zip_name}"

    size_1 = sftp.stat(remote_path).st_size
    time.sleep(STABLE_SECONDS)
    size_2 = sftp.stat(remote_path).st_size

    return size_1 == size_2 and size_1 > 0


def download_and_extract_zip(sftp, zip_name: str) -> None:
    remote_zip = f"{REMOTE_DIR}/{zip_name}"
    local_zip = LOCAL_DIR / zip_name
    extract_dir = LOCAL_DIR / zip_name.replace(".zip", "")

    if local_zip.exists():
        print(f"Already downloaded: {zip_name}")
        return

    if not remote_file_is_stable(sftp, zip_name):
        print(f"Skipping active/incomplete file: {zip_name}")
        return

    temp_zip = local_zip.with_suffix(".zip.part")

    print(f"\nDownloading {zip_name}...")
    sftp.get(remote_zip, str(temp_zip))
    temp_zip.rename(local_zip)

    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(local_zip, "r") as z:
            z.extractall(extract_dir)
        print(f"Extracted to: {extract_dir}")

    except BadZipFile:
        print(f"Bad/incomplete ZIP, deleting local copy: {zip_name}")
        local_zip.unlink(missing_ok=True)


def main():
    transport, sftp = connect_sftp()

    try:
        print(f"Watching {REMOTE_DIR} for new ZIP files...")

        while True:
            files = sftp.listdir(REMOTE_DIR)

            zip_files = sorted(
                [f for f in files if f.lower().endswith(".zip")],
                key=extract_timestamp,
            )

            for zip_name in zip_files:
                local_zip = LOCAL_DIR / zip_name

                if not local_zip.exists():
                    download_and_extract_zip(sftp, zip_name)

            time.sleep(POLL_SECONDS)

    finally:
        sftp.close()
        transport.close()


if __name__ == "__main__":
    main()
    