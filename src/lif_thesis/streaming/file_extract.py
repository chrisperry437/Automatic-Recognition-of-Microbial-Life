import re
from pathlib import Path
from zipfile import ZipFile
from datetime import datetime

import paramiko


HOST = "192.168.1.103"
PORT = 22
USERNAME = "Rapid-E-user"
PASSWORD = "QEYKvnnw"

REMOTE_DIR = "/DATA/D_00001"
LOCAL_DIR = Path("data/live_rapid_e/experiment")
LOCAL_DIR.mkdir(exist_ok=True)

# Set to None to download newest file.
# Or set to a specific file, e.g. "D_000013314_202606171027.zip"
TARGET_ZIP = None

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

    print(f"\nExtracting to {extract_dir}...")
    with ZipFile(local_zip, "r") as z:
        print("\nZIP contents:")
        for name in z.namelist():
            print(name)
        z.extractall(extract_dir)

    return extract_dir


def inspect_raw_files(extract_dir: Path) -> None:
    raw_files = list(extract_dir.rglob("*.raw"))

    print(f"\nFound {len(raw_files)} RAW files.")

    for raw_file in raw_files:
        size_mb = raw_file.stat().st_size / 1024 / 1024

        print("\n--------------------------------")
        print(f"RAW file: {raw_file}")
        print(f"Size: {size_mb:.2f} MB")

        with open(raw_file, "rb") as f:
            first_bytes = f.read(128)

        print("\nFirst 128 bytes:")
        print(first_bytes)

        print("\nHEX preview:")
        print(first_bytes.hex())


def main():
    transport, sftp = connect_sftp()

    try:
        files = sftp.listdir(REMOTE_DIR)

        zip_files = sorted(
            [f for f in files if f.lower().endswith(".zip")],
            key=extract_timestamp,
        )

        print(f"\nFound {len(zip_files)} ZIP files in {REMOTE_DIR}.")

        if not zip_files:
            print("No ZIP files found.")
            return

        if TARGET_ZIP is not None:
            if TARGET_ZIP not in zip_files:
                raise FileNotFoundError(
                    f"{TARGET_ZIP} was not found in {REMOTE_DIR}"
                )
            zip_name = TARGET_ZIP
        else:
            zip_name = zip_files[-1]

        print(f"\nSelected ZIP: {zip_name}")
        print(f"Parsed timestamp: {extract_timestamp(zip_name)}")

        extract_dir = download_and_extract_zip(sftp, zip_name)
        inspect_raw_files(extract_dir)

        print("\nDone.")

    finally:
        sftp.close()
        transport.close()


if __name__ == "__main__":
    main()