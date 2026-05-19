"""Download and unpack the UCI SMS Spam Collection dataset."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DEFAULT_OUTPUT_DIR = Path("data/raw")


def download_dataset(output_dir: Path = DEFAULT_OUTPUT_DIR, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "SMSSpamCollection"
    zip_path = output_dir / "sms_spam_collection.zip"

    if dataset_path.exists() and not force:
        print(f"Dataset already exists: {dataset_path}")
        return dataset_path

    if force and dataset_path.exists():
        dataset_path.unlink()

    print(f"Downloading dataset from {DATA_URL}")
    with urllib.request.urlopen(DATA_URL) as response, zip_path.open("wb") as file:
        shutil.copyfileobj(response, file)

    print(f"Unpacking {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    readme_path = output_dir / "readme"
    if readme_path.exists():
        readme_path.rename(output_dir / "dataset_readme.txt")

    if not dataset_path.exists():
        raise FileNotFoundError("Expected SMSSpamCollection file was not found after extraction.")

    print(f"Dataset ready: {dataset_path}")
    return dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Download again even if the dataset exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_dataset(args.output_dir, args.force)


if __name__ == "__main__":
    main()

