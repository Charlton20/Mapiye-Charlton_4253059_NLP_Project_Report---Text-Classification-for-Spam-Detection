"""Data loading and dataset summary helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = Path("data/raw/SMSSpamCollection")


def load_sms_spam_dataset(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the UCI SMS Spam Collection dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {path}. Run `python scripts/download_dataset.py` first."
        )

    dataframe = pd.read_csv(path, sep="\t", names=["label", "message"], encoding="latin-1")
    dataframe["label"] = dataframe["label"].str.strip().str.lower()
    dataframe["message"] = dataframe["message"].astype(str)
    dataframe["target"] = dataframe["label"].map({"ham": 0, "spam": 1})

    if dataframe["target"].isna().any():
        invalid = sorted(dataframe.loc[dataframe["target"].isna(), "label"].unique())
        raise ValueError(f"Unexpected labels found in dataset: {invalid}")

    return dataframe


def write_dataset_summary(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Write basic dataset statistics for reporting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_counts = dataframe["label"].value_counts().to_dict()
    summary = {
        "total_messages": int(len(dataframe)),
        "label_counts": {key: int(value) for key, value in label_counts.items()},
        "label_percentages": {
            key: round(float(value / len(dataframe) * 100), 2) for key, value in label_counts.items()
        },
        "average_message_length_characters": round(float(dataframe["message"].str.len().mean()), 2),
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

