"""Classify new messages with the trained spam detection model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib


DEFAULT_MODEL_PATH = Path("outputs/best_model.joblib")


def predict_message(message: str, model_path: Path = DEFAULT_MODEL_PATH) -> str:
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run `PYTHONPATH=src python -m spam_detection.train`.")

    pipeline = joblib.load(model_path)
    prediction = int(pipeline.predict([message])[0])
    return "spam" if prediction == 1 else "legitimate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", help="Message text to classify.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(predict_message(args.message, args.model_path))


if __name__ == "__main__":
    main()

