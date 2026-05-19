"""Train and evaluate spam detection models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from spam_detection.data import DEFAULT_DATA_PATH, load_sms_spam_dataset, write_dataset_summary
from spam_detection.modeling import build_model_specs, build_pipeline, evaluate_predictions


DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_SAMPLE_SIZES = [500, 1500, 3000, -1]


def stratified_subset(dataframe: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if sample_size == -1 or sample_size >= len(dataframe):
        return dataframe.copy()

    _, subset = train_test_split(
        dataframe,
        test_size=sample_size,
        stratify=dataframe["target"],
        random_state=random_state,
    )
    return subset.reset_index(drop=True)


def train_and_evaluate(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_sizes: list[int] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_sizes = sample_sizes or DEFAULT_SAMPLE_SIZES

    dataframe = load_sms_spam_dataset(data_path)
    write_dataset_summary(dataframe, output_dir / "dataset_summary.json")

    model_specs = build_model_specs()
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    for sample_size in sample_sizes:
        subset = stratified_subset(dataframe, sample_size, random_state)
        train_df, test_df = train_test_split(
            subset,
            test_size=test_size,
            stratify=subset["target"],
            random_state=random_state,
        )

        for spec in model_specs:
            pipeline = build_pipeline(spec)
            pipeline.fit(train_df["message"], train_df["target"])
            predictions = pipeline.predict(test_df["message"])
            metrics = evaluate_predictions(test_df["target"], predictions)

            row = {
                "dataset_size": len(subset),
                "train_size": len(train_df),
                "test_size": len(test_df),
                "spam_train_count": int(train_df["target"].sum()),
                "spam_test_count": int(test_df["target"].sum()),
                "model": spec.name,
                "vectorizer": spec.vectorizer,
                **metrics,
            }
            rows.append(row)

            sort_key = (
                len(subset),
                metrics["f1"],
                metrics["recall"],
                metrics["precision"],
                metrics["accuracy"],
            )
            if best is None or sort_key > best["sort_key"]:
                best = {
                    "pipeline": pipeline,
                    "spec": spec,
                    "metrics": metrics,
                    "sort_key": sort_key,
                    "test_messages": test_df["message"],
                    "test_target": test_df["target"],
                    "predictions": predictions,
                    "dataset_size": len(subset),
                }

    metrics_df = pd.DataFrame(rows).sort_values(
        ["f1", "recall", "precision", "accuracy"], ascending=False
    )
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    if best is None:
        raise RuntimeError("No model was trained.")

    joblib.dump(best["pipeline"], output_dir / "best_model.joblib")
    report = classification_report(
        best["test_target"],
        best["predictions"],
        target_names=["legitimate", "spam"],
        zero_division=0,
    )
    (output_dir / "best_model_report.txt").write_text(
        (
            f"Best model: {best['spec'].name} with {best['spec'].vectorizer}\n"
            f"Dataset size: {best['dataset_size']}\n\n"
            f"{report}"
        ),
        encoding="utf-8",
    )

    matrix = confusion_matrix(best["test_target"], best["predictions"])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["legitimate", "spam"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Best Model: {best['spec'].name} ({best['spec'].vectorizer})")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    return metrics_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=DEFAULT_SAMPLE_SIZES)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_df = train_and_evaluate(
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_sizes=args.sample_sizes,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(metrics_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
