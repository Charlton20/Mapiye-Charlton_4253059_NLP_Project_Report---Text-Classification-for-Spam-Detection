# NLP Project: Text Classification for Spam Detection

This project develops a supervised NLP system for binary spam detection. It uses the UCI SMS Spam Collection dataset and compares multiple text-classification pipelines across progressively larger training subsets.

## Project Goals

- Collect a labeled spam/legitimate message dataset.
- Preprocess messages with text normalization, stop-word removal, and stemming.
- Encode text with Bag-of-Words and TF-IDF features.
- Train and compare supervised models.
- Evaluate generalization on a held-out test set.
- Save metrics, plots, and a trained model for classifying new messages.

## Recommended Workflow

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_dataset.py
PYTHONPATH=src python -m spam_detection.train
PYTHONPATH=src python -m spam_detection.predict "Congratulations, you won a free prize. Reply now!"
```

If `python3 -m venv` is unavailable, dependencies can be installed into a project-local directory:

```bash
python3 -m pip install --target .python_packages -r requirements.txt
python3 scripts/download_dataset.py
PYTHONPATH=.python_packages:src python3 -m spam_detection.train
PYTHONPATH=.python_packages:src python3 -m spam_detection.predict "Congratulations, you won a free prize. Reply now!"
```

## Outputs

After training, generated artifacts are written to `outputs/`:

- `metrics.csv`: model results for each dataset size and model configuration.
- `best_model.joblib`: best performing trained pipeline.
- `best_model_report.txt`: classification report for the best model.
- `confusion_matrix.png`: confusion matrix for the best model.
- `dataset_summary.json`: dataset size and class distribution.

## Dataset

The project uses the UCI SMS Spam Collection dataset:

> Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results. Proceedings of the 11th ACM Symposium on Document Engineering.

Dataset page: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
