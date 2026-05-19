"""Model construction and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spam_detection.preprocessing import tokenize_and_stem


@dataclass(frozen=True)
class ModelSpec:
    name: str
    vectorizer: str
    estimator: BaseEstimator


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("MultinomialNB", "count", MultinomialNB()),
        ModelSpec("MultinomialNB", "tfidf", MultinomialNB()),
        ModelSpec("LinearSVC", "tfidf", LinearSVC(class_weight="balanced", random_state=42)),
        ModelSpec(
            "LogisticRegression",
            "tfidf",
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        ),
    ]


def build_pipeline(spec: ModelSpec) -> Pipeline:
    if spec.vectorizer == "count":
        vectorizer = CountVectorizer(tokenizer=tokenize_and_stem, token_pattern=None, ngram_range=(1, 2))
    elif spec.vectorizer == "tfidf":
        vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, token_pattern=None, ngram_range=(1, 2))
    else:
        raise ValueError(f"Unsupported vectorizer: {spec.vectorizer}")

    return Pipeline([("vectorizer", vectorizer), ("classifier", spec.estimator)])


def evaluate_predictions(y_true: pd.Series, y_pred: list[int] | pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

