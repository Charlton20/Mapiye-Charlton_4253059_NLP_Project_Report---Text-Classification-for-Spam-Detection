"""Text preprocessing used by the vectorizers."""

from __future__ import annotations

import re
import string
from functools import lru_cache

from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")
PUNCTUATION_TRANSLATION = str.maketrans("", "", string.punctuation)


@lru_cache(maxsize=1)
def get_stemmer() -> SnowballStemmer:
    return SnowballStemmer("english")


def normalize_text(text: str) -> str:
    """Normalize casing, URLs, phone numbers, and repeated whitespace."""
    normalized = text.lower()
    normalized = re.sub(r"https?://\S+|www\.\S+", " urltoken ", normalized)
    normalized = re.sub(r"\b\d+(?:[.,]\d+)?\b", " numbertoken ", normalized)
    normalized = normalized.translate(PUNCTUATION_TRANSLATION)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize_and_stem(text: str) -> list[str]:
    """Tokenize text, remove English stop words, and apply stemming."""
    stemmer = get_stemmer()
    normalized = normalize_text(text)
    tokens = TOKEN_PATTERN.findall(normalized)
    return [
        stemmer.stem(token)
        for token in tokens
        if token not in ENGLISH_STOP_WORDS and len(token) > 1
    ]

