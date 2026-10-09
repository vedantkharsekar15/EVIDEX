"""Tokenization and token normalization for Module 2.

This module deliberately operates only on strings and token lists.  Page and
sentence cleaning remain responsibilities of the other preprocessing stages.
"""

from __future__ import annotations

from nltk.tokenize import word_tokenize


# Kept local and deterministic so stop-word filtering does not require the
# optional NLTK ``stopwords`` corpus.  Punctuation is intentionally absent:
# normalize_tokens() must not remove punctuation as a side effect.
_ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


def tokenize(text: str) -> list[str]:
    """Tokenize text with NLTK's deterministic Penn Treebank rules.

    ``preserve_line=True`` selects the Treebank tokenizer without invoking
    sentence segmentation.  Thus punctuation is returned as separate tokens
    where Treebank rules define it, while original token case is preserved.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        return []
    return list(word_tokenize(text, preserve_line=True))


def normalize_tokens(
    tokens: list[str],
    lowercase: bool = True,
    remove_stopwords: bool = False,
) -> list[str]:
    """Apply optional lowercasing and English stop-word removal.

    Token boundaries and punctuation are preserved.  Stop-word matching is
    case-insensitive even when ``lowercase`` is false, so the option has the
    same meaning for raw-case and lowercased token lists.
    """

    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise TypeError("tokens must be a list of strings")

    normalized: list[str] = []
    for token in tokens:
        if remove_stopwords and token.lower() in _ENGLISH_STOPWORDS:
            continue
        normalized.append(token.lower() if lowercase else token)
    return normalized


def preprocess_query(text: str, cfg: dict) -> list[str]:
    """Tokenize and normalize a query using the corpus preprocessing path."""

    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dictionary")
    preprocessing = cfg.get("preprocessing", {})
    if preprocessing is None:
        preprocessing = {}
    if not isinstance(preprocessing, dict):
        raise ValueError("cfg['preprocessing'] must be a dictionary")

    lowercase = preprocessing.get("lowercase", True)
    remove_stopwords = preprocessing.get("remove_stopwords", False)
    return normalize_tokens(
        tokenize(text),
        lowercase=lowercase,
        remove_stopwords=remove_stopwords,
    )
