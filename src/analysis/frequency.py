"""Corpus-level term frequency analysis.

Computes raw and filtered term frequencies, skips references sections,
and generates structured pandas DataFrames for tabular display and analysis.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

# Standard fallback stopwords in case NLTK data is unavailable
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def _get_stopwords() -> set[str]:
    """Retrieve stopwords set from NLTK if available, otherwise return default set."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        return DEFAULT_STOPWORDS


def term_frequencies(
    records: list[dict[str, Any]],
    top_n: int | None = None,
    remove_stopwords: bool = False,
) -> list[tuple[str, int]]:
    """Calculate term frequencies from a list of sentence records.

    Skips records where in_references is True.

    Args:
        records: List of SentenceRecord dictionaries.
        top_n: Optional maximum number of terms to return.
        remove_stopwords: Whether to filter out standard English stopwords.

    Returns:
        List of (term, count) tuples sorted by count descending, then alphabetically.
    """
    if not records:
        return []

    stopwords_set = _get_stopwords() if remove_stopwords else set()
    counter: Counter[str] = Counter()

    for record in records:
        # Skip sentences in references section
        if record.get("in_references", False):
            continue

        # Prefer normalized_tokens, fallback to tokens or text
        tokens = record.get("normalized_tokens")
        if tokens is None:
            raw_tokens = record.get("tokens")
            if raw_tokens:
                tokens = [t.lower() for t in raw_tokens if re.match(r"^[a-zA-Z0-9_-]+$", t)]
            else:
                text = record.get("text", "")
                tokens = [t.lower() for t in re.findall(r"\b\w+\b", text)]

        for token in tokens:
            cleaned = token.strip().lower()
            if not cleaned or (remove_stopwords and cleaned in stopwords_set):
                continue
            counter[cleaned] += 1

    # Sort by count desc, then alphabetically asc
    sorted_items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

    if top_n is not None and top_n > 0:
        return sorted_items[:top_n]
    return sorted_items


def frequency_table(
    records: list[dict[str, Any]],
    top_n: int = 50,
    remove_stopwords: bool = False,
) -> pd.DataFrame:
    """Generate a ranked pandas DataFrame of term frequencies.

    Args:
        records: List of SentenceRecord dictionaries.
        top_n: Number of top terms to include.
        remove_stopwords: Whether to remove stopwords.

    Returns:
        pandas.DataFrame with columns: ['rank', 'term', 'count'] (or ['term', 'count', 'rank']).
    """
    freqs = term_frequencies(records, top_n=top_n, remove_stopwords=remove_stopwords)
    if not freqs:
        return pd.DataFrame(columns=["rank", "term", "count"])

    data = [
        {"rank": rank, "term": term, "count": count}
        for rank, (term, count) in enumerate(freqs, start=1)
    ]
    df = pd.DataFrame(data)
    return df[["rank", "term", "count"]]
