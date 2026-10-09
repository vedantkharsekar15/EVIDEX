"""Unit tests for Term Frequency and Word Cloud analysis."""

import json
from pathlib import Path

import pandas as pd
import pytest
from src.analysis.frequency import frequency_table, term_frequencies
from src.analysis.wordcloud import build_wordcloud


@pytest.fixture
def sample_records() -> list[dict]:
    """Load records fixture with references and normal sentences."""
    fixture_path = Path(__file__).parent / "fixtures" / "sentences_nlp.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_term_frequencies_and_references_excluded(sample_records: list[dict]) -> None:
    """Check frequency counts and verify that references are excluded."""
    # In fixture, reference sentences have terms like 'vaswani' and 'neurips'
    all_freqs = dict(term_frequencies(sample_records, remove_stopwords=False))

    # 'transformer' and 'model' should be present
    assert "transformer" in all_freqs
    assert "model" in all_freqs

    # 'vaswani' is only in an in_references=True sentence, so it must NOT appear
    assert "vaswani" not in all_freqs
    assert "neurips" not in all_freqs


def test_stopwords_filter_toggle(sample_records: list[dict]) -> None:
    """Stop-word toggle removes stopwords such as 'the', 'is', 'on'."""
    with_stops = dict(term_frequencies(sample_records, remove_stopwords=False))
    without_stops = dict(term_frequencies(sample_records, remove_stopwords=True))

    assert "the" in with_stops
    assert "the" not in without_stops
    assert "model" in without_stops


def test_top_n_and_empty_input() -> None:
    """top_n limits output size; empty records return empty list."""
    records = [
        {"normalized_tokens": ["apple", "banana", "apple", "cherry"], "in_references": False}
    ]
    top_2 = term_frequencies(records, top_n=2)
    assert len(top_2) == 2
    assert top_2[0] == ("apple", 2)

    # Empty inputs
    assert term_frequencies([]) == []
    assert term_frequencies([{"normalized_tokens": [], "in_references": False}]) == []


def test_frequency_table(sample_records: list[dict]) -> None:
    """frequency_table returns DataFrame with rank, term, count columns."""
    df = frequency_table(sample_records, top_n=10, remove_stopwords=True)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["rank", "term", "count"]
    assert len(df) <= 10
    if not df.empty:
        assert df["rank"].iloc[0] == 1
        assert df["count"].iloc[0] >= df["count"].iloc[-1]

    # Empty df check
    df_empty = frequency_table([])
    assert isinstance(df_empty, pd.DataFrame)
    assert list(df_empty.columns) == ["rank", "term", "count"]
    assert df_empty.empty


def test_wordcloud_generation() -> None:
    """Word Cloud returns PIL Image for valid frequencies, and None for empty input."""
    from PIL.Image import Image

    # Empty freqs -> None
    assert build_wordcloud({}) is None
    assert build_wordcloud({"   ": 5}) is None
    assert build_wordcloud({"word": 0, "another": -1}) is None

    # Valid freqs -> Image
    freqs = {"transformer": 10, "attention": 8, "language": 5, "retrieval": 7}
    img = build_wordcloud(freqs, max_words=50)
    assert img is not None
    assert isinstance(img, Image)
    assert img.size[0] > 0 and img.size[1] > 0
