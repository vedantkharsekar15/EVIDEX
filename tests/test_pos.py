"""Unit tests for HMM POS Tagger."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from src.pos.hmm_tagger import HMMTagger, MissingNLTKDataError


@pytest.fixture
def tiny_tagged_corpus() -> list[list[tuple[str, str]]]:
    """Tiny tagged corpus with ambiguous words (e.g. 'can' as MD vs NN, 'run' as VB vs NN)."""
    return [
        [("I", "PRP"), ("can", "MD"), ("see", "VB"), ("the", "DT"), ("can", "NN"), (".", ".")],
        [("They", "PRP"), ("can", "MD"), ("run", "VB"), ("fast", "RB"), (".", ".")],
        [("The", "DT"), ("run", "NN"), ("was", "VBD"), ("long", "JJ"), (".", ".")],
        [("She", "PRP"), ("will", "MD"), ("open", "VB"), ("a", "DT"), ("can", "NN"), (".", ".")],
    ]


def test_tiny_corpus_training_and_disambiguation(tiny_tagged_corpus: list[list[tuple[str, str]]]) -> None:
    """HMM on tiny corpus learns transition/emission and disambiguates words based on context."""
    tagger = HMMTagger(alpha=0.01, beta=0.001)
    tagger.train(tiny_tagged_corpus)

    assert "MD" in tagger.tag_set
    assert "NN" in tagger.tag_set
    assert "DT" in tagger.tag_set

    # In "They can see the can":
    # 1st 'can' after 'They' (PRP) should be 'MD' (modal verb)
    # 2nd 'can' after 'the' (DT) should be 'NN' (noun)
    sent = ["They", "can", "see", "the", "can", "."]
    tagged = tagger.tag(sent)

    tags = [t for _, t in tagged]
    assert tags[1] == "MD"  # "can" after PRP
    assert tags[3] == "DT"
    assert tags[4] == "NN"  # "can" after DT


def test_unknown_word_heuristics() -> None:
    """Unknown word does not crash and gets plausible tags (capitalized -> NNP, -ing -> VBG, -ly -> RB)."""
    tagger = HMMTagger()
    # Simple training so tagger has standard tag inventory and transitions
    training_data = [
        [("Alice", "NNP"), ("is", "VBZ"), ("reading", "VBG"), ("quickly", "RB"), (".", ".")],
        [("Bob", "NNP"), ("was", "VBD"), ("walking", "VBG"), ("slowly", "RB"), (".", ".")],
        [("The", "DT"), ("model", "NN"), ("improved", "VBD"), ("greatly", "RB"), (".", ".")],
        [("Researchers", "NNS"), ("tested", "VBD"), ("Stanford", "NNP"), ("data", "NN"), (".", ".")],
    ]
    tagger.train(training_data)

    # 1. Capitalized OOV in non-start position -> NNP
    res1 = tagger.tag(["The", "researcher", "visited", "Stanford", "."])
    tags1 = dict(res1)
    assert tags1["Stanford"] in ("NNP", "NNPS", "NN")

    # 2. -ing OOV -> VBG
    res2 = tagger.tag(["Alice", "is", "benchmarking", "the", "model", "."])
    tags2 = dict(res2)
    assert tags2["benchmarking"] == "VBG"

    # 3. -ly OOV -> RB
    res3 = tagger.tag(["Bob", "walked", "briskly", "."])
    tags3 = dict(res3)
    assert tags3["briskly"] == "RB"


def test_empty_tokens_and_evaluation() -> None:
    """Empty token list returns []; held-out accuracy is float in [0, 1] and deterministic."""
    tagger = HMMTagger()
    assert tagger.tag([]) == []

    # Test training from treebank
    acc1 = tagger.train_from_treebank(test_split=0.1, seed=42)
    assert isinstance(acc1, float)
    assert 0.0 <= acc1 <= 1.0
    assert acc1 > 0.70  # HMM on Treebank with smoothing and morphology typically achieves > 80%

    # Repeat with same seed should yield exact same held-out accuracy
    tagger2 = HMMTagger()
    acc2 = tagger2.train_from_treebank(test_split=0.1, seed=42)
    assert acc1 == acc2


def test_missing_treebank_error() -> None:
    """Missing treebank raises MissingNLTKDataError with download command."""
    tagger = HMMTagger()

    with patch("nltk.corpus.treebank.tagged_sents", side_effect=LookupError("Treebank not found")):
        with pytest.raises(MissingNLTKDataError) as exc_info:
            tagger.train_from_treebank()

        assert "python -m nltk.downloader treebank" in str(exc_info.value)


def test_pos_serialization(tiny_tagged_corpus: list[list[tuple[str, str]]]) -> None:
    """Test to_dict and from_dict serialization."""
    tagger = HMMTagger(alpha=0.05, beta=0.005)
    tagger.train(tiny_tagged_corpus)

    d = tagger.to_dict()
    restored = HMMTagger.from_dict(d)

    assert restored.alpha == tagger.alpha
    assert restored.beta == tagger.beta
    assert restored.tags == tagger.tags
    assert restored.vocab == tagger.vocab

    sent = ["The", "run", "was", "long", "."]
    assert tagger.tag(sent) == restored.tag(sent)
