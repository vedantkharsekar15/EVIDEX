"""Unit tests for N-Gram Language Model."""

import json
import math
import random
from pathlib import Path

import pytest
from src.language_model.ngram_model import NGramModel


@pytest.fixture
def toy_corpus() -> list[list[str]]:
    """Three-sentence toy corpus for exact hand-calculated checks."""
    return [
        ["the", "model", "improves"],
        ["the", "model", "runs"],
        ["a", "model", "improves"],
    ]


@pytest.fixture
def nlp_sentences() -> list[list[str]]:
    """Load normalized tokens from sentences_nlp.json fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "sentences_nlp.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [rec["normalized_tokens"] for rec in records if not rec.get("in_references", False)]


def test_bigram_hand_calculated_probability(toy_corpus: list[list[str]]) -> None:
    """Bigram probability matches a hand-calculated value on a 3-sentence toy corpus."""
    # Sentences:
    # 1: <s> <s> the model improves </s>
    # 2: <s> <s> the model runs </s>
    # 3: <s> <s> a model improves </s>
    #
    # Vocabulary V = {the, model, improves, runs, a, </s>} -> |V| = 6
    # Let's check bigram P(model | the) with k = 0.1:
    # Context (the,): occurs 2 times (in sent 1 and sent 2)
    # Bigram (the, model): occurs 2 times
    # Hand-calc: (Count("the", "model") + k) / (Count("the") + k * |V|)
    #           = (2 + 0.1) / (2 + 0.1 * 6) = 2.1 / 2.6 = 0.8076923...
    model = NGramModel(n_values=(1, 2, 3), k=0.1)
    model.fit(toy_corpus)

    assert "model" in model.vocab
    assert len(model.vocab) == 6

    calculated_prob = model.prob("model", context=("the",))
    expected_prob = (2 + 0.1) / (2 + 0.1 * 6)
    assert math.isclose(calculated_prob, expected_prob, rel_tol=1e-5)

    # Check P(improves | model) with k=0.1:
    # Context (model,): occurs 3 times
    # Bigram (model, improves): occurs 2 times
    # Hand-calc: (2 + 0.1) / (3 + 0.1 * 6) = 2.1 / 3.6 = 0.5833333...
    p_improves = model.prob("improves", context=("model",))
    assert math.isclose(p_improves, (2 + 0.1) / (3 + 0.6), rel_tol=1e-5)


def test_probabilities_sum_to_one_and_unseen_nonzero(toy_corpus: list[list[str]]) -> None:
    """Probabilities over the vocabulary sum to ~1 with smoothing; unseen n-gram gives non-zero prob."""
    model = NGramModel(n_values=(1, 2, 3), k=0.1).fit(toy_corpus)

    # Unigram distribution sum
    total_unigram_prob = sum(model.prob(w) for w in model.vocab)
    assert math.isclose(total_unigram_prob, 1.0, rel_tol=1e-5)

    # Bigram conditional distribution sum given context ("the",)
    total_bigram_prob = sum(model.prob(w, context=("the",)) for w in model.vocab)
    assert math.isclose(total_bigram_prob, 1.0, rel_tol=1e-5)

    # Bigram conditional distribution sum given unseen context ("unknown",)
    total_unseen_ctx_prob = sum(model.prob(w, context=("unknown",)) for w in model.vocab)
    assert math.isclose(total_unseen_ctx_prob, 1.0, rel_tol=1e-5)

    # Trigram conditional distribution sum
    total_trigram_prob = sum(model.prob(w, context=("the", "model")) for w in model.vocab)
    assert math.isclose(total_trigram_prob, 1.0, rel_tol=1e-5)

    # Unseen word/ngram probability should be non-zero
    p_unseen = model.prob("unseen_word", context=("the", "model"))
    assert p_unseen > 0.0


def test_predict_next_and_top_ngrams(toy_corpus: list[list[str]]) -> None:
    """predict_next returns most frequent continuation first; top_ngrams sorted correctly."""
    model = NGramModel(n_values=(1, 2, 3), k=0.1).fit(toy_corpus)

    # Context ("the",) -> followed by "model" (count 2)
    predictions = model.predict_next(context=("the",), top=3)
    assert len(predictions) == 3
    assert predictions[0][0] == "model"
    assert predictions[0][1] > predictions[1][1]

    # Context ("model",) -> followed by "improves" (count 2) and "runs" (count 1)
    model_predictions = model.predict_next(context=("model",), top=3)
    assert model_predictions[0][0] == "improves"
    assert model_predictions[1][0] == "runs"
    assert model_predictions[0][1] > model_predictions[1][1]

    # Top unigrams
    top_unigrams = model.top_ngrams(n=1, top=10)
    assert len(top_unigrams) > 0
    # Both "model" and "</s>" appear 3 times in toy corpus
    unigram_dict = dict(top_unigrams)
    assert unigram_dict.get(("model",)) == 3
    assert top_unigrams[0][1] == 3

    # Top bigrams
    top_bigrams = model.top_ngrams(n=2, top=10)
    # Check descending order of counts
    counts = [c for _, c in top_bigrams]
    assert counts == sorted(counts, reverse=True)


def test_perplexity_finite_and_lower_on_seen_text(nlp_sentences: list[list[str]]) -> None:
    """Perplexity is finite and lower on seen than on scrambled text."""
    model = NGramModel(n_values=(1, 2, 3), k=0.1).fit(nlp_sentences)

    # Perplexity on training sentences
    pp_seen = model.perplexity(nlp_sentences)
    assert math.isfinite(pp_seen)
    assert pp_seen > 1.0

    # Scramble tokens within sentences to break n-gram patterns
    scrambled = []
    rng = random.Random(42)
    for sent in nlp_sentences:
        s = list(sent)
        rng.shuffle(s)
        scrambled.append(s)

    pp_scrambled = model.perplexity(scrambled)
    assert math.isfinite(pp_scrambled)
    # Natural seen sequences should have substantially lower perplexity than scrambled text
    assert pp_seen < pp_scrambled


def test_ngram_serialization(nlp_sentences: list[list[str]]) -> None:
    """Test to_dict and from_dict roundtrip."""
    model = NGramModel(n_values=(1, 2, 3), k=0.2).fit(nlp_sentences)
    d = model.to_dict()

    restored = NGramModel.from_dict(d)
    assert restored.n_values == model.n_values
    assert restored.k == model.k
    assert restored.vocab == model.vocab
    assert restored.total_unigrams == model.total_unigrams

    # Check matching probability
    p1 = model.prob("attention", context=("the", "model"))
    p2 = restored.prob("attention", context=("the", "model"))
    assert math.isclose(p1, p2)
