"""Statistical N-Gram Language Model with Add-k Smoothing.

Supports unigrams, bigrams, trigrams, boundary tokens (<s>, </s>),
conditional probabilities, next-word prediction, and perplexity evaluation.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


class NGramModel:
    """N-Gram statistical language model supporting add-k smoothing."""

    START_TOKEN = "<s>"
    END_TOKEN = "</s>"

    def __init__(self, n_values: tuple[int, ...] = (1, 2, 3), k: float = 0.1) -> None:
        """Initialize the NGramModel.

        Args:
            n_values: Tuple of n values to maintain (e.g. (1, 2, 3) for unigram, bigram, trigram).
            k: Add-k smoothing parameter (must be non-negative).
        """
        if not n_values:
            raise ValueError("n_values cannot be empty.")
        if k < 0:
            raise ValueError("Smoothing parameter k must be non-negative.")

        self.n_values = tuple(sorted(n_values))
        self.max_n = max(self.n_values)
        self.k = float(k)
        self.ngram_counts: dict[int, Counter[tuple[str, ...]]] = {
            n: Counter() for n in self.n_values
        }
        # Also maintain context counts for denominators if needed (e.g., prefix (n-1)-grams)
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.vocab: set[str] = set()
        self.total_unigrams: int = 0

    def fit(self, sentences: list[list[str]]) -> NGramModel:
        """Fit the n-gram model on a list of tokenized sentences.

        Args:
            sentences: List of sentences, where each sentence is a list of normalized tokens.

        Returns:
            self for method chaining.
        """
        # Reset internal counters
        self.ngram_counts = {n: Counter() for n in self.n_values}
        self.context_counts = Counter()
        self.vocab = set()
        self.total_unigrams = 0

        pad_length = self.max_n - 1

        for sent in sentences:
            if not isinstance(sent, list):
                sent = list(sent)

            # Collect vocabulary words (excluding padding start tokens, including end token)
            for token in sent:
                self.vocab.add(token)
            self.vocab.add(self.END_TOKEN)

            # Padded sentence: [<s>, <s>, ..., token1, token2, ..., </s>]
            padded_sent = [self.START_TOKEN] * pad_length + sent + [self.END_TOKEN]

            # Count unigrams: count from the original sentence + </s>
            eval_tokens = sent + [self.END_TOKEN]
            for token in eval_tokens:
                self.ngram_counts[1][(token,)] += 1
                self.total_unigrams += 1

            # Count higher-order n-grams and context prefixes
            for n in self.n_values:
                if n == 1:
                    continue
                # Extract sliding windows of length n
                for i in range(len(padded_sent) - n + 1):
                    ngram = tuple(padded_sent[i : i + n])
                    # Ensure the n-gram doesn't consist solely of START_TOKENs
                    if all(t == self.START_TOKEN for t in ngram):
                        continue
                    self.ngram_counts[n][ngram] += 1
                    context = ngram[:-1]
                    self.context_counts[context] += 1

        return self

    def count(self, ngram: tuple[str, ...]) -> int:
        """Return the raw count of a given n-gram tuple.

        Args:
            ngram: Tuple of tokens representing the n-gram.

        Returns:
            Raw occurrence count (0 if unseen).
        """
        n = len(ngram)
        if n in self.ngram_counts:
            return self.ngram_counts[n].get(ngram, 0)
        # Fallback search if length wasn't in self.n_values
        return 0

    def prob(self, word: str, context: tuple[str, ...] = ()) -> float:
        """Calculate the add-k smoothed conditional probability P(word | context).

        Args:
            word: Target word token.
            context: Preceding context tuple.

        Returns:
            Smoothed conditional probability value.
        """
        vocab_size = len(self.vocab)
        if vocab_size == 0:
            # Unfitted model fallback
            return 0.0

        if not context:
            # Unigram probability: P(word) = (Count(word) + k) / (N + k * |V|)
            word_count = self.ngram_counts.get(1, Counter()).get((word,), 0)
            denominator = self.total_unigrams + self.k * vocab_size
            if denominator == 0:
                return 1.0 / vocab_size
            return (word_count + self.k) / denominator

        # Truncate context to match max_n - 1
        if len(context) >= self.max_n:
            context = tuple(context[-(self.max_n - 1) :])

        ngram = context + (word,)
        n = len(ngram)

        # Retrieve ngram count
        ngram_count = 0
        if n in self.ngram_counts:
            ngram_count = self.ngram_counts[n].get(ngram, 0)

        # Context count (denominator)
        if n == 2:
            # Context is a unigram (or <s>)
            if context == (self.START_TOKEN,):
                # Count of <s> as context is the context_counts[(<s>,)]
                ctx_count = self.context_counts.get(context, 0)
            else:
                ctx_count = self.ngram_counts.get(1, Counter()).get(context, 0)
                if ctx_count == 0:
                    ctx_count = self.context_counts.get(context, 0)
        else:
            ctx_count = self.context_counts.get(context, 0)
            if ctx_count == 0 and (n - 1) in self.ngram_counts:
                ctx_count = self.ngram_counts[n - 1].get(context, 0)

        denominator = ctx_count + self.k * vocab_size
        if denominator == 0:
            return 1.0 / vocab_size
        return (ngram_count + self.k) / denominator

    def top_ngrams(self, n: int, top: int = 20) -> list[tuple[tuple[str, ...], int]]:
        """Return the top most frequent n-grams of order n.

        Args:
            n: Order of n-gram (1, 2, 3, etc.).
            top: Maximum number of n-grams to return.

        Returns:
            List of (ngram_tuple, count) pairs sorted by frequency descending.
        """
        if n not in self.ngram_counts:
            return []

        counts = self.ngram_counts[n]
        # Sort by count desc, then alphabetically by ngram for deterministic order
        sorted_items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return sorted_items[:top]

    def predict_next(self, context: tuple[str, ...], top: int = 5) -> list[tuple[str, float]]:
        """Predict the most likely next words given a context tuple.

        Args:
            context: Tuple of preceding tokens.
            top: Maximum number of candidate predictions to return.

        Returns:
            List of (next_word, probability) sorted descending by probability.
        """
        if not self.vocab:
            return []

        # Predict among vocabulary tokens
        candidates = [w for w in self.vocab if w != self.START_TOKEN]
        scored: list[tuple[str, float]] = []

        for word in candidates:
            p = self.prob(word, context)
            scored.append((word, p))

        # Sort by probability descending, then word ascending
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top]

    def perplexity(self, sentences: list[list[str]]) -> float:
        """Calculate the perplexity of the model on a test set of sentences.

        Args:
            sentences: List of tokenized test sentences.

        Returns:
            Perplexity score (finite float >= 1.0). Lower is better.
        """
        if not sentences or not self.vocab:
            return float("inf")

        log_prob_sum = 0.0
        total_tokens = 0
        pad_length = self.max_n - 1

        for sent in sentences:
            if not isinstance(sent, list):
                sent = list(sent)

            padded_sent = [self.START_TOKEN] * pad_length + sent + [self.END_TOKEN]
            eval_tokens = sent + [self.END_TOKEN]

            for idx, token in enumerate(eval_tokens):
                # Target token index in padded_sent is idx + pad_length
                actual_idx = idx + pad_length
                # Context is the preceding pad_length tokens
                context = tuple(padded_sent[actual_idx - pad_length : actual_idx])

                p = self.prob(token, context)
                if p <= 0:
                    p = 1e-15  # Numerical safety
                log_prob_sum += math.log(p)
                total_tokens += 1

        if total_tokens == 0:
            return float("inf")

        cross_entropy = -log_prob_sum / total_tokens
        try:
            return math.exp(cross_entropy)
        except OverflowError:
            return float("inf")

    def to_dict(self) -> dict:
        """Serialize the NGramModel state to a Python dictionary."""
        # Convert tuple keys to lists for JSON / YAML friendliness
        serialized_counts: dict[str, list[dict]] = {}
        for n, counter in self.ngram_counts.items():
            serialized_counts[str(n)] = [
                {"ngram": list(ngram), "count": count} for ngram, count in counter.items()
            ]

        serialized_context = [
            {"context": list(ctx), "count": count}
            for ctx, count in self.context_counts.items()
        ]

        return {
            "n_values": list(self.n_values),
            "k": self.k,
            "vocab": sorted(list(self.vocab)),
            "total_unigrams": self.total_unigrams,
            "ngram_counts": serialized_counts,
            "context_counts": serialized_context,
        }

    @classmethod
    def from_dict(cls, d: dict) -> NGramModel:
        """Reconstruct an NGramModel instance from a dictionary."""
        n_values = tuple(d.get("n_values", [1, 2, 3]))
        k = float(d.get("k", 0.1))
        model = cls(n_values=n_values, k=k)

        model.vocab = set(d.get("vocab", []))
        model.total_unigrams = int(d.get("total_unigrams", 0))

        raw_counts = d.get("ngram_counts", {})
        for n_str, items in raw_counts.items():
            n = int(n_str)
            counter = Counter()
            for item in items:
                counter[tuple(item["ngram"])] = int(item["count"])
            model.ngram_counts[n] = counter

        ctx_items = d.get("context_counts", [])
        for item in ctx_items:
            model.context_counts[tuple(item["context"])] = int(item["count"])

        return model
