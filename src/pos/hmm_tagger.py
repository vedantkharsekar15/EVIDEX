"""Hidden Markov Model (HMM) Part-of-Speech Tagger with Viterbi Decoding.

Trains on tagged corpora (such as Penn Treebank), estimates smoothed transition
and emission probabilities, handles unknown words via morphological and capitalization
heuristics, and performs Viterbi decoding in log space.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Sequence


class MissingNLTKDataError(Exception):
    """Raised when required NLTK dataset (e.g. Penn Treebank) is not installed."""

    def __init__(
        self,
        message: str = (
            "Required NLTK Penn Treebank data is missing. "
            "Please install it using: python -m nltk.downloader treebank"
        ),
    ) -> None:
        super().__init__(message)


class HMMTagger:
    """Part-of-Speech tagger based on a First-Order Hidden Markov Model."""

    START_TAG = "^START^"

    def __init__(self, alpha: float = 0.01, beta: float = 0.001) -> None:
        """Initialize the HMM Tagger.

        Args:
            alpha: Laplace smoothing parameter for transitions.
            beta: Laplace smoothing parameter for emissions.
        """
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.tags: list[str] = []
        self.tag_set: set[str] = set()
        self.vocab: set[str] = set()

        # Counts
        self.tag_counts: Counter[str] = Counter()
        self.transition_counts: Counter[tuple[str, str]] = Counter()  # (prev_tag, curr_tag)
        self.emission_counts: Counter[tuple[str, str]] = Counter()    # (tag, word)
        self.word_counts: Counter[str] = Counter()
        self.num_sentences: int = 0

        # Log probability caches
        self.log_start_probs: dict[str, float] = {}
        self.log_trans_probs: dict[tuple[str, str], float] = {}
        self.log_emission_probs: dict[tuple[str, str], float] = {}
        self.default_tag: str = "NN"

    def train(self, tagged_sents: list[list[tuple[str, str]]]) -> None:
        """Train the HMM parameters on a list of tagged sentences.

        Args:
            tagged_sents: List of sentences, where each sentence is a list of (word, tag) tuples.
        """
        if not tagged_sents:
            return

        self.tag_counts = Counter()
        self.transition_counts = Counter()
        self.emission_counts = Counter()
        self.word_counts = Counter()
        self.tag_set = set()
        self.vocab = set()
        self.num_sentences = len(tagged_sents)

        for sent in tagged_sents:
            prev_tag = self.START_TAG
            for word, tag in sent:
                self.tag_set.add(tag)
                self.vocab.add(word)
                self.tag_counts[tag] += 1
                self.word_counts[word] += 1
                self.transition_counts[(prev_tag, tag)] += 1
                self.emission_counts[(tag, word)] += 1
                prev_tag = tag

        self.tags = sorted(list(self.tag_set))
        if "NN" in self.tag_set:
            self.default_tag = "NN"
        elif self.tags:
            self.default_tag = self.tag_counts.most_common(1)[0][0]

        self._compute_log_probabilities()

    def _compute_log_probabilities(self) -> None:
        """Pre-compute log probability tables with add-alpha and add-beta smoothing."""
        num_tags = len(self.tags)
        num_words = len(self.vocab)

        if num_tags == 0:
            return

        self.log_start_probs = {}
        self.log_trans_probs = {}
        self.log_emission_probs = {}

        # 1. Initial tag log-probabilities from START_TAG
        denom_start = self.num_sentences + self.alpha * num_tags
        for tag in self.tags:
            c = self.transition_counts.get((self.START_TAG, tag), 0)
            p = (c + self.alpha) / denom_start
            self.log_start_probs[tag] = math.log(max(p, 1e-15))

        # 2. Transition log-probabilities
        for prev_tag in self.tags:
            denom_trans = self.tag_counts[prev_tag] + self.alpha * num_tags
            for curr_tag in self.tags:
                c = self.transition_counts.get((prev_tag, curr_tag), 0)
                p = (c + self.alpha) / denom_trans
                self.log_trans_probs[(prev_tag, curr_tag)] = math.log(max(p, 1e-15))

        # 3. Emission log-probabilities for seen (tag, word)
        for tag in self.tags:
            denom_emiss = self.tag_counts[tag] + self.beta * (num_words + 1)
            for (t, word), c in self.emission_counts.items():
                if t == tag:
                    p = (c + self.beta) / denom_emiss
                    self.log_emission_probs[(tag, word)] = math.log(max(p, 1e-15))

    def _get_log_emission(self, tag: str, word: str, is_sentence_start: bool = False) -> float:
        """Compute the log emission probability for a given (tag, word), handling OOV."""
        num_words = len(self.vocab)
        denom_emiss = self.tag_counts[tag] + self.beta * (num_words + 1)

        # 1. Known word with observed emission for this tag
        if (tag, word) in self.log_emission_probs:
            return self.log_emission_probs[(tag, word)]

        # 2. Known word, but unseen with this particular tag (smoothed)
        if word in self.vocab:
            p = self.beta / denom_emiss
            return math.log(max(p, 1e-15))

        # 3. Unknown (OOV) word handling with morphology and capitalization heuristics
        heuristic_boost = self._oov_tag_weight(tag, word, is_sentence_start)
        p = (self.beta * heuristic_boost) / denom_emiss
        return math.log(max(p, 1e-15))

    def _oov_tag_weight(self, tag: str, word: str, is_sentence_start: bool) -> float:
        """Return a heuristic multiplier for unknown words based on morphology."""
        lower = word.lower()

        # Numbers / digits
        if any(char.isdigit() for char in word):
            if tag == "CD":
                return 100.0
            return 0.1

        # Capitalized words (proper nouns)
        if word[0].isupper():
            if not is_sentence_start:
                if tag in ("NNP", "NNPS"):
                    return 80.0
            else:
                # First word of sentence might be capitalized just because of punctuation
                if tag in ("NNP", "NNPS", "NN", "JJ", "RB"):
                    return 10.0

        # Hyphenated compound words
        if "-" in word:
            if tag in ("JJ", "NN"):
                return 30.0

        # Suffix heuristics
        if lower.endswith("ing"):
            if tag == "VBG":
                return 500.0
            if tag in ("NN", "JJ"):
                return 5.0
            return 0.01
        elif lower.endswith("ed"):
            if tag in ("VBD", "VBN"):
                return 500.0
            if tag == "JJ":
                return 15.0
            return 0.01
        elif lower.endswith("ly"):
            if tag == "RB":
                return 500.0
            if tag == "JJ":
                return 10.0
            return 0.01
        elif lower.endswith(("tion", "sion", "ment", "ness", "ity", "ance", "ence", "ism", "ist")):
            if tag == "NN":
                return 500.0
            return 0.01
        elif lower.endswith(("able", "ible", "ous", "ive", "ful", "less", "al", "ic")):
            if tag == "JJ":
                return 500.0
            return 0.01
        elif lower.endswith(("ize", "ise", "ify", "ate")):
            if tag in ("VB", "VBP"):
                return 100.0
            return 0.01
        elif lower.endswith("s") and not lower.endswith("ss"):
            if tag in ("NNS", "VBZ"):
                return 100.0
            return 0.01

        # General open-class fallback prior
        if tag in ("NN", "NNS", "JJ", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "RB", "NNP"):
            return 5.0

        # Closed-class tags are unlikely for OOV words
        return 0.05

    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        """Tag a sequence of raw-case tokens using Viterbi decoding in log space.

        Args:
            tokens: List of raw-case word tokens.

        Returns:
            List of (word, tag) tuples.
        """
        if not tokens:
            return []

        if not self.tags:
            return [(token, "NN") for token in tokens]

        T = len(tokens)
        num_tags = len(self.tags)

        # Viterbi table: viterbi[t_idx][tag_idx] = max_log_prob
        # Backpointers: backpointers[t_idx][tag_idx] = best_prev_tag_idx
        viterbi = [[-float("inf")] * num_tags for _ in range(T)]
        backpointers = [[0] * num_tags for _ in range(T)]

        first_word = tokens[0]
        # Step 0: Initialization
        for j, tag in enumerate(self.tags):
            log_init = self.log_start_probs.get(tag, -100.0)
            log_emiss = self._get_log_emission(tag, first_word, is_sentence_start=True)
            viterbi[0][j] = log_init + log_emiss

        # Steps 1 to T-1: Recursion
        for t in range(1, T):
            word = tokens[t]
            for j, curr_tag in enumerate(self.tags):
                log_emiss = self._get_log_emission(curr_tag, word, is_sentence_start=False)
                best_val = -float("inf")
                best_prev_idx = 0

                for i, prev_tag in enumerate(self.tags):
                    log_trans = self.log_trans_probs.get((prev_tag, curr_tag), -100.0)
                    val = viterbi[t - 1][i] + log_trans
                    if val > best_val:
                        best_val = val
                        best_prev_idx = i

                viterbi[t][j] = best_val + log_emiss
                backpointers[t][j] = best_prev_idx

        # Step T-1: Termination / Best final state
        best_last_tag_idx = max(range(num_tags), key=lambda idx: viterbi[T - 1][idx])

        # Backtracking
        best_path_indices = [0] * T
        best_path_indices[-1] = best_last_tag_idx
        for t in range(T - 1, 0, -1):
            best_path_indices[t - 1] = backpointers[t][best_path_indices[t]]

        result = [(tokens[t], self.tags[best_path_indices[t]]) for t in range(T)]
        return result

    def evaluate(self, tagged_sents: list[list[tuple[str, str]]]) -> float:
        """Evaluate accuracy on a test set of tagged sentences.

        Args:
            tagged_sents: Gold standard test sentences with (word, tag) tuples.

        Returns:
            Accuracy score as a float in [0.0, 1.0].
        """
        if not tagged_sents:
            return 0.0

        total_tokens = 0
        correct_tokens = 0

        for sent in tagged_sents:
            if not sent:
                continue
            tokens = [word for word, _ in sent]
            gold_tags = [tag for _, tag in sent]
            preds = self.tag(tokens)

            for (_, pred_tag), gold_tag in zip(preds, gold_tags):
                if pred_tag == gold_tag:
                    correct_tokens += 1
                total_tokens += 1

        if total_tokens == 0:
            return 0.0
        return correct_tokens / total_tokens

    def train_from_treebank(self, test_split: float = 0.1, seed: int = 42) -> float:
        """Load Penn Treebank from NLTK, train on 1-test_split, and evaluate on test_split.

        Args:
            test_split: Fraction of sentences to reserve for testing (e.g. 0.1 for 10%).
            seed: Random seed for deterministic train/test partition.

        Returns:
            Held-out accuracy float in [0.0, 1.0].

        Raises:
            MissingNLTKDataError: If NLTK treebank corpus is not installed.
        """
        try:
            import nltk
            from nltk.corpus import treebank
            sents = list(treebank.tagged_sents())
            if not sents:
                raise MissingNLTKDataError()
        except (ImportError, LookupError, OSError, AttributeError) as exc:
            raise MissingNLTKDataError() from exc

        rng = random.Random(seed)
        indices = list(range(len(sents)))
        rng.shuffle(indices)

        split_idx = int(len(sents) * (1.0 - test_split))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_sents = [sents[i] for i in train_indices]
        test_sents = [sents[i] for i in test_indices]

        self.train(train_sents)
        acc = self.evaluate(test_sents)
        return acc

    def to_dict(self) -> dict:
        """Serialize the HMMTagger state to a dictionary."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "tags": self.tags,
            "vocab": sorted(list(self.vocab)),
            "num_sentences": self.num_sentences,
            "default_tag": self.default_tag,
            "tag_counts": dict(self.tag_counts),
            "word_counts": dict(self.word_counts),
            "transition_counts": [
                {"prev": prev, "curr": curr, "count": count}
                for (prev, curr), count in self.transition_counts.items()
            ],
            "emission_counts": [
                {"tag": tag, "word": word, "count": count}
                for (tag, word), count in self.emission_counts.items()
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> HMMTagger:
        """Reconstruct an HMMTagger instance from a serialized dictionary."""
        tagger = cls(
            alpha=float(d.get("alpha", 0.01)),
            beta=float(d.get("beta", 0.001)),
        )
        tagger.tags = list(d.get("tags", []))
        tagger.tag_set = set(tagger.tags)
        tagger.vocab = set(d.get("vocab", []))
        tagger.num_sentences = int(d.get("num_sentences", 0))
        tagger.default_tag = str(d.get("default_tag", "NN"))

        tagger.tag_counts = Counter(d.get("tag_counts", {}))
        tagger.word_counts = Counter(d.get("word_counts", {}))

        tagger.transition_counts = Counter()
        for item in d.get("transition_counts", []):
            tagger.transition_counts[(item["prev"], item["curr"])] = int(item["count"])

        tagger.emission_counts = Counter()
        for item in d.get("emission_counts", []):
            tagger.emission_counts[(item["tag"], item["word"])] = int(item["count"])

        tagger._compute_log_probabilities()
        return tagger
