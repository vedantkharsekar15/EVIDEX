# EVIDEX Module 4: N-Gram, HMM POS Tagger, Frequency & Word Cloud

**Owner:** Shlok Parte (coding)
**Branch:** `feat/nlp-analysis`
**Depends on:** only the Shared Contract v1 and NLTK `treebank`. The HMM trains on Treebank, so it needs no other module.

## Contract (input/output)
```
N-Gram   IN : list[list[str]]   (each = SentenceRecord.normalized_tokens)
HMM      IN : list[str]         (SentenceRecord.tokens, raw case)  OUT: list[tuple[str, str]]
Freq     IN : list[SentenceRecord]  (skips in_references)            OUT: ranked counts
```

## Files
```
src/language_model/{__init__,ngram_model}.py
src/pos/{__init__,hmm_tagger}.py
src/analysis/{__init__,frequency,wordcloud}.py
app/pages/2_NLP_Analysis.py
tests/fixtures/sentences_nlp.json     # hand-written, ~15 sentences with tokens + normalized_tokens
tests/{test_ngram,test_pos,test_frequency}.py
```
`src/analysis/wordcloud.py` imports the third-party `wordcloud` package. This is fine with absolute imports; never put `src/analysis` itself on `sys.path`.

## Steps
1. Hand-write `sentences_nlp.json` from the contract.
2. `ngram_model.py`: unigram/bigram/trigram counts with `<s>`/`</s>`, add-k smoothing, probabilities, top n-grams, next-word prediction, perplexity.
3. `hmm_tagger.py`: load Treebank (clear error with the download command if missing), 90/10 split with fixed seed, transition and emission estimation (smoothed), unknown-word handling (capitalization + suffix heuristics), Viterbi in log space, held-out accuracy.
4. `frequency.py`: token counts excluding `in_references`, optional stop-words, ranked table.
5. `wordcloud.py`: image from frequencies, safe on empty input.
6. `app/pages/2_NLP_Analysis.py`: top N-Grams, next-word demo, POS demo for a chosen sentence, tagger accuracy shown with the note "trained on news text; exploratory only, does not drive retrieval", frequency table + Word Cloud.
7. Tests, then run on a real corpus.

## Interfaces
```python
# ngram_model.py
class NGramModel:
    def __init__(self, n_values: tuple[int, ...] = (1, 2, 3), k: float = 0.1)
    def fit(self, sentences: list[list[str]]) -> "NGramModel"
    def count(self, ngram: tuple[str, ...]) -> int
    def prob(self, word: str, context: tuple[str, ...] = ()) -> float        # add-k smoothed
    def top_ngrams(self, n: int, top: int = 20) -> list[tuple[tuple[str, ...], int]]
    def predict_next(self, context: tuple[str, ...], top: int = 5) -> list[tuple[str, float]]
    def perplexity(self, sentences: list[list[str]]) -> float
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "NGramModel"
# hmm_tagger.py
class MissingNLTKDataError(Exception)      # message includes: python -m nltk.downloader treebank
class HMMTagger:
    def train(self, tagged_sents: list[list[tuple[str, str]]]) -> None
    def train_from_treebank(self, test_split: float = 0.1, seed: int = 42) -> float   # returns held-out accuracy
    def tag(self, tokens: list[str]) -> list[tuple[str, str]]            # Viterbi
    def evaluate(self, tagged_sents: list[list[tuple[str, str]]]) -> float
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "HMMTagger"
# frequency.py
term_frequencies(records: list[dict], top_n: int | None = None, remove_stopwords: bool = False) -> list[tuple[str, int]]
frequency_table(records: list[dict], top_n: int = 50) -> "pandas.DataFrame"      # columns: term, count, rank
# wordcloud.py
build_wordcloud(freqs: dict[str, int], max_words: int = 100) -> "PIL.Image.Image | None"   # None if empty
```
Serialization uses `to_dict`/`from_dict`. Module 1 persists the dicts via `save_artifact`; do not write files yourself.

## Tests (`pytest tests/test_ngram.py tests/test_pos.py tests/test_frequency.py`)
- Bigram probability matches a hand-calculated value on a 3-sentence toy corpus (with `<s>`, `</s>`).
- Probabilities over the vocabulary sum to ≈ 1 with smoothing; unseen n-gram gives a non-zero probability.
- `predict_next` returns the most frequent continuation first; `top_ngrams` sorted correctly.
- Perplexity is finite and lower on seen than on scrambled text.
- HMM on a tiny hand-made tagged corpus: transition/emission values correct; Viterbi gives the expected tags on an ambiguous word.
- Unknown word does not crash and gets a plausible tag (capitalized → NNP, `-ing` → VBG).
- Empty token list returns `[]`; held-out accuracy is a float in [0, 1] and the same on repeat runs (seeded).
- Missing treebank raises `MissingNLTKDataError` with the download command.
- Frequency: counts correct, references excluded, stop-word toggle works, empty input gives an empty result; Word Cloud returns `None` for empty input, an image otherwise.

## Handoff
- **To Module 1 (Vedant):** `NGramModel.fit/to_dict/from_dict`, `term_frequencies` for `build_corpus`; `HMMTagger` trained once and cached as an artifact (lazy-loaded by the NLP page).
- **From Module 2:** `tokens` / `normalized_tokens` fields (until merged, use your fixture).
- **From Module 1:** `st.session_state["corpus"]` for the page (keys: `records`, `ngram_model`, `frequencies`).
- **Report to the team:** HMM held-out accuracy (needed for the Definition of Done and the final report).
- **PR checklist:** tests pass, type hints, no new deps beyond NLTK/NumPy/Pandas/WordCloud/Matplotlib, signatures unchanged.

## Git Workflow
1. `git checkout main && git pull`, then `git checkout -b feat/nlp-analysis` (after the scaffold PR is merged).
2. Commit small and often: `feat(lm): add bigram smoothing`, `feat(pos): add viterbi decoding`, `test(pos): add unknown-word case`.
3. Touch only `src/language_model/`, `src/pos/`, `src/analysis/`, `app/pages/2_NLP_Analysis.py`, your tests and fixture. Ask Vedant for any change to the contract or config.
4. Consider separate PRs per component (N-Gram, HMM, frequency/wordcloud, app page) to keep reviews small.
5. `git fetch && git rebase origin/main`, then `pytest` locally.
6. Open the PR to `main` with the template. CI green + 1 approval, squash merge, delete the branch.
