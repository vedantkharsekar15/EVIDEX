# EVIDEX — Mini-Project Implementation Plan

> **Planning document.** In the repository, keep this file at `docs/mini_project_plan.md`. The root `README.md` stays short.
>
> **Scope.** This document covers the academic mini-project only. Semantic retrieval, NLI, RAG, voice and quantization belong to the full-project continuation and are not part of this version.

---

## 1. Project Overview

**EVIDEX** is an NLP-based research intelligence application that helps a user work with a collection of research papers and understand the evidence contained in them.

The mini-project focuses on building a reliable NLP pipeline that can:

- accept research-paper PDFs supplied by the user,
- extract and clean their text,
- preserve document and page-level provenance,
- analyse the language structure of the corpus,
- identify important sentences that can act as research-claim candidates,
- retrieve potentially relevant evidence from the uploaded papers,
- compare text using TF-IDF and cosine similarity,
- expose the result through a simple interactive application,
- and provide useful visual analysis of the corpus.

The goal is to build **one coherent working application**, not a set of disconnected demonstrations.

### 1.1 What makes EVIDEX different

Many research tools search large public paper databases. EVIDEX is built for a different case: **the user's own collection of papers, processed locally, with every result traceable to a page.**

The USPs of this version:

- **Private, user-supplied corpus.** No external paper database or API is needed. Papers stay on the user's machine.
- **Traceable by construction.** Every sentence, claim candidate and retrieval result carries document, page and sentence ID.
- **Transparent lexical baseline.** TF-IDF and cosine scores are visible and reproducible. In the full project this baseline is the reference every advanced component is measured against.
- **Honest labelling.** Claim candidates are labelled as candidates. Similarity is never presented as support.
- **Inspectable classical NLP.** The N-Gram model and the HMM POS tagger are implemented in-house, so every probability can be inspected and explained.
- **Measured, not assumed.** Retrieval is evaluated with Precision@K, Recall@K and MRR on a manually labelled set.

What this version does **not** claim: it does not verify scientific truth, and it does not decide whether a paper supports or contradicts a claim. That comes later and is evaluated separately.

---

## 2. Mini-Project Goal

The completed mini-project should allow a user to:

1. start the application,
2. upload one or more research papers in PDF format,
3. build a searchable research corpus from those papers,
4. inspect the extracted and processed content,
5. analyse linguistic patterns,
6. enter a claim or research statement,
7. find text passages in the uploaded papers that are relevant to that claim,
8. view where each retrieved passage came from,
9. inspect claim–evidence similarity scores (high lexical similarity means a passage is related, **not** that it supports the claim),
10. view corpus-level keyword/frequency analysis.

The system should work with **new papers supplied by the user**. No fixed external research-paper dataset is required for the application to function.

---

## 3. Scope of This Version

This version establishes the complete foundational NLP application.

### Included

- PDF ingestion
- text extraction
- document/page metadata
- text cleaning
- sentence segmentation
- tokenization
- normalization
- corpus creation
- N-Gram language modelling
- HMM-based POS tagging
- TF-IDF representation
- cosine-similarity retrieval
- candidate claim identification
- evidence passage retrieval
- provenance tracking
- Word Cloud / frequency visualisation
- interactive Streamlit application
- unit tests
- evaluation using user-supplied papers
- logging and error handling

### Not included in this version

Do not add advanced semantic retrieval, natural-language-inference classification, a generative RAG answer layer, voice interaction, or model quantization to this version.

Keep the implementation focused enough that the whole mini-project can be completed, tested, demonstrated, and understood end-to-end.

---

## 4. Core User Flow

```text
User
  ↓
Upload Research Papers
  ↓
PDF Validation
  ↓
Text Extraction
  ↓
Document + Page Metadata
  ↓
Preprocessing
  ↓
Sentence / Token Representation
  ↓
Corpus Analysis
  ├── N-Gram Language Model
  ├── HMM POS Tagging
  ├── TF-IDF Representation
  └── Frequency / Word Cloud
  ↓
Claim Input / Claim Candidate Detection
  ↓
TF-IDF + Cosine Similarity Retrieval
  ↓
Relevant Evidence Passages
  ↓
Source / Page Provenance
  ↓
Interactive Results
```

---

## 5. Important Design Principle

The project must maintain the difference between:

- **raw source content** — the original uploaded PDF,
- **processed content** — cleaned and segmented text,
- **analysis artifacts** — N-Grams, POS tags, TF-IDF vectors, frequencies,
- **claim candidates** — sentences identified as potentially claim-like,
- **retrieved evidence** — passages returned for a particular claim,
- **user-facing results** — what is displayed by the application.

Never overwrite the original source papers.

Every derived text unit should retain enough metadata to identify:

- document,
- page,
- sentence,
- and source text location.

This provenance is essential for a research-oriented application.

---

# 6. Input Data

## 6.1 Research Papers

The application is designed for **user-provided research papers**.

The user can later upload papers on any supported research topic. The implementation should therefore avoid assumptions such as:

- fixed authors,
- fixed paper titles,
- fixed number of papers,
- fixed keywords,
- fixed topic,
- fixed labels.

A useful initial development corpus may contain a small number of open-access papers purely for testing, but those papers are **test inputs**, not a permanent project dataset.

## 6.2 Supported Initial Input

Primary input:

- `.pdf`

Optional future formats should not be added unless there is a real requirement.

## 6.3 Input Validation

Before processing:

- verify that the file is actually a PDF (check the `%PDF` header, not just the extension),
- reject empty or corrupted files,
- prevent duplicate ingestion,
- assign a stable document ID (derived from the SHA-256 hash of the file content),
- record filename and ingestion timestamp,
- report extraction failures clearly.

---

# 7. Repository Structure

Use a clean project structure from the beginning.

```text
EVIDEX/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   └── metadata.py
│   │
│   ├── preprocessing/
│   │   ├── cleaning.py
│   │   ├── sentence_splitter.py
│   │   └── tokenizer.py
│   │
│   ├── language_model/
│   │   └── ngram_model.py
│   │
│   ├── pos/
│   │   └── hmm_tagger.py
│   │
│   ├── similarity/
│   │   ├── tfidf.py
│   │   └── retrieval.py
│   │
│   ├── claims/
│   │   └── claim_candidates.py
│   │
│   ├── analysis/
│   │   ├── frequency.py
│   │   └── wordcloud.py
│   │
│   ├── pipeline/
│   │   └── build_corpus.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── file_utils.py
│
├── data/
│   ├── raw/
│   │   └── papers/
│   ├── processed/
│   └── artifacts/
│
├── evaluation/
│   ├── claims.json
│   └── relevance_labels.json
│
├── tests/
│   ├── fixtures/
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_ngram.py
│   ├── test_pos.py
│   ├── test_tfidf.py
│   ├── test_claims.py
│   ├── test_retrieval.py
│   ├── test_frequency.py
│   └── test_pipeline.py
│
├── docs/
│   └── mini_project_plan.md
│
├── requirements.txt
├── .gitignore
├── README.md
└── config.yaml
```

Every package directory under `src/` contains an `__init__.py` (omitted above for readability). Empty data directories are kept in Git with a `.gitkeep` file.

The structure is intentionally modular. Each major NLP component should be independently testable.

---

# 8. What Each Directory Is For

## `app/`

Contains only the application/UI layer.

It should call reusable functions from `src/` instead of containing the NLP implementation itself.

## `src/`

Contains the actual EVIDEX implementation.

This is the main development area.

## `data/raw/papers/`

Stores user-provided research-paper PDFs during local development.

This directory is an **input location**, not a committed dataset.

Do not commit private or copyrighted papers to the repository.

## `data/processed/`

Stores cleaned/structured representations created from source papers.

Examples:

- extracted text,
- sentence records,
- normalized corpus.

## `data/artifacts/`

Stores reusable outputs generated by the pipeline.

Examples:

- N-Gram counts,
- TF-IDF vocabulary,
- serialized TF-IDF matrix,
- frequency statistics.

Generated artifacts should be reproducible from the source papers.

## `evaluation/`

Stores the manually prepared evaluation set: the claims used for testing and the sentences labelled as relevant to each claim (see Section 27). It contains labels only, not the papers themselves.

## `tests/`

Contains automated tests for individual components and small end-to-end checks.

`tests/fixtures/` holds small test inputs. Generate tiny test PDFs programmatically or use clearly open-licensed documents. Do not commit copyrighted papers.

## `docs/`

Holds the planning documents.

---

# 9. Environment Setup

Use **Python 3.11** in an isolated virtual environment.

Example:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the required NLTK data once (the sentence tokenizer and the Penn Treebank sample used to train the POS tagger):

```bash
python -m nltk.downloader punkt punkt_tab treebank
```

(`punkt_tab` is required by newer NLTK versions; `punkt` by older ones.)

The initial dependency set should remain small and purposeful. Pin versions in `requirements.txt`.

Typical libraries:

```text
PyMuPDF
NLTK
NumPy
Pandas
Scikit-learn
Matplotlib
WordCloud
Streamlit
PyYAML
pytest
```

Do not add libraries merely because they are popular. Add a dependency only when it is required by an implemented module.

---

# 10. Configuration

Keep configurable values outside the source code.

Examples:

```yaml
app:
  title: "EVIDEX"

paths:
  raw_papers: "data/raw/papers"
  processed: "data/processed"
  artifacts: "data/artifacts"

processing:
  min_sentence_length: 5
  max_sentence_length: 500
  exclude_references: true
  min_alpha_ratio: 0.6

preprocessing:
  lowercase: true
  remove_stopwords: false   # decide by retrieval evaluation, not assumption

tfidf:
  ngram_range: [1, 2]
  min_df: 1
  sublinear_tf: true

retrieval:
  top_k: 5
  min_similarity: 0.10
  exclude_source_sentence: true

claims:
  min_candidate_score: 2
  max_candidates: 50

language_model:
  n_values: [1, 2, 3]
  smoothing: "add_k"
  k: 0.1

pos:
  training_corpus: "nltk_treebank"
  test_split: 0.1
```

The exact values should be tuned during testing rather than treated as universal constants.

---

# 11. PDF Ingestion

Implement a dedicated PDF ingestion module.

### Input

One or more PDF files.

### Output

Structured document records such as:

```python
{
    "document_id": "...",
    "filename": "...",
    "page_number": 1,
    "text": "..."
}
```

### Requirements

- extract page-by-page,
- preserve page number,
- preserve document ID,
- normalize extraction output,
- handle empty pages,
- handle extraction errors without crashing the entire batch,
- for two-column papers, extract text blocks and sort them in reading order instead of using raw line order.

The application should report which documents were successfully processed and which failed.

---

# 12. Text Preprocessing

Create a reproducible preprocessing pipeline.

Recommended order:

```text
Raw page text
    ↓
Unicode normalization
    ↓
Whitespace normalization
    ↓
Removal of unwanted extraction artifacts
    ↓
Sentence segmentation
    ↓
Tokenization
    ↓
Case normalization
    ↓
Optional stop-word handling
```

Do not aggressively remove content before understanding the effect.

For example, punctuation may be unnecessary for some frequency calculations but useful during sentence processing.

Maintain separate representations where necessary instead of destroying information globally.

### 12.1 Extraction artifacts to handle explicitly

Research PDFs are messy. The cleaning module should handle at least:

- repeated headers and footers (lines that repeat across many pages),
- page numbers,
- end-of-line hyphenation (`evalu-\nation` → `evaluation`),
- the **references section**: detect it and mark its sentences (`in_references = true`). They stay in the processed corpus but are excluded from the retrieval index and from claim candidates by default (`processing.exclude_references`),
- figure and table captions (keep, but do not treat as claim candidates),
- equation and table debris: filter sentences with a very low alphabetic-character ratio (`processing.min_alpha_ratio`),
- sentences split across a page break.

### 12.2 Two token representations

Keep both:

- `tokens` — raw-case tokens, used by the HMM POS tagger (capitalization is a useful signal),
- `normalized_tokens` — lowercased, optionally stop-word filtered, used by the N-Gram model, TF-IDF and frequency analysis.

---

# 13. Sentence-Level Data Model

Represent every processed sentence as a structured record.

Example:

```python
{
    "sentence_id": "...",
    "document_id": "...",
    "document_name": "...",
    "page_number": 3,
    "sentence_index": 12,
    "text": "...",
    "tokens": [...],
    "normalized_tokens": [...],
    "in_references": False
}
```

This record becomes the central unit used by later modules.

### ID rules

- `document_id` — derived from the SHA-256 hash of the PDF content (the first 12 hex characters may be used for display).
- `sentence_index` — zero-based reading-order index **within the document** (not per page).
- `sentence_id` — deterministic: `{document_id}_s{sentence_index:05d}`.
- `page_number` — the page where the sentence **starts**. If a sentence crosses a page break, also store `page_end`.

The same input must always produce the same IDs. Later modules, labels and the full-project passage layer depend on this.

---

# 14. N-Gram Language Model

Implement a small statistical language model over the processed corpus.

Support:

- Unigram
- Bigram
- Trigram

The model should calculate frequencies and conditional probabilities.

For a bigram:

\[
P(w_i|w_{i-1}) =
\frac{Count(w_{i-1},w_i)}
{Count(w_{i-1})}
\]

The system should be able to:

- generate N-Grams,
- count frequencies,
- calculate probabilities,
- inspect common sequences,
- predict a likely next word from a context.

Use sentence boundary tokens (`<s>`, `</s>`) so sentence starts and ends are modelled.

Handle unseen sequences safely with a documented smoothing method (add-k smoothing by default; backoff is an acceptable alternative). Optionally report perplexity on held-out sentences.

The N-Gram module is mainly a corpus-language analysis component; it should not be forced into the evidence-retrieval logic where it does not provide a clear benefit.

---

# 15. HMM POS Tagging

Implement POS analysis using an HMM-based tagger.

Conceptually:

```text
Observed words
      ↓
Hidden POS states
      ↓
Transition probabilities
      +
Emission probabilities
      ↓
Viterbi decoding
      ↓
Most probable POS sequence
```

The implementation should support:

- transition probability estimation,
- emission probability estimation,
- smoothing for unseen cases,
- Viterbi decoding,
- sentence-level POS output.

Example representation:

```python
[
    ("The", "DT"),
    ("model", "NN"),
    ("improves", "VBZ")
]
```

The tagger should be isolated from the rest of the application so that errors or changes in POS logic do not affect PDF ingestion or retrieval.

### 15.1 Training data

An HMM needs a tagged corpus. The research papers are untagged, so:

- train on the NLTK Penn Treebank sample (`treebank`), which uses the Penn tag set shown above,
- hold out about 10% of the tagged sentences and report tagging accuracy,
- handle unknown words explicitly (smoothing, optionally with suffix or capitalization heuristics),
- state clearly in the application that the tagger was trained on news text and may be less accurate on scientific text. The POS output is exploratory analysis. It does not drive retrieval or claim detection.

---

# 16. TF-IDF Representation

The core retrieval baseline should represent each sentence using TF-IDF.

Conceptually:

```text
Research corpus
      ↓
Documents / sentences
      ↓
TF-IDF vectorization
      ↓
Numerical representation
```

Use a consistent fitted vocabulary for the corpus.

Store:

- vocabulary,
- vectorizer,
- sentence IDs associated with rows,
- processed corpus metadata.

Do not refit the vectorizer separately for every query.

**Rebuild rule.** The vectorizer is fitted on the whole corpus, so it must be refitted whenever the set of documents changes (a paper is added or removed). Store a corpus fingerprint (hash of the sorted document IDs plus the preprocessing config) with the artifacts. Reuse the artifacts only when the fingerprint matches.

Queries must go through exactly the same preprocessing and vectorizer as the corpus. Whether to remove stop words is decided by retrieval evaluation (Section 27), not assumed.

---

# 17. Cosine Similarity Retrieval

The user enters a research claim or statement.

Example:

```text
RAG reduces hallucination in domain-specific question answering.
```

Process the query using the same preprocessing and TF-IDF representation.

Calculate cosine similarity between the query vector and sentence vectors.

Conceptually:

\[
\cos(\theta)=
\frac{A \cdot B}
{\|A\|\|B\|}
\]

Return the highest-ranking relevant passages.

Each result should include:

- similarity score,
- sentence ID,
- sentence text,
- document title/name,
- page number,
- document ID.

The result must remain traceable to the original PDF.

---

# 18. Claim Candidate Identification

The mini-project should support two ways of obtaining a claim:

### User-entered claim

The user directly types a claim.

### Claim candidates

The system may identify potentially claim-like sentences from the corpus using lightweight linguistic heuristics.

Candidate signals may include:

- declarative sentence structure,
- scientific reporting language,
- verbs such as *shows*, *demonstrates*, *improves*, *reduces*, *increases*, *outperforms*,
- comparative language,
- measurable/result-oriented statements.

Exclude sentences from the references section, questions, and sentences below the minimum length.

This is **candidate detection**, not a guarantee that a sentence is scientifically validated.

The application must clearly label such outputs as candidates.

### Records

A candidate is stored with a link back to its source sentence:

```python
{
    "candidate_id": "...",
    "sentence_id": "...",
    "document_id": "...",
    "page_number": 3,
    "text": "...",
    "score": 3,
    "signals": ["result_verb", "comparative", "measurable"]
}
```

A user-entered claim uses the same minimal shape:

```python
{
    "claim_id": "...",
    "text": "...",
    "source": "user"
}
```

The full-project continuation extends these records, so keep the field names stable.

---

# 19. Evidence Retrieval Flow

For each claim:

```text
Claim
  ↓
Preprocess claim
  ↓
TF-IDF transform
  ↓
Compare against corpus sentences
  ↓
Cosine similarity
  ↓
Exclude the claim's own source sentence (if it came from a paper)
  ↓
Rank results
  ↓
Apply relevance threshold
  ↓
Display top evidence passages
  ↓
Show source provenance
```

When the claim is a claim candidate taken from a paper, the sentence it came from would otherwise return itself with similarity 1.0. It is excluded by default (`retrieval.exclude_source_sentence`). Whether evidence from the same paper is allowed at all should be a visible option in the application.

The retrieval function should be reusable independently of the UI.

---

# 20. Provenance

Every displayed evidence result must show enough information for a user to locate it.

Minimum display:

```text
Source: paper_name.pdf
Page: 7
Similarity: 0.63
Passage: "..."
```

Where possible, provide the original uploaded document name rather than relying on internal IDs alone.

Do not generate evidence text that cannot be traced back to a source passage.

---

# 21. Word Frequency and Word Cloud

Create a corpus-analysis view that shows common terms.

The process is:

```text
Corpus
  ↓
Clean tokens
  ↓
Frequency count
  ↓
Top terms
  ↓
Word Cloud / table
```

The Word Cloud is an exploratory visualization, not a semantic interpretation of the research literature.

Also display a ranked frequency table so the user can inspect the actual counts.

---

# 22. Application Interface

Use a simple Streamlit application.

Suggested interface:

### Home

- project description,
- upload papers,
- processing status.

### Corpus

- uploaded documents,
- page counts,
- sentence counts,
- token statistics.

### NLP Analysis

- N-Gram results,
- POS tagging example,
- top terms,
- Word Cloud.

### Claim & Evidence

- claim input,
- claim candidates,
- similarity results,
- source/page provenance.

Avoid unnecessary UI complexity.

The application should be understandable within a few minutes of first use.

---

# 23. Processing Pipeline

Create one orchestration function that performs the complete processing sequence.

Example:

```python
build_corpus(uploaded_papers)
```

Conceptually:

```text
Upload
  ↓
Validate
  ↓
Extract
  ↓
Normalize
  ↓
Segment sentences
  ↓
Tokenize
  ↓
Create sentence records
  ↓
Run corpus analysis
  ↓
Build TF-IDF index
  ↓
Store artifacts
  ↓
Application ready
```

The UI should call the orchestration layer instead of manually executing every processing step.

---

# 24. Caching and Reuse

Do not repeatedly process unchanged papers.

Use a content hash or document fingerprint.

For example:

```text
PDF
 ↓
SHA-256 hash
 ↓
Document ID
```

If the same document is uploaded again, the system should detect that it has already been processed where appropriate.

Per-document results (extracted text, sentence records) are cached by document ID. Corpus-level artifacts (TF-IDF vectorizer and matrix, N-Gram counts, frequency statistics) are rebuilt whenever the document set changes and are keyed by the corpus fingerprint from Section 16.

Cache expensive preprocessing and vectorization operations during an application session.

---

# 25. Error Handling

The application must fail gracefully.

Handle:

- invalid PDFs,
- empty PDFs,
- extraction failures,
- no extractable text,
- duplicate documents,
- empty claims,
- claims with no meaningful terms,
- zero-vector TF-IDF queries,
- missing artifacts,
- corrupted serialized objects,
- missing NLTK data (show the download command).

Show understandable user-facing messages.

Keep technical details in logs.

Load serialized artifacts only from the project's own `data/artifacts/` directory.

---

# 26. Testing Strategy

Testing should be built alongside implementation.

### Unit tests

Test independently:

- PDF extraction,
- cleaning (headers/footers, hyphenation, references detection),
- sentence segmentation,
- tokenization,
- N-Gram probabilities,
- HMM transition/emission calculations,
- Viterbi output,
- TF-IDF creation,
- cosine similarity,
- claim candidate detection,
- frequency analysis,
- deterministic ID generation.

### Integration test

Run:

```text
PDF → extraction → preprocessing → indexing → query → results
```

using a small controlled corpus from `tests/fixtures/` (`test_pipeline.py`).

### Application test

Verify that a user can:

1. upload papers,
2. process them,
3. open the analysis pages,
4. enter a claim,
5. retrieve evidence,
6. identify the source page.

---

# 27. Evaluation

The project should be evaluated with a small manually prepared test set after the core pipeline is working.

For selected claims, record expected relevant passages.

**Labelling unit.** Label relevance at the **`sentence_id`** level. This keeps the labels valid when the full project later moves to passage-level retrieval (a passage counts as relevant if it contains a relevant sentence).

Store the test set in `evaluation/`:

```text
evaluation/
    claims.json             # claim_id, claim text
    relevance_labels.json   # claim_id -> list of relevant sentence_ids
```

A reasonable size for the mini-project is 3–5 papers on one topic and 15–25 labelled claims. Keep a small portion of the claims aside and do not tune against them.

Useful retrieval metrics include:

- Precision@K
- Recall@K
- MRR

Also measure system-level properties:

- processing time,
- query latency,
- number of documents processed,
- number of extracted sentences,
- failure rate.

Report the HMM tagger's held-out accuracy (Section 15.1) alongside these.

Do not tune the system against only one paper.

The final evaluation should use multiple user-provided research papers from a coherent topic so the retrieval task is meaningful.

---

# 28. Development Workflow

Build in this order:

### Stage A — Project foundation

- repository,
- environment,
- configuration,
- logging,
- directory structure.

### Stage B — Paper ingestion

- PDF validation,
- extraction,
- metadata,
- raw/processed separation.

### Stage C — Preprocessing

- cleaning,
- sentence segmentation,
- tokenization,
- sentence records.

### Stage D — NLP analysis

- N-Gram model,
- HMM POS tagger,
- frequency analysis,
- Word Cloud.

### Stage E — Retrieval

- TF-IDF,
- cosine similarity,
- claim input,
- evidence ranking,
- provenance.

### Stage F — Claim candidates

- lightweight candidate identification,
- ranking/filtering,
- candidate display.

### Stage G — Application

- Streamlit interface,
- processing controls,
- analysis pages,
- claim/evidence interface.

### Stage H — Validation

- unit tests,
- integration testing,
- real-paper testing,
- retrieval evaluation,
- performance measurement.

Do not build the UI first and then put the NLP inside it. Build reusable backend modules first and connect the interface afterward.

---

# 29. AI-Assisted Development Workflow

When using an AI coding assistant such as Codex or ChatGPT, use it as an implementation assistant rather than allowing it to redesign the project.

For each module:

1. give the assistant the relevant section of this README,
2. ask it to implement only that module,
3. require type hints and error handling,
4. ask for unit tests,
5. run the tests locally,
6. inspect the implementation,
7. integrate only after the tests pass.

A useful prompt format is:

```text
Implement only the [MODULE NAME] described below.

Requirements:
- follow the existing EVIDEX repository structure,
- do not change unrelated modules,
- keep the implementation modular,
- add type hints,
- handle edge cases,
- write unit tests,
- do not introduce unnecessary dependencies.

[PASTE THE RELEVANT README SECTION]
```

Do not ask an AI coding assistant to generate the entire repository in one step.

---

# 30. Definition of Done

The mini-project is complete when all of the following are true:

- the application starts successfully,
- one or more user-provided PDFs can be uploaded,
- papers are extracted and stored with provenance,
- text is cleaned and segmented (including references handling),
- sentence records are generated with deterministic IDs,
- N-Gram analysis works,
- HMM POS tagging works and its held-out accuracy is reported,
- TF-IDF indexing works,
- cosine-similarity retrieval works,
- claim candidates can be identified,
- evidence passages can be retrieved,
- every result retains source/page information,
- frequency analysis and Word Cloud work,
- the Streamlit application exposes the complete workflow,
- unit and integration tests pass,
- real research papers can be processed,
- retrieval behaviour has been evaluated on the labelled set in `evaluation/`,
- errors are handled cleanly,
- and the repository can be reproduced from a clean environment.

---

# 31. Final Expected Outcome

At the end of the mini-project, EVIDEX should be a functioning research-paper NLP application rather than a notebook collection.

A user should be able to provide papers such as:

```text
paper_1.pdf
paper_2.pdf
paper_3.pdf
...
```

and then interact with the system through the application to:

```text
Upload Papers
      ↓
Build Research Corpus
      ↓
Analyse Corpus
      ↓
Enter / Identify Claim
      ↓
Retrieve Relevant Passages
      ↓
Inspect Similarity
      ↓
Trace Evidence to Paper + Page
```

The implementation should remain simple, modular, reproducible, and easy to extend.

---

# 32. Repository Hygiene and Release

- `.gitignore` excludes `.venv/`, `data/raw/`, `data/processed/`, `data/artifacts/` and Python cache files. The `.gitkeep` files keep the empty folders in Git.
- Do not commit papers. Document where to place them instead.
- When the Definition of Done in Section 30 is met, tag the repository:

```bash
git tag v0.1-miniproject
```

- That tag is the version submitted for the academic mini-project. All continuation work (hybrid retrieval, NLI, RAG and so on) happens afterwards on a separate branch, so the submitted scope stays exactly as described in this document.
- The root `README.md` stays short: what the project is, how to install it, how to run it, and a screenshot.
