# EVIDEX Module 1: Foundation, Ingestion, Pipeline, GitHub

**Owner:** Vedant Kharsekar (coding + GitHub integration)
**Branch:** `feat/foundation-ingestion`
**Depends on:** nobody for the scaffold and ingestion. Pipeline wiring comes last, after Modules 2-4 merge.

## Shared Contract v1 (frozen; implement in `src/utils/schemas.py`)
```
PageRecord     {document_id, filename, page_number, text}
SentenceRecord {sentence_id, document_id, document_name, page_number, page_end?,
                sentence_index, text, tokens, normalized_tokens, in_references, is_caption}
Claim          {claim_id, text, source:"user"}
ClaimCandidate {candidate_id, sentence_id, document_id, page_number, text, score, signals}
Evidence       {rank, similarity, sentence_id, document_id, document_name, page_number, text}
document_id = sha256(file bytes).hexdigest()[:12]
sentence_id = f"{document_id}_s{sentence_index:05d}"   # index is zero-based per document
```
Field names are stable. Only additive changes are allowed, via PR + team notice.

## Files
```
src/utils/{__init__,config,logger,file_utils,schemas}.py
src/ingestion/{__init__,pdf_loader,metadata}.py
src/pipeline/{__init__,build_corpus}.py
app/streamlit_app.py            # Home + Corpus page
config.yaml  requirements.txt  .gitignore  README.md  docs/mini_project_plan.md
data/{raw/papers,processed,artifacts}/.gitkeep
tests/fixtures/make_pdfs.py     # generates tiny test PDFs programmatically
tests/{test_ingestion,test_pipeline}.py
.github/{workflows/ci.yml,pull_request_template.md,CODEOWNERS}
```

## Steps
1. **Scaffold PR (merge within day 1):** folder tree, `config.yaml` (Section 10 values), pinned `requirements.txt`, `.gitignore`, `.gitkeep` files, `schemas.py` (TypedDicts), `logger.py`, `config.py`, CI workflow. Others branch from this.
2. `file_utils.py`: hashing, JSON IO, artifact save/load restricted to `data/artifacts/`, corpus fingerprint.
3. `pdf_loader.py`: validate (`%PDF` header, non-empty, not corrupt), page-by-page PyMuPDF extraction in block reading order for two-column layouts, per-file error isolation.
4. `metadata.py`: document registry in `data/processed/documents.json`, duplicate detection by hash.
5. `make_pdfs.py` + `test_ingestion.py`.
6. `app/streamlit_app.py` shell: upload, "Process" button, status/failures, Corpus page (docs, pages, sentences, token stats). Store the result in `st.session_state["corpus"]`. Modules 3 and 4 add pages in `app/pages/`.
7. **After Modules 2-4 merge:** wire `build_corpus`, run the integration test, run real-paper tests, build `evaluation/` baseline with Module 3.
8. Final: Definition of Done check (plan Section 30), tag release.

## Interfaces
```python
# config.py / logger.py
load_config(path="config.yaml") -> dict
get_logger(name: str) -> logging.Logger
# file_utils.py
sha256_bytes(b: bytes) -> str;  document_id_from_bytes(b: bytes) -> str
save_json(obj, path); load_json(path)
save_artifact(obj, name: str); load_artifact(name: str) -> object | None   # artifacts dir only
corpus_fingerprint(doc_ids: list[str], cfg: dict) -> str   # hash(sorted ids + preprocessing/tfidf cfg)
# pdf_loader.py
class PDFValidationError(Exception)
validate_pdf(path: Path) -> bytes
extract_pages(path: Path) -> list[PageRecord]
load_pdfs(paths: list[Path]) -> tuple[list[PageRecord], list[dict]]   # (pages, failures[{filename,error}])
# metadata.py
build_document_meta(path, page_count) -> dict   # document_id, filename, sha256, page_count, ingested_at
is_duplicate(document_id: str) -> bool
# build_corpus.py
build_corpus(paths: list[Path], cfg: dict | None = None, force: bool = False) -> dict
# returns {records, documents, failed, fingerprint, tfidf_index, ngram_model, frequencies, timings}
```
`build_corpus` calls: M2 `build_sentence_records(pages, cfg)`; M3 `TfidfIndex.build(records, cfg)`; M4 `NGramModel.fit(...)` and `term_frequencies(...)`. It reuses artifacts when the fingerprint matches.

## Tests (`pytest tests/test_ingestion.py tests/test_pipeline.py`)
- Rejects a `.pdf`-named text file, an empty file and a corrupt PDF; the batch continues.
- Same bytes give the same `document_id`; a duplicate upload is detected.
- Pages come out numbered 1..N, empty pages are handled, and a two-column fixture is in reading order.
- Fingerprint changes when a doc is added or the config changes.
- Pipeline integration: fixture PDF → records → index → query returns a result with filename and page.
- Failure path: a bad PDF in the batch is reported, not fatal.

## Handoff
- **To everyone (day 1):** scaffold PR, `schemas.py`, `config.yaml`, CI.
- **To M2:** `PageRecord` list from `load_pdfs`.
- **To M3/M4:** `session_state["corpus"]` structure for their app pages.
- **From M2/M3/M4:** import paths and signatures exactly as listed in their READMEs.
- **Final deliverable:** tagged `v0.1-miniproject`, clean-environment run verified.

## Git Workflow (owner + integrator)
- **Repo setup:** create the GitHub repo, add the 3 teammates as collaborators, and set up labels (`module-1..4`, `bug`, `integration`) and milestones (Stage A-H).
- **Branch protection on `main`:** PR required, 1 approval, CI must pass, no direct pushes, squash merge.
- **CI (`ci.yml`):** Python 3.11, `pip install -r requirements.txt`, `python -m nltk.downloader punkt punkt_tab treebank`, `pytest`.
- **Also add:** PR template (module, tests run, contract changes?) and `CODEOWNERS` (each `src/<pkg>` → its owner).
- **Merge order:** scaffold → M2, M3, M4 (any order) → pipeline wiring PR → app pages → evaluation.
- **Own work:** commits `feat(ingestion): ...`; rebase on `main` before opening a PR.
- **Never commit:** papers, `data/*`, `.venv/`.
- **Release:** `git tag v0.1-miniproject && git push --tags`, then branch `continuation/full-project` for later work.
