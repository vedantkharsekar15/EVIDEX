import time
from pathlib import Path
from typing import List, Dict, Any, Union

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.utils.file_utils import save_artifact, corpus_fingerprint, save_json
from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.metadata import is_duplicate, _get_registry_path, _load_registry
from src.preprocessing.sentence_splitter import build_sentence_records
from src.analysis.frequency import frequency_table, term_frequencies
from src.language_model.ngram_model import NGramModel
from src.pos.hmm_tagger import HMMTagger, MissingNLTKDataError
from src.similarity.tfidf import TfidfIndex

logger = get_logger(__name__)

def build_corpus(paths: List[Union[str, Path]], cfg: dict = None, force: bool = False) -> dict:
    if cfg is None:
        cfg = load_config()

    path_objs = [Path(p) for p in paths]

    # 1. Load PDFs
    pages, failures = load_pdfs(path_objs)

    if not pages:
        logger.warning("No valid pages extracted.")
        return {
            "records": [],
            "tfidf_index": None,
            "fingerprint": None,
            "failures": failures,
            "metadata": {}
        }

    # Filter out duplicates
    unique_docs = {}
    valid_pages = []

    for p in pages:
        doc_id = p["document_id"]
        if not force and is_duplicate(doc_id):
            logger.info(f"Skipping duplicate document: {doc_id}")
            continue
        if doc_id not in unique_docs:
            unique_docs[doc_id] = {"filename": p["filename"], "page_count": 0}
        unique_docs[doc_id]["page_count"] += 1
        valid_pages.append(p)

    if not valid_pages:
        logger.warning("No valid pages after duplicate filtering.")
        return {
            "records": [],
            "tfidf_index": None,
            "fingerprint": None,
            "failures": failures,
            "metadata": {}
        }

    # 2. Build Sentence Records
    try:
        records = build_sentence_records(valid_pages, cfg)
    except Exception as e:
        logger.error(f"Error building sentence records: {e}")
        return {
            "records": [],
            "tfidf_index": None,
            "fingerprint": None,
            "failures": failures + [{"error": str(e), "stage": "preprocessing"}],
            "metadata": {}
        }

    if not records:
        logger.warning("No sentences extracted.")
        return {
            "records": [],
            "tfidf_index": None,
            "fingerprint": None,
            "failures": failures,
            "metadata": {}
        }

    # 3. NLP Analysis (Frequency, N-Gram, HMM)
    try:
        freq_list = term_frequencies(records)
        freq_table_df = frequency_table(records)

        ngram = NGramModel()
        token_lists = [r.get("normalized_tokens", []) for r in records if "normalized_tokens" in r]
        if token_lists:
            ngram.fit(token_lists)

        tagger = HMMTagger()

    except MissingNLTKDataError as e:
        logger.warning(f"Missing NLTK data: {e}")
    except Exception as e:
        logger.warning(f"NLP analysis failed: {e}", exc_info=True)

    # 4. TF-IDF Index
    try:
        tfidf_index = TfidfIndex.build(records, cfg)
    except Exception as e:
        logger.error(f"TF-IDF index build failed: {e}")
        return {
            "records": records,
            "tfidf_index": None,
            "fingerprint": None,
            "failures": failures + [{"error": str(e), "stage": "tfidf"}],
            "metadata": {}
        }

    # 5. Fingerprint & Save Artifacts
    doc_ids = list(unique_docs.keys())
    fingerprint = corpus_fingerprint(doc_ids, cfg)

    save_artifact(records, "records")
    tfidf_index.save(fingerprint)

    # 6. Update Registry / Metadata
    registry_path = _get_registry_path()
    registry = _load_registry()
    for doc_id, info in unique_docs.items():
        registry[doc_id] = {
            "document_id": doc_id,
            "filename": info["filename"],
            "page_count": info["page_count"],
            "ingested_at": time.time()
        }

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(registry, registry_path)

    return {
        "records": records,
        "tfidf_index": tfidf_index,
        "fingerprint": fingerprint,
        "failures": failures,
        "metadata": registry
    }
