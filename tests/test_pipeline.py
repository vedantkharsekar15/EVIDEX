import pytest
from pathlib import Path
from src.pipeline.build_corpus import build_corpus
from src.similarity.retrieval import retrieve
from src.utils.config import load_config
from src.claims.claim_candidates import find_claim_candidates

@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"

def test_full_pipeline(fixtures_dir, tmp_path, monkeypatch):
    # Setup temporary config for paths
    cfg = load_config()
    cfg["paths"] = {
        "artifacts": str(tmp_path / "artifacts"),
        "processed": str(tmp_path / "processed")
    }

    pdf_path = fixtures_dir / "valid.pdf"
    
    # 1. Build Corpus
    import logging
    from src.pipeline.build_corpus import logger as pipeline_logger
    
    warnings = []
    def mock_warning(msg, *args, **kwargs):
        warnings.append(msg)
    monkeypatch.setattr(pipeline_logger, "warning", mock_warning)

    result = build_corpus([pdf_path], cfg=cfg, force=True)
    
    records = result["records"]
    tfidf_index = result["tfidf_index"]
    fingerprint = result["fingerprint"]
    
    # Check that NLP analysis did not fail
    for w in warnings:
        assert not w.startswith("NLP analysis failed"), f"NLP analysis failed unexpectedly: {w}"
    
    # 2. Check basic outputs
    assert len(records) > 0, "Pipeline should produce sentence records."
    assert tfidf_index is not None, "Pipeline should produce TF-IDF index."
    assert fingerprint is not None, "Pipeline should produce fingerprint."
    
    # 3. Check Provenance (Schema constraints)
    for record in records:
        assert "sentence_id" in record
        assert "document_id" in record
        assert "document_name" in record
        assert record["document_name"] == "valid.pdf"
        assert "page_number" in record
        assert "text" in record

    # 4. Find claim candidates
    candidates = find_claim_candidates(records, cfg)
    
    # 5. Retrieval
    records_by_id = {r["sentence_id"]: r for r in records}
    claim = {"text": "This is a test claim for the pipeline."}
    if candidates:
        claim = candidates[0]
        
    results = retrieve(
        claim=claim,
        index=tfidf_index,
        records_by_id=records_by_id,
        cfg=cfg,
        allow_same_document=True
    )
    
    # Even if results is empty due to min similarity, the pipeline ran end-to-end.
    # The return type should be a list.
    assert isinstance(results, list)
    
    # Verify duplicates handling
    result_dup = build_corpus([pdf_path], cfg=cfg, force=False)
    # Should skip due to being in registry, returning 0 records
    assert len(result_dup["records"]) == 0
