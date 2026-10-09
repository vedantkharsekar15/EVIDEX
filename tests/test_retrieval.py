import pytest
from src.similarity.tfidf import TfidfIndex
from src.similarity.retrieval import retrieve

@pytest.fixture
def mock_cfg():
    return {
        "tfidf": {
            "ngram_range": [1, 2],
            "min_df": 1,
            "sublinear_tf": True
        },
        "retrieval": {
            "top_k": 5,
            "min_similarity": 0.10,
            "exclude_source_sentence": True
        }
    }

@pytest.fixture
def sample_records():
    return [
        {
            "sentence_id": "doc1_s001",
            "document_id": "doc1",
            "document_name": "doc1.pdf",
            "page_number": 1,
            "text": "The cat sat on the mat.",
            "normalized_tokens": ["the", "cat", "sat", "on", "the", "mat"],
            "in_references": False
        },
        {
            "sentence_id": "doc1_s002",
            "document_id": "doc1",
            "document_name": "doc1.pdf",
            "page_number": 1,
            "text": "Another sentence about cats.",
            "normalized_tokens": ["another", "sentence", "about", "cats"],
            "in_references": False
        },
        {
            "sentence_id": "doc2_s001",
            "document_id": "doc2",
            "document_name": "doc2.pdf",
            "page_number": 1,
            "text": "Dogs are different.",
            "normalized_tokens": ["dogs", "are", "different"],
            "in_references": False
        },
        {
            "sentence_id": "doc2_s002",
            "document_id": "doc2",
            "document_name": "doc2.pdf",
            "page_number": 2,
            "text": "The dog sat on the log.",
            "normalized_tokens": ["the", "dog", "sat", "on", "the", "log"],
            "in_references": False
        },
        {
            "sentence_id": "doc3_s001",
            "document_id": "doc3",
            "document_name": "doc3.pdf",
            "page_number": 1,
            "text": "A reference about cats.",
            "normalized_tokens": ["a", "reference", "about", "cats"],
            "in_references": True
        }
    ]

@pytest.fixture
def records_by_id(sample_records):
    return {r["sentence_id"]: r for r in sample_records}

@pytest.fixture
def tfidf_index(sample_records, mock_cfg):
    return TfidfIndex.build(sample_records, mock_cfg)

def test_correct_ranking(tfidf_index, records_by_id, mock_cfg):
    claim = {"text": "The cat sat"}
    results = retrieve(claim, tfidf_index, records_by_id, mock_cfg)
    
    assert len(results) > 0
    # doc1_s001 should be highly relevant
    assert results[0]["sentence_id"] == "doc1_s001"
    assert results[0]["rank"] == 1
    assert "similarity" in results[0]
    
    # Verify Evidence structure exactly matches Contract v1
    expected_keys = {"rank", "similarity", "sentence_id", "document_id", "document_name", "page_number", "text"}
    assert set(results[0].keys()) == expected_keys

def test_exclude_source_sentence(tfidf_index, records_by_id, mock_cfg):
    claim = {
        "text": "The cat sat on the mat.",
        "sentence_id": "doc1_s001",
        "document_id": "doc1"
    }
    # It would be identical, so without exclusion it would be top rank.
    results = retrieve(claim, tfidf_index, records_by_id, mock_cfg)
    
    for r in results:
        assert r["sentence_id"] != "doc1_s001"

def test_allow_same_document_false(tfidf_index, records_by_id, mock_cfg):
    claim = {
        "text": "The cat sat on the mat.",
        "sentence_id": "doc1_s002",
        "document_id": "doc1"
    }
    # With allow_same_document=False, doc1_s001 should be excluded because it's in doc1
    results = retrieve(claim, tfidf_index, records_by_id, mock_cfg, allow_same_document=False)
    
    for r in results:
        assert r["document_id"] != "doc1"

def test_threshold_and_top_k(tfidf_index, records_by_id, mock_cfg):
    mock_cfg["retrieval"]["top_k"] = 1
    mock_cfg["retrieval"]["min_similarity"] = 0.50
    claim = {"text": "The cat sat on the mat."}
    results = retrieve(claim, tfidf_index, records_by_id, mock_cfg)
    
    assert len(results) == 1
    assert results[0]["similarity"] >= 0.50

def test_descending_similarity(tfidf_index, records_by_id, mock_cfg):
    claim = {"text": "cat sat sentence"}
    results = retrieve(claim, tfidf_index, records_by_id, mock_cfg)
    
    if len(results) > 1:
        for i in range(len(results) - 1):
            assert results[i]["similarity"] >= results[i+1]["similarity"]

def test_empty_or_unknown_query(tfidf_index, records_by_id, mock_cfg):
    claim_empty = {"text": ""}
    results_empty = retrieve(claim_empty, tfidf_index, records_by_id, mock_cfg)
    assert len(results_empty) == 0
    
    claim_unknown = {"text": "zxqwy jklp"}
    results_unknown = retrieve(claim_unknown, tfidf_index, records_by_id, mock_cfg)
    assert len(results_unknown) == 0

def test_references_do_not_appear(tfidf_index, records_by_id, mock_cfg):
    claim = {"text": "A reference about cats."}
    results = retrieve(claim, tfidf_index, records_by_id, mock_cfg)
    
    for r in results:
        assert r["sentence_id"] != "doc3_s001"

def test_custom_preprocessor(tfidf_index, records_by_id, mock_cfg):
    claim = {"text": "The cat sat."}
    
    def my_preprocessor(text):
        return ["dogs", "are", "different"]
        
    results = retrieve(claim, tfidf_index, records_by_id, mock_cfg, query_preprocessor=my_preprocessor)
    # The preprocessor changes the query to match doc2_s001
    assert len(results) > 0
    assert results[0]["sentence_id"] == "doc2_s001"

