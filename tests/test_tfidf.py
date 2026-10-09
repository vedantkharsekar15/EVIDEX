import pytest
import os
import json
from src.similarity.tfidf import TfidfIndex

def compute_cosine(vec1, vec2):
    score = 0.0
    for idx, w1 in vec1.items():
        if idx in vec2:
            score += w1 * vec2[idx]
    return score

@pytest.fixture
def sample_records():
    return [
        {
            "sentence_id": "doc1_s001",
            "text": "The cat sat on the mat.",
            "normalized_tokens": ["the", "cat", "sat", "on", "the", "mat"],
            "in_references": False
        },
        {
            "sentence_id": "doc1_s002",
            "text": "Smith (2020) Reference.",
            "normalized_tokens": ["smith", "2020", "reference"],
            "in_references": True
        },
        {
            "sentence_id": "doc2_s001",
            "text": "A completely different sentence.",
            "normalized_tokens": ["a", "completely", "different", "sentence"],
            "in_references": False
        },
        {
            "sentence_id": "doc2_s002",
            "text": "",
            "normalized_tokens": [],
            "in_references": False
        }
    ]

@pytest.fixture
def mock_cfg():
    return {
        "paths": {
            "artifacts": "data/artifacts"
        },
        "tfidf": {
            "ngram_range": [1, 2],
            "min_df": 1,
            "sublinear_tf": True
        }
    }

def test_references_excluded(sample_records, mock_cfg):
    index = TfidfIndex.build(sample_records, mock_cfg)
    assert "doc1_s002" not in index.sentence_ids
    assert len(index.sentence_ids) == 3

def test_query_no_vocab_change(sample_records, mock_cfg):
    index = TfidfIndex.build(sample_records, mock_cfg)
    vocab_size = len(index.vocab)
    index.transform_query(["new", "unseen", "tokens"])
    assert len(index.vocab) == vocab_size

def test_identical_sentence_similarity(sample_records, mock_cfg):
    index = TfidfIndex.build(sample_records, mock_cfg)
    query_tokens = ["the", "cat", "sat", "on", "the", "mat"]
    query_vec = index.transform_query(query_tokens)
    doc_vec = index.tfidf_matrix[index.sentence_ids.index("doc1_s001")]
    
    score = compute_cosine(query_vec, doc_vec)
    assert abs(score - 1.0) < 1e-6

def test_disjoint_sentence_similarity(sample_records, mock_cfg):
    index = TfidfIndex.build(sample_records, mock_cfg)
    query_tokens = ["the", "cat", "sat", "on", "the", "mat"]
    query_vec = index.transform_query(query_tokens)
    doc_vec = index.tfidf_matrix[index.sentence_ids.index("doc2_s001")]
    
    score = compute_cosine(query_vec, doc_vec)
    assert score == 0.0

def test_save_and_load(sample_records, mock_cfg, tmp_path, monkeypatch):
    mock_cfg["paths"]["artifacts"] = str(tmp_path)
    index = TfidfIndex.build(sample_records, mock_cfg)
    
    fingerprint = "test_fp_123"
    index.save(fingerprint)
    
    # Mock load_config
    import src.similarity.tfidf
    monkeypatch.setattr(src.similarity.tfidf, "load_config", lambda: mock_cfg)
    
    loaded_index = TfidfIndex.load(fingerprint)
    
    assert loaded_index is not None
    assert loaded_index.sentence_ids == index.sentence_ids
    assert loaded_index.vocab == index.vocab
    
def test_mismatched_fingerprint_returns_none(sample_records, mock_cfg, tmp_path, monkeypatch):
    mock_cfg["paths"]["artifacts"] = str(tmp_path)
    index = TfidfIndex.build(sample_records, mock_cfg)
    index.save("test_fp_123")
    
    import src.similarity.tfidf
    monkeypatch.setattr(src.similarity.tfidf, "load_config", lambda: mock_cfg)
    
    bad_index = TfidfIndex.load("wrong_fp")
    assert bad_index is None

def test_missing_or_corrupt_artifact_fails_safely(mock_cfg, tmp_path, monkeypatch):
    mock_cfg["paths"]["artifacts"] = str(tmp_path)
    import src.similarity.tfidf
    monkeypatch.setattr(src.similarity.tfidf, "load_config", lambda: mock_cfg)
    
    # Missing
    missing_index = TfidfIndex.load("missing_fp")
    assert missing_index is None
    
    # Corrupt
    fingerprint = "corrupt_fp"
    os.makedirs(str(tmp_path), exist_ok=True)
    path = os.path.join(str(tmp_path), f"tfidf_{fingerprint}.json")
    with open(path, "w") as f:
        f.write("invalid json {")
        
    corrupt_index = TfidfIndex.load(fingerprint)
    assert corrupt_index is None

def test_empty_records_handled_safely(mock_cfg):
    index = TfidfIndex.build([], mock_cfg)
    assert len(index.sentence_ids) == 0
    assert len(index.vocab) == 0
    assert len(index.tfidf_matrix) == 0
    
def test_sentence_id_mapping(sample_records, mock_cfg):
    index = TfidfIndex.build(sample_records, mock_cfg)
    assert index.sentence_ids[0] == "doc1_s001"
    assert index.sentence_ids[1] == "doc2_s001"
    assert index.sentence_ids[2] == "doc2_s002"
    assert len(index.tfidf_matrix) == len(index.sentence_ids)
