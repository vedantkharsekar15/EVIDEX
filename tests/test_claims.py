import pytest
from src.claims.claim_candidates import find_claim_candidates, make_user_claim

@pytest.fixture
def mock_cfg():
    return {
        "processing": {
            "min_sentence_length": 5
        },
        "claims": {
            "min_candidate_score": 2.0,
            "max_candidates": 3
        }
    }

@pytest.fixture
def sample_records():
    return [
        {
            "sentence_id": "doc1_s001",
            "document_id": "doc1",
            "page_number": 1,
            "text": "The proposed algorithm achieves 95% accuracy.",
            "tokens": ["The", "proposed", "algorithm", "achieves", "95", "%", "accuracy", "."],
            "in_references": False,
            "is_caption": False
        },
        {
            "sentence_id": "doc1_s002",
            "document_id": "doc1",
            "page_number": 1,
            "text": "Smith (2020) achieved better results.",
            "tokens": ["Smith", "2020", "achieved", "better", "results", "."],
            "in_references": True,
            "is_caption": False
        },
        {
            "sentence_id": "doc1_s003",
            "document_id": "doc1",
            "page_number": 1,
            "text": "Figure 1 shows the result.",
            "tokens": ["Figure", "1", "shows", "the", "result", "."],
            "in_references": False,
            "is_caption": True
        },
        {
            "sentence_id": "doc1_s004",
            "document_id": "doc1",
            "page_number": 1,
            "text": "Does it perform better?",
            "tokens": ["Does", "it", "perform", "better", "?"],
            "in_references": False,
            "is_caption": False
        },
        {
            "sentence_id": "doc1_s005",
            "document_id": "doc1",
            "page_number": 1,
            "text": "It is fast.",
            "tokens": ["It", "is", "fast", "."],
            "in_references": False,
            "is_caption": False
        },
        {
            "sentence_id": "doc2_s001",
            "document_id": "doc2",
            "page_number": 1,
            "text": "Method A is significantly faster than Method B.",
            "tokens": ["Method", "A", "is", "significantly", "faster", "than", "Method", "B", "."],
            "in_references": False,
            "is_caption": False
        },
        {
            "sentence_id": "doc2_s002",
            "document_id": "doc2",
            "page_number": 1,
            "text": "Our approach outperformed the baseline by 10 points.",
            "tokens": ["Our", "approach", "outperformed", "the", "baseline", "by", "10", "points", "."],
            "in_references": False,
            "is_caption": False
        },
        {
            "sentence_id": "doc2_s003",
            "document_id": "doc2",
            "page_number": 1,
            "text": "This model showed higher precision of 0.88.",
            "tokens": ["This", "model", "showed", "higher", "precision", "of", "0.88", "."],
            "in_references": False,
            "is_caption": False
        },
        {
            "sentence_id": "doc2_s004",
            "document_id": "doc2",
            "page_number": 1,
            "text": "Another declarative sentence with a result.",
            "tokens": ["Another", "declarative", "sentence", "with", "a", "result", "."],
            "in_references": False,
            "is_caption": False
        }
    ]

def test_find_candidates_scoring_and_ordering(sample_records, mock_cfg):
    candidates = find_claim_candidates(sample_records, mock_cfg)
    
    # Max candidates is 3
    assert len(candidates) == 3
    
    # Highest score first: doc2_s003 (score 4.0: result_verb, comparative, measurable, declarative)
    assert candidates[0]["sentence_id"] == "doc2_s003"
    assert candidates[0]["score"] == 4.0
    
    # Tie for 2nd: doc1_s001 (3.0), doc2_s002 (3.0) -> alphabetical by sentence_id
    assert candidates[1]["sentence_id"] == "doc1_s001"
    assert candidates[2]["sentence_id"] == "doc2_s002"

def test_exclusions(sample_records, mock_cfg):
    candidates = find_claim_candidates(sample_records, mock_cfg)
    extracted_ids = [c["sentence_id"] for c in candidates]
    
    assert "doc1_s002" not in extracted_ids # Reference
    assert "doc1_s003" not in extracted_ids # Caption
    assert "doc1_s004" not in extracted_ids # Question
    assert "doc1_s005" not in extracted_ids # Short

def test_candidate_output_fields(sample_records, mock_cfg):
    candidates = find_claim_candidates(sample_records, mock_cfg)
    c = candidates[0]
    expected_keys = {"candidate_id", "sentence_id", "document_id", "page_number", "text", "score", "signals"}
    assert set(c.keys()) == expected_keys
    assert c["candidate_id"] == f"cand_{c['sentence_id']}"

def test_make_user_claim_valid():
    claim = make_user_claim(" Algorithm X is fast. ")
    assert claim["text"] == "Algorithm X is fast."
    assert claim["source"] == "user"
    assert claim["claim_id"].startswith("claim_")
    assert len(claim["claim_id"]) == 6 + 10 # "claim_" + 10 chars

def test_make_user_claim_empty():
    with pytest.raises(ValueError):
        make_user_claim("   ")
        
    with pytest.raises(ValueError):
        make_user_claim("")
