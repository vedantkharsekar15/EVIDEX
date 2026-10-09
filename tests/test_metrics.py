import pytest
from src.similarity.metrics import precision_at_k, recall_at_k, mrr

def test_precision_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = {"A", "C", "E"}
    
    assert precision_at_k(retrieved, relevant, 1) == 1.0  # A is relevant
    assert precision_at_k(retrieved, relevant, 2) == 0.5  # A relevant, B not
    assert precision_at_k(retrieved, relevant, 3) == 2/3  # A, C relevant
    
    # Invalid K
    assert precision_at_k(retrieved, relevant, 0) == 0.0
    assert precision_at_k(retrieved, relevant, -1) == 0.0
    
    # Empty inputs
    assert precision_at_k([], relevant, 3) == 0.0
    assert precision_at_k(retrieved, set(), 3) == 0.0

def test_recall_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = {"A", "C", "E"}  # Missed "E"
    
    assert recall_at_k(retrieved, relevant, 1) == 1/3
    assert recall_at_k(retrieved, relevant, 2) == 1/3
    assert recall_at_k(retrieved, relevant, 3) == 2/3
    assert recall_at_k(retrieved, relevant, 4) == 2/3
    assert recall_at_k(retrieved, relevant, 10) == 2/3
    
    # Invalid K
    assert recall_at_k(retrieved, relevant, 0) == 0.0
    
    # Empty inputs
    assert recall_at_k([], relevant, 3) == 0.0
    assert recall_at_k(retrieved, set(), 3) == 0.0

def test_mrr():
    all_retrieved = [
        ["A", "B", "C"], # rank 1 (A)
        ["X", "Y", "Z"], # rank 2 (Y)
        ["M", "N"],      # missed
        ["P"]            # no relevant labels
    ]
    all_relevant = [
        {"A"},
        {"Y", "Z"},
        {"Q"},
        set()
    ]
    
    score = mrr(all_retrieved, all_relevant)
    # (1.0 + 0.5 + 0.0 + 0.0) / 4 = 1.5 / 4 = 0.375
    assert score == 0.375
    
    assert mrr([], []) == 0.0
    assert mrr([["A"]], [{"A"}, {"B"}]) == 0.0 # Length mismatch returns 0.0 safely
