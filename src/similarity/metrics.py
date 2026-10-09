from typing import List, Set

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Computes Precision@K.
    If K <= 0, returns 0.0.
    If relevant set is empty, returns 0.0.
    """
    if k <= 0 or not relevant:
        return 0.0
        
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
        
    relevant_retrieved = sum(1 for item in top_k if item in relevant)
    return relevant_retrieved / len(top_k)

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Computes Recall@K.
    If K <= 0, returns 0.0.
    If relevant set is empty, returns 0.0.
    """
    if k <= 0 or not relevant:
        return 0.0
        
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
        
    relevant_retrieved = sum(1 for item in top_k if item in relevant)
    return relevant_retrieved / len(relevant)

def mrr(all_retrieved: List[List[str]], all_relevant: List[Set[str]]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR).
    A query with no relevant result contributes zero to its reciprocal rank.
    If input lists are empty or lengths don't match, returns 0.0.
    """
    if not all_retrieved or not all_relevant or len(all_retrieved) != len(all_relevant):
        return 0.0
        
    rr_sum = 0.0
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        if not relevant:
            continue
            
        for rank, item in enumerate(retrieved, start=1):
            if item in relevant:
                rr_sum += 1.0 / rank
                break
                
    return rr_sum / len(all_retrieved)
