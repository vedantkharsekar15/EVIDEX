import json
import os
import sys

# Ensure the parent directory is in the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.similarity.tfidf import TfidfIndex
from src.similarity.retrieval import retrieve
from src.similarity.metrics import precision_at_k, recall_at_k, mrr
from src.utils.config import load_config

def main():
    cfg = load_config()
    k = cfg.get("retrieval", {}).get("top_k", 5)
    
    # Load fixture records
    records_path = "tests/fixtures/sentences_retrieval.json"
    if not os.path.exists(records_path):
        print(f"Error: Fixture {records_path} missing.")
        return
        
    with open(records_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    records_by_id = {r["sentence_id"]: r for r in records}
    
    # Build index
    index = TfidfIndex.build(records, cfg)
    
    # Load claims
    claims_path = "evaluation/claims.json"
    if not os.path.exists(claims_path):
        print(f"Error: {claims_path} missing.")
        return
        
    with open(claims_path, "r", encoding="utf-8") as f:
        claims = json.load(f)
        
    # Load labels
    labels_path = "evaluation/relevance_labels.json"
    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} missing.")
        return
        
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)
        
    all_retrieved = []
    all_relevant = []
    
    print(f"--- Evaluation Results (Top K={k}) ---")
    
    for claim in claims:
        claim_id = claim["claim_id"]
        relevant_set = set(labels_dict.get(claim_id, []))
        
        # Exclude source sentence logic: eval claims aren't candidates but just for completeness
        results = retrieve(claim, index, records_by_id, cfg, allow_same_document=True)
        retrieved_ids = [r["sentence_id"] for r in results]
        
        all_retrieved.append(retrieved_ids)
        all_relevant.append(relevant_set)
        
        if not relevant_set:
            print(f"Claim: {claim_id} | No relevant labels available.")
            print()
            continue
            
        p_at_k = precision_at_k(retrieved_ids, relevant_set, k)
        r_at_k = recall_at_k(retrieved_ids, relevant_set, k)
        
        print(f"Claim: {claim_id}")
        print(f"  Query: '{claim['text']}'")
        print(f"  Precision@{k}: {p_at_k:.2f}")
        print(f"  Recall@{k}:    {r_at_k:.2f}")
        print(f"  Retrieved:   {retrieved_ids}")
        print(f"  Relevant:    {list(relevant_set)}")
        print()
        
    overall_mrr = mrr(all_retrieved, all_relevant)
    print(f"Aggregate MRR: {overall_mrr:.4f}")

if __name__ == "__main__":
    main()
