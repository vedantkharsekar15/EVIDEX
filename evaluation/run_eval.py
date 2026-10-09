import json
import os
import sys
import argparse
import hashlib

# Ensure the parent directory is in the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.similarity.tfidf import TfidfIndex
from src.similarity.retrieval import retrieve
from src.similarity.metrics import precision_at_k, recall_at_k, mrr
from src.utils.config import load_config
from src.utils.file_utils import load_artifact

def main():
    parser = argparse.ArgumentParser(description="Run EVIDEX evaluation.")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic demo evaluation")
    args = parser.parse_args()

    cfg = load_config()
    k = cfg.get("retrieval", {}).get("top_k", 5)
    
    if args.synthetic:
        print("--- Running Synthetic Demo Evaluation ---")
        records_path = "tests/fixtures/sentences_retrieval.json"
        claims_path = "tests/fixtures/eval/synthetic_claims.json"
        labels_path = "tests/fixtures/eval/synthetic_labels.json"
        
        if not os.path.exists(records_path):
            print(f"Error: Fixture {records_path} missing.")
            return
            
        with open(records_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        print("--- Running Real-Paper Evaluation ---")
        claims_path = "evaluation/claims.json"
        labels_path = "evaluation/relevance_labels.json"
        records = load_artifact("records")
        if not records:
            print("Error: Real corpus records not found in data/artifacts/records.json.")
            print("Please process your PDF papers through the pipeline (e.g., via Streamlit) before evaluating.")
            return
            
    records_by_id = {r["sentence_id"]: r for r in records}
    
    # Build index dynamically for evaluation (since we need to retrieve)
    # The real index could be loaded, but building is fast and ensures sync.
    index = TfidfIndex.build(records, cfg)
    
    if not os.path.exists(claims_path):
        print(f"Error: {claims_path} missing.")
        return
        
    with open(claims_path, "r", encoding="utf-8") as f:
        claims = json.load(f)
        
    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} missing.")
        return
        
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)
        
    # Split claims into tuning and test based on 'split' field or deterministic hash
    test_claims = []
    tuning_claims = []
    
    for claim in claims:
        split = claim.get("split")
        if not split:
            # Deterministic 20% test split based on claim_id
            h = int(hashlib.md5(claim["claim_id"].encode()).hexdigest(), 16)
            split = "test" if (h % 5 == 0) else "tuning"
            claim["split"] = split
            
        if split == "test":
            test_claims.append(claim)
        else:
            tuning_claims.append(claim)
            
    print(f"Found {len(test_claims)} test claims and {len(tuning_claims)} tuning claims.")
    
    # We will evaluate on the TEST claims. 
    # (Tuning claims are left out of final metric aggregation)
    all_retrieved = []
    all_relevant = []
    unlabeled_claims = []
    
    print(f"\n--- Test Set Evaluation Results (Top K={k}) ---")
    
    for claim in test_claims:
        claim_id = claim["claim_id"]
        relevant_list = labels_dict.get(claim_id, [])
        relevant_set = set(relevant_list)
        
        # Exclude source sentence logic: eval claims aren't candidates but just for completeness
        results = retrieve(claim, index, records_by_id, cfg, allow_same_document=True)
        retrieved_ids = [r["sentence_id"] for r in results]
        
        if not relevant_set:
            unlabeled_claims.append(claim_id)
            print(f"Claim: {claim_id} | WARNING: No relevant labels available. Skipping from metrics.")
            print()
            continue
            
        all_retrieved.append(retrieved_ids)
        all_relevant.append(relevant_set)
        
        p_at_k = precision_at_k(retrieved_ids, relevant_set, k)
        r_at_k = recall_at_k(retrieved_ids, relevant_set, k)
        
        print(f"Claim: {claim_id}")
        print(f"  Query: '{claim['text']}'")
        print(f"  Precision@{k}: {p_at_k:.2f}")
        print(f"  Recall@{k}:    {r_at_k:.2f}")
        print(f"  Retrieved:   {retrieved_ids}")
        print(f"  Relevant:    {list(relevant_set)}")
        print()
        
    if unlabeled_claims:
        print(f"Skipped {len(unlabeled_claims)} claims due to missing labels: {unlabeled_claims}")
        
    if all_retrieved:
        overall_mrr = mrr(all_retrieved, all_relevant)
        avg_precision = sum(precision_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)) / len(all_retrieved)
        avg_recall = sum(recall_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)) / len(all_retrieved)
        
        print(f"\n--- Aggregate Test Metrics ---")
        print(f"Total Evaluated Claims: {len(all_retrieved)}")
        print(f"Mean Precision@{k}: {avg_precision:.4f}")
        print(f"Mean Recall@{k}:    {avg_recall:.4f}")
        print(f"Mean Reciprocal Rank (MRR): {overall_mrr:.4f}")
    else:
        print("\nNo labeled test claims were evaluated.")

if __name__ == "__main__":
    main()
