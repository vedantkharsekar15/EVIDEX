import re
from typing import Callable, List, Dict, Optional
from src.similarity.tfidf import TfidfIndex

def retrieve(
    claim: dict,
    index: TfidfIndex,
    records_by_id: Dict[str, dict],
    cfg: dict,
    query_preprocessor: Optional[Callable[[str], List[str]]] = None,
    allow_same_document: bool = True,
) -> List[dict]:
    
    text = claim.get("text", "")
    
    if query_preprocessor:
        tokens = query_preprocessor(text)
    else:
        # Fallback lowercase regex tokenization
        tokens = [t for t in re.sub(r'[^\w\s]', '', text.lower()).split() if t]
        
    query_vec = index.transform_query(tokens)
    if not query_vec:
        return []
        
    top_k = cfg.get("retrieval", {}).get("top_k", 5)
    min_sim = cfg.get("retrieval", {}).get("min_similarity", 0.10)
    exclude_source = cfg.get("retrieval", {}).get("exclude_source_sentence", True)
    
    claim_sentence_id = claim.get("sentence_id")
    claim_doc_id = claim.get("document_id")
    
    similarities = []
    
    for i, doc_vec in enumerate(index.tfidf_matrix):
        # Compute cosine similarity (dot product of L2-normalized vectors)
        score = sum(query_vec.get(idx, 0.0) * w for idx, w in doc_vec.items() if idx in query_vec)
        
        if score < min_sim:
            continue
            
        sentence_id = index.sentence_ids[i]
        
        # Exclude the claim's own source sentence if configured
        if exclude_source and claim_sentence_id and sentence_id == claim_sentence_id:
            continue
            
        record = records_by_id.get(sentence_id)
        if not record:
            continue
            
        # Exclude results from the same document if configured
        if not allow_same_document and claim_doc_id and record.get("document_id") == claim_doc_id:
            continue
            
        similarities.append((score, sentence_id, record))
        
    # Sort deterministically: descending similarity, then alphabetical by sentence_id for ties
    similarities.sort(key=lambda x: (-x[0], x[1]))
    
    results = []
    for rank, (score, sentence_id, record) in enumerate(similarities[:top_k], start=1):
        evidence = {
            "rank": rank,
            "similarity": score,
            "sentence_id": sentence_id,
            "document_id": record.get("document_id", ""),
            "document_name": record.get("document_name", ""),
            "page_number": record.get("page_number", 0),
            "text": record.get("text", "")
        }
        results.append(evidence)
        
    return results
