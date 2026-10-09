import hashlib
import re
from typing import List, Dict

RESULT_VERBS = {
    "achieve", "achieves", "achieved", "show", "shows", "showed", 
    "reduce", "reduces", "reduced", "increase", "increases", "increased", 
    "outperform", "outperforms", "outperformed", "result", "results", "resulted", 
    "demonstrate", "demonstrates", "demonstrated", "improve", "improves", "improved", 
    "yield", "yields", "yielded"
}

COMPARATIVES = {
    "better", "faster", "more", "less", "than", "compared", 
    "higher", "lower", "significant", "significantly"
}

def calculate_signals(text: str, tokens: List[str]) -> Dict[str, float]:
    signals = {}
    lower_tokens = {t.lower() for t in tokens}
    
    if lower_tokens.intersection(RESULT_VERBS):
        signals["result_verb"] = 1.0
        
    if lower_tokens.intersection(COMPARATIVES):
        signals["comparative"] = 1.0
        
    if re.search(r'\d', text) or '%' in text:
        signals["measurable"] = 1.0
        
    if text and not text.endswith('?') and text.endswith('.'):
        signals["declarative"] = 1.0
        
    return signals

def find_claim_candidates(records: List[dict], cfg: dict) -> List[dict]:
    min_length = cfg.get("processing", {}).get("min_sentence_length", 5)
    min_score = cfg.get("claims", {}).get("min_candidate_score", 2.0)
    max_cands = cfg.get("claims", {}).get("max_candidates", 50)
    
    candidates = []
    
    for record in records:
        if record.get("in_references", False) or record.get("is_caption", False):
            continue
            
        text = record.get("text", "").strip()
        if text.endswith('?'):
            continue
            
        tokens = record.get("tokens", [])
        if len(tokens) < min_length:
            continue
            
        signals = calculate_signals(text, tokens)
        score = sum(signals.values())
        
        if score >= min_score:
            sentence_id = record["sentence_id"]
            candidates.append({
                "candidate_id": f"cand_{sentence_id}",
                "sentence_id": sentence_id,
                "document_id": record.get("document_id", ""),
                "page_number": record.get("page_number", 0),
                "text": text,
                "score": score,
                "signals": signals
            })
            
    # Deterministic sort: descending score, then ascending sentence_id
    candidates.sort(key=lambda x: (-x["score"], x["sentence_id"]))
    
    return candidates[:max_cands]

def make_user_claim(text: str) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("Claim text cannot be empty.")
        
    text_bytes = text.encode("utf-8")
    claim_id = "claim_" + hashlib.sha1(text_bytes).hexdigest()[:10]
    
    return {
        "claim_id": claim_id,
        "text": text,
        "source": "user"
    }
