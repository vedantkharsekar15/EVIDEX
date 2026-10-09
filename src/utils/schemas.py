from typing import TypedDict, Optional, List

class PageRecord(TypedDict):
    document_id: str
    filename: str
    page_number: int
    text: str

class SentenceRecord(TypedDict):
    sentence_id: str
    document_id: str
    document_name: str
    page_number: int
    page_end: Optional[int]
    sentence_index: int
    text: str
    tokens: List[str]
    normalized_tokens: List[str]
    in_references: bool
    is_caption: bool

class Claim(TypedDict):
    claim_id: str
    text: str
    source: str

class ClaimCandidate(TypedDict):
    candidate_id: str
    sentence_id: str
    document_id: str
    page_number: int
    text: str
    score: float
    signals: dict

class Evidence(TypedDict):
    rank: int
    similarity: float
    sentence_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
