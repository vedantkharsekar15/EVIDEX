import time
from pathlib import Path
from typing import Dict, Any
from src.utils.file_utils import document_id_from_bytes, sha256_bytes, load_json, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def build_document_meta(path: str | Path, page_count: int) -> Dict[str, Any]:
    """Build metadata for a document based on its file."""
    p = Path(path)
    b = p.read_bytes()
    
    doc_id = document_id_from_bytes(b)
    sha_hash = sha256_bytes(b)
    
    return {
        "document_id": doc_id,
        "filename": p.name,
        "sha256": sha_hash,
        "page_count": page_count,
        "ingested_at": time.time()
    }

def _get_registry_path() -> Path:
    cfg = load_config()
    proc_path = cfg.get("paths", {}).get("processed", "data/processed")
    p = Path(proc_path) / "documents.json"
    return p

def _load_registry() -> Dict[str, Any]:
    p = _get_registry_path()
    if p.exists():
        try:
            return load_json(p)
        except Exception:
            return {}
    return {}

def is_duplicate(document_id: str) -> bool:
    """Check if a document is already registered."""
    registry = _load_registry()
    return document_id in registry
