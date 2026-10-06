import hashlib
import json
from pathlib import Path
from typing import Any, Optional, List, Dict
from src.utils.config import load_config

def sha256_bytes(b: bytes) -> str:
    """Return SHA-256 hex digest of bytes."""
    return hashlib.sha256(b).hexdigest()

def document_id_from_bytes(b: bytes) -> str:
    """Generate a document ID from bytes based on SHA-256, first 12 chars."""
    return sha256_bytes(b)[:12]

def save_json(obj: Any, path: str | Path) -> None:
    """Save object as JSON, creating parent directories if necessary."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)

def load_json(path: str | Path) -> Any:
    """Load object from JSON."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_artifacts_dir() -> Path:
    cfg = load_config()
    artifacts_path = cfg.get("paths", {}).get("artifacts", "data/artifacts")
    p = Path(artifacts_path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_artifact(obj: Any, name: str) -> None:
    """Save object as an artifact (restricted to artifacts dir)."""
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError("Arbitrary paths not allowed for artifacts.")
    p = _get_artifacts_dir() / f"{name}.json"
    save_json(obj, p)

def load_artifact(name: str) -> Optional[Any]:
    """Load object from artifacts dir."""
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError("Arbitrary paths not allowed for artifacts.")
    p = _get_artifacts_dir() / f"{name}.json"
    if p.exists():
        return load_json(p)
    return None

def corpus_fingerprint(doc_ids: List[str], cfg: Dict[str, Any]) -> str:
    """Generate deterministic corpus fingerprint from doc IDs and relevant config."""
    sorted_docs = sorted(doc_ids)
    
    relevant_cfg = {
        "processing": cfg.get("processing", {}),
        "preprocessing": cfg.get("preprocessing", {})
    }
    
    state = {
        "docs": sorted_docs,
        "config": relevant_cfg
    }
    
    state_bytes = json.dumps(state, sort_keys=True).encode("utf-8")
    return sha256_bytes(state_bytes)[:12]
