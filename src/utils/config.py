import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except Exception:
        return {}
