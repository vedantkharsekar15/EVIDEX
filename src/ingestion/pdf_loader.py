import fitz
from pathlib import Path
from typing import List, Tuple, Dict, Any
from src.utils.schemas import PageRecord
from src.utils.file_utils import document_id_from_bytes
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PDFValidationError(Exception):
    """Raised when a PDF fails validation."""
    pass

def validate_pdf(path: Path) -> bytes:
    """Validate a PDF and return its bytes."""
    if not path.exists():
        raise PDFValidationError(f"File not found: {path}")
    
    if not path.is_file():
        raise PDFValidationError(f"Not a file: {path}")
        
    try:
        b = path.read_bytes()
    except Exception as e:
        raise PDFValidationError(f"Could not read file {path}: {e}")
        
    if not b:
        raise PDFValidationError(f"Empty file: {path}")
        
    if not b.startswith(b"%PDF"):
        raise PDFValidationError(f"Not a valid PDF, missing %PDF header: {path}")
        
    # Additional check using PyMuPDF to catch corrupt files
    try:
        with fitz.open(stream=b, filetype="pdf") as doc:
            if not doc.is_pdf:
                raise PDFValidationError(f"PyMuPDF could not parse PDF: {path}")
    except Exception as e:
        raise PDFValidationError(f"Invalid or corrupt PDF {path}: {e}")
        
    return b

def extract_pages(path: Path) -> List[PageRecord]:
    """Extract pages from a PDF file."""
    b = validate_pdf(path)
    doc_id = document_id_from_bytes(b)
    filename = path.name
    
    records = []
    
    with fitz.open(stream=b, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            page_num = i + 1
            # Simple block-based extraction (can provide better reading order)
            text_blocks = page.get_text("blocks")
            # Sort blocks by vertical position, then horizontal
            text_blocks.sort(key=lambda block: (block[1], block[0]))
            
            # extract text from blocks (index 4 is text, index 6 is block_type where 0 is text)
            page_text = "\n".join([block[4] for block in text_blocks if block[6] == 0])
            
            # If no blocks (or fails), fallback to basic text extraction
            if not page_text.strip():
                page_text = page.get_text()
                
            records.append({
                "document_id": doc_id,
                "filename": filename,
                "page_number": page_num,
                "text": page_text
            })
            
    return records

def load_pdfs(paths: List[Path]) -> Tuple[List[PageRecord], List[Dict[str, Any]]]:
    """Load multiple PDFs, returning valid pages and a list of failures."""
    all_pages = []
    failures = []
    
    for p in paths:
        try:
            pages = extract_pages(p)
            all_pages.extend(pages)
            logger.info(f"Successfully extracted {len(pages)} pages from {p.name}")
        except Exception as e:
            logger.error(f"Failed to load {p}: {e}")
            failures.append({
                "filename": p.name if hasattr(p, 'name') else str(p),
                "error": str(e),
                "path": str(p)
            })
            
    return all_pages, failures
