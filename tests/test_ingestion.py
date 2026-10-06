import pytest
from pathlib import Path
from src.ingestion.pdf_loader import validate_pdf, extract_pages, load_pdfs, PDFValidationError
from src.ingestion.metadata import build_document_meta, is_duplicate
from src.utils.file_utils import document_id_from_bytes, sha256_bytes
from tests.fixtures.make_pdfs import generate_all_fixtures

@pytest.fixture(scope="session", autouse=True)
def setup_fixtures():
    generate_all_fixtures()

@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"

def test_valid_pdf_accepted(fixtures_dir):
    p = fixtures_dir / "valid.pdf"
    b = validate_pdf(p)
    assert b.startswith(b"%PDF")

def test_fake_pdf_rejected(fixtures_dir):
    p = fixtures_dir / "fake.pdf"
    with pytest.raises(PDFValidationError, match="missing %PDF header"):
        validate_pdf(p)

def test_empty_file_rejected(fixtures_dir):
    p = fixtures_dir / "empty.pdf"
    with pytest.raises(PDFValidationError, match="Empty file"):
        validate_pdf(p)

def test_corrupt_pdf_rejected(fixtures_dir):
    p = fixtures_dir / "corrupt.pdf"
    with pytest.raises(PDFValidationError):
        validate_pdf(p)

def test_deterministic_doc_id(fixtures_dir):
    p = fixtures_dir / "valid.pdf"
    b = p.read_bytes()
    doc_id1 = document_id_from_bytes(b)
    doc_id2 = document_id_from_bytes(b)
    assert doc_id1 == doc_id2

def test_pages_numbered(fixtures_dir):
    p = fixtures_dir / "multipage.pdf"
    records = extract_pages(p)
    assert len(records) == 3
    assert records[0]["page_number"] == 1
    assert records[1]["page_number"] == 2
    assert records[2]["page_number"] == 3

def test_empty_page_extraction(fixtures_dir):
    p = fixtures_dir / "empty_page.pdf"
    records = extract_pages(p)
    assert len(records) == 1
    assert records[0]["text"].strip() == ""

def test_load_pdfs_multiple(fixtures_dir):
    p1 = fixtures_dir / "valid.pdf"
    p2 = fixtures_dir / "multipage.pdf"
    p3 = fixtures_dir / "fake.pdf"
    
    pages, failures = load_pdfs([p1, p2, p3])
    
    assert len(pages) == 4 # 1 from valid, 3 from multipage
    assert len(failures) == 1 # fake.pdf failed
    assert failures[0]["filename"] == "fake.pdf"

def test_metadata_generation(fixtures_dir):
    p = fixtures_dir / "valid.pdf"
    meta = build_document_meta(p, 1)
    
    assert "document_id" in meta
    assert meta["filename"] == "valid.pdf"
    assert "sha256" in meta
    assert meta["page_count"] == 1
    assert "ingested_at" in meta

def test_provenance_preserved(fixtures_dir):
    p = fixtures_dir / "valid.pdf"
    records = extract_pages(p)
    b = p.read_bytes()
    doc_id = document_id_from_bytes(b)
    
    assert records[0]["document_id"] == doc_id
    assert records[0]["filename"] == "valid.pdf"

def test_two_column_reading_order(fixtures_dir):
    p = fixtures_dir / "two_column.pdf"
    records = extract_pages(p)
    text = records[0]["text"]
    assert text.find("Left column") < text.find("Right column")
