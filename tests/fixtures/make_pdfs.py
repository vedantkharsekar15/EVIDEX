import fitz
from pathlib import Path

def create_valid_pdf(path: Path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello World", fontsize=12)
    doc.save(str(path))
    doc.close()

def create_multipage_pdf(path: Path):
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((50, 50), f"Page {i+1}", fontsize=12)
    doc.save(str(path))
    doc.close()

def create_empty_page_pdf(path: Path):
    doc = fitz.open()
    doc.new_page() # blank page
    doc.save(str(path))
    doc.close()

def create_corrupt_pdf(path: Path):
    import os
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4\n" + os.urandom(1024))

def create_fake_pdf(path: Path):
    with open(path, "wb") as f:
        f.write(b"This is just a text file but named .pdf")

def create_empty_file(path: Path):
    with open(path, "wb") as f:
        f.write(b"")

def create_two_column_pdf(path: Path):
    doc = fitz.open()
    page = doc.new_page()
    # Left column
    page.insert_text((50, 50), "Left column text 1", fontsize=12)
    page.insert_text((50, 150), "Left column text 2", fontsize=12)
    # Right column
    page.insert_text((300, 50), "Right column text 1", fontsize=12)
    page.insert_text((300, 150), "Right column text 2", fontsize=12)
    doc.save(str(path))
    doc.close()

def generate_all_fixtures():
    base_dir = Path(__file__).parent
    create_valid_pdf(base_dir / "valid.pdf")
    create_multipage_pdf(base_dir / "multipage.pdf")
    create_empty_page_pdf(base_dir / "empty_page.pdf")
    create_corrupt_pdf(base_dir / "corrupt.pdf")
    create_fake_pdf(base_dir / "fake.pdf")
    create_empty_file(base_dir / "empty.pdf")
    create_two_column_pdf(base_dir / "two_column.pdf")

if __name__ == "__main__":
    generate_all_fixtures()
