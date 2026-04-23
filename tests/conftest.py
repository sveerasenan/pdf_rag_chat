import pytest
import os
from pypdf import PdfWriter


@pytest.fixture(scope="session")
def generated_pdf_path(tmp_path_factory):
    """Creates a minimal 3-page PDF (blank pages) for automated testing."""
    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = tmp_dir / "test_sample.pdf"
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=612, height=792)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return str(pdf_path)


@pytest.fixture(scope="session")
def real_pdf_path():
    """
    Points to tests/sample_pdfs/sample.pdf.
    Tests using this fixture are skipped when the file is absent.
    """
    path = os.path.join(os.path.dirname(__file__), "sample_pdfs", "sample.pdf")
    if not os.path.exists(path):
        pytest.skip("tests/sample_pdfs/sample.pdf not found — place a real PDF there to run integration tests")
    return path
