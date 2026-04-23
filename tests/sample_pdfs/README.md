# Place your PDF files here for integration tests

Put a file named `sample.pdf` in this directory to enable the integration test
suite in `tests/test_pdf_service.py`.

Integration tests that depend on a real PDF are automatically **skipped** when
`sample.pdf` is absent, so the rest of the test suite still passes without it.

> **Never commit confidential PDFs.**  Add `tests/sample_pdfs/*.pdf` to
> `.gitignore` if your documents are sensitive.
