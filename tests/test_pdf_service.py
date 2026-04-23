"""
Unit and integration tests for services/pdf_service.py
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from services.pdf_service import load_and_split_pdf


def _make_docs(count: int):
    """Helper: create fake Document objects with 0-indexed page metadata (as PyPDFLoader does)."""
    return [
        Document(
            page_content=f"Content for page {i}",
            metadata={"page": i, "source": "test.pdf"},
        )
        for i in range(count)
    ]


class TestPageNumberNormalization:
    def test_page_numbers_are_converted_to_1_indexed(self):
        fake_docs = _make_docs(3)  # pages 0, 1, 2
        with patch("services.pdf_service.PyPDFLoader") as mock_loader_cls, \
             patch("services.pdf_service.RecursiveCharacterTextSplitter") as mock_splitter_cls:
            mock_loader_cls.return_value.load.return_value = fake_docs
            mock_splitter_cls.return_value.split_documents.return_value = fake_docs

            chunks = load_and_split_pdf("test.pdf")

        pages = [c.metadata["page"] for c in chunks]
        assert pages == [1, 2, 3], f"Expected [1, 2, 3], got {pages}"

    def test_single_page_document_starts_at_1(self):
        fake_docs = [Document(page_content="Only page", metadata={"page": 0, "source": "test.pdf"})]
        with patch("services.pdf_service.PyPDFLoader") as mock_loader_cls, \
             patch("services.pdf_service.RecursiveCharacterTextSplitter") as mock_splitter_cls:
            mock_loader_cls.return_value.load.return_value = fake_docs
            mock_splitter_cls.return_value.split_documents.return_value = fake_docs

            chunks = load_and_split_pdf("test.pdf")

        assert chunks[0].metadata["page"] == 1

    def test_returns_list(self):
        fake_docs = _make_docs(2)
        with patch("services.pdf_service.PyPDFLoader") as mock_loader_cls, \
             patch("services.pdf_service.RecursiveCharacterTextSplitter") as mock_splitter_cls:
            mock_loader_cls.return_value.load.return_value = fake_docs
            mock_splitter_cls.return_value.split_documents.return_value = fake_docs

            result = load_and_split_pdf("test.pdf")

        assert isinstance(result, list)
        assert len(result) == 2

    def test_page_metadata_key_always_present(self):
        fake_docs = _make_docs(2)
        with patch("services.pdf_service.PyPDFLoader") as mock_loader_cls, \
             patch("services.pdf_service.RecursiveCharacterTextSplitter") as mock_splitter_cls:
            mock_loader_cls.return_value.load.return_value = fake_docs
            mock_splitter_cls.return_value.split_documents.return_value = fake_docs

            chunks = load_and_split_pdf("test.pdf")

        for chunk in chunks:
            assert "page" in chunk.metadata


class TestIntegration:
    def test_load_real_pdf_returns_documents(self, generated_pdf_path):
        """Integration: load a real (auto-generated blank) PDF."""
        chunks = load_and_split_pdf(generated_pdf_path)
        assert isinstance(chunks, list)

    def test_load_real_pdf_page_numbers_start_at_1(self, generated_pdf_path):
        chunks = load_and_split_pdf(generated_pdf_path)
        for chunk in chunks:
            assert chunk.metadata.get("page", 0) >= 1, (
                f"Expected page >= 1, got {chunk.metadata.get('page')}"
            )

    def test_load_sample_pdf_if_present(self, real_pdf_path):
        """Integration with a real user-supplied PDF (skipped if absent)."""
        chunks = load_and_split_pdf(real_pdf_path)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata.get("page", 0) >= 1
