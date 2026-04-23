"""
Unit tests for services/embedding_service.py
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from services.embedding_service import create_embeddings, create_vectorstore, create_retriever


def _make_chunks():
    return [
        Document(page_content="Chunk one", metadata={"page": 1}),
        Document(page_content="Chunk two", metadata={"page": 2}),
    ]


class TestCreateEmbeddings:
    def test_openai_provider_returns_openai_embeddings(self):
        # Import happens inside the function body, so patch at the source module
        mock_emb = MagicMock()
        with patch("langchain_openai.OpenAIEmbeddings", return_value=mock_emb) as mock_cls:
            result = create_embeddings("openai", "fake-key")

        mock_cls.assert_called_once_with(api_key="fake-key", model="text-embedding-3-small")
        assert result is mock_emb

    def test_gemini_provider_returns_google_embeddings(self):
        mock_emb = MagicMock()
        with patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", return_value=mock_emb) as mock_cls:
            result = create_embeddings("gemini", "fake-key")

        mock_cls.assert_called_once_with(
            google_api_key="fake-key",
            model="models/gemini-embedding-001",
        )
        assert result is mock_emb

    def test_unknown_provider_defaults_to_openai(self):
        """Any non-gemini provider string falls through to the OpenAI branch."""
        mock_emb = MagicMock()
        with patch("langchain_openai.OpenAIEmbeddings", return_value=mock_emb):
            result = create_embeddings("unknown-provider", "fake-key")

        assert result is mock_emb


class TestCreateVectorstore:
    def test_calls_faiss_from_documents(self):
        mock_vs = MagicMock()
        mock_embeddings = MagicMock()
        chunks = _make_chunks()

        with patch("services.embedding_service.FAISS") as mock_faiss:
            mock_faiss.from_documents.return_value = mock_vs
            result = create_vectorstore(chunks, mock_embeddings)

        mock_faiss.from_documents.assert_called_once_with(chunks, mock_embeddings)
        assert result is mock_vs


class TestCreateRetriever:
    def test_returns_retriever_with_default_k(self):
        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        result = create_retriever(mock_vs)

        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        assert result is mock_retriever

    def test_returns_retriever_with_custom_k(self):
        mock_vs = MagicMock()
        create_retriever(mock_vs, k=5)
        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
