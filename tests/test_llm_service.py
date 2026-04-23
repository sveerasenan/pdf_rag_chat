"""
Unit tests for services/llm_service.py
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from services.llm_service import build_prompt, get_answer, create_llm


def _make_docs(pages_and_content):
    return [
        Document(page_content=content, metadata={"page": page})
        for page, content in pages_and_content
    ]


class TestBuildPrompt:
    def test_includes_page_numbers_in_context(self):
        docs = _make_docs([(3, "Some content about topic A")])
        prompt = build_prompt("What is topic A?", docs)
        assert "[Page 3]" in prompt

    def test_includes_all_pages(self):
        docs = _make_docs([(1, "First page content"), (5, "Fifth page content")])
        prompt = build_prompt("Summarise.", docs)
        assert "[Page 1]" in prompt
        assert "[Page 5]" in prompt

    def test_includes_question(self):
        docs = _make_docs([(1, "Some content")])
        question = "What is the capital of France?"
        prompt = build_prompt(question, docs)
        assert question in prompt

    def test_includes_doc_content(self):
        docs = _make_docs([(2, "Paris is the capital.")])
        prompt = build_prompt("Question?", docs)
        assert "Paris is the capital." in prompt

    def test_handles_missing_page_metadata(self):
        doc = Document(page_content="Content without page", metadata={})
        prompt = build_prompt("Question?", [doc])
        assert "[Page ?]" in prompt

    def test_returns_string(self):
        docs = _make_docs([(1, "content")])
        result = build_prompt("question", docs)
        assert isinstance(result, str)

    def test_multiple_docs_all_appear_in_prompt(self):
        docs = _make_docs([(1, "Alpha"), (2, "Beta"), (3, "Gamma")])
        prompt = build_prompt("Test?", docs)
        assert "Alpha" in prompt
        assert "Beta" in prompt
        assert "Gamma" in prompt


class TestGetAnswer:
    def test_returns_llm_content(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="The answer is 42.")
        docs = _make_docs([(1, "Context text")])

        result = get_answer(mock_llm, "What is the answer?", docs)

        assert result == "The answer is 42."

    def test_calls_llm_invoke_once(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")
        docs = _make_docs([(1, "Context")])

        get_answer(mock_llm, "Question?", docs)

        mock_llm.invoke.assert_called_once()

    def test_prompt_passed_to_invoke_contains_question(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")
        docs = _make_docs([(1, "Context")])
        question = "What does the document say about testing?"

        get_answer(mock_llm, question, docs)

        called_prompt = mock_llm.invoke.call_args[0][0]
        assert question in called_prompt


class TestCreateLlm:
    def test_openai_provider_creates_chatopenai(self):
        # Import happens inside the function body, so patch at the source module
        mock_llm = MagicMock()
        with patch("langchain_openai.ChatOpenAI", return_value=mock_llm) as mock_cls:
            result = create_llm("openai", "fake-key")

        mock_cls.assert_called_once_with(api_key="fake-key", model="gpt-4o-mini")
        assert result is mock_llm

    def test_gemini_provider_creates_chatgoogle(self):
        mock_llm = MagicMock()
        with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm) as mock_cls:
            result = create_llm("gemini", "fake-key")

        mock_cls.assert_called_once_with(google_api_key="fake-key", model="gemini-2.5-flash")
        assert result is mock_llm
