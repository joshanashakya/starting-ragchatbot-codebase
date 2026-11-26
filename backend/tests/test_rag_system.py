from unittest.mock import MagicMock

import pytest

from config import Config
from rag_system import RAGSystem


@pytest.fixture
def rag(tmp_path, fake_embeddings, monkeypatch):
    monkeypatch.setattr("ai_generator.anthropic.Anthropic", MagicMock())
    cfg = Config(ANTHROPIC_API_KEY="k", CHROMA_PATH=str(tmp_path / "chroma"))
    system = RAGSystem(cfg)
    system.ai_generator.generate_response = MagicMock(return_value="the answer")
    return system


def test_search_tool_is_registered(rag):
    assert [d["name"] for d in rag.tool_manager.get_tool_definitions()] == ["search_course_content"]


class TestIngestion:
    def test_add_course_document(self, rag, sample_course_file):
        course, n = rag.add_course_document(sample_course_file)
        assert course.title == "Intro to Testing" and n == 3
        assert rag.get_course_analytics() == {"total_courses": 1, "course_titles": ["Intro to Testing"]}

    def test_add_course_document_failure_returns_none(self, rag, tmp_path):
        assert rag.add_course_document(str(tmp_path / "missing.txt")) == (None, 0)

    def test_add_course_folder_only_reads_supported_extensions(self, rag, docs_folder):
        courses, chunks = rag.add_course_folder(docs_folder)
        assert courses == 2 and chunks == 4
        assert set(rag.get_course_analytics()["course_titles"]) == {"Intro to Testing", "Cooking Basics"}

    def test_existing_courses_are_skipped(self, rag, docs_folder):
        rag.add_course_folder(docs_folder)
        assert rag.add_course_folder(docs_folder) == (0, 0)
        assert rag.get_course_analytics()["total_courses"] == 2

    def test_clear_existing_rebuilds(self, rag, docs_folder):
        rag.add_course_folder(docs_folder)
        assert rag.add_course_folder(docs_folder, clear_existing=True) == (2, 4)

    def test_missing_folder(self, rag, tmp_path):
        assert rag.add_course_folder(str(tmp_path / "nope")) == (0, 0)


class TestQuery:
    def test_wraps_prompt_and_passes_tools(self, rag):
        rag.query("What is MCP?")
        kwargs = rag.ai_generator.generate_response.call_args.kwargs
        assert kwargs["query"] == "Answer this question about course materials: What is MCP?"
        assert kwargs["conversation_history"] is None
        assert kwargs["tool_manager"] is rag.tool_manager
        assert [t["name"] for t in kwargs["tools"]] == ["search_course_content"]

    def test_returns_answer_and_sources_then_resets(self, rag):
        rag.search_tool.last_sources = [{"text": "C - Lesson 1", "url": "u"}]
        answer, sources = rag.query("q")
        assert answer == "the answer"
        assert sources == [{"text": "C - Lesson 1", "url": "u"}]
        assert rag.search_tool.last_sources == []
        assert rag.query("q again")[1] == []  # no stale sources on next request

    def test_session_history_round_trip(self, rag):
        sid = rag.session_manager.create_session()
        rag.query("first", sid)
        rag.query("second", sid)
        history = rag.ai_generator.generate_response.call_args.kwargs["conversation_history"]
        assert history == "User: first\nAssistant: the answer"
        assert rag.session_manager.get_conversation_history(sid).endswith("User: second\nAssistant: the answer")

    def test_no_session_means_nothing_stored(self, rag):
        rag.query("q")
        assert rag.session_manager.sessions == {}

    def test_end_to_end_search_populates_sources(self, rag, sample_course_file):
        """Fake the model's tool call so the real tool + vector store are exercised."""
        rag.add_course_document(sample_course_file)

        def fake_generate(query, conversation_history, tools, tool_manager):
            tool_manager.execute_tool("search_course_content", query="fixtures", course_name="Intro to Testing")
            return "ok"

        rag.ai_generator.generate_response = fake_generate
        answer, sources = rag.query("tell me about fixtures")
        assert answer == "ok"
        assert sources and all(s["text"].startswith("Intro to Testing - Lesson") for s in sources)
