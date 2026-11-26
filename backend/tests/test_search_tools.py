from unittest.mock import MagicMock

import pytest

from search_tools import CourseSearchTool, Tool, ToolManager
from vector_store import SearchResults


def results(*pairs):
    return SearchResults(
        documents=[d for d, _ in pairs], metadata=[m for _, m in pairs], distances=[0.0] * len(pairs)
    )


@pytest.fixture
def store():
    s = MagicMock()
    s.get_lesson_link.side_effect = lambda title, n: f"http://lesson/{n}"
    s.get_course_link.return_value = "http://course"
    return s


@pytest.fixture
def tool(store):
    return CourseSearchTool(store)


class TestCourseSearchTool:
    def test_definition(self, tool):
        d = tool.get_tool_definition()
        assert d["name"] == "search_course_content"
        assert d["input_schema"]["required"] == ["query"]
        assert set(d["input_schema"]["properties"]) == {"query", "course_name", "lesson_number"}

    def test_passes_args_to_store(self, tool, store):
        store.search.return_value = SearchResults.empty("x")
        tool.execute("q", course_name="C", lesson_number=2)
        store.search.assert_called_once_with(query="q", course_name="C", lesson_number=2)

    def test_returns_store_error(self, tool, store):
        store.search.return_value = SearchResults.empty("No course found matching 'Z'")
        assert tool.execute("q", course_name="Z") == "No course found matching 'Z'"

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({}, "No relevant content found."),
            ({"course_name": "C"}, "No relevant content found in course 'C'."),
            ({"lesson_number": 3}, "No relevant content found in lesson 3."),
            (
                {"course_name": "C", "lesson_number": 3},
                "No relevant content found in course 'C' in lesson 3.",
            ),
        ],
    )
    def test_empty_results_message(self, tool, store, kwargs, expected):
        store.search.return_value = SearchResults([], [], [])
        assert tool.execute("q", **kwargs) == expected

    def test_formats_results_with_headers(self, tool, store):
        store.search.return_value = results(
            ("alpha", {"course_title": "C", "lesson_number": 1}),
            ("beta", {"course_title": "C", "lesson_number": 2}),
        )
        out = tool.execute("q")
        assert out == "[C - Lesson 1]\nalpha\n\n[C - Lesson 2]\nbeta"

    def test_sources_carry_lesson_urls_and_are_deduplicated(self, tool, store):
        store.search.return_value = results(
            ("a", {"course_title": "C", "lesson_number": 1}),
            ("b", {"course_title": "C", "lesson_number": 1}),
            ("c", {"course_title": "C", "lesson_number": 2}),
        )
        tool.execute("q")
        assert tool.last_sources == [
            {"text": "C - Lesson 1", "url": "http://lesson/1"},
            {"text": "C - Lesson 2", "url": "http://lesson/2"},
        ]

    def test_lesson_zero_is_not_treated_as_missing(self, tool, store):
        store.search.return_value = results(("a", {"course_title": "C", "lesson_number": 0}))
        assert tool.execute("q").startswith("[C - Lesson 0]")
        assert tool.last_sources[0]["text"] == "C - Lesson 0"

    def test_chunk_without_lesson_falls_back_to_course_link(self, tool, store):
        store.search.return_value = results(("a", {"course_title": "C", "lesson_number": None}))
        out = tool.execute("q")
        assert out.startswith("[C]\n")
        assert tool.last_sources == [{"text": "C", "url": "http://course"}]

    def test_works_against_real_store(self, populated_store):
        t = CourseSearchTool(populated_store)
        out = t.execute("fixtures", course_name="Intro to Testing", lesson_number=1)
        assert "[Intro to Testing - Lesson 1]" in out
        assert t.last_sources == [{"text": "Intro to Testing - Lesson 1", "url": "https://example.com/testing/1"}]


class DummyTool(Tool):
    def get_tool_definition(self):
        return {"name": "dummy"}

    def execute(self, **kwargs):
        return f"ran {kwargs}"


class TestToolManager:
    def test_register_and_list(self, tool):
        tm = ToolManager()
        tm.register_tool(tool)
        tm.register_tool(DummyTool())
        assert [d["name"] for d in tm.get_tool_definitions()] == ["search_course_content", "dummy"]

    def test_register_requires_name(self):
        class Nameless(DummyTool):
            def get_tool_definition(self):
                return {}

        with pytest.raises(ValueError):
            ToolManager().register_tool(Nameless())

    def test_execute(self):
        tm = ToolManager()
        tm.register_tool(DummyTool())
        assert tm.execute_tool("dummy", a=1) == "ran {'a': 1}"

    def test_execute_unknown_tool(self):
        assert ToolManager().execute_tool("x") == "Tool 'x' not found"

    def test_sources_lifecycle(self, tool, store):
        tm = ToolManager()
        tm.register_tool(tool)
        tm.register_tool(DummyTool())  # has no last_sources attribute
        assert tm.get_last_sources() == []
        store.search.return_value = results(("a", {"course_title": "C", "lesson_number": 1}))
        tm.execute_tool("search_course_content", query="q")
        assert tm.get_last_sources() == [{"text": "C - Lesson 1", "url": "http://lesson/1"}]
        tm.reset_sources()
        assert tm.get_last_sources() == []
