import json

from vector_store import SearchResults, VectorStore


class TestSearchResults:
    def test_from_chroma(self):
        r = SearchResults.from_chroma(
            {"documents": [["a"]], "metadatas": [[{"k": 1}]], "distances": [[0.1]]}
        )
        assert r.documents == ["a"] and r.metadata == [{"k": 1}] and r.distances == [0.1]
        assert not r.is_empty()

    def test_from_chroma_empty(self):
        r = SearchResults.from_chroma({"documents": [], "metadatas": [], "distances": []})
        assert r.is_empty() and r.error is None

    def test_empty_with_error(self):
        r = SearchResults.empty("boom")
        assert r.is_empty() and r.error == "boom"


class TestBuildFilter:
    def test_filters(self, vector_store):
        f = vector_store._build_filter
        assert f(None, None) is None
        assert f("C", None) == {"course_title": "C"}
        assert f(None, 0) == {"lesson_number": 0}  # lesson 0 must not be treated as "no filter"
        assert f("C", 3) == {"$and": [{"course_title": "C"}, {"lesson_number": 3}]}


class TestCatalog:
    def test_counts_and_titles(self, populated_store):
        assert populated_store.get_course_count() == 2
        assert set(populated_store.get_existing_course_titles()) == {"Intro to Testing", "Cooking Basics"}

    def test_links(self, populated_store):
        assert populated_store.get_course_link("Intro to Testing") == "https://example.com/testing"
        assert populated_store.get_lesson_link("Intro to Testing", 1) == "https://example.com/testing/1"

    def test_missing_links_are_none(self, populated_store):
        assert populated_store.get_lesson_link("Intro to Testing", 99) is None
        assert populated_store.get_lesson_link("Intro to Testing", 2) is None  # lesson without link
        assert populated_store.get_lesson_link("Nope", 0) is None
        assert populated_store.get_course_link("Nope") is None

    def test_all_courses_metadata_parses_lessons(self, populated_store):
        meta = {m["title"]: m for m in populated_store.get_all_courses_metadata()}
        lessons = meta["Intro to Testing"]["lessons"]
        assert [l["lesson_number"] for l in lessons] == [0, 1, 2]
        assert "lessons_json" not in meta["Intro to Testing"]

    def test_add_course_content_with_no_chunks_is_noop(self, vector_store):
        vector_store.add_course_content([])
        assert vector_store.course_content.count() == 0


class TestSearch:
    def test_unfiltered_search_finds_relevant_chunk(self, populated_store):
        r = populated_store.search("sharp knife onions")
        assert r.error is None and not r.is_empty()
        assert r.metadata[0]["course_title"] == "Cooking Basics"

    def test_course_name_is_resolved_semantically(self, populated_store):
        r = populated_store.search("anything", course_name="Cooking")
        assert {m["course_title"] for m in r.metadata} == {"Cooking Basics"}

    def test_lesson_filter(self, populated_store):
        r = populated_store.search("fixtures", course_name="Intro to Testing", lesson_number=1)
        assert r.metadata and all(m["lesson_number"] == 1 for m in r.metadata)

    def test_lesson_zero_filter_is_applied(self, populated_store):
        r = populated_store.search("tests", course_name="Intro to Testing", lesson_number=0)
        assert r.metadata and all(m["lesson_number"] == 0 for m in r.metadata)

    def test_limit_overrides_max_results(self, populated_store):
        assert len(populated_store.search("tests", limit=1).documents) == 1

    def test_unknown_course_returns_error(self, vector_store):
        r = vector_store.search("x", course_name="Whatever")
        assert r.error == "No course found matching 'Whatever'"

    def test_query_failure_is_returned_as_error(self, populated_store, monkeypatch):
        def boom(**kw):
            raise RuntimeError("db down")

        monkeypatch.setattr(populated_store.course_content, "query", boom)
        assert populated_store.search("x").error == "Search error: db down"


def test_clear_all_data(populated_store):
    populated_store.clear_all_data()
    assert populated_store.get_course_count() == 0
    assert populated_store.course_content.count() == 0


def test_data_persists_across_instances(populated_store, tmp_path):
    again = VectorStore(str(tmp_path / "chroma"), "ignored", 5)
    assert again.get_course_count() == 2
