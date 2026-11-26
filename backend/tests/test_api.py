"""API tests run against app.py with RAGSystem mocked out."""


class TestQueryEndpoint:
    def test_creates_session_when_missing(self, client, mock_rag):
        r = client.post("/api/query", json={"query": "hello"})
        assert r.status_code == 200
        assert r.json() == {
            "answer": "An answer",
            "sources": [{"text": "Course - Lesson 1", "url": "https://x/1"}],
            "session_id": "session_1",
        }
        mock_rag.query.assert_called_once_with("hello", "session_1")

    def test_reuses_provided_session(self, client, mock_rag):
        r = client.post("/api/query", json={"query": "hi", "session_id": "session_9"})
        assert r.json()["session_id"] == "session_9"
        mock_rag.session_manager.create_session.assert_not_called()
        mock_rag.query.assert_called_once_with("hi", "session_9")

    def test_source_without_url(self, client, mock_rag):
        mock_rag.query.return_value = ("a", [{"text": "T"}])
        assert client.post("/api/query", json={"query": "q"}).json()["sources"] == [{"text": "T", "url": None}]

    def test_empty_sources(self, client, mock_rag):
        mock_rag.query.return_value = ("a", [])
        assert client.post("/api/query", json={"query": "q"}).json()["sources"] == []

    def test_missing_query_is_422(self, client):
        assert client.post("/api/query", json={}).status_code == 422

    def test_rag_errors_become_500(self, client, mock_rag):
        mock_rag.query.side_effect = RuntimeError("model unavailable")
        r = client.post("/api/query", json={"query": "q"})
        assert r.status_code == 500 and r.json()["detail"] == "model unavailable"

    def test_wrong_method(self, client):
        assert client.get("/api/query").status_code == 405


class TestCoursesEndpoint:
    def test_stats(self, client):
        r = client.get("/api/courses")
        assert r.status_code == 200
        assert r.json() == {"total_courses": 2, "course_titles": ["A", "B"]}

    def test_errors_become_500(self, client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("chroma down")
        r = client.get("/api/courses")
        assert r.status_code == 500 and r.json()["detail"] == "chroma down"


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()
