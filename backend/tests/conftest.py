"""Shared fixtures. Everything here runs offline: no API key, no model download."""
import hashlib
import math
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import chromadb
import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

SAMPLE_COURSE = """Course Title: Intro to Testing
Course Link: https://example.com/testing
Course Instructor: Ada Lovelace

Lesson 0: Why Test
Lesson Link: https://example.com/testing/0
Tests catch regressions early. Good tests are fast and deterministic. They document behaviour.

Lesson 1: Fixtures
Lesson Link: https://example.com/testing/1
Fixtures provide reusable setup. Pytest fixtures can be scoped to a function or a session.

Lesson 2: Mocking
Mocks replace slow or unreliable dependencies. Use them at the boundaries of your system.
"""

OTHER_COURSE = """Course Title: Cooking Basics
Course Link: https://example.com/cooking
Course Instructor: Julia Child

Lesson 0: Knives
Lesson Link: https://example.com/cooking/0
A sharp knife is safer than a dull knife. Learn the claw grip before chopping onions.
"""


class FakeEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic bag-of-words embedding so similar texts land close together."""

    DIM = 64

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def name() -> str:
        return "fake-bow"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return FakeEmbeddingFunction()

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for text in input:
            vec = [0.0] * self.DIM
            for word in re.findall(r"[a-z]+", text.lower()):
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
def sample_course_file(tmp_path):
    path = tmp_path / "course.txt"
    path.write_text(SAMPLE_COURSE, encoding="utf-8")
    return str(path)


@pytest.fixture
def docs_folder(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "testing.txt").write_text(SAMPLE_COURSE, encoding="utf-8")
    (folder / "cooking.txt").write_text(OTHER_COURSE, encoding="utf-8")
    (folder / "ignored.md").write_text("not a course", encoding="utf-8")
    return str(folder)


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Swap the SentenceTransformer embedder for a deterministic offline one."""
    monkeypatch.setattr(
        chromadb.utils.embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        FakeEmbeddingFunction,
    )


@pytest.fixture
def vector_store(tmp_path, fake_embeddings):
    """A real VectorStore on a temp Chroma dir with a fake embedding function."""
    from vector_store import VectorStore

    return VectorStore(str(tmp_path / "chroma"), "ignored-model", max_results=5)


@pytest.fixture
def populated_store(vector_store, docs_folder):
    from document_processor import DocumentProcessor

    processor = DocumentProcessor(800, 100)
    for name in ("testing.txt", "cooking.txt"):
        course, chunks = processor.process_course_document(os.path.join(docs_folder, name))
        vector_store.add_course_metadata(course)
        vector_store.add_course_content(chunks)
    return vector_store


@pytest.fixture(scope="session")
def app_module():
    """Import app.py once with RAGSystem patched out (it would load models/Chroma)."""
    import rag_system

    original = rag_system.RAGSystem
    rag_system.RAGSystem = MagicMock()
    cwd = os.getcwd()
    os.chdir(BACKEND_DIR)  # app.py mounts ../frontend relative to cwd
    sys.modules.pop("app", None)
    try:
        import app as app_module
    finally:
        os.chdir(cwd)
        rag_system.RAGSystem = original
    return app_module


@pytest.fixture
def mock_rag(app_module):
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "session_1"
    rag.query.return_value = ("An answer", [{"text": "Course - Lesson 1", "url": "https://x/1"}])
    rag.get_course_analytics.return_value = {"total_courses": 2, "course_titles": ["A", "B"]}
    app_module.rag_system = rag
    return rag


@pytest.fixture
def client(app_module, mock_rag):
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)
