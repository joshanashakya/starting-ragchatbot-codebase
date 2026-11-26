# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                      # install dependencies (Python >= 3.13, managed with uv)
cp .env.example .env         # then set ANTHROPIC_API_KEY
./run.sh                     # start the app; same as: cd backend && uv run uvicorn app:app --reload --port 8000
```

- App: http://localhost:8000, API docs: http://localhost:8000/docs
- **Run the server from `backend/`.** Paths are relative to it: `../docs`, `../frontend`, `./chroma_db`.
- There are no tests, linter, or build step configured. `main.py` at the repo root is an unused stub.

## Architecture

A RAG chatbot over course transcripts. FastAPI (`backend/app.py`) serves both the JSON API and the static vanilla-JS frontend (`frontend/`). There is no frontend build.

Request flow for `POST /api/query`:
`app.py` → `RAGSystem.query()` → `AIGenerator.generate_response()` → Claude → (optional) `ToolManager` → `CourseSearchTool` → `VectorStore` (ChromaDB) → second Claude call → `(answer, sources)`.

Key design points that span several files:

- **Retrieval is a Claude tool, not a fixed pipeline.** Claude decides whether to call `search_course_content` (`search_tools.py`). `ai_generator.py` makes up to two API calls: the first with tools, the second (after tool results) **without tools**, so only one search round is possible. The system prompt in `AIGenerator.SYSTEM_PROMPT` also says "one search per query maximum".
- **New tools** subclass `Tool` in `search_tools.py` and are registered with `ToolManager` in `RAGSystem.__init__`.
- **Sources use shared state.** `CourseSearchTool._format_results()` sets `last_sources` on the tool instance as a side effect. `RAGSystem.query()` reads it with `ToolManager.get_last_sources()` and then calls `reset_sources()`. It is not request-scoped.
- **Two Chroma collections** in `vector_store.py`: `course_catalog` (one record per course, embeds the title only, lesson list stored as a JSON string in metadata) and `course_content` (one record per chunk). A partial `course_name` from Claude is resolved by a top-1 semantic query on `course_catalog`, and the resulting full title becomes the filter on `course_content`.
- **Conversation history** (`session_manager.py`) is in memory only. It is formatted as plain text and appended to the **system prompt**, not sent as `messages`. `MAX_HISTORY` counts exchanges (it keeps `MAX_HISTORY * 2` messages). Session IDs are sequential (`session_N`) and reset on restart.
- `config.py` holds the tunables (model, chunk size/overlap, `MAX_RESULTS`, `MAX_HISTORY`, `CHROMA_PATH`, embedding model).

## Document ingestion

On startup `app.py` calls `RAGSystem.add_course_folder("../docs")`. `document_processor.py` expects this format:

```
Course Title: ...
Course Link: ...
Course Instructor: ...

Lesson 0: <title>
Lesson Link: ...
<transcript text>
Lesson 1: ...
```

- Only `.txt` really works. `.pdf` and `.docx` are accepted by extension but are read as plain UTF-8 text.
- Text is split by sentence into ~800-character chunks with 100-character overlap, per lesson. Chunks get a text prefix (`Lesson N content:` / `Course <title> Lesson N content:`), and the prefixing is inconsistent between the final lesson and the others.
- A course's title is its unique ID. Already-loaded titles are skipped on later startups, so editing a transcript has no effect until the data is rebuilt (`add_course_folder(..., clear_existing=True)`, or delete `backend/chroma_db/`).
- The embedding model (`all-MiniLM-L6-v2`) is downloaded on first run.
