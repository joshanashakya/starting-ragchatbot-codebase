from document_processor import DocumentProcessor


def make(size=800, overlap=100):
    return DocumentProcessor(size, overlap)


class TestChunkText:
    def test_short_text_is_single_chunk(self):
        assert make().chunk_text("Hello there. General Kenobi.") == ["Hello there. General Kenobi."]

    def test_whitespace_is_normalised(self):
        assert make().chunk_text("One   sentence\n\nwith   gaps.") == ["One sentence with gaps."]

    def test_empty_text_gives_no_chunks(self):
        assert make().chunk_text("   \n ") == []

    def test_chunks_respect_size(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(50))
        chunks = make(size=100, overlap=0).chunk_text(text)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_no_overlap_means_no_repeated_sentences(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(20))
        chunks = make(size=100, overlap=0).chunk_text(text)
        assert " ".join(chunks) == text

    def test_overlap_repeats_trailing_sentence(self):
        text = "Alpha is first. Bravo is second. Charlie is third. Delta is fourth."
        chunks = make(size=35, overlap=20).chunk_text(text)
        assert len(chunks) > 1
        for prev, nxt in zip(chunks, chunks[1:]):
            assert prev.split(". ")[-1].rstrip(".") in nxt

    def test_oversized_sentence_still_emitted(self):
        long_sentence = "Word " * 50 + "end."
        chunks = make(size=40, overlap=0).chunk_text(long_sentence)
        assert chunks == [long_sentence.strip()]

    def test_always_terminates_with_large_overlap(self):
        text = " ".join(f"Sentence {i} here." for i in range(10))
        assert make(size=40, overlap=1000).chunk_text(text)


class TestProcessCourseDocument:
    def test_parses_metadata(self, sample_course_file):
        course, _ = make().process_course_document(sample_course_file)
        assert course.title == "Intro to Testing"
        assert course.course_link == "https://example.com/testing"
        assert course.instructor == "Ada Lovelace"

    def test_parses_lessons_and_links(self, sample_course_file):
        course, _ = make().process_course_document(sample_course_file)
        assert [(l.lesson_number, l.title, l.lesson_link) for l in course.lessons] == [
            (0, "Why Test", "https://example.com/testing/0"),
            (1, "Fixtures", "https://example.com/testing/1"),
            (2, "Mocking", None),  # sample lesson 2 has no link
        ]

    def test_chunks_have_sequential_indexes_and_lesson_numbers(self, sample_course_file):
        _, chunks = make().process_course_document(sample_course_file)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert [c.lesson_number for c in chunks] == [0, 1, 2]
        assert all(c.course_title == "Intro to Testing" for c in chunks)

    def test_lesson_link_line_is_not_in_content(self, sample_course_file):
        _, chunks = make().process_course_document(sample_course_file)
        assert not any("Lesson Link" in c.content for c in chunks)

    def test_chunk_prefixes(self, sample_course_file):
        """Documents current behaviour: the last lesson gets a course-title prefix too."""
        _, chunks = make().process_course_document(sample_course_file)
        assert chunks[0].content.startswith("Lesson 0 content: ")
        assert chunks[-1].content.startswith("Course Intro to Testing Lesson 2 content: ")

    def test_missing_instructor_is_none(self, tmp_path):
        f = tmp_path / "c.txt"
        f.write_text("Course Title: T\nCourse Link: http://x\n\nLesson 0: A\nHello world.\n")
        course, _ = make().process_course_document(str(f))
        assert course.instructor is None

    def test_title_falls_back_to_first_line(self, tmp_path):
        f = tmp_path / "c.txt"
        f.write_text("Just a heading\nCourse Link: http://x\nCourse Instructor: Me\n\nLesson 0: A\nHello.\n")
        course, _ = make().process_course_document(str(f))
        assert course.title == "Just a heading"

    def test_document_without_lessons_becomes_unnumbered_chunks(self, tmp_path):
        f = tmp_path / "c.txt"
        f.write_text("Course Title: T\nCourse Link: http://x\nCourse Instructor: Me\n\nSome plain body text here.\n")
        course, chunks = make().process_course_document(str(f))
        assert course.lessons == []
        assert len(chunks) == 1 and chunks[0].lesson_number is None

    def test_invalid_utf8_is_tolerated(self, tmp_path):
        f = tmp_path / "c.txt"
        f.write_bytes(b"Course Title: T\nCourse Link: x\nCourse Instructor: Me\n\nLesson 0: A\nCaf\xe9 text here.\n")
        _, chunks = make().process_course_document(str(f))
        assert chunks
