import pytest

from app.ingestion.chunking import chunk_blocks, embedding_text, estimate_tokens
from app.ingestion.parsers import Block

SIZES = {"target_tokens": 60, "max_tokens": 80, "overlap_tokens": 10}


def sentence(n: int) -> str:
    return f"Sentence number {n} talks about topic {n} in some detail."


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") == 3  # 2 words * 1.3, rounded up
    # Dense text without spaces is counted by characters instead.
    assert estimate_tokens("x" * 400) == 100


def test_small_section_is_one_chunk() -> None:
    blocks = [Block("Short intro.", section_path=("Intro",))]
    chunks = chunk_blocks(blocks, **SIZES)
    assert len(chunks) == 1
    assert chunks[0].text == "Short intro."
    assert chunks[0].section_path == ("Intro",)
    assert chunks[0].index == 0


def test_chunks_never_cross_sections() -> None:
    blocks = [
        Block("About leave.", section_path=("Leave",)),
        Block("About travel.", section_path=("Travel",)),
        Block("More on travel.", section_path=("Travel",)),
    ]
    chunks = chunk_blocks(blocks, **SIZES)
    assert [c.section_path for c in chunks] == [("Leave",), ("Travel",)]
    assert chunks[1].text == "About travel.\n\nMore on travel."


def test_long_section_respects_max_and_overlaps() -> None:
    blocks = [Block(sentence(i), section_path=("S",)) for i in range(30)]
    chunks = chunk_blocks(blocks, **SIZES)
    assert len(chunks) > 3
    assert all(c.token_count <= SIZES["max_tokens"] for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Each chunk after the first starts with the tail of the one before it.
    for previous, current in zip(chunks, chunks[1:], strict=False):
        first_word_of_current = current.text.split()[0]
        assert first_word_of_current in previous.text.split()[-8:]
    # Nothing is lost: every sentence appears somewhere.
    joined = " ".join(c.text for c in chunks)
    assert all(sentence(i) in joined for i in range(30))


def test_oversized_paragraph_is_split_on_sentences() -> None:
    paragraph = " ".join(sentence(i) for i in range(20))
    chunks = chunk_blocks([Block(paragraph)], **SIZES)
    assert len(chunks) > 1
    assert all(c.token_count <= SIZES["max_tokens"] for c in chunks)


def test_single_huge_sentence_is_split_on_words() -> None:
    chunks = chunk_blocks([Block("word " * 500)], **SIZES)
    assert all(c.token_count <= SIZES["max_tokens"] for c in chunks)
    assert sum(c.text.split().count("word") for c in chunks) >= 500


def test_page_range_is_tracked() -> None:
    blocks = [Block("On page one.", page=1), Block("On page two.", page=2)]
    chunks = chunk_blocks(blocks, **SIZES)
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_invalid_sizes_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_blocks([], target_tokens=100, max_tokens=50, overlap_tokens=10)


def test_embedding_text_has_location_header() -> None:
    chunk = chunk_blocks([Block("Capped at 20 days.", section_path=("Leave", "Annual"))], **SIZES)[
        0
    ]
    assert embedding_text(chunk, "Handbook") == "Handbook > Leave > Annual\n\nCapped at 20 days."
    assert embedding_text(chunk, None) == "Leave > Annual\n\nCapped at 20 days."
