"""Unit tests for transcript parsing and chunking."""

from app.chunking import (
    Chunk,
    add_token_overlap,
    count_tokens,
    extract_last_tokens,
    parse_transcript,
    split_at_sentences,
)


def test_count_tokens_nonempty():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0


def test_split_at_sentences_breaks_on_punctuation():
    parts = split_at_sentences("One. Two. Three.", max_tokens=1000)
    assert len(parts) == 1  # all fit under the limit, stays together
    assert "One." in parts[0] and "Three." in parts[0]


def test_split_at_sentences_respects_max_tokens():
    text = "First sentence here. Second sentence here. Third sentence here."
    parts = split_at_sentences(text, max_tokens=4)
    assert len(parts) > 1  # forced to split
    assert all(p.strip() for p in parts)  # no empty fragments


def test_split_at_sentences_no_punctuation():
    assert split_at_sentences("no sentence boundary here", max_tokens=10) == [
        "no sentence boundary here"
    ]


def test_extract_last_tokens_short_text_returned_whole():
    assert extract_last_tokens("short", num_tokens=50) == "short"


def test_extract_last_tokens_truncates_long_text():
    text = " ".join(f"word{i}" for i in range(200))
    tail = extract_last_tokens(text, num_tokens=5)
    assert tail.endswith("word199")
    assert len(tail) < len(text)


def test_add_token_overlap_first_chunk_unchanged():
    chunks = [
        Chunk(timestamp="00:00:01", text="alpha beta gamma", index=0),
        Chunk(timestamp="00:00:02", text="delta epsilon", index=1),
    ]
    out = add_token_overlap(chunks, overlap_tokens=2)

    # First chunk is untouched.
    assert out[0].text == "alpha beta gamma"
    # Second chunk's embedded text carries overlap from the previous chunk...
    assert "delta epsilon" in out[1].text
    assert out[1].text != "delta epsilon"
    # ...but its citation text stays clean.
    assert out[1].source_text == "delta epsilon"


def test_add_token_overlap_single_chunk_noop():
    chunks = [Chunk(timestamp="00:00:01", text="only one", index=0)]
    assert add_token_overlap(chunks, overlap_tokens=5) == chunks


def _write_transcript(tmp_path):
    content = (
        "00:00:05\n"
        "The first segment of speech.\n"
        "\n"
        "00:01:10\n"
        "The second segment of speech.\n"
    )
    p = tmp_path / "transcript.txt"
    p.write_text(content, encoding="utf-8")
    return p


def test_parse_transcript_extracts_timestamps_and_text(tmp_path):
    chunks = parse_transcript(_write_transcript(tmp_path))

    assert len(chunks) == 2
    assert chunks[0].timestamp == "00:00:05"
    assert chunks[1].timestamp == "00:01:10"
    assert "first segment" in chunks[0].source_text
    assert "second segment" in chunks[1].source_text


def test_parse_transcript_indices_are_sequential(tmp_path):
    chunks = parse_transcript(_write_transcript(tmp_path))
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_parse_transcript_source_text_has_no_overlap(tmp_path):
    chunks = parse_transcript(_write_transcript(tmp_path))
    # Citation text never bleeds across the timestamp boundary.
    assert chunks[1].source_text == "The second segment of speech."


def test_parse_transcript_missing_file_raises(tmp_path):
    try:
        parse_transcript(tmp_path / "nope.txt")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")
