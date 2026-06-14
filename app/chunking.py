"""Parse transcript by timestamp, split large segments at sentence boundaries, add token overlap."""

import re
from pathlib import Path
from typing import List

import tiktoken

from app.config import settings


class Chunk:
    """A transcript chunk with timestamp and text."""

    def __init__(self, timestamp: str, text: str, index: int):
        self.timestamp = timestamp
        self.text = text
        self.index = index

    def __repr__(self) -> str:
        return f"Chunk(timestamp={self.timestamp}, length={len(self.text)})"


# Initialize tokenizer once (cl100k_base works for most modern models)
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    # Fallback to a simple approximation if tiktoken fails
    tokenizer = None


def count_tokens(text: str) -> int:
    """
    Count tokens accurately using tiktoken.
    Fallback to rough approximation if tokenizer unavailable.
    """
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4


def split_at_sentences(text: str, max_tokens: int) -> List[str]:
    """
    Split text into chunks at sentence boundaries, respecting max_tokens.

    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk

    Returns:
        List of text chunks
    """
    if not text:
        return []

    # Simple sentence splitting: . ! ?
    sentence_endings = re.findall(r"[^.!?]+[.!?]+", text)

    if not sentence_endings:
        # No sentence boundaries found, fall back to character split
        return [text]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentence_endings:
        sentence_tokens = count_tokens(sentence)

        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        else:
            # Start new chunk
            if current_chunk:
                chunks.append("".join(current_chunk).strip())
            current_chunk = [sentence]
            current_tokens = sentence_tokens

    # Add final chunk
    if current_chunk:
        chunks.append("".join(current_chunk).strip())

    return chunks if chunks else [text]


def extract_last_tokens(text: str, num_tokens: int = 50) -> str:
    """
    Extract the last N tokens from text.

    Args:
        text: Source text
        num_tokens: Number of tokens to extract

    Returns:
        Last N tokens as text, or empty if not enough tokens
    """
    if not tokenizer:
        # Fallback: approximate tokens by characters
        char_count = num_tokens * 4
        if len(text) <= char_count:
            return text
        return text[-char_count:].strip()

    # Get token IDs
    tokens = tokenizer.encode(text)

    if len(tokens) <= num_tokens:
        return text

    # Get last N tokens and decode back
    last_tokens = tokens[-num_tokens:]
    return tokenizer.decode(last_tokens).strip()


def add_token_overlap(chunks: List[Chunk], overlap_tokens: int = 50) -> List[Chunk]:
    """
    Add token overlap between adjacent chunks.

    Takes the last N tokens from chunk[i] and prepends to chunk[i+1].

    Args:
        chunks: List of Chunk objects
        overlap_tokens: Number of tokens to overlap

    Returns:
        New list of chunks with overlap applied
    """
    if len(chunks) <= 1:
        return chunks

    overlapped_chunks = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            # First chunk stays as-is
            overlapped_chunks.append(chunk)
        else:
            # Get overlap from previous chunk
            prev_chunk = chunks[i - 1]
            overlap_text = extract_last_tokens(prev_chunk.text, overlap_tokens)

            if overlap_text:
                # Prepend overlap to current chunk
                new_text = f"{overlap_text} {chunk.text}".strip()
                overlapped_chunk = Chunk(
                    timestamp=chunk.timestamp, text=new_text, index=chunk.index
                )
                overlapped_chunks.append(overlapped_chunk)
            else:
                # No overlap available, keep original
                overlapped_chunks.append(chunk)

    return overlapped_chunks


def parse_transcript(
    path: str | Path = settings.TRANSCRIPT_PATH,
    max_tokens: int = 500,
    overlap_tokens: int = settings.OVERLAP_TOKENS,
) -> List[Chunk]:
    """
    Parse the transcript into timestamp-aware chunks with token overlap.

    This strategy:
    1. Parses by timestamp boundaries (preserves natural segments)
    2. Splits large segments at sentence boundaries (keeps same timestamp)
    3. Keeps small segments as-is (no merging)
    4. Adds N tokens of overlap between adjacent chunks

    Args:
        path: Path to the transcript file
        max_tokens: Maximum tokens per chunk (default: 500)
        overlap_tokens: Number of tokens to overlap (default: 50)

    Returns:
        List of Chunk objects with timestamp, text, and sequential index
    """
    transcript_path = Path(path)
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    # Parse into timestamp segments
    segments = []
    i = 0
    timestamp_pattern = re.compile(r"^\d{2}:\d{2}:\d{2}$")

    while i < len(lines):
        line = lines[i]

        if timestamp_pattern.match(line):
            timestamp = line
            i += 1
            content_parts = []

            # Collect all lines until the next timestamp
            while i < len(lines) and not timestamp_pattern.match(lines[i]):
                if lines[i]:  # Skip empty lines
                    content_parts.append(lines[i])
                i += 1

            text = " ".join(content_parts)
            if text:
                segments.append({"timestamp": timestamp, "text": text})
        else:
            i += 1

    # Split large segments and build chunks
    all_chunks = []
    chunk_index = 0

    for segment in segments:
        timestamp = segment["timestamp"]
        text = segment["text"]
        token_count = count_tokens(text)

        if token_count > max_tokens:
            # Split large segment at sentence boundaries
            split_texts = split_at_sentences(text, max_tokens)

            for split_text in split_texts:
                all_chunks.append(
                    Chunk(timestamp=timestamp, text=split_text, index=chunk_index)
                )
                chunk_index += 1
        else:
            # Keep segment as-is
            all_chunks.append(Chunk(timestamp=timestamp, text=text, index=chunk_index))
            chunk_index += 1

    # Add token overlap between adjacent chunks
    final_chunks = add_token_overlap(all_chunks, overlap_tokens)

    return final_chunks


def chunk_texts(chunks: List[Chunk]) -> List[str]:
    """Extract just the text from chunks for embedding."""
    return [chunk.text for chunk in chunks]


if __name__ == "__main__":
    path = "data/transcript.txt"
    chunks = chunk_texts(parse_transcript(path))
    print(chunks)
