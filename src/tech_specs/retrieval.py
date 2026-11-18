"""BM25 lexical retrieval over the structured JSONL corpus."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Okapi

# Keep identifiers like 0x3F, TIM2, GPIOA_MODER, R/W and 16-bit intact.
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[._/-][^\W_]+)*", re.UNICODE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
SEPARATOR_PATTERN = re.compile(r"[._/-]+")
REQUIRED_FIELDS = {"chunk_id", "document", "page", "content", "content_type"}


def tokenize(text: str) -> list[str]:
    """Normalize technical identifiers consistently in queries and chunks.

    Compound identifiers are indexed both intact and by their components.
    Example: '16-bit' -> ['16-bit', '16', 'bit'].
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = HTML_TAG_PATTERN.sub(" ", normalized)
    tokens = []

    for token in TOKEN_PATTERN.findall(normalized):
        tokens.append(token)
        if SEPARATOR_PATTERN.search(token):
            tokens.extend(
                part for part in SEPARATOR_PATTERN.split(token) if part
            )

    return tokens


def load_chunks(path: Path | str) -> list[dict]:
    """Read and validate a JSONL corpus, preserving document order."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus not found: {path}")

    chunks = []
    ids = set()

    with path.open("r", encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if not isinstance(chunk, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            missing = REQUIRED_FIELDS - chunk.keys()
            if missing:
                raise ValueError(
                    f"Line {line_number}: missing fields {sorted(missing)}"
                )
            if not isinstance(chunk["content"], str):
                raise ValueError(f"Line {line_number}: content must be a string")
            if not chunk["content"].strip():
                raise ValueError(f"Line {line_number}: content is empty")
            chunk_id = chunk["chunk_id"]
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"Line {line_number}: invalid chunk_id")
            if chunk_id in ids:
                raise ValueError(f"Duplicate chunk_id: {chunk_id}")
            ids.add(chunk_id)
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"Corpus is empty: {path}")
    return chunks



class BM25Retriever:
    """In-memory BM25 baseline."""

    def __init__(self, chunks: list[dict]):

        if not chunks:
            raise ValueError("At least one chunk is required")

        self.chunks = chunks

        # Prepare documents for BM25 indexing.
        tokenized_corpus = [
            tokenize(
                "\n".join([
                    chunk["document"],
                    chunk.get("section") or "",
                    chunk["content"],
                ])
            )
            for chunk in chunks
        ]

        if not any(tokenized_corpus):
            raise ValueError("No searchable tokens found")

        self.model = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
        document: str | None = None,
    ) -> list[dict]:
        """
        Search chunks using BM25.

        Always returns a list:
        - Ranked hits when matches exist.
        - An empty list when no matches exist.
        """

        # 1. Validate parameters
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        # 2. Tokenize the query
        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        # 3. Calculate BM25 scores for all chunks
        scores = self.model.get_scores(query_tokens)

        # 4. Keep positive scores and apply document filter
        candidates = [
            (float(score), idx)
            for idx, score in enumerate(scores)
            if score > 0
            and (
                document is None
                or self.chunks[idx]["document"] == document
            )
        ]

        # 5. Sort by relevance (highest score first)
        candidates.sort(
            key=lambda item: (
                -item[0],
                self.chunks[item[1]]["chunk_id"],
            )
        )

        # 6. Build the result list
        results = [
            {
                "rank": rank,
                "score": round(score, 6),
                "chunk": self.chunks[idx],
            }
            for rank, (score, idx) in enumerate(
                candidates[:top_k],
                start=1,
            )
        ]

        # 7. Always return a list
        return results
