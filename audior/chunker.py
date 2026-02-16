from __future__ import annotations

from typing import List

import tiktoken

from .utils import split_paragraphs, split_sentences


def get_tokenizer(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str) -> int:
    enc = get_tokenizer(model)
    return len(enc.encode(text))


def chunk_text(text: str, max_tokens: int, overlap_tokens: int, model: str) -> List[str]:
    paragraphs = split_paragraphs(text)
    enc = get_tokenizer(model)

    chunks: List[str] = []
    current_tokens = 0
    current_parts: List[str] = []

    def flush():
        nonlocal current_tokens, current_parts
        if not current_parts:
            return
        chunk = "\n\n".join(current_parts).strip()
        if chunk:
            chunks.append(chunk)
        current_tokens = 0
        current_parts = []

    for para in paragraphs:
        para_tokens = len(enc.encode(para))
        if para_tokens > max_tokens:
            sentences = split_sentences(para)
            for sentence in sentences:
                sent_tokens = len(enc.encode(sentence))
                if sent_tokens > max_tokens:
                    pieces = [sentence[i : i + 800] for i in range(0, len(sentence), 800)]
                    for piece in pieces:
                        piece_tokens = len(enc.encode(piece))
                        if current_tokens + piece_tokens > max_tokens:
                            flush()
                        current_parts.append(piece)
                        current_tokens += piece_tokens
                    continue
                if current_tokens + sent_tokens > max_tokens:
                    flush()
                current_parts.append(sentence)
                current_tokens += sent_tokens
            continue

        if current_tokens + para_tokens > max_tokens:
            flush()
        current_parts.append(para)
        current_tokens += para_tokens

    flush()

    if overlap_tokens <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: List[str] = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
            continue
        prev = chunks[i - 1]
        prev_tokens = enc.encode(prev)
        overlap = enc.decode(prev_tokens[-overlap_tokens:]) if len(prev_tokens) > overlap_tokens else prev
        overlapped.append(f"{overlap}\n\n{chunk}")
    return overlapped
