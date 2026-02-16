import re
from typing import Iterable, List


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dehyphenate(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    return parts


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_list(items: Iterable[str], max_chars: int) -> List[str]:
    chunks: List[str] = []
    current = ""
    for item in items:
        if not current:
            current = item
            continue
        if len(current) + 1 + len(item) <= max_chars:
            current = f"{current} {item}"
        else:
            chunks.append(current)
            current = item
    if current:
        chunks.append(current)
    return chunks


def sanitize_for_tts(text: str) -> str:
    # Remove markdown headings and bullets, and common outline markers.
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"^[\-\*•]+\s*", "", line)
        line = re.sub(r"^[IVXLCDM]+\.\s+", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^[0-9]+\.\s+", "", line)
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = normalize_whitespace(cleaned)
    return cleaned
