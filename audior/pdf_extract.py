from __future__ import annotations

from collections import Counter
from typing import List

from pypdf import PdfReader

from .utils import normalize_whitespace, dehyphenate


def extract_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages


def remove_repeated_page_lines(pages: List[str], ratio: float = 0.25) -> List[str]:
    if not pages:
        return pages

    line_counts: Counter[str] = Counter()
    page_lines = []
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        page_lines.append(lines)
        for line in set(lines):
            line_counts[line] += 1

    threshold = max(2, int(len(pages) * ratio))
    repeated = {line for line, count in line_counts.items() if count >= threshold and len(line) >= 6}

    cleaned_pages = []
    for lines in page_lines:
        kept = [line for line in lines if line not in repeated]
        cleaned_pages.append("\n".join(kept))
    return cleaned_pages


def clean_text(text: str) -> str:
    text = dehyphenate(text)
    text = normalize_whitespace(text)
    return text
