from __future__ import annotations

import re
from typing import List, Tuple


CHAPTER_PATTERN = re.compile(
    r"^(chapter|chap\\.?|part)\\s+([0-9]+|[ivxlcdm]+)\\b.*",
    re.IGNORECASE,
)


def split_chapters(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    indices: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        if CHAPTER_PATTERN.match(line.strip()):
            indices.append((i, line.strip()))

    if not indices:
        return [("Full Text", text)]

    chapters: List[Tuple[str, str]] = []
    for idx, (start, title) in enumerate(indices):
        end = indices[idx + 1][0] if idx + 1 < len(indices) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        chapters.append((title, body))

    return chapters
