from __future__ import annotations

from typing import Callable, List, Optional

from openai import OpenAI

from .chunker import count_tokens


MAP_PROMPT = """You are summarizing a chunk of a larger document.
Extract key facts, terms, and claims. Keep it concise and factual.
Return bullet points only.
"""

REDUCE_PROMPT = """You are combining summaries from a long document.
Produce a structured outline with sections and key bullets.
Keep it concise and factual.
"""

CHAPTER_MAP_PROMPT = """You are summarizing part of a chapter from a book.
Capture key events, people, claims, and explanations. Be detailed.
Write in plain sentences. Avoid markdown, bullets, or numbering.
"""

CHAPTER_REDUCE_PROMPT = """You are writing a detailed chapter summary.
Include key events, people, concepts, and conclusions.
Add 3 to 7 highlight insights as plain sentences.
Avoid markdown, bullets, or numbering.
"""


def _call_llm(client: OpenAI, model: str, prompt: str, text: str, max_output_tokens: int) -> str:
    response = client.responses.create(
        model=model,
        input=f"{prompt}\n\n<text>\n{text}\n</text>",
        max_output_tokens=max_output_tokens,
    )
    return response.output_text.strip()


def map_summaries(
    client: OpenAI,
    model: str,
    chunks: List[str],
    max_output_tokens: int,
    on_progress: Optional[Callable[[int, int], None]] = None,
    prompt: str = MAP_PROMPT,
) -> List[str]:
    summaries = []
    total = len(chunks)
    for i, chunk in enumerate(chunks, start=1):
        summaries.append(_call_llm(client, model, prompt, chunk, max_output_tokens))
        if on_progress:
            on_progress(i, total)
    return summaries


def reduce_summaries(
    client: OpenAI,
    model: str,
    summaries: List[str],
    max_output_tokens: int,
    prompt: str = REDUCE_PROMPT,
) -> str:
    combined = "\n\n".join(summaries)
    return _call_llm(client, model, prompt, combined, max_output_tokens)


def budgeted_reduce(
    client: OpenAI,
    model: str,
    summaries: List[str],
    max_output_tokens: int,
    max_input_tokens: int,
    prompt: str = REDUCE_PROMPT,
) -> str:
    combined = "\n\n".join(summaries)
    if count_tokens(combined, model) <= max_input_tokens:
        return reduce_summaries(client, model, summaries, max_output_tokens, prompt=prompt)

    batches: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0
    for summary in summaries:
        tokens = count_tokens(summary, model)
        if current_tokens + tokens > max_input_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(summary)
        current_tokens += tokens
    if current:
        batches.append(current)

    reduced = [reduce_summaries(client, model, batch, max_output_tokens, prompt=prompt) for batch in batches]
    if len(reduced) == 1:
        return reduced[0]
    return budgeted_reduce(client, model, reduced, max_output_tokens, max_input_tokens, prompt=prompt)


def summarize_long_text(
    client: OpenAI,
    model: str,
    text: str,
    max_input_tokens: int,
    map_output_tokens: int,
    reduce_output_tokens: int,
    on_progress: Optional[Callable[[int, int], None]] = None,
    map_prompt: str = MAP_PROMPT,
    reduce_prompt: str = REDUCE_PROMPT,
) -> str:
    if count_tokens(text, model) <= max_input_tokens:
        return _call_llm(client, model, map_prompt, text, reduce_output_tokens)

    from .chunker import chunk_text

    chunks = chunk_text(text, max_input_tokens, overlap_tokens=200, model=model)
    summaries = map_summaries(client, model, chunks, map_output_tokens, on_progress=on_progress, prompt=map_prompt)
    return budgeted_reduce(
        client,
        model,
        summaries,
        reduce_output_tokens,
        max_input_tokens,
        prompt=reduce_prompt,
    )
