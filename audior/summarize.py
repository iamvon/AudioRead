from __future__ import annotations

from typing import List

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


def _call_llm(client: OpenAI, model: str, prompt: str, text: str, max_output_tokens: int) -> str:
    response = client.responses.create(
        model=model,
        input=f"{prompt}\n\n<text>\n{text}\n</text>",
        max_output_tokens=max_output_tokens,
    )
    return response.output_text.strip()


def map_summaries(client: OpenAI, model: str, chunks: List[str], max_output_tokens: int) -> List[str]:
    summaries = []
    for chunk in chunks:
        summaries.append(_call_llm(client, model, MAP_PROMPT, chunk, max_output_tokens))
    return summaries


def reduce_summaries(client: OpenAI, model: str, summaries: List[str], max_output_tokens: int) -> str:
    combined = "\n\n".join(summaries)
    return _call_llm(client, model, REDUCE_PROMPT, combined, max_output_tokens)


def budgeted_reduce(client: OpenAI, model: str, summaries: List[str], max_output_tokens: int, max_input_tokens: int) -> str:
    combined = "\n\n".join(summaries)
    if count_tokens(combined, model) <= max_input_tokens:
        return reduce_summaries(client, model, summaries, max_output_tokens)

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

    reduced = [reduce_summaries(client, model, batch, max_output_tokens) for batch in batches]
    if len(reduced) == 1:
        return reduced[0]
    return budgeted_reduce(client, model, reduced, max_output_tokens, max_input_tokens)
