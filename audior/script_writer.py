from __future__ import annotations

from openai import OpenAI


STYLE_PROMPTS = {
    "narration": "Write a clear, engaging narration for general listeners.",
    "podcast": "Write a two-speaker podcast dialogue. Use Speaker 1 and Speaker 2 labels.",
    "lecture": "Write a single-speaker lecture script for students.",
    "summary": "Write a concise spoken summary.",
}


def generate_script(client: OpenAI, model: str, outline: str, style: str, max_output_tokens: int) -> str:
    style_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["narration"])
    response = client.responses.create(
        model=model,
        input=(
            "You are writing a spoken script based on an outline. "
            "Keep it coherent and easy to follow.\n\n"
            f"Style: {style_prompt}\n\n"
            f"<outline>\n{outline}\n</outline>"
        ),
        max_output_tokens=max_output_tokens,
    )
    return response.output_text.strip()
