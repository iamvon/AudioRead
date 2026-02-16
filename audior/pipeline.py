from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from openai import OpenAI
from rich.progress import Progress

from .chunker import chunk_text
from .config import AudioReadConfig
from .pdf_extract import extract_pages, remove_repeated_page_lines, clean_text
from .summarize import map_summaries, budgeted_reduce
from .script_writer import generate_script
from .tts import script_to_audio_chunks, concat_mp3
from .utils import normalize_whitespace


def run_pipeline(
    pdf_path: Path,
    output_dir: Path,
    config: AudioReadConfig,
    api_key: str,
) -> Dict[str, str]:
    client = OpenAI(api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("Extracting PDF", total=1)
        pages = extract_pages(str(pdf_path))
        progress.advance(task)

        if config.remove_repeated_lines:
            task = progress.add_task("Removing headers/footers", total=1)
            pages = remove_repeated_page_lines(pages, ratio=config.repeated_line_ratio)
            progress.advance(task)

        task = progress.add_task("Cleaning text", total=1)
        if config.clean_text:
            pages = [clean_text(page) for page in pages]
        text = normalize_whitespace("\n\n".join(pages))
        progress.advance(task)

        (output_dir / "input.txt").write_text(text)

        task = progress.add_task("Chunking", total=1)
        chunks = chunk_text(text, config.max_input_tokens, config.overlap_tokens, config.text_model)
        progress.advance(task)

        summaries = []
        task = progress.add_task("Summarizing", total=len(chunks))
        for chunk in chunks:
            summaries.extend(map_summaries(client, config.text_model, [chunk], config.map_summary_tokens))
            progress.advance(task)

        (output_dir / "summaries.txt").write_text("\n\n".join(summaries))

        task = progress.add_task("Reducing summaries", total=1)
        outline = budgeted_reduce(
            client,
            config.text_model,
            summaries,
            config.reduce_summary_tokens,
            config.max_input_tokens,
        )
        progress.advance(task)

        (output_dir / "outline.txt").write_text(outline)

        task = progress.add_task("Writing script", total=1)
        script = generate_script(client, config.text_model, outline, config.style, max_output_tokens=2000)
        progress.advance(task)

        (output_dir / "script.txt").write_text(script)

        task = progress.add_task("Text-to-speech", total=1)
        chunks_dir = output_dir / "audio_chunks"
        audio_chunks = script_to_audio_chunks(
            client,
            config.tts_model,
            config.voice,
            script,
            config.tts_max_chars,
            chunks_dir,
        )
        progress.advance(task)

        task = progress.add_task("Merging audio", total=1)
        audio_path = output_dir / "output.mp3"
        concat_mp3(audio_chunks, audio_path)
        progress.advance(task)

    meta = {
        "input": str(pdf_path),
        "output_dir": str(output_dir),
        "chunks": len(chunks),
        "audio_chunks": len(audio_chunks),
        "text_model": config.text_model,
        "tts_model": config.tts_model,
        "voice": config.voice,
        "style": config.style,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return {
        "audio": str(audio_path),
        "script": str(output_dir / "script.txt"),
        "outline": str(output_dir / "outline.txt"),
        "summaries": str(output_dir / "summaries.txt"),
        "input_text": str(output_dir / "input.txt"),
        "meta": str(output_dir / "meta.json"),
    }
