from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Callable, List, Optional

from openai import OpenAI

from .utils import split_paragraphs, split_sentences, chunk_list


def _tts_bytes(client: OpenAI, model: str, voice: str, text: str) -> bytes:
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
    ) as response:
        with io.BytesIO() as buffer:
            for chunk in response.iter_bytes():
                buffer.write(chunk)
            return buffer.getvalue()


def script_to_audio_chunks(
    client: OpenAI,
    model: str,
    voice: str,
    script: str,
    max_chars: int,
    output_dir: Path,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paragraphs = split_paragraphs(script)
    pieces: List[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        sentences = split_sentences(para)
        pieces.extend(chunk_list(sentences, max_chars))

    paths: List[Path] = []
    total = len(pieces)
    if on_progress:
        on_progress(0, total)
    for i, piece in enumerate(pieces, start=1):
        audio_bytes = _tts_bytes(client, model, voice, piece)
        chunk_path = output_dir / f"chunk_{i:04d}.mp3"
        with open(chunk_path, "wb") as f:
            f.write(audio_bytes)
        paths.append(chunk_path)
        if on_progress:
            on_progress(i, total)
    return paths


def concat_mp3(chunks: List[Path], output_path: Path) -> None:
    if not chunks:
        raise ValueError("No audio chunks to merge")

    with open(output_path, "wb") as out_f:
        for chunk in chunks:
            with open(chunk, "rb") as in_f:
                out_f.write(in_f.read())
