from __future__ import annotations

import os
from pathlib import Path

import typer

from .config import AudioReadConfig
from .pipeline import run_pipeline
from dotenv import load_dotenv

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """AudioRead CLI."""
    return


@app.command()
def convert(
    pdf: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(Path("outputs"), help="Output directory"),
    text_model: str = typer.Option("gpt-4o-mini", help="OpenAI text model"),
    tts_model: str = typer.Option("gpt-4o-mini-tts", help="OpenAI TTS model"),
    voice: str = typer.Option("alloy", help="TTS voice"),
    style: str = typer.Option("narration", help="Script style"),
    max_input_tokens: int = typer.Option(12000, help="Max tokens per chunk"),
    overlap_tokens: int = typer.Option(200, help="Overlap tokens between chunks"),
    map_summary_tokens: int = typer.Option(800, help="Max output tokens per chunk summary"),
    reduce_summary_tokens: int = typer.Option(1200, help="Max output tokens for combined outline"),
    tts_max_chars: int = typer.Option(3500, help="Max chars per TTS call"),
    no_clean: bool = typer.Option(False, help="Disable text cleanup"),
    keep_headers: bool = typer.Option(False, help="Disable repeated header/footer removal"),
    api_key: str = typer.Option(None, help="OpenAI API key (or set OPENAI_API_KEY)")
) -> None:
    load_dotenv()
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise typer.BadParameter("Missing OpenAI API key. Set OPENAI_API_KEY or pass --api-key")

    output_dir = out / pdf.stem

    config = AudioReadConfig(
        text_model=text_model,
        tts_model=tts_model,
        voice=voice,
        style=style,
        max_input_tokens=max_input_tokens,
        overlap_tokens=overlap_tokens,
        map_summary_tokens=map_summary_tokens,
        reduce_summary_tokens=reduce_summary_tokens,
        tts_max_chars=tts_max_chars,
        clean_text=not no_clean,
        remove_repeated_lines=not keep_headers,
    )

    results = run_pipeline(pdf, output_dir, config, key)
    typer.echo("Done")
    for name, path in results.items():
        typer.echo(f"{name}: {path}")


if __name__ == "__main__":
    app()
