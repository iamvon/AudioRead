# AudioRead (CLI)

Convert a PDF into a narrated audio file using a chunked LLM pipeline that avoids context window limits.

## Acknowledgements

This project is inspired by PDF2Audio:

```
https://github.com/lamm-mit/PDF2Audio
```

## Install (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

```bash
cp .env.example .env
# edit .env to set OPENAI_API_KEY
audior convert /path/to/book.pdf
```

Options:
- `--style narration|podcast|lecture|summary`
- `--mode summary|full|chapter`
- `--text-model gpt-4o-mini`
- `--tts-model gpt-4o-mini-tts`
- `--voice alloy`
- `--max-input-tokens 12000`
- `--overlap-tokens 200`
- `--tts-max-chars 3500`

Outputs are placed under `outputs/<pdf_name>/`.

## Contributing

- Create an issue to report bugs or suggest features.
- Open a PR for improvements or fixes.

## License

MIT

## Notes
- Audio chunks are concatenated as MP3 bytes. Most players handle this fine.
- For higher fidelity merges, we can add optional ffmpeg/pydub later.
- `script_tts.txt` is a cleaned version used to avoid speaking markdown symbols.

## Demo

Summary audio sample:
[docs/demo-output.mp3](https://github.com/iamvon/AudioRead/tree/main/docs/demo-output.mp3)

Source book:
[Great Physicists: The Life and Times of Leading Physicists from Galileo to Hawking](https://www.amazon.com/Great-Physicists-Leading-Galileo-Hawking/dp/0195137485)
