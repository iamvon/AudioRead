from pydantic import BaseModel, Field


class AudioReadConfig(BaseModel):
    text_model: str = Field(default="gpt-4o-mini")
    tts_model: str = Field(default="gpt-4o-mini-tts")
    voice: str = Field(default="alloy")

    max_input_tokens: int = Field(default=12000)
    map_summary_tokens: int = Field(default=800)
    reduce_summary_tokens: int = Field(default=1200)
    overlap_tokens: int = Field(default=200)

    tts_max_chars: int = Field(default=3500)
    style: str = Field(default="narration")

    clean_text: bool = Field(default=True)
    remove_repeated_lines: bool = Field(default=True)
    repeated_line_ratio: float = Field(default=0.25)
