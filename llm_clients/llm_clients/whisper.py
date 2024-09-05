import re

import faster_whisper
import faster_whisper.tokenizer
import streamlit

from llm_clients import types


@streamlit.cache_resource(show_spinner=False)
def _transcribe(file_path: str, regex: str) -> list[types.Transcript]:
    model = faster_whisper.WhisperModel("large-v3", compute_type="float32")
    tokenizer = faster_whisper.tokenizer.Tokenizer(
        model.hf_tokenizer, multilingual=True, task="transcribe", language="ja"
    )

    pattern = re.compile(regex)
    tokens = (
        [-1]
        + [
            i
            for i in range(tokenizer.eot)
            if not any(pattern.fullmatch(c) for c in tokenizer.decode([i]))
        ]
        + [50363, 50364]
    )

    transcipts: list[types.Transcript] = []
    segments, _ = model.transcribe(
        file_path,
        temperature=0,
        beam_size=10,
        suppress_tokens=tokens,
        word_timestamps=True,
        condition_on_previous_text=False,
        no_repeat_ngram_size=10,
        vad_filter=True,
    )

    for segment in segments:
        transcipts.append(
            types.Transcript(
                start=round(segment.start, 3),
                end=round(segment.end, 3),
                text="".join(pattern.findall(segment.text.strip())),
            )
        )

    return transcipts


class Whisper:
    def __init__(self, regex: str = "[a-zA-Zぁ-んァ-ン！？ ]+"):
        self.regex = regex

    def transcribe(self, file_path: str) -> list[types.Transcript]:
        return _transcribe(file_path, self.regex)
