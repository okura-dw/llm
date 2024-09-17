import dataclasses
import os
import re

import demucs.separate
import faster_whisper
import faster_whisper.audio
import faster_whisper.tokenizer
import faster_whisper.vad
import pydub
import streamlit

from llm_clients import logger, types

SAMPLE_RATE = 16_000


@dataclasses.dataclass
class SplittedFile:
    fname: str
    start: float
    end: float


def _vocal_extract(audio_path: str, dirname: str) -> str:
    audio_fname, _ = os.path.splitext(os.path.basename(audio_path))
    vocal_path = f"{dirname}/htdemucs/{audio_fname}/vocals.wav"
    if not os.path.exists(vocal_path):
        demucs.separate.main(["--two-stems", "vocals", audio_path, "-o", dirname])
    return vocal_path


@streamlit.cache_resource(show_spinner=False)
def _vad_split(audio_path: str, vocal_path: str) -> list[SplittedFile]:
    dir_path, _ = os.path.splitext(audio_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    timestamps = faster_whisper.vad.get_speech_timestamps(
        faster_whisper.audio.decode_audio(vocal_path, sampling_rate=SAMPLE_RATE),  # pyright: ignore
        vad_options=faster_whisper.vad.VadOptions(),
    )

    audio: pydub.AudioSegment = pydub.AudioSegment.from_file(audio_path)
    ret: list[SplittedFile] = []
    for t in timestamps:
        logger.logger.debug(f"split: {t['start']} -> {t['end']}")
        start = t["start"] / SAMPLE_RATE
        end = t["end"] / SAMPLE_RATE
        fname = f"{dir_path}/{start}-{end}.mp3"

        splitted_audio = audio[start * 1000 : end * 1000]
        splitted_audio.export(fname, "mp3")  # pyright: ignore
        ret.append(SplittedFile(fname=fname, start=start, end=end))

    return ret


@streamlit.cache_resource(show_spinner=False)
def _transcribe(
    audio_path: str, vocal_path: str, regex: str, model_name: str
) -> list[types.Transcript]:
    progress_bar = streamlit.progress(0)

    model = faster_whisper.WhisperModel(model_name, compute_type="float32")
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

    splitted_files = _vad_split(audio_path, vocal_path)
    transcipts: list[types.Transcript] = []
    for i, f in enumerate(splitted_files):
        segments, _ = model.transcribe(
            f.fname,
            temperature=0,
            beam_size=10,
            suppress_tokens=tokens,
            word_timestamps=True,
            condition_on_previous_text=False,
            no_repeat_ngram_size=10,
        )

        for segment in segments:
            transcipts.append(
                types.Transcript(
                    start=round(segment.start + f.start, 3),
                    end=round(segment.end + f.start, 3),
                    text="".join(pattern.findall(segment.text.strip())),
                )
            )

        progress_bar.progress((i + 1) / len(splitted_files))
    progress_bar.empty()

    return transcipts


class Whisper:
    def __init__(self, regex: str = "[a-zA-Zぁ-んァ-ン！？ ]+"):
        self.regex = regex

    def transcribe(
        self,
        audio_path: str,
        use_vocal: bool = False,
        vocal_dirname: str = "./",
        model_name: str = "large-v2",
    ) -> list[types.Transcript]:
        vocal_path = _vocal_extract(audio_path, vocal_dirname)
        if use_vocal:
            audio_path = vocal_path
        return _transcribe(audio_path, vocal_path, self.regex, model_name)
