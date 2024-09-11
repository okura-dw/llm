import math
import re

import llm_clients
import llm_clients.types
import openai.types.chat
import streamlit

import vocacolle
from vocacolle import entities, types

ROOT_DIR = "downloads"
USDJPY = 142.16


def sec2str(second: int | float | None) -> str:
    if second is None:
        return "--:--:--,---"
    ms = str(second % 1)[2:5].ljust(3, "0")
    second = math.floor(second)
    minute = second // 60
    ss = str(second % 60).zfill(2)
    mm = str(minute % 60).zfill(2)
    hh = str(minute // 60).zfill(2)
    return f"{hh}:{mm}:{ss},{ms}"


def output_srt(
    lyrics: types.Lyrics, prev_lyrics: types.Lyrics | None, linebreak: str = "\n"
) -> str:
    srt_list: list[str] = []
    srt_list.append(str(lyrics.lyrics_row))
    if prev_lyrics is not None and prev_lyrics.start_second is None:
        srt_list.append(
            f'<b><font color="blue">{sec2str(lyrics.start_second)} --> {sec2str(lyrics.end_second)}</font></b>'
        )
    else:
        srt_list.append(f"{sec2str(lyrics.start_second)} --> {sec2str(lyrics.end_second)}")
    srt_list.append(lyrics.lyrics)
    return linebreak.join(srt_list)


def write_srt(lyrics_list: list[types.Lyrics], prev_lyrics_list: list[types.Lyrics] | None = None):
    if prev_lyrics_list is None:
        texts_str = "<br>".join(output_srt(l, None, linebreak="<br>") for l in lyrics_list)
    else:
        texts_str = "<br>".join(
            output_srt(l, prev_l, linebreak="<br>")
            for l, prev_l in zip(lyrics_list, prev_lyrics_list)
        )
    texts_str = f'<div style="max-width: 600px; max-height: 600px; overflow: auto; overflow-y: scroll;"><span style="white-space: nowrap; font-family: monospace; font-size: 90%;">{texts_str}</br></span></div>'
    streamlit.write(texts_str, unsafe_allow_html=True)


def _display(content_id: str, lyrics: str, is_audio_input: bool):
    i = 0
    lyrics_list: list[types.Lyrics] = []
    for lyric in lyrics.split("\n"):
        if re.sub(r"^\s+$", "", lyric) != "":
            lyrics_list.append(
                types.Lyrics(start_second=None, end_second=None, lyrics=lyric, lyrics_row=i)
            )

    if is_audio_input:
        output, alignment = streamlit.tabs(["出力", "位置合わせ結果"])
    else:
        output, transcript, alignment = streamlit.tabs(["出力", "文字起こし結果", "位置合わせ結果"])

    with output:
        if f"output_{content_id}" in streamlit.session_state:
            write_srt(streamlit.session_state[f"output_{content_id}"])

    if not is_audio_input:
        with transcript:  # pyright: ignore
            if f"transcripts_{content_id}" in streamlit.session_state:
                transcripts: list[llm_clients.types.Transcript] = streamlit.session_state[
                    f"transcripts_{content_id}"
                ]
                transcripts_str = "\n\n".join(f"{t.start} - {t.end}: {t.text}" for t in transcripts)
                streamlit.write(transcripts_str, unsafe_allow_html=True)

    with alignment:
        if f"alignment_{content_id}" in streamlit.session_state:
            with streamlit.expander("プロンプト"):
                messages: list[openai.types.chat.ChatCompletionMessageParam] = (
                    streamlit.session_state[f"alignment_{content_id}"][1]
                )
                for message in messages:
                    if message["role"] == "system" or (
                        message["role"] == "user" and isinstance(message["content"], str)
                    ):
                        streamlit.code(message["content"])
            write_srt(streamlit.session_state[f"alignment_{content_id}"][0], lyrics_list)


def _run(api_key: str, model: str, content_id: str, lyrics: str, is_audio_input: bool):
    with streamlit.spinner("動画をダウンロード中..."):
        pass

    audio_path = f"{ROOT_DIR}/{content_id}.wav"
    audio_path = "downloads/Kienai_natuno_kaori_Remix_free_ver.mp3"
    streamlit.audio(audio_path)

    if is_audio_input:
        with streamlit.spinner("位置合わせ中..."):
            alignment_audio = entities.AlignmentWithAudio(api_key=api_key, model=model)
            alignment_lyrics = alignment_audio.run(lyrics, audio_path)
            alignment_messages = alignment_audio.messages
            print(f"¥{alignment_audio.llm.fee * USDJPY}")

    else:
        with streamlit.spinner("文字起こし中..."):
            whisper = llm_clients.Whisper()
            transcripts = whisper.transcribe(audio_path)
            streamlit.session_state[f"transcripts_{content_id}"] = transcripts

        with streamlit.spinner("位置合わせ中..."):
            alignment = entities.Alignment(api_key=api_key, model=model)
            alignment_lyrics = alignment.run(lyrics, transcripts)
            alignment_messages = alignment.messages

    streamlit.session_state[f"alignment_{content_id}"] = alignment_lyrics, alignment_messages

    with streamlit.spinner("LLMで補間中..."):
        pass

    with streamlit.spinner("線形補間中..."):
        pass

    streamlit.session_state[f"output_{content_id}"] = alignment_lyrics


if __name__ == "__main__":
    streamlit.set_page_config(layout="wide")

    with streamlit.sidebar:
        logger = vocacolle.get_logger()

        api_key = streamlit.text_input("API key")
        model_name = streamlit.radio("model", ["OpenAI", "Gemini"])

        match model_name:
            case "OpenAI":
                model = streamlit.selectbox("model", ["gpt-4o-2024-08-06"])
                llm = llm_clients.OpenAI(api_key, model)
                is_audio_input = False
            case "Gemini":
                model = streamlit.selectbox(
                    "model",
                    [
                        "gemini-1.5-pro",
                        "gemini-1.5-flash",
                        "gemini-1.5-pro-exp-0827",
                        "gemini-1.5-flash-8b-exp-0827",
                    ],
                )
                llm = llm_clients.Gemini(api_key, model)
                is_audio_input = streamlit.checkbox("音声ファイルを入力する")

            case _:
                raise ValueError

    content_id = streamlit.text_input("content_id")
    lyrics = streamlit.text_area("歌詞")

    if streamlit.button("run"):
        _run(api_key, model, content_id, lyrics, is_audio_input)

    _display(content_id, lyrics, is_audio_input)
