import math
import re

import llm_clients.types
import streamlit

import sync_lyrics
from sync_lyrics import entities, types

ROOT_DIR = "downloads"


def sec2str(second: int | float | None) -> str:
    """秒をSRT形式の文字列(hh:mm:ss,xxx)に変換して返す

    Parameters
    ----------
    second
    """
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
    lyrics: types.Lyrics, prev_lyrics: types.Lyrics | None, linebreak: str = "<br>"
) -> str:
    """streamlit.write で表示するために html でSRT形式を記述して返す

    Parameters
    ----------
    lyrics
        表示したいタイムスタンプ付きの歌詞
    prev_lyrics
        複数段階で処理を行う場合、前段階の処理結果の歌詞
        指定した場合、前回との差分で新たにタイムスタンプが付与された部分が青く表示される
    linebreak
        改行記号
    """
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
    """SRT形式で表示する

    lyrics_list
        表示したいタイムスタンプ付きの歌詞のリスト
    prev_lyrics_list
        複数段階で処理を行う場合、前段階の処理結果の歌詞のリスト
        指定した場合、前回との差分で新たにタイムスタンプが付与された部分が青く表示される
    """
    if prev_lyrics_list is None:
        texts_str = "<br>".join(output_srt(l, None) for l in lyrics_list)
    else:
        texts_str = "<br>".join(
            output_srt(l, prev_l) for l, prev_l in zip(lyrics_list, prev_lyrics_list)
        )
    texts_str = f'<div style="max-width: 600px; max-height: 600px; overflow: auto; overflow-y: scroll;"><span style="white-space: nowrap; font-family: monospace; font-size: 90%;">{texts_str}</br></span></div>'
    streamlit.write(texts_str, unsafe_allow_html=True)


def _display(content_id: str, lyrics: str, use_vocal: bool):
    """結果の表示

    Parameters
    ----------
    content_id
    lyrics
    use_vocal
        ボーカル抽出した音声ファイルを入力するかどうか
    """
    i = 0
    lyrics_list: list[types.Lyrics] = []
    for lyric in lyrics.split("\n"):
        if re.sub(r"^\s+$", "", lyric) != "":
            lyrics_list.append(
                types.Lyrics(start_second=None, end_second=None, lyrics=lyric, lyrics_row=i)
            )

    output, alignment, fee = streamlit.tabs(["出力", "位置合わせ結果", "料金"])

    with output:
        if f"output_{content_id}_{use_vocal}" in streamlit.session_state:
            write_srt(streamlit.session_state[f"output_{content_id}_{use_vocal}"])

    with alignment:
        if f"alignment_{content_id}_{use_vocal}" in streamlit.session_state:
            with streamlit.expander("プロンプト"):
                messages: list[llm_clients.types.TupleMessage] = streamlit.session_state[
                    f"alignment_{content_id}_{use_vocal}"
                ][1]
                for message in messages:
                    if not message.role == "user" or isinstance(message.content, str):
                        streamlit.code(message.content)
            write_srt(
                streamlit.session_state[f"alignment_{content_id}_{use_vocal}"][0], lyrics_list
            )

    with fee:
        if f"fee_{content_id}_{use_vocal}" in streamlit.session_state:
            usd_jpy = float(streamlit.text_input("USD/JPY", value=150))  # pyright: ignore

            table_str = "処理 | 料金\n" "--- | ---\n"
            sum_fee = 0
            for process, usd_fee in streamlit.session_state[
                f"fee_{content_id}_{use_vocal}"
            ].items():
                jpy_fee = usd_fee * usd_jpy
                sum_fee += jpy_fee
                table_str += f"{process} | ¥{round(jpy_fee,3)}\n"
            table_str += f"計 | ¥{round(sum_fee,3)}\n"
            streamlit.write(table_str)


def _run(api_key: str, model: str, content_id: str, lyrics: str, use_vocal: bool):
    """メインの処理
    結果は streamlit.session_state に格納される

    Parameters
    ----------
    api_key
    model
    content_id
    lyrics
    use_vocal
        ボーカル抽出した音声ファイルを入力するかどうか
    """
    fee_dict: dict[str, float] = {}
    with streamlit.spinner("動画をダウンロード中..."):
        pass

    audio_path = f"{ROOT_DIR}/music/{content_id}.mp3"
    streamlit.audio(audio_path)
    if use_vocal:
        audio_path = llm_clients.vocal_extract(audio_path, ROOT_DIR)

    with streamlit.spinner("位置合わせ中..."):
        alignment_audio = entities.AlignmentWithAudio(api_key=api_key, model=model)
        alignment_lyrics = alignment_audio.run(lyrics, audio_path)
        alignment_messages = alignment_audio.messages
        fee_dict["位置合わせ"] = alignment_audio.llm.fee

    streamlit.session_state[f"alignment_{content_id}_{use_vocal}"] = (
        alignment_lyrics,
        alignment_messages,
    )

    streamlit.session_state[f"output_{content_id}_{use_vocal}"] = alignment_lyrics
    streamlit.session_state[f"fee_{content_id}_{use_vocal}"] = fee_dict


if __name__ == "__main__":
    streamlit.set_page_config(layout="wide")

    with streamlit.sidebar:
        logger = sync_lyrics.get_logger()

        api_key = streamlit.text_input("API key")
        model = streamlit.selectbox(
            "model",
            ["gemini-1.5-pro-exp-0827", "gemini-1.5-flash-8b-exp-0827"],
        )

        use_vocal = streamlit.checkbox("ボーカル抽出")

    content_id = streamlit.text_input("曲名")
    try:
        with open(f"{ROOT_DIR}/lyrics/{content_id}.txt") as f:
            lyrics = f.read()
        with streamlit.expander("歌詞"):
            streamlit.code(lyrics)
    except:
        streamlit.warning(f"「{content_id}」の歌詞が、lyrics/ フォルダ以下に見つかりませんでした")
        lyrics = ""

    if streamlit.button("run"):
        _run(api_key, model, content_id, lyrics, use_vocal)

    _display(content_id, lyrics, use_vocal)
