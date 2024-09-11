import re

import llm_clients
import llm_clients.whisper
import openai.types.chat
import pydantic

from vocacolle import types

SYSTEM_PROMPT = """\
## 役割
あなたはプロの編集者です。ルールを遵守し、出力の形式に沿って「歌詞」と「歌の自動文字起こし結果」の位置合わせをしてください。
## ルール
- 全ての「歌詞」を一行ずつ出力してください。
- 「歌詞」と「歌の自動文字起こし結果」の文字の一致度が50%を超える箇所だけ位置合わせしてください。
- 「歌の自動文字起こし結果」には誤字があります。適宜誤字を推測して、文字が合っている位置に合わせてください。
- 「歌詞」には同じ文が何度も登場することがあります。「歌の自動文字起こし結果」の発話開始秒と「歌詞」のパラグラフも考慮して適切な位置に合わせてください。
- 「歌詞」と「歌の自動文字起こし結果」の改行位置は異なる場合があります。「歌詞」に準拠して位置合わせをしてください。
"""

INFORMATION_PROMPT = """\
「歌詞」
{lyrics}


「歌の自動文字起こし結果」
{transcripts}
"""


class Response(pydantic.BaseModel, frozen=True):
    lyrics_list: list[types.Lyrics]


class Alignment:
    def __init__(self, api_key: str, model: str):
        self.messages: list[openai.types.chat.ChatCompletionMessageParam] = []
        if model.startswith("gpt"):
            self.llm = llm_clients.OpenAI(api_key, model)
        elif model.startswith("gemini"):
            self.llm = llm_clients.Gemini(api_key, model)
        else:
            raise ValueError

    def run(self, lyrics: str, transcripts: list[llm_clients.Transcript]) -> list[types.Lyrics]:
        lyrics_list = [l for l in lyrics.split("\n")]

        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": INFORMATION_PROMPT.format(
                    lyrics=self._lyrics2str(lyrics_list),
                    transcripts=self._transcripts2str(transcripts),
                ),
            },
        ]

        response = self.llm.fetch(llm_clients.message2tuple(self.messages), Response)
        if response is None:
            raise ValueError

        return response.lyrics_list

    @staticmethod
    def _lyrics2str(lyrics: list[str]) -> str:
        i = 1
        lyrics_str = "行番号. 歌詞\n"
        for lyric in lyrics:
            if re.sub(r"^\s+$", "", lyric) != "":
                lyrics_str += f"{i}. {lyric}\n"
                i += 1
            else:
                lyrics_str += "\n"
        return lyrics_str

    @staticmethod
    def _transcripts2str(transcripts: list[llm_clients.Transcript]) -> str:
        transcripts_str = "行番号. 発話開始秒: 文字起こし内容\n"
        for i, t in enumerate(transcripts, start=1):
            transcripts_str += f"{i}. {t.start}: {t.text}"
        return transcripts_str
