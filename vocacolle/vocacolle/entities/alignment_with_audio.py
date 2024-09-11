import re

import llm_clients
import llm_clients.whisper
import openai.types.chat
import pydantic

from vocacolle import types

PROMPT = """\
次の音声データの曲にタイムスタンプを付けて、例のようなJSON形式で出力してください。
提示したすべての<歌詞>を一行ずつ出力してください。
## 例
{{
    "lyrics_list": [
        {{
            "start_time": "00:10.003",
            "end_time": "00:12.455",
            "lyrics": あいうえお,
            "lyrics_row": 1
        }},
        {{
            "start_time": "00:12.562",
            "end_time": "00:16.419",
            "lyrics": かきくけこ,
            "lyrics_row": 2
        }},
        {{
            "start_time": "01:02.110",
            "end_time": "01:08.978",
            "lyrics": さしすせそ,
            "lyrics_row": 3
        }}
    ]
}}

「歌詞」
{lyrics}
"""


class Lyrics(pydantic.BaseModel, frozen=True):
    start_time: str = pydantic.Field(description="歌詞の発話開始時間")
    end_time: str = pydantic.Field(description="歌詞の発話終了時間")
    lyrics: str = pydantic.Field(description="歌詞")
    lyrics_row: int = pydantic.Field(description="歌詞の行番号")


class Response(pydantic.BaseModel, frozen=True):
    lyrics_list: list[Lyrics]


class AlignmentWithAudio:
    def __init__(self, api_key: str, model: str):
        self.messages: list[openai.types.chat.ChatCompletionMessageParam] = []
        if model.startswith("gemini"):
            self.llm = llm_clients.Gemini(api_key, model)
        else:
            raise ValueError

    def run(self, lyrics: str, audio_path: str) -> list[types.Lyrics]:
        lyrics_list = [l for l in lyrics.split("\n") if re.sub(r"^\s+$", "", l) != ""]

        self.messages = [
            {
                "role": "user",
                "content": PROMPT.format(lyrics=self._lyrics2str(lyrics_list)),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": audio_path, "detail": "auto"}}
                ],
            },
        ]

        response = self.llm.fetch(llm_clients.message2tuple(self.messages))
        response_json = re.search(r"{.*}", response, flags=re.DOTALL)
        if response_json is None:
            raise ValueError(f"Cannot parse response: {response}")
        response = Response.model_validate_json(response_json.group())

        return [
            types.Lyrics(
                lyrics=l.lyrics,
                lyrics_row=l.lyrics_row,
                start_second=self._str2sec(l.start_time),
                end_second=self._str2sec(l.end_time),
            )
            for l in response.lyrics_list
        ]

    @staticmethod
    def _str2sec(second_str: str) -> float:
        """
        Parameters
        ----------
        second_str
            xx:xx:xx.xxx 形式の文字列
        """
        nums = second_str.split(":")
        return sum(float(n) * (60**i) for i, n in enumerate(reversed(nums)))

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
