import re

import llm_clients.gemini
import llm_clients.types
import pydantic

from vocacolle import types

PROMPT = """\
次の音声データの曲にタイムスタンプを付けて、例のようなSRT形式で出力してください。
提示したすべての「歌詞」を一行ずつ出力してください。
## 例
1
00:00:10,003 --> 00:00:12,455
あいうえお
2
00:00:12,562 --> 00:00:16,419
かきくけこ
3
00:01:02,110 --> 00:01:08,978
さしすせそ

「歌詞」
{lyrics}
"""

JSON_PROMPT = """\
次のSRT形式のデータを、JSONに整形してください。

「SRT形式のデータ」
{srt}
"""


class Lyrics(pydantic.BaseModel, frozen=True):
    start_time: str = pydantic.Field(description="歌詞の発話開始時間")
    end_time: str = pydantic.Field(description="歌詞の発話終了時間")
    lyrics: str = pydantic.Field(description="歌詞")
    lyrics_row: int = pydantic.Field(description="歌詞の行番号")


class Response(pydantic.BaseModel, frozen=True):
    lyrics_list: list[Lyrics]


class AlignmentWithAudio:
    """音声ファイルを入力して歌詞にタイムスタンプをつける

    Attributes
    ----------
    messages
    llm
    """

    def __init__(self, api_key: str, model: str):
        """init

        Parameters
        ----------
        api_key
        model
        """
        self.messages: list[llm_clients.types.TupleMessage] = []
        if model.startswith("gemini"):
            self.llm = llm_clients.gemini.Gemini(api_key, model)
        else:
            raise ValueError

    def fetch(self) -> type[Response]:
        """fetch 一度文字列でタイムスタンプをつけた後にJSONに成形する"""
        response = self.llm.fetch(tuple(self.messages))
        self.messages.append(
            llm_clients.types.TupleMessageAssistant(role="assistant", content=response)
        )
        response = self.llm.fetch(
            (
                llm_clients.types.TupleMessageUser(
                    role="user", content=JSON_PROMPT.format(srt=response)
                ),
            ),
            Response,
        )
        return response

    def run(self, lyrics: str, audio_path: str) -> list[types.Lyrics]:
        """実行

        Parameters
        ----------
        lyrics
        audio_path
        """
        lyrics_list = [l for l in lyrics.split("\n") if re.sub(r"^\s+$", "", l) != ""]

        self.messages = [
            llm_clients.types.TupleMessageUser(
                role="user",
                content=PROMPT.format(lyrics=self._lyrics2str(lyrics_list)),
            ),
            llm_clients.types.TupleMessageUser(
                role="user",
                content=(
                    llm_clients.types.TupleContentParam(type="image_url", content=audio_path),
                ),
            ),
        ]

        response = self.fetch()

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
        """xx:xx:xx.xxx または xx:xx:xx,xxx 形式の文字列を秒に変換する

        Parameters
        ----------
        second_str
            xx:xx:xx.xxx または xx:xx:xx,xxx 形式の文字列
        """
        nums = second_str.replace(",", ".").split(":")
        return sum(float(n) * (60**i) for i, n in enumerate(reversed(nums)))

    @staticmethod
    def _lyrics2str(lyrics: list[str]) -> str:
        """歌詞のリストを行番号つきの文字列に変換する

        Parameters
        ----------
        lyrics
        """
        i = 1
        lyrics_str = ""
        for lyric in lyrics:
            if re.sub(r"^\s+$", "", lyric) != "":
                lyrics_str += f"{i}\n{lyric}\n"
                i += 1
            else:
                lyrics_str += "\n"
        return lyrics_str
