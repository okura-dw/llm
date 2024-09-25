import re

import llm_clients.gemini
import llm_clients.types
import pydub

from vocacolle import types

PROMPT = """\
次の音声データの曲にタイムスタンプを付けて、例のようなSRT形式で出力してください。
提示したすべての「歌詞」を一行ずつ、歌詞中の記号や空白を変えずに出力してください。
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

    def fetch(self) -> list[types.Lyrics]:
        """fetch 一度文字列でタイムスタンプをつけた後にJSONに成形する"""
        response = self.llm.fetch(tuple(self.messages))
        self.messages.append(llm_clients.types.TupleMessageAssistant(content=response))
        response = self._srt2model(response)
        return response

    def run(self, lyrics: str, audio_path: str) -> list[types.Lyrics]:
        """実行

        Parameters
        ----------
        lyrics
        audio_path
        """
        lyrics_list = lyrics.split("\n")

        self.messages = [
            llm_clients.types.TupleMessageUser(
                content=PROMPT.format(lyrics=self._lyrics2str(lyrics_list)),
            ),
            llm_clients.types.TupleMessageUser(
                content=(
                    llm_clients.types.TupleContentParam(type="image_url", content=audio_path),
                ),
            ),
        ]

        for _ in range(3):  # バリデーションに失敗した場合は２回までやり直す
            response = self.fetch()
            valid = self._validate_lyrics(lyrics_list, response)
            valid = self._validate_timestamp(audio_path, response) and valid
            if valid:
                break

        return response  # pyright: ignore[reportPossiblyUnboundVariable]

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

    def _srt2model(self, srt: str) -> list[types.Lyrics]:
        """SRT形式の文字列を types.Lyrics のリスト形式に変換する

        Parameters
        ----------
        srt
            SRT形式の文字列
        """
        pattern = re.compile(r"(\d+)\n(.*) --> (.*)\n(.*)(?:$|\n)")
        ret: list[types.Lyrics] = []
        for row, start, end, lyric in pattern.findall(srt):
            ret.append(
                types.Lyrics(
                    start_second=self._str2sec(start),
                    end_second=self._str2sec(end),
                    lyrics=lyric,
                    lyrics_row=row,
                )
            )
        return ret

    def _validate_lyrics(self, lyrics_list: list[str], response: list[types.Lyrics]) -> bool:
        """歌詞周りのバリデーション
        誤りがある場合は self.messages にその旨を追加する

        Parameters
        ----------
        lyrics_list
        response
        """
        lyrics_list = [l for l in lyrics_list if re.sub(r"^\s+$", "", l) != ""]

        error_message = ""
        if len(lyrics_list) != len(response):
            error_message += f"歌詞の行数が誤っています。正しい歌詞の行数は{len(lyrics_list)}行ですが、出力は{len(response)}行になっています。\n"
        for i, lyric in enumerate(lyrics_list, start=1):
            resp = [r for r in response if r.lyrics_row == i]
            if not resp:
                error_message += f"歌詞が誤っています。正しい{i}行目の歌詞「{lyric}」が、出力の{i}行目にありません。\n"
                continue
            if len(resp) > 1:
                error_message += f"出力の仕様が違います。出力に歌詞の{i}行目が複数含まれています。歌詞は分割せず一行ずつ出力してください。\n"
                continue
            if lyric.split() != resp[0].lyrics.split():
                error_message += f"歌詞が誤っています。正しい{i}行目の歌詞は「{lyric}」なのに対して、出力の{i}行目の歌詞は「{resp[0].lyrics}」になっています。\n"

        if error_message != "":
            self.messages.append(llm_clients.types.TupleMessageUser(content=error_message))
            return False
        return True

    def _validate_timestamp(self, audio_path: str, response: list[types.Lyrics]) -> bool:
        """タイムスタンプ周りのバリデーション
        誤りがある場合は self.messages にその旨を追加する

        Parameters
        ----------
        lyrics_list
        response
        """
        audio_segment = pydub.AudioSegment.from_file(audio_path)
        error_message = ""

        i = 1
        last_second = response[-1].end_second
        while last_second is None:
            last_second = response[-(i + 1)].end_second
            i += 1
        if audio_segment.duration_seconds < last_second:
            error_message += f"タイムスタンプが誤っています。音声ファイルの秒数{round(audio_segment.duration_seconds, 3)}秒を上回って出力のタイムスタンプがつけられています。\n"

        for i, resp in enumerate(response):
            if resp.end_second is None:
                continue
            j = i + 1
            next_start_second = None
            while j < len(response) and next_start_second is None:
                next_start_second = response[j].start_second
                j += 1
            if next_start_second is None:
                continue
            if resp.end_second > next_start_second:
                error_message += f"タイムスタンプが誤っています。{resp.lyrics_row}行目の終了時刻「{resp.end_second}」を、{response[j-1].lyrics_row}行目の開始時刻「{next_start_second}」が下回っています。\n"

        if error_message != "":
            self.messages.append(llm_clients.types.TupleMessageUser(content=error_message))
            return False
        return True
