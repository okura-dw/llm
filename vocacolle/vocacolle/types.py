import pydantic


class Lyrics(pydantic.BaseModel, frozen=True):
    start_second: float | None = pydantic.Field(description="歌詞の発話開始秒")
    end_second: float | None = pydantic.Field(description="歌詞の発話終了秒")
    lyrics: str = pydantic.Field(description="歌詞")
    lyrics_row: int = pydantic.Field(description="歌詞の行番号")
