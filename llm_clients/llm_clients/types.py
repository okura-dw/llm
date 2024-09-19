from typing import Literal, NamedTuple


# キャッシュできるようにタプルでメッセージを持っておく
class TupleContentParam(NamedTuple):
    type: Literal["text", "image_url"]
    content: str


class TupleMessageUser(NamedTuple):
    role: Literal["user"]
    content: str | tuple[TupleContentParam, ...]


class TupleMessageAssistant(NamedTuple):
    role: Literal["assistant"]
    content: str


class TupleMessageSystem(NamedTuple):
    role: Literal["system"]
    content: str


class TupleMessageTool(NamedTuple):
    role: Literal["tool"]
    content: str


TupleMessage = TupleMessageUser | TupleMessageAssistant | TupleMessageSystem | TupleMessageTool


class Transcript(NamedTuple):
    start: float
    end: float
    text: str
