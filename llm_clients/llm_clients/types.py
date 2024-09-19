from typing import Literal, NamedTuple


# キャッシュできるようにタプルでメッセージを持っておく
class TupleContentParam(NamedTuple):
    type: Literal["text", "image_url"]
    content: str


class TupleMessageUser(NamedTuple):
    content: str | tuple[TupleContentParam, ...]
    role: Literal["user"] = "user"


class TupleMessageAssistant(NamedTuple):
    content: str
    role: Literal["assistant"] = "assistant"


class TupleMessageSystem(NamedTuple):
    content: str
    role: Literal["system"] = "system"


class TupleMessageTool(NamedTuple):
    content: str
    role: Literal["tool"] = "tool"


TupleMessage = TupleMessageUser | TupleMessageAssistant | TupleMessageSystem | TupleMessageTool


class Transcript(NamedTuple):
    start: float
    end: float
    text: str
