from typing import Literal, NamedTuple


class TupleContentParam(NamedTuple):
    type: Literal["text", "image_url"]
    content: str


class TupleMessageUser(NamedTuple):
    role: Literal["user"]
    content: str | tuple[TupleContentParam, ...]


class TupleMessage(NamedTuple):
    role: Literal["assistant", "system", "tool"]
    content: str


class Transcript(NamedTuple):
    start: float
    end: float
    text: str
