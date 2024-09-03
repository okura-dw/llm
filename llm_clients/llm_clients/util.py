from typing import Literal, NamedTuple

import openai.types.chat


class TupleContentParam(NamedTuple):
    type: Literal["text", "image_url"]
    content: str


class TupleMessageUser(NamedTuple):
    role: Literal["user"]
    content: str | tuple[TupleContentParam, ...]


class TupleMessage(NamedTuple):
    role: Literal["assistant", "system", "tool"]
    content: str


def message2tuple(
    messages: list[openai.types.chat.ChatCompletionMessageParam],
) -> tuple[TupleMessage | TupleMessageUser, ...]:
    tuple_messages: list[TupleMessage | TupleMessageUser] = []
    for message in messages:
        match message["role"]:
            case "user":
                if isinstance(message["content"], str):
                    tuple_messages.append(
                        TupleMessageUser(role=message["role"], content=message["content"])
                    )
                else:
                    contents: list[TupleContentParam] = []
                    for param in message["content"]:
                        match param["type"]:
                            case "text":
                                contents.append(
                                    TupleContentParam(type=param["type"], content=param["text"])
                                )
                            case "image_url":
                                contents.append(
                                    TupleContentParam(
                                        type=param["type"], content=param["image_url"]["url"]
                                    )
                                )
                    tuple_messages.append(TupleMessageUser("user", tuple(contents)))
            case _:
                tuple_messages.append(
                    TupleMessage(
                        role=message["role"], content=message["content"]  # pyright: ignore
                    )
                )
    return tuple(tuple_messages)
