from typing import overload

import openai
import openai.types.chat
import pydantic
import streamlit

from llm_clients import logger, types


def tuple2message(
    tuple_messages: tuple[types.TupleMessage | types.TupleMessageUser, ...]
) -> list[openai.types.chat.ChatCompletionMessageParam]:
    messages: list[openai.types.chat.ChatCompletionMessageParam] = []
    for tuple_message in tuple_messages:
        match tuple_message.role:
            case "user":
                if isinstance(tuple_message.content, str):
                    messages.append(
                        openai.types.chat.ChatCompletionUserMessageParam(
                            role=tuple_message.role, content=tuple_message.content
                        )
                    )
                else:
                    contents: list[openai.types.chat.ChatCompletionContentPartParam] = []
                    for tuple_content in tuple_message.content:
                        match tuple_content.type:
                            case "text":
                                contents.append(
                                    {"type": tuple_content.type, "text": tuple_content.content}
                                )
                            case "image_url":
                                contents.append(
                                    {
                                        "type": tuple_content.type,
                                        "image_url": {
                                            "url": tuple_content.content,
                                            "detail": "low",
                                        },
                                    }
                                )
            case "assistant":
                messages.append(
                    openai.types.chat.ChatCompletionAssistantMessageParam(
                        role=tuple_message.role, content=tuple_message.content
                    )
                )
            case "system":
                messages.append(
                    openai.types.chat.ChatCompletionSystemMessageParam(
                        role=tuple_message.role, content=tuple_message.content
                    )
                )
            case "tool":
                messages.append(
                    openai.types.chat.ChatCompletionToolMessageParam(
                        role=tuple_message.role, content=tuple_message.content, tool_call_id=""
                    )
                )

    return messages


@overload
def _cached_fetch(
    api_key: str,
    model: str,
    messages: tuple[types.TupleMessage | types.TupleMessageUser, ...],
    response_format: None,
) -> openai.types.chat.ChatCompletion: ...


@overload
def _cached_fetch[
    T: type[pydantic.BaseModel]
](
    api_key: str,
    model: str,
    messages: tuple[types.TupleMessage | types.TupleMessageUser, ...],
    response_format: T,
) -> openai.types.chat.ParsedChatCompletion[T]: ...


@streamlit.cache_resource(show_spinner=False)
def _cached_fetch[
    T: type[pydantic.BaseModel]
](
    api_key: str,
    model: str,
    messages: tuple[types.TupleMessage | types.TupleMessageUser, ...],
    response_format: T | None,
) -> (openai.types.chat.ParsedChatCompletion[T] | openai.types.chat.ChatCompletion):
    logger.logger.info("don't use cache")
    client = openai.OpenAI(api_key=api_key)

    if response_format is not None:
        return client.beta.chat.completions.parse(
            model=model, messages=tuple2message(messages), response_format=response_format
        )
    else:
        return client.chat.completions.create(model=model, messages=tuple2message(messages))


class OpenAI:
    def __init__(self, api_key: str, model: str = "gpt-4o-2024-08-06") -> None:
        self.api_key = api_key
        self.model = model

    @overload
    def fetch(
        self,
        messages: tuple[types.TupleMessage | types.TupleMessageUser, ...],
        response_format: None,
    ) -> str: ...

    @overload
    def fetch(self, messages: tuple[types.TupleMessage | types.TupleMessageUser, ...]) -> str: ...

    @overload
    def fetch[
        T: type[pydantic.BaseModel]
    ](
        self,
        messages: tuple[types.TupleMessage | types.TupleMessageUser, ...],
        response_format: T,
    ) -> (T | None): ...

    def fetch[
        T: type[pydantic.BaseModel]
    ](
        self,
        messages: tuple[types.TupleMessage | types.TupleMessageUser, ...],
        response_format: T | None = None,
    ) -> (str | T | None):
        if response_format is not None:
            response = _cached_fetch(self.api_key, self.model, messages, response_format)
            logger.logger.debug(response)
            return response.choices[0].message.parsed
        else:
            response = _cached_fetch(self.api_key, self.model, messages, response_format)
            logger.logger.debug(response)
            return response.choices[0].message.content
