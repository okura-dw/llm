import mimetypes
from typing import TypedDict, overload

import google.generativeai
import google.generativeai.models
import pydantic
import pydub
import streamlit

from llm_clients import logger, types


def tuple2message(
    tuple_messages: tuple[types.TupleMessage | types.TupleMessageUser, ...]
) -> list[google.generativeai.types.ContentDict]:
    messages: list[google.generativeai.types.ContentDict] = []
    for tuple_message in tuple_messages:
        match tuple_message.role:
            case "user":
                if isinstance(tuple_message.content, str):
                    messages.append(
                        google.generativeai.types.ContentDict(
                            role="user", parts=[tuple_message.content]
                        )
                    )
                else:
                    contents: list[str | google.generativeai.types.File] = []
                    for tuple_content in tuple_message.content:
                        match tuple_content.type:
                            case "text":
                                contents.append(tuple_content.content)
                            case "image_url":
                                file = google.generativeai.upload_file(path=tuple_content.content)
                                contents.append(file)
                    messages.append(
                        google.generativeai.types.ContentDict(role="user", parts=contents)
                    )
            case "assistant":
                messages.append(
                    google.generativeai.types.ContentDict(
                        role="model", parts=[tuple_message.content]
                    )
                )
            case "system":
                messages.append(
                    google.generativeai.types.ContentDict(
                        role="user", parts=[tuple_message.content]
                    )
                )
            case "tool":
                messages.append(
                    google.generativeai.types.ContentDict(
                        role="user", parts=[tuple_message.content]
                    )
                )

    return messages


def pydantic_to_typed_dict(model: type[pydantic.BaseModel]) -> type[TypedDict]:  # pyright: ignore
    class_dict = {k: v for k, v in model.__annotations__.items()}
    return TypedDict(model.__name__ + "Dict", class_dict)  # pyright: ignore


@streamlit.cache_resource(show_spinner=False)
def _cached_fetch(
    api_key: str,
    model: str,
    messages: tuple[types.TupleMessage | types.TupleMessageUser, ...],
    response_format: type[pydantic.BaseModel] | None,
) -> google.generativeai.types.GenerateContentResponse:
    logger.logger.info("don't use cache")
    google.generativeai.configure(api_key=api_key)
    client = google.generativeai.GenerativeModel(model)

    if response_format is not None:
        generation_config = google.generativeai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=pydantic_to_typed_dict(response_format),  # pyright: ignore
        )
    else:
        generation_config = None

    return client.generate_content(
        contents=tuple2message(messages),
        generation_config=generation_config,
        safety_settings={
            google.generativeai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: google.generativeai.types.HarmBlockThreshold.BLOCK_NONE,
            google.generativeai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: google.generativeai.types.HarmBlockThreshold.BLOCK_NONE,
            google.generativeai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: google.generativeai.types.HarmBlockThreshold.BLOCK_NONE,
            google.generativeai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: google.generativeai.types.HarmBlockThreshold.BLOCK_NONE,
        },
    )


class Gemini:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        self.api_key = api_key
        self.model = model
        self.fee = 0.0

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
    ) -> T: ...

    def fetch[
        T: type[pydantic.BaseModel]
    ](
        self,
        messages: tuple[types.TupleMessage | types.TupleMessageUser, ...],
        response_format: T | None = None,
    ) -> (T | str):
        response = _cached_fetch(self.api_key, self.model, messages, response_format)
        logger.logger.debug(response)
        self.calc_fee(messages, response)
        if response_format is not None:
            return response_format.model_validate_json(response.text)
        else:
            return response.text

    def calc_fee(
        self,
        messages: tuple[types.TupleMessage | types.TupleMessageUser, ...],
        response: google.generativeai.types.GenerateContentResponse,
    ):
        if self.model.startswith("gemini-1.5-flash"):
            input_token_price = 0.00001875 / 1_000
            output_token_price = 0.000075 / 1_000
            audio_price = 0.000002
            image_price = 0.00002
            video_price = 0.00002
        elif self.model.startswith("gemini-1.5-pro"):
            input_token_price = 0.00125 / 1_000
            output_token_price = 0.00375 / 1_000
            audio_price = 0.000125
            image_price = 0.001315
            video_price = 0.001315
        else:
            input_token_price = 0
            output_token_price = 0
            audio_price = 0
            image_price = 0
            video_price = 0

        for message in messages:
            if message.role != "user" or isinstance(message.content, str):
                continue
            for content in message.content:
                if content.type == "text":
                    continue
                file_type, _ = mimetypes.guess_type(content.content)
                if file_type is None:
                    continue
                if file_type.startswith("audio/"):
                    self.fee += (
                        audio_price * pydub.AudioSegment.from_file(content.content).duration_seconds
                    )
                elif file_type.startswith("image/"):
                    self.fee += image_price
                elif file_type.startswith("video/"):
                    self.fee += (
                        video_price * pydub.AudioSegment.from_file(content.content).duration_seconds
                    )

        self.fee += (
            response.usage_metadata.prompt_token_count * input_token_price
            + response.usage_metadata.candidates_token_count * output_token_price
        )
