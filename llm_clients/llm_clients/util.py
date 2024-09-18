import os

import demucs.separate
import openai.types.chat

import llm_clients.types


def message2tuple(
    messages: list[openai.types.chat.ChatCompletionMessageParam],
) -> tuple[llm_clients.types.TupleMessage, ...]:
    """キャッシュできるように OpenAI のメッセージをタプルに変換する

    Parameters
    ----------
    messages
    """
    tuple_messages: list[llm_clients.types.TupleMessage] = []
    for message in messages:
        match message["role"]:
            case "user":
                if isinstance(message["content"], str):
                    tuple_messages.append(
                        llm_clients.types.TupleMessageUser(
                            role=message["role"], content=message["content"]
                        )
                    )
                else:
                    contents: list[llm_clients.types.TupleContentParam] = []
                    for param in message["content"]:
                        match param["type"]:
                            case "text":
                                contents.append(
                                    llm_clients.types.TupleContentParam(
                                        type=param["type"], content=param["text"]
                                    )
                                )
                            case "image_url":
                                contents.append(
                                    llm_clients.types.TupleContentParam(
                                        type=param["type"], content=param["image_url"]["url"]
                                    )
                                )
                    tuple_messages.append(
                        llm_clients.types.TupleMessageUser("user", tuple(contents))
                    )
            case _:
                tuple_messages.append(
                    llm_clients.types.TupleMessage(
                        role=message["role"], content=message["content"]  # pyright: ignore
                    )
                )
    return tuple(tuple_messages)


def vocal_extract(audio_path: str, dirname: str) -> str:
    """ボーカル抽出してファイルのパスを返す

    audio_path
        ボーカル抽出したい音声ファイルのパス
    dirname
        ボーカルのファイルを置きたいパス {dirname}/htdemucs/ 以下に作られる
    """
    audio_fname, _ = os.path.splitext(os.path.basename(audio_path))
    vocal_path = f"{dirname}/htdemucs/{audio_fname}/vocals.wav"
    if not os.path.exists(vocal_path):
        demucs.separate.main(["--two-stems", "vocals", audio_path, "-o", dirname])
    return vocal_path
