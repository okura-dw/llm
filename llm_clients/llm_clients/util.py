import openai.types.chat

import llm_clients.types


def message2tuple(
    messages: list[openai.types.chat.ChatCompletionMessageParam],
) -> tuple[llm_clients.types.TupleMessage | llm_clients.types.TupleMessageUser, ...]:
    tuple_messages: list[llm_clients.types.TupleMessage | llm_clients.types.TupleMessageUser] = []
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
