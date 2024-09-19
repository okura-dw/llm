import pydantic
import streamlit

import llm_clients.gemini
import llm_clients.openai
import llm_clients.types


class Response(pydantic.BaseModel, frozen=True):
    message: str = pydantic.Field(description="会話の応答")
    sources: list[str] = pydantic.Field(description="出典となるURL")


if __name__ == "__main__":
    with streamlit.sidebar:
        api_key = streamlit.text_input("API key")
        model_name = streamlit.radio("model", ["OpenAI", "Gemini"])

        match model_name:
            case "OpenAI":
                model = streamlit.selectbox("model", ["gpt-4o-2024-08-06"])
                llm = llm_clients.openai.OpenAI(api_key, model)
            case "Gemini":
                model = streamlit.selectbox(
                    "model",
                    [
                        "gemini-1.5-pro",
                        "gemini-1.5-flash",
                        "gemini-1.5-pro-exp-0827",
                        "gemini-1.5-flash-8b-exp-0827",
                    ],
                )
                llm = llm_clients.gemini.Gemini(api_key, model)

            case _:
                raise ValueError
    response = None

    user_message = streamlit.text_area("input")
    messages: list[llm_clients.types.TupleMessage] = [
        llm_clients.types.TupleMessageSystem(
            role="system", content="返答には必ず情報源となるURLを添えてください"
        ),
        llm_clients.types.TupleMessageUser(role="user", content=user_message),
    ]
    if streamlit.button("run"):
        response = llm.fetch(tuple(messages), response_format=Response)
    if response is not None:
        print(response)
        streamlit.write(response.message)
