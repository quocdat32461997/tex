import base64
from functools import partial
from typing import List

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from tex.agents.schemas import FormInput
from tex.model import ModelFactory


def create_extract_info(  # state and config are two default runtime params.
    # Other static parameters
    model_name: str,
    url: str,
    tools: List[str] = [],
):
    def extract_info(
        # state and config are two default runtime params.
        state: FormInput,
        config: RunnableConfig,
        # Other static parameters
        url: str,
        model_name: str,
        tools: List[str] = [],
    ):
        """
        Follow below script to get specific on run (https://langchain-ai.github.io/langgraph/how-tos/graph-api/#add-runtime-configuration). # noqa
            MODELS = {
                "anthropic": init_chat_model("anthropic:claude-3-5-haiku-latest"),
                "openai": init_chat_model("openai:gpt-4.1-mini"),
            }

            def call_model(state: MessagesState, config: RunnableConfig):
                model = config["configurable"].get("model", "anthropic")
                model = MODELS[model]
                response = model.invoke(state["messages"])
                return {"messages": [response]}
        """
        # Get model
        model = ModelFactory.get(model_name)

        # Binding tools in run time
        if len(tools) > 0:
            model = model.bind_tools(tools)

        with open(url, "rb") as file:
            image_data = file.read()
            image_data = base64.b64encode(image_data).decode("utf-8")
        # Invoke model
        response = model.invoke(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract info from the given document, except sensitive infomration such as Social Security Number.",  # noqa
                        },
                        {
                            "type": "image",
                            "source_type": "base64",
                            "data": image_data,
                            "mime_type": "image/png",
                        },
                    ],
                }
            ]
        )
        return Command(
            update={
                "messages": [response],
                "statements": [response],
            }
        )

    return partial(
        extract_info,
        model_name=model_name,
        tools=tools,
        url=url,
    )
