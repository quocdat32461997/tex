import base64
import json

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from tex.agents.schemas import ExtractStatementInput, FormInput
from tex.constants import NUM_TRIALS, STATUS
from tex.registry import ModelRegistry, ReActRegistry
from tex.utils.json_format import clean_json_string


@ReActRegistry.register("extract_from_image")
def extract_from_image(
    # state and config are two default runtime params.
    state: FormInput,
    config: RunnableConfig,
    model_name: str,
    next_agent: str,
) -> FormInput:
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
    model = ModelRegistry.get(model_name)

    # Get human input
    input = interrupt(value="Please upload the file.")
    # input = "tex/db/statement_db/w2.png"
    if isinstance(input, str) is True:
        with open(input, "rb") as file:
            image_data = file.read()
            image_data = base64.b64encode(image_data).decode("utf-8")
    elif isinstance(input, bytes) is True:
        image_data = base64.b64encode(input).decode("utf-8")
    else:
        raise ValueError("Input must be a file path or bytes.")

    # Invoke model
    result = {}
    response = ""
    for _ in range(NUM_TRIALS):
        try:
            response = model.invoke(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Extract info from the given document, except sensitive infomration such as Social Security Number. Must return in JSON format with key-value pairs. For example, {{key: value}}""",  # noqa
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
            result = json.loads(clean_json_string(response.content))
        except Exception as e:
            print(f"Error parsing JSON: {response}", e)
            continue
    return Command(
        goto=next_agent,
        update={
            "messages": [],
            "statements": [result],
            "status": STATUS.SUCCESS,
        },  # [response],
    )


@ReActRegistry.register("extract_statement")
def extract_statement(
    # state and config are two default runtime params.
    state: ExtractStatementInput,
    config: RunnableConfig,
) -> FormInput:
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
    model = ModelRegistry.get("gemini_chat")

    # Get human input
    # input = interrupt(value="Please upload the file.")
    input = "tex/db/statement_db/w2.png"
    if isinstance(input, str) is True:
        with open(input, "rb") as file:
            image_data = file.read()
            image_data = base64.b64encode(image_data).decode("utf-8")
    elif isinstance(input, bytes) is True:
        image_data = base64.b64encode(input).decode("utf-8")
    else:
        raise ValueError("Input must be a file path or bytes.")

    # Invoke model
    result = {}
    response = ""
    for _ in range(NUM_TRIALS):
        try:
            response = model.invoke(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Extract info from the given document, except sensitive infomration such as Social Security Number. Must return in JSON format with key-value pairs. 
                                For example, {{key: value}}""",  # noqa
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
            result = clean_json_string(response.content)
            break
        except Exception as e:
            print(f"Error parsing JSON: {response}", e)
            continue
    return Command(
        goto=state["next_agent"],
        update={"statements": {state["statement_name"]: result}},
    )


__all__ = ["extract_from_image", "extract_statement"]
