from typing import List

from langchain_core.runnables import RunnableConfig

from tex.agents.schemas import FormInput
from tex.constants import STATUS
from tex.registry import ModelRegistry, ReActRegistry


@ReActRegistry.register("call_model")
def call_model(
    # state and config are two default runtime params.
    state: FormInput,
    config: RunnableConfig,
    # Other static parameters
    model_name: str,
    tools: List[str] = [],
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

    # Binding tools in run time
    if len(tools) > 0:
        model = model.bind_tools(tools)

    # Invoke model
    response = model.invoke(state["messages"])

    return {
        "messages": [response],
        "status": STATUS.SUCCESS,
    }


__all__ = ["call_model"]
