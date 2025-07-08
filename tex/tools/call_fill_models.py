from typing import List

from langchain_core.runnables import RunnableConfig

from tex.agents.schemas import FormInput
from tex.model import ModelFactory


def call_fill_model(
    # runtime parameters
    state: FormInput,
    config: RunnableConfig,
    # Other static parameters
    model_name: str,
    form_name: str,
    line: str,
    tools: List[str] = [],
):
    # Get model
    model = ModelFactory.get(model_name)
    print("***********", state)
    # Binding tools in run time
    if len(tools) > 0:
        model.bind_tools(tools)
    messages = state["messages"]
    statements = state["statments"]
    response = model.invoke(messages)
    return {
        "messages": [response],
        # "forms": state["forms"][form_name].update(response),
    }
