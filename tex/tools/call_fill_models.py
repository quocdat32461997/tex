from typing import List

from langchain_core.messages import HumanMessage
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
    question: str,
    instruction: str,
    tools: List[str] = [],
):
    # Get model
    model = ModelFactory.get(model_name)
    print("***********", state)
    # Binding tools in run time
    if len(tools) > 0:
        model.bind_tools(tools)

    # Add instruction
    # TO-DO: next line (line 31) attempts to combine the prompt to look at the extracted W2 info to return the total amount of income.
    # However, the result is "```tool_code\nprint(default_api.get_form_1040_line_item(item='1a'))\n```". Problem is how to parse them.
    state["messages"].append(HumanMessage(content=line + question))
    response = model.invoke(state["messages"])

    return {
        "messages": [response],
        "forms": [instruction.format(answer=response)],
    }
