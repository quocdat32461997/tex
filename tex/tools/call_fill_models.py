from typing import List

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from tex.agents.schemas import FormInput
from tex.registry import ModelRegistry, ReActRegistry


def extract_value(input) -> str:
    if input.count("{") != 1 or input.count("}") != 1:
        raise ValueError(f"Wrong format: {input}")
    return input.split("{")[-1].split("}")[0]


@ReActRegistry.register("call_fill_model")
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
    model = ModelRegistry.get(model_name)

    # Binding tools in run time
    if len(tools) > 0:
        model.bind_tools(tools)

    # All extracted info is stored in state["statements"].
    # Combine the statements' info with question, the model
    # now can do the work. However, lines (rows 39-44) show
    # complex and lengthy for-loop(s) and text joining.
    # TO-DO: how to store statements more efficiently.
    state["messages"].append(HumanMessage(content=" ".join([line, question])))
    response = model.invoke(
        " ".join(
            [
                " ".join([x.content for x in state["statements"]]),
                state["messages"][-1].content,
            ]
        )
    )

    return {
        "messages": [response],
        "forms": [instruction.format(answer=extract_value(response.content))],
    }
