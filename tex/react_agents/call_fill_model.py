from typing import List

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from tex.agents.schemas import FormInput
from tex.constants import NUM_TRIALS, STATUS
from tex.registry import ModelRegistry, ReActRegistry


def extract_value(input) -> str:
    if input.count("{") != 1 and input.count("}") != 1:
        return input.split("{")[-1].split("}")[0]
    elif input.isnumeric() is True:
        return input
    else:
        raise ValueError(f"Wrong format: {input}")


@ReActRegistry.register("_call_fill_model")
def _call_fill_model(
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
        state["messages"][-1].content,
    )

    return {
        "messages": [response],
        "forms": [instruction.format(answer=extract_value(response.content))],
        "status": STATUS.SUCCESS,
    }


@ReActRegistry.register("call_fill_model")
def call_fill_model(
    # runtime parameters
    state: FormInput,
    # Other static parameters
    model_name: str,
    form_name: str,
    line: str,
    question: str,
    instruction: str,
    tools: List[str] = [],
) -> FormInput:
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

    STATEMENT_LIST = "\n".join(
        [
            f"Statement name: {name} - Content: {statement_list}"
            for name, statement_list in state["statements"].items()
        ]
    ).strip()
    PROMPT = f"""Given the below statements and instruction,
        Statements: {STATEMENT_LIST}
        Instruction: {line}

        Follow the below command.
        {question}
        """
    state["messages"].append(HumanMessage(content=PROMPT))
    response = None
    result = 0.0
    for _ in range(NUM_TRIALS):
        try:
            response = model.invoke(PROMPT)
            result = extract_value(response.content)
            break
        except Exception as e:
            print(f"Error parsing JSON: {response}", e)
            continue
    return {
        "messages": [response],
        "forms": {form_name: instruction.format(answer=result)},  # noqa
    }


__all__ = ["call_fill_model"]
