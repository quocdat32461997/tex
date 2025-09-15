import json
from typing import List

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from tex.agents.schemas import FormInput
from tex.constants import NUM_TRIALS, STATUS
from tex.registry import ModelRegistry, ReActRegistry
from tex.utils.json_format import clean_json_string


def extract_values(input: str) -> str:
    """
    Function to extract values in JSON format.
    """
    # Get rid of {}
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
        "forms": [instruction.format(answer=extract_values(response.content))],  # noqa
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
            result = extract_values(response.content)
            break
        except Exception as e:
            print(f"Error parsing JSON: {response}", e)
            continue
    return {
        "messages": [response],
        "forms": {form_name: instruction.format(answer=result)},  # noqa
    }


@ReActRegistry.register("clf_model")
def clf_model(
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
    """
    (C)all-(L)oad-(F)ill model attempts to
    - Call another form agent (if required) to fill and aggregates numbers on
        another forms. The numbers wouuld be used in the current form.
    - Load any missing statements.
    - Given all info, start filling current form
    """
    message_list = []
    form_dict = state["forms"]
    # Get model
    model = ModelRegistry.get(model_name)

    # Binding tools in run time
    if len(tools) > 0:
        model.bind_tools(tools)

    # STATEMENT_LIST = "\n".join(
    #     [
    #         f"Statement name: {name} - Content: {statement_list}"
    #         for name, statement_list in state["statements"].items()
    #     ]
    # ).strip()
    FORM_LIST = "\n".join(
        [
            f"Form name: {name} - Content: {form_list}"
            for name, form_list in form_dict.items()
        ]
    ).strip()

    """Step 1: Call another form agent to aggregate numbers that would be used in the current form."""
    PROMPT = f"""Given the below statements, forms, and instruction,
        Forms: {FORM_LIST}
        Instruction: {line}

        Your tasks are:
        1. Decide which forms are missing.
        2. Then, return the list of missing forms or statements in the following JSON format: {{"result": [W-2, Form 2441]}}
        3. If found no missing forms, return the empty like the following JSON format: {{"result": []}}.
        """

    response = None
    result = {"result": []}
    for _ in range(NUM_TRIALS):
        try:
            response = model.invoke(PROMPT)
            result = json.loads(clean_json_string(response.content))

            message_list.append(HumanMessage(content=PROMPT))
            message_list.append(response)
            break
        except Exception as e:
            print(f"Error parsing JSON: {response}", e)
            continue

    """Step 2: Load any other missing statements."""
    if len(result["result"]) > 0:
        form_dict.update(result)
        pass

    """Step 3: Fill form w/ all given info."""
    FORM_LIST = "\n".join(
        [
            f"Form name: {name} - Content: {form_list}"
            for name, form_list in form_dict.items()
        ]
    ).strip()
    PROMPT = f"""Given the below statements and instruction,
        Forms: {FORM_LIST}
        Instruction: {line}

        Follow the below command.
        {question}
        """
    response = None
    result = 0.0
    for _ in range(NUM_TRIALS):
        try:
            response = model.invoke(PROMPT)
            result = extract_values(response.content)
            message_list.append(HumanMessage(content=PROMPT))
            message_list.append(response)
            break
        except Exception as e:
            print(f"Error parsing JSON: {response}", e)
            continue
    return {
        "messages": message_list,
        "forms": {form_name: instruction.format(answer=result)},  # noqa
    }
    return {
        "messages": [response],
        "forms": {form_name: instruction.format(answer=result)},  # noqa
    }


__all__ = ["call_fill_model", "clf_model"]
