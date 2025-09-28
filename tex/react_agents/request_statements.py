import json

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from tex.agents.schemas import FormInput, StatementInput
from tex.constants import NUM_TRIALS
from tex.registry import ModelRegistry, ReActRegistry
from tex.utils.json_format import clean_json_string
from tex.utils.wait import wait


@ReActRegistry.register("request_statements")
def request_statements(state: FormInput) -> StatementInput:
    """
    Request user to provide W-2 form.
    """
    print("request statement")
    prompt = f"""Based on the given below list of staetments,
        Forms: {state["forms"]}
        Decide:
        1. Which income forms are missing.
        2. If a form is missing, set the corresponding field of the form to true.
        No more reasoning. Must return in format: {{"need_w2": true/false, "need_1099": true/false}}.
        """
    # Get model
    model = ModelRegistry.get("gemini_chat")
    result = {}
    for _ in range(NUM_TRIALS):
        try:
            response = model.invoke(prompt)
            result = json.loads(
                clean_json_string(response.content),
            )
            print("request_statements result", result)
            break
        except Exception as e:
            print("Error parsing JSON:", e)
            wait()
            continue

    return StatementInput(**result)


@ReActRegistry.register("request_statements_with_instruction")
def request_statements_with_instruction(
    state: FormInput,
    line: str,
) -> FormInput:
    """
    Request user to provide W-2 form.
    """
    print("request statement")
    FORM_LIST = "\n".join(
        [f"Form name: {name}" for name in state["forms"].keys()]
    ).strip()
    PROMPT = f"""Given the below listi of forms, and instruction,
        Forms: {FORM_LIST}
        Instruction: {line}

        Your tasks are:
        1. Decide which forms are required in the above instruction and do not exist in the above given form list.
        2. Then, return the list of missing forms in the following JSON format: {{"result": [W-2, Form 2441]}}. No more reasoning
        3. If found no missing forms or duplicate forms, return the empty like the following JSON format: {{"result": []}}.
        """

    # Get model
    model = ModelRegistry.get("gemini_chat")
    message_list = []
    result = {}
    for _ in range(NUM_TRIALS):
        try:
            response = model.invoke(PROMPT)
            result = json.loads(
                clean_json_string(response.content),
            )
            message_list.append(HumanMessage(content=PROMPT))
            message_list.append(response)

            print("request_statements result", result)
            break
        except Exception as e:
            print("Error parsing JSON:", e)
            wait()
            continue

    # return StatementsInput(statement_name_list=result["result"])
    return Command(
        update={
            "messages": message_list,
            "missing_forms": result["result"],
        }
    )


__all__ = ["request_statements", "request_statements_with_instruction"]
