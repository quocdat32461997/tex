import json
import re

from tex.agents.schemas import FormInput, StatementInput
from tex.constants import NUM_TRIALS
from tex.registry import ModelRegistry, ReActRegistry


@ReActRegistry.register("request_statements")
def request_statements(state: FormInput) -> StatementInput:
    """
    Request user to provide W-2 form.
    """

    prompt = f"""Based on the given below list of staetments,
        Statements: {state["statements"]}
        Decide:
        1. Which income statements are missing.
        2. If a statement is missing, set the corresponding field of the statement to true.
        No more reasoning. Must return in format: {{"need_w2": true/false, "need_1099": true/false}}.
        """
    # Get model
    model = ModelRegistry.get("gemini_chat")
    print("request_statements prompt", prompt)
    result = {}
    for _ in range(NUM_TRIALS):
        try:
            response = model.invoke(prompt)
            result = json.loads(
                re.sub(
                    f"`*(json)*\n*\s*",  # noqa
                    "",
                    response.content,
                )
            )
            print("request_statements result", result)
            break
        except Exception as e:
            print("Error parsing JSON:", e)
            continue

    return StatementInput(**result)


__all__ = ["request_statements"]
