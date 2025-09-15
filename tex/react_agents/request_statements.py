import json

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


__all__ = ["request_statements"]
