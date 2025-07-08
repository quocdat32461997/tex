from typing import Annotated, Any, Dict, List

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState, add_messages  # noqa
from typing_extensions import TypedDict


class FormInput(MessagesState):
    """
    A shared message stage between graphs.
    {form: {
        form_name: 1040,
        lines: {
            line_number: 0,
            context: "Line 1: Total amount on W-2 Form(s)",
            result: 100.0,
            }
        }
    }
    """

    messages: Annotated[List[AnyMessage], add_messages]
    statements: Annotated[List[AnyMessage], add_messages]
    forms: Annotated[List[AnyMessage], add_messages]


class ConfigSchema(TypedDict):
    mode: str
