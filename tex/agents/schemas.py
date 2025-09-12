from typing import Annotated, Any, Dict, List, NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState, add_messages  # noqa
from typing_extensions import TypedDict

from tex.agents.utils import update_lookup


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
    statements: Annotated[Dict[str, List[Any]], update_lookup]  #

    forms: Annotated[List[AnyMessage], add_messages]


class ConfigSchema(TypedDict):
    mode: str


class HumanInput(TypedDict):
    input: str


class StatementInput(TypedDict):
    need_w2: NotRequired[bool]
    need_1099: NotRequired[bool]


__all__ = ["FormInput", "ConfigSchema", "StatementInput", "HumanInput"]
