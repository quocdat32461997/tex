from typing import Annotated, Any, Dict, List, NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState, add_messages  # noqa
from typing_extensions import TypedDict

from tex.agents.utils import update_list, update_lookup


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
    statements: Annotated[Dict[str, List[Any]], update_lookup]
    forms: Annotated[Dict[str, List[Any]], update_lookup]
    missing_forms: Annotated[List[str], update_list]


class ConfigSchema(TypedDict):
    mode: str


class HumanInput(TypedDict):
    input: str


class StatementInput(TypedDict):
    need_w2: NotRequired[bool]
    need_1099: NotRequired[bool]


class ExtractStatementInput(TypedDict):
    statement_name_list: str
    next_agent: NotRequired[str]


class StatementsInput(TypedDict):
    statement_name_list: List[str]
    next_agent: NotRequired[str]


class ToFillInput(TypedDict):
    form_name: str
    line: str
    question: str
    instruction: str
    tools: List[str]


__all__ = [
    "FormInput",
    "ConfigSchema",
    "StatementInput",
    "HumanInput",
    "ExtractStatementInput",
    "StatementsInput",
]
