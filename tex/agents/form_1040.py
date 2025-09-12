from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph  # noqa
from langgraph.types import Command, Send

from tex.agents.base_agent import BaseAgent
from tex.agents.schemas import FormInput, StatementInput
from tex.db.utils import get_form_lines
from tex.registry import ReActRegistry, ToolRegistry


def should_request_info(state: StatementInput):
    # last_message = state["messages"][-1]
    # print("should request_info", last_message)

    # if last_message.tool_calls:
    #     return "tools"
    routes = []
    if state["need_w2"] is True:
        routes.append(
            Send(
                node="extract_statement",
                arg={
                    "statement_name": "w2",
                    "next_agent": "request_statement",
                },
            )
        )
        # return "need_w2"

    if state["need_1099"] is True:
        # return "need_1099"
        routes.append(
            Send(
                node="extract_statement",
                arg={
                    "statement_name": "1099",
                    "next_agent": "request_statement",
                },
            )
        )

    if len(routes) > 0:
        return routes

    return "process_f1040"


def check_missing_statements(state: FormInput):
    return {
        "messages": AIMessage(content="Checking for any missing statements."),
    }


def process_f1040(state: FormInput) -> FormInput:
    """
    Proceed to process form 1040.
    """

    return {"messages": AIMessage(content="Starting to file form 1040.")}


class Form1040Agent(BaseAgent):
    name: str = "form_1040"
    model_name: str = "gemini_chat"

    def __init__(
        self,
        year: int,
    ) -> None:
        workflow = StateGraph(FormInput)

        # Get lines in form 1040
        lines = get_form_lines(
            year=year,
            form_name=self.name,
        )

        # Add nodes and edges representing lines in form.
        workflow.add_node(
            "call_model",
            ReActRegistry.get(
                "call_model",
                model_name=self.model_name,
            ),
        )
        workflow.add_node(
            "extract_statement",
            ReActRegistry.get(
                "extract_statement",
            ),
        )

        workflow.add_edge(START, "check_missing_statements")
        workflow.add_node("check_missing_statements", check_missing_statements)

        workflow.add_edge("check_missing_statements", "request_statement")
        workflow.add_node(
            "request_statement", ReActRegistry.get("request_statements")
        )  # noqa

        # workflow.add_node("tools", ToolNode([extract_w2]))
        workflow.add_node("process_f1040", process_f1040)

        workflow.add_conditional_edges(
            "request_statement",
            should_request_info,
        )

        prev_line = "process_f1040"
        for line in lines[:3]:
            workflow.add_node(
                line["name"],
                ReActRegistry.get(
                    name="call_fill_model",
                    form_name=self.name,
                    line=line["context"],
                    question=line["question"],
                    instruction=line["comprehensive_context"],
                    model_name=self.model_name,
                    tools=[ToolRegistry.get("retrieve_instructions")],
                ),
            )
            workflow.add_edge(prev_line, line["name"])
            prev_line = line["name"]
        workflow.add_edge(prev_line, END)
        self.workflow = workflow.compile()

    def get(self):
        return self.workflow


__all__ = ["Form1040Agent"]
