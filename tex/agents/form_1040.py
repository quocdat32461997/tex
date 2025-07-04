from functools import partial

from langgraph.graph import END, START, StateGraph  # noqa

from tex.agents.base_agent import BaseAgent
from tex.agents.schemas import FormInput
from tex.data.utils import get_form_lines
from tex.tools import ToolFactory, call_fill_model, retrieve_instructions  # noqa


def should_continue(state: FormInput):
    # messages = state["messages"]
    # last_message = messages[-1]

    # if len(last_message.tool_calls) == 1 and :
    #     # Return the form name.
    #     return last_message.tool_calls[0]["name"]
    return END


def finish_response(state: FormInput):
    """
    Notify when finish filing tax form.
    """

    # Return notification of finishing tax filing
    return {"final_response": "Finished filing form 1040."}


class Form1040Agent(BaseAgent):
    name: str = "form_1040"
    model_name: str = "gemini_chat"

    def __init__(
        self,
        year: int,
    ) -> None:
        self.workflow = StateGraph(FormInput)

        # Get lines in form 1040
        lines = get_form_lines(
            year=year,
            form_name=self.name,
        )

        # Add nodes and edges representing lines in form.
        self.workflow.add_node("finish_response", finish_response)
        prev_line = START
        for line in lines:
            self.workflow.add_node(
                line["name"],
                partial(
                    call_fill_model,
                    form_name=self.name,
                    line=line["context"],
                    model_name=self.model_name,
                    tools=[ToolFactory.get("retrieve_instructions")],
                ),
            )
            self.workflow.add_edge(prev_line, line["name"])
            prev_line = line["name"]

        self.workflow.add_edge(prev_line, "finish_response")
        self.workflow.add_edge("finish_response", END)
        self.workflow = self.workflow.compile()

    def get(self):
        return self.workflow
