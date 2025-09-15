from langgraph.graph import END, START, StateGraph  # noqa
from langgraph.prebuilt import ToolNode

from tex.agents.base_agent import BaseAgent
from tex.agents.form_1040 import Form1040Agent
from tex.agents.schemas import ConfigSchema, FormInput
from tex.registry import ReActRegistry, ToolRegistry


def should_continue(state: FormInput):
    last_message = state["messages"][-1]
    print("should continue", last_message.tool_calls)

    if last_message.tool_calls:
        return "tools"

    return END


class TexAgent(BaseAgent):
    name: str = "form"
    model_name: str = "gemini_chat"

    def __init__(
        self,
    ) -> None:
        workflow = StateGraph(
            FormInput,
            config_schema=ConfigSchema,
        )

        # Define agents
        f1040_agent = Form1040Agent(year=2024)  # .get()

        # Define tools
        f1040_agent_tool = ToolRegistry.get(
            "handoff_tool",
            agent_name=f1040_agent.name,
            description="Transfer user to an agent to file tax form 1040.",
        )  # noqa

        # Define nodes and edges
        workflow.add_node("tools", ToolNode([f1040_agent_tool]))
        workflow.add_node(f1040_agent.name, f1040_agent.get())

        workflow.add_edge(START, "call_model")
        workflow.add_node(
            "call_model",
            ReActRegistry.get(
                "call_model",
                model_name=self.model_name,
                tools=[
                    f1040_agent_tool,
                    ToolRegistry.get("multiply"),
                ],
            ),
        )

        workflow.add_conditional_edges(
            "call_model",
            should_continue,
            {
                "tools": "tools",  # f1040_agent.name,
                END: END,
            },
        )  # noqa
        workflow.add_edge(f1040_agent.name, "call_model")

        self.workflow = workflow.compile()

    def get(self):
        return self.workflow


__all__ = ["TexAgent"]
