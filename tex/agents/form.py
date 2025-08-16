from langgraph.graph import END, START, StateGraph  # noqa
from langgraph.prebuilt import ToolNode

from tex.agents.base_agent import BaseAgent
from tex.agents.form_1040 import Form1040Agent
from tex.agents.schemas import ConfigSchema, FormInput
from tex.registry import ReActRegistry, ToolRegistry
from tex.tools.call_agents import create_handoff_tool

# from tex.tools.call_model import create_call_model


def should_continue(state: FormInput):
    last_message = state["messages"][-1]
    print("should continue", last_message)

    if last_message.tool_calls:
        return "tools"

    return END


class FormAgent(BaseAgent):
    name: str = "form"
    model_name: str = "gemini_chat"

    def __init__(
        self,
    ) -> None:
        self.workflow = StateGraph(
            FormInput,
            config_schema=ConfigSchema,
        )

        # Define agents
        f1040_agent = Form1040Agent(year=2024)  # .get()

        # Define tools
        f1040_agent_tool = create_handoff_tool(
            agent_name=f1040_agent.name,
            description="Transfer user to an agent to file tax form 1040.",
        )

        call_model = ReActRegistry.get(
            "call_model",
            model_name=self.model_name,
            tools=[
                f1040_agent_tool,
                ToolRegistry.get("multiply"),
            ],
        )

        tool_node = ToolNode([f1040_agent_tool])
        # Add nodes
        self.workflow.add_node("call_model", call_model)
        self.workflow.add_node(f1040_agent.name, f1040_agent.get())
        self.workflow.add_node("tools", tool_node)

        # Add edges
        self.workflow.add_edge(START, "call_model")
        self.workflow.add_conditional_edges(
            "call_model",
            should_continue,
            {
                "tools": "tools",  # f1040_agent.name,
                END: END,
            },
        )  # noqa
        # self.workflow.add_edge("tools", "call_model")

        self.workflow = self.workflow.compile()

    def get(self):
        return self.workflow
