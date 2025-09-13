from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from tex.agents.schemas import FormInput
from tex.registry import ToolRegistry


@ToolRegistry.register("handoff_tool")
def handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer to {agent_name}"

    @tool(name, description=description)
    def _handoff_tool(
        state: Annotated[FormInput, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        print("*********", state)
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }

        return Command(
            goto=agent_name,
            update={
                "messages": state["messages"] + [tool_message],
            },
            # graph=Command.PARENT,# Implement later if required to back to parent.
        )

    return _handoff_tool


__all__ = ["handoff_tool"]
