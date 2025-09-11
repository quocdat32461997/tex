from langchain_core.messages import ToolMessage

from tex.agents.schemas import FormInput
from tex.registry import ToolRegistry


# Define our tool node: This function is not working. Use ToolNode instead.
def call_tools(state: FormInput) -> FormInput:
    outputs = []
    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool by name
        tool = ToolRegistry.get(name=tool_call["name"])

        if tool_call["name"].startswith("transfer_to"):
            # print(
            #     "call_tools",
            tool_result = tool.invoke(
                {
                    "state": state,
                    "tool_call_id": tool_call["id"],
                }
            )
            # )
        else:
            tool_result = tool.invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


__all__ = ["call_tools"]
