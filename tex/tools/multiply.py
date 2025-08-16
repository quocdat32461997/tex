from langchain.tools import tool

from tex.tools.tool_registry import ToolRegistry


@ToolRegistry.register("multiply")
@tool
def multiply(a: int, b: int):
    """
    Multiply a and b.
    """
    return a * b
