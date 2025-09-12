from langchain.tools import tool

from tex.registry import ToolRegistry


@ToolRegistry.register("multiply")
@tool
def multiply(a: int, b: int):
    """
    Multiply a and b.
    """
    return a * b
