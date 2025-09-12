from langgraph.types import interrupt

from tex.agents.schemas import HumanInput
from tex.registry import ReActRegistry


@ReActRegistry.register("human_input")
def human_input(state) -> HumanInput:
    input = interrupt()
    return {"input": input}


__all__ = ["human_input"]
