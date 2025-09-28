from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from tex.agents.base_agent import BaseAgent
from tex.agents.schemas import FormInput
from tex.db.utils import get_form_lines
from tex.registry import ReActRegistry, ToolRegistry


class ScheduleBAgent(BaseAgent):
    name: str = "schedule_b"
    model_name: str = "gemini_chat"

    def __init__(
        self,
        year: int,
    ) -> None:
        workflow = StateGraph(FormInput)

        # Get lines in Schedule B
        lines = get_form_lines(year=year, form_name=self.name)

    def get(self):
        return self.workflow
