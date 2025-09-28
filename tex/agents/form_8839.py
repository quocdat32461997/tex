from tex.agents.base_agent import BaseAgent
from tex.agents.schemas import FormInput, StatementInput
from tex.db.utils import get_form_lines
from tex.registry import ReActRegistry, ToolRegistry


class Form8839Agent(BaseAgent):
    name: str = "form_8839"
    model_name: str = "gemini_chat"

    def __init__(self):
        super().__init__()
