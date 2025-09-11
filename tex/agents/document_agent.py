from langgraph.graph import END, START, StateGraph  # noqa

from tex.agents.base_agent import BaseAgent
from tex.agents.schemas import ConfigSchema, FormInput
from tex.tools.call_model import create_call_model


class DocumentAgent(BaseAgent):
    name: str = "document_agent"
    model_name: str = "gemini_chat"

    def __init__(self):
        self.workflow = StateGraph(
            FormInput,
            config_schema=ConfigSchema,
        )

        # Define call model
        call_model = create_call_model(
            model_name=self.model_name,
            tools=[],
        )

        # Add nodes
        self.workflow.add_node("call_model", call_model)
