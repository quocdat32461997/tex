from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore

from tex.agents.schemas import FormInput
from tex.models.gemini import call_gemini_embedding
from tex.tools.tool_registry import ToolRegistry

vector_store = InMemoryVectorStore(call_gemini_embedding)


@ToolRegistry.register("retrieve_forms")
@tool
def retrieve_forms(state: FormInput):
    """
    Function to retrieve formss
    """
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}
