from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore

from tex.agents.schemas import FormInput
from tex.model import call_gemini_embedding
from tex.RAG.app import retrieve_from_rag
from tex.tools.tool_factory import ToolFactory


@ToolFactory.register("retrieve_instructions")
def retrieve_instructions(query: str):
    """
    Function to retrieve instructions given the query input.

    Parameters:
        query: string contains the query input.

    Returns:
        A list of retrieved documents.
    """
    # retrieved_docs = vector_store.similarity_search(state["question"])
    return retrieve_from_rag(
        query=query,
        k=5,
    )
