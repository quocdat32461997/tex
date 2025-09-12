from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore

from tex.agents.schemas import FormInput
from tex.models.gemini import call_gemini_embedding
from tex.rag.app import retrieve_from_rag
from tex.registry import ToolRegistry

vector_store = InMemoryVectorStore(call_gemini_embedding)


@ToolRegistry.register("retrieve_instructions")
@tool
def retrieve_instructions(query: str):
    """
    Function to retrieve instructions given the query input.

    Parameters:
        query: string contains the query input.

    Returns:
        A list of retrieved documents.
    """

    return retrieve_from_rag(
        query=query,
        k=5,
    )


@ToolRegistry.register("retrieve_")
@tool
def retrieve_forms(state: FormInput):
    """
    Function to retrieve formss
    """
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


@ToolRegistry.register("retrieve_statements")
@tool
def retrieve_statements(state: FormInput):
    """
    Function to retrieve statements
    """
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


__all__ = ["retrieve_instructions", "retrieve_forms", "retrieve_statements"]
