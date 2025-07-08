from langchain.tools import tool

from tex.RAG.app import retrieve_from_rag
from tex.tools.tool_factory import ToolFactory


@ToolFactory.register("retrieve_instructions")
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
