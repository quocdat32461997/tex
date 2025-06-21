from tex.RAG.rag import RAG

rag = RAG(path_to_vectorstore="abc")


# endpoint
def retrieve_from_rag(query: str, k: int):
    return rag.retrieve(
        query=query,
        k=k,
    )
