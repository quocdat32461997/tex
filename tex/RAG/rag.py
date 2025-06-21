import time
import warnings
from datetime import datetime as dt
from typing import List

import faiss
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndex

# Splitting data to managebale chunks and vectorizing in memory-store
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import FAISS

# Langchain Imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_communnity.document_loaders import PyPDFLoader

# HuggingFace Imports
from sentence_transformers import SentenceTransformer

from tex.model import ModelFactory
from tex.RAG.constants import LOCAL_DISK_PATH, VectorStorePaths

warnings.filterwarnings("ignore")

# EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-large-en-v1.5")


class RAG:
    def __init__(self, path_to_vectorstore: str = None):

        self.vectorstore = None
        if path_to_vectorstore is not None:
            self.vectorstore = FAISS.load_local(path_to_vectorstore)

    def _create_vectorstore(
        self,
        documents: List[str],
    ):
        # Chunkging text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        split_docs = text_splitter.split_documents(documents)

        # Build vector store
        return FAISS.from_documents(
            split_docs,
            # EMBEDDING_MODEL,
            ModelFactory.get("alibaba_embedding_model"),
        )

    def build_vectorstore(
        path: str,
        year: int,
        name_to_save: str,
    ) -> None:
        # Load documents
        if path.endswith(".pdf"):
            documents = PyPDFLoader(path).load()
        else:
            raise f"File {path} is not supported."

        # create vector store
        vectorstore = self._create_vectorstore(documents=documents)

        # Save to local disk.
        vectorstore.save_local(
            VectorStorePaths.INSTRUCTIONS.format(
                dir_path=LOCAL_DISK_PATH,
                year=year,
                form_name=name_to_save,
            )
        )

    def retrieve(self, query: str, k=5):
        if self.vectorstore is None:
            raise ValueError(
                "Vectorstore is not initialized, Run create_vectorstore first."
            )
        return self.vectorstore.similarity_search(query, k=k)


"""
[a, b, c] -> 1
[b, c, d] -> 2
[b, d, d, z] -> 3

ranking/training -> ranking bundle 1 or 2 or 3. 
explicit labeling -> so many combinations of products -> 
    infinite number of bundles. 
    -> no need a set of candidate bundles. 1M bundles. 

Marketing team: fixed 5K bundles. 
[a, b, c] -> [1, 1, 1, 0, 0, ...] - 1# 26 chars
[b, c, d] -> [0, 1, 1, 1, 0, ...] - 120
[b, c, d, z] -> [0, 1, 1, 1, 0, ..., 1] - 121
"""
