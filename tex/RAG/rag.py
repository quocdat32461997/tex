import time
import warnings

from datetime import datetime as dt

warnings.filterwarnings("ignore")

from langchain_communnity.document_loaders import PyPDFLoader

#Splitting data to managebale chunks and vectorizing in memory-store 
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndex

#Langchain Imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA 
from langchain.docstore.document import Document 

#HuggingFace Imports
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import faiss

class RAG:
    def __init__(self, pdf_path, model_name='Alibaba-NLP/gte-large-en-v1.5'):
        self.pdf_path = pdf_path
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None

    def load_pdf(self):
        """
        Loads the PDF file and returns its contents as documents.

        Returns:
        --------
        List[Document]
            A list of document chunks extracted from the PDF.

        Notes:
        ------
        Uses `PyPDFLoader` to read and parse the contents of the PDF file.
            """
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        return documents

    def create_vectorstore(self, documents):
        """
    Creates a vector store from a list of documents using chunking and embedding.

    Parameters:
    -----------
    documents : List[Document]
        A list of document objects to be added to the vector store.
        These documents will be split into smaller chunks before being embedded..


    Process:
    --------
    1. Documents are split into overlapping chunks using RecursiveCharacterTextSplitter.
       - `chunk_size=1000`: Each chunk is up to 1000 characters long.
       - `chunk_overlap=200`: Adjacent chunks share 200 characters for context continuity.
    
    2. Each chunk is then converted into embeddings using the `self.embeddings` model.
    
    3. The resulting embedded documents are stored in a FAISS vector store,
       which is assigned to `self.vectorstore` for later retrieval.

    Returns:
    --------
    None
        This method sets the `self.vectorstore` attribute and does not return anything.

    """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        

    def retrieve(self, query, k=5):
        """
    Retrieve the top-k most similar documents from the vector store based on a query.

    Parameters:
    -----------
    query : str
        The search query string used to find similar documents in the vector store.
    k : int, optional (default=5)
        The number of top similar documents to retrieve. 
        

    Returns:
    --------
    List
        A list of the top-k documents most similar to the query.
    """
        if self.vectorstore is None:
            raise ValueError("Vectorstore is not initialized, Run create_vectorstore first.")
        result = self.vectorstore.similarity_search(query, k=k)
        return result