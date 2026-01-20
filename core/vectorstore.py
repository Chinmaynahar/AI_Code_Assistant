from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    PERSIST_DIRECTORY,
    RETRIEVAL_SEARCH_TYPE,
    RETRIEVAL_K,
    RETRIEVAL_FETCH_K
)


class VectorStoreManager:
    def __init__(self):
        self.vectorstore: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=CHUNK_SEPARATORS
        )
    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)
    
    def create_vectorstore(
        self, 
        chunks: List[Document], 
        embeddings: HuggingFaceEmbeddings
    ) -> Chroma:
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        return self.vectorstore
    
    def get_retriever(self):
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Load documents first.")
        
        return self.vectorstore.as_retriever(
            search_type=RETRIEVAL_SEARCH_TYPE,
            search_kwargs={"k": RETRIEVAL_K, "fetch_k": RETRIEVAL_FETCH_K}
        )
    
    def delete_collection(self):
        if self.vectorstore:
            self.vectorstore.delete_collection()
            self.vectorstore = None