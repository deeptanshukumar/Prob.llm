"""ChromaDB vector store initialization and management functions"""

import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Optional


def initialize_chroma_db(chroma_db_dir, embedding_model_name, collection_name="prob_lm_documents"):
    """Initialize ChromaDB client and collection and return Chroma vectorstore instance"""
    client = chromadb.PersistentClient(path=chroma_db_dir) #persistent client for local storage
    embeddings = OllamaEmbeddings(model=embedding_model_name)
    
    # Delete the collection if it exists to avoid embedding dimension conflicts
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # Collection might not exist
    
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    return vectorstore


def clear_vectorstore(vectorstore):
    """clear all documents from the vectorstore"""
    try:
        # Get all existing IDs and delete them
        existing_data = vectorstore.get()
        if existing_data['ids']:
            vectorstore.delete(ids=existing_data['ids'])
        return True
    except Exception:
        return False


def add_documents_to_vectorstore(vectorstore, documents):
    """Add documents to the vectorstore and return success status"""
    try:
        vectorstore.add_documents(documents)
        return True
    except Exception as e:
        print(f"Error adding documents to vectorstore: {e}")
        return False


def get_retriever(vectorstore):    
    return vectorstore.as_retriever(search_kwargs={"k": 3})
