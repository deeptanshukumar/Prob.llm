"""
Main module for the Prob.lm Study Assistant.
This module contains the core logic and cached functions used by the Streamlit app.
"""

import os
import streamlit as st
from typing import Tuple, Optional, List
from langchain_core.documents import Document

# Import core modules
from core.document_loaders import ingest_documents_from_dir
from core.text_processing import chunk_documents, prepare_documents_for_vectorstore
from core.vector_store import initialize_chroma_db, clear_vectorstore, add_documents_to_vectorstore
from core.llm_interface import setup_rag_chain, query_rag_chain, generate_fallback_response

# Constants
UPLOAD_DIR = "./uploaded_docs"
CHROMA_DB_DIR = "./chroma_data"
TEMP_IMAGES_DIR = "./temp_images"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "llama3.2:1b"


def ensure_directories():
    """Ensure all required directories exist."""
    for directory in [UPLOAD_DIR, CHROMA_DB_DIR, TEMP_IMAGES_DIR]:
        os.makedirs(directory, exist_ok=True)


@st.cache_resource(show_spinner="Initializing knowledge base... This may take a moment.")
def perform_ingestion_and_setup_rag(upload_dir: str, chroma_db_dir: str) -> Tuple[Optional[object], Optional[object]]:
    """Cached function for document ingestion and RAG setup."""
    ensure_directories()

    # Load documents
    all_loaded_documents = ingest_documents_from_dir(upload_dir)
    
    if not all_loaded_documents:
        st.sidebar.warning("No supported documents found in 'uploaded_docs' folder. Please upload or place files.")
        return None, None

    # Process documents
    text_chunks = chunk_documents(all_loaded_documents, chunk_size=1000, chunk_overlap=200)
    prepared_documents = prepare_documents_for_vectorstore(text_chunks)

    # Initialize vector store
    vectorstore = initialize_chroma_db(chroma_db_dir, EMBEDDING_MODEL_NAME)
    
    # Clear existing documents
    clear_vectorstore(vectorstore)

    # Add new documents
    success = add_documents_to_vectorstore(vectorstore, prepared_documents)
    if not success:
        st.error("Failed to add documents to vector store.")
        return None, None
    
    # Setup RAG chain with hybrid retrieval (pass text_chunks for BM25)
    rag_chain = setup_rag_chain(vectorstore, LLM_MODEL_NAME, documents=text_chunks)
    
    return vectorstore, rag_chain


def process_user_query(rag_chain, query, llm_model_name):
    """Process a user query through the RAG chain."""
    if not rag_chain:
        return {
            "success": False,
            "message": "Please ingest documents first to activate the RAG system.",
            "answer": None,
            "source_docs": [],
            "is_fallback": False
        }
    
    # Query the RAG chain
    response = query_rag_chain(rag_chain, query)
    
    if not response["success"]:
        return {
            "success": False,
            "message": f"An error occurred: {response['error']}. Please ensure Ollama server is running and the {llm_model_name} model is available.",
            "answer": None,
            "source_docs": [],
            "is_fallback": False
        }
    
    answer = response["result"]
    source_docs = response["source_documents"]
    
    # Check if we need to provide a fallback response
    if not source_docs and "I don't have enough information" in answer:
        fallback_response = generate_fallback_response(llm_model_name, query)
        return {
            "success": True,
            "message": "I don't have enough information in my current knowledge base to answer that directly. Here is a general answer from my training data:",
            "answer": fallback_response,
            "source_docs": [],
            "is_fallback": True
        }
    
    return {
        "success": True,
        "message": None,
        "answer": answer,
        "source_docs": source_docs,
        "is_fallback": False
    }


def format_source_documents(source_docs):
    unique_sources = set()
    for doc in source_docs:
        source_info = f"Source: {doc.metadata.get('source', 'N/A')}"
        if doc.metadata.get('page') is not None:
            source_info += f", Page: {doc.metadata.get('page', 'N/A')}"
        if doc.metadata.get('slide_number') is not None:
            source_info += f", Slide: {doc.metadata.get('slide_number', 'N/A')}"
        unique_sources.add(source_info)
    
    return sorted(list(unique_sources))


# Export constants and functions for use in streamlit_app.py
__all__ = [
    'UPLOAD_DIR',
    'CHROMA_DB_DIR', 
    'TEMP_IMAGES_DIR',
    'EMBEDDING_MODEL_NAME',
    'LLM_MODEL_NAME',
    'ensure_directories',
    'perform_ingestion_and_setup_rag',
    'process_user_query',
    'format_source_documents'
]
