"""Text processing and chunking functions."""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def chunk_documents(documents, chunk_size=800, chunk_overlap=150):
    """splitting documents into smaller chunks for better retrieval 
    and returns a list of langchain Document objects."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    
    text_chunks = text_splitter.split_documents(documents)
    
    # Add chunk metadata
    for i, chunk in enumerate(text_chunks):
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "unknown_source_file"
        chunk.metadata["chunk_id"] = i
    
    return text_chunks


def prepare_documents_for_vectorstore(documents):
    """Prepare documents for vector store ingestion by adding necessary metadata 
    and returns a list prepared of Document objects."""
    prepared_docs = []
    
    for i, doc in enumerate(documents):
        # Ensure each document has required metadata
        if "source" not in doc.metadata:
            doc.metadata["source"] = "unknown_source_file"
        
        if "chunk_id" not in doc.metadata:
            doc.metadata["chunk_id"] = i
            
        prepared_docs.append(doc)
    
    return prepared_docs
