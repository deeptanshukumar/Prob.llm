# Document loading and processing functions for various file types.
import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain_core.documents import Document
from typing import List


def load_pdf_document(file_path):
    """load PDF document using PyMuPDFLoader."""
    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = os.path.basename(file_path)
        return documents
    except Exception as e:
        st.error(f"Error loading PDF {file_path}: {e}")
        return []


def load_docx_document(file_path):
    """load DOCX document using Docx2txtLoader."""
    try:
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = os.path.basename(file_path)
        return documents
    except Exception as e:
        st.error(f"Error loading DOCX {file_path}: {e}")
        return []


def load_image_document(file_path):
    """load image document using UnstructuredImageLoader for OCR."""
    try:
        loader = UnstructuredImageLoader(file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = os.path.basename(file_path)
        return documents
    except Exception as e:
        st.error(f"Error loading image {file_path}: {e}")
        return []

def load_text_document(file_path):
    """load text document from a plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        doc = Document(page_content=content, metadata={"source": os.path.basename(file_path)})
        return [doc]
    except Exception as e:
        st.error(f"Error loading text file {file_path}: {e}")
        return []


"""ingest all supported documents from a directory."""
"""this is the main function that gets called to load documents using above other functions"""
def ingest_documents_from_dir(directory):
    all_documents = []
    
    if not os.path.exists(directory):
        return all_documents
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == ".pdf":
            all_documents.extend(load_pdf_document(filepath))
        elif file_ext == ".docx":
            all_documents.extend(load_docx_document(filepath))
        elif file_ext in [".jpg", ".jpeg", ".png"]:
            all_documents.extend(load_image_document(filepath))
        elif file_ext == ".txt":
            all_documents.extend(load_text_document(filepath))
        # more file types can be added here when needed        
    
    return all_documents
