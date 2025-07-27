import os
import chromadb # NEW IMPORT
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings # NEW IMPORT
from langchain_community.vectorstores import Chroma # NEW IMPORT
from typing import List

UPLOAD_DIR = "./uploaded_docs"
CHROMA_DB_DIR = "./chroma_data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def initialize_chroma_db():
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR) #becauee we want to keep the data even after program is stopped
    #this is the embedding model we will use for chroma db
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        collection_name="prob_lm_documents",
        embedding_function=embeddings,
        client=client
    )
    print("ChromaDB initialized.")
    return vectorstore

def load_pdf_document(file_path: str) -> List[dict]:
    print(f"Loading PDF: {file_path}")

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} pages from PDF.")

    return documents

if __name__ == "__main__":
    # --- For Testing This Step ---
    # Ensure UPLOAD_DIR exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    # Ensure CHROMA_DB_DIR exists (ChromaDB will create it if not, but good to ensure parent)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True) # NEW LINE: Ensure ChromaDB directory exists
    
    sample_pdf_path = os.path.join(UPLOAD_DIR, "sample.pdf")
    
    if not os.path.exists(sample_pdf_path):
        print(f"Creating a dummy PDF for testing at {sample_pdf_path}...")
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(sample_pdf_path, pagesize=letter)
        c.drawString(100, 750, "This is a sample document for testing the RAG system.")
        c.drawString(100, 730, "It contains some basic information about RAG and LLMs.")
        c.drawString(100, 710, "Retrieval Augmented Generation (RAG) is an AI framework that combines")
        c.drawString(100, 690, "retrieval mechanisms with generative models to produce more accurate and")
        c.drawString(100, 670, "contextually relevant outputs.")
        c.drawString(100, 650, "Large Language Models (LLMs) like TinyLlama are powerful but can hallucinate.")
        c.drawString(100, 630, "RAG helps by providing external knowledge as context.")
        c.save()
        print("Dummy PDF created.")
    
    loaded_docs = load_pdf_document(sample_pdf_path) 

    text_chunks = [] # Initialize empty list for chunks
    if loaded_docs:
        print("\n--- Starting Chunking ---")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,        # Max characters per chunk
            chunk_overlap=200,      # Characters to overlap between chunks
            length_function=len,    # Use standard Python len() for measuring chunk size
            add_start_index=True,   # Add metadata with the starting character index of the chunk
        )
        text_chunks = text_splitter.split_documents(loaded_docs)
        
        print(f"Split into {len(text_chunks)} chunks.")
        print("\nFirst chunk content snippet (from app.py):")
        print(text_chunks[0].page_content[:300] + "...")
        print("\nFirst chunk metadata:")
        print(text_chunks[0].metadata)
    else:
        print("No documents were loaded, skipping chunking.")

    # NEW SECTION: Initialize ChromaDB and Add Chunks
    print("\n--- Initializing ChromaDB and Adding Chunks ---")
    vectorstore = initialize_chroma_db() # Call the new function to get the vectorstore

    if text_chunks:
        # Before adding, let's add some custom metadata to each chunk
        # This is useful for knowing the original source of the chunk
        for i, chunk in enumerate(text_chunks):
            chunk.metadata["source"] = os.path.basename(sample_pdf_path) # Add source filename
            chunk.metadata["chunk_id"] = i # Add a simple chunk ID for debugging
            # Note: PyMuPDFLoader already adds 'page' to metadata, which is good!

        # Add the documents (chunks) to the vectorstore. 
        # Chroma will use the embedding_function we defined to embed them.
        vectorstore.add_documents(text_chunks)
        print(f"Successfully added {len(text_chunks)} chunks to ChromaDB.")
        print(f"ChromaDB data stored in: {CHROMA_DB_DIR}")
    else:
        print("No text chunks to add to ChromaDB.")

    # We will add the RAG query logic in the next step.

