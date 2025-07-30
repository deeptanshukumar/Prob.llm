"""
Streamlit application for the Prob.lm Study Assistant.
This is the main UI entry point that uses the core functionality from main.py.
"""

import streamlit as st
import os
import shutil
import time

# Import from main orchestration module
from main import (
    UPLOAD_DIR, 
    CHROMA_DB_DIR, 
    LLM_MODEL_NAME,
    ensure_directories,
    perform_ingestion_and_setup_rag,
    process_user_query,
    format_source_documents
)

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Prob.lm Study Assistant", layout="wide")
st.title("📚 Prob.lm Study Assistant")

# --- Initialize Session State ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Auto-ingest existing documents on first load ---
if not st.session_state.rag_chain and os.path.exists(UPLOAD_DIR) and os.listdir(UPLOAD_DIR):
    try:
        st.session_state.vectorstore, st.session_state.rag_chain = perform_ingestion_and_setup_rag(UPLOAD_DIR, CHROMA_DB_DIR)
        if st.session_state.rag_chain:
            st.success("Knowledge base loaded automatically from existing files in 'uploaded_docs'. Ready to chat!")
        else:
            st.error("Automatic knowledge base loading failed. Please check logs or try uploading new documents.")
    except Exception as e:
        st.error(f"Failed to auto-load documents: {e}")
        st.info("Please upload documents manually to continue.")

# --- Sidebar for Document Management ---
st.sidebar.header("Document Management")

# File Uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload new study documents (PDF, DOCX, PPTX, JPG, PNG)",
    type=["pdf", "docx", "pptx", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="file_uploader"
)

# Ingest Documents Button
if st.sidebar.button("Ingest Uploaded Documents"):
    if uploaded_files:
        st.sidebar.info("Preparing for fresh ingestion...")
        
        # Clear cache and session state
        perform_ingestion_and_setup_rag.clear() 
        st.session_state.rag_chain = None 
        st.session_state.vectorstore = None 
        st.session_state.messages = []
        
        time.sleep(0.5)  # Small delay

        # Clear existing directories
        if os.path.exists(CHROMA_DB_DIR):
            try:
                shutil.rmtree(CHROMA_DB_DIR)
            except OSError as e:
                st.sidebar.error(f"Error clearing old knowledge base: {e}")
        
        if os.path.exists(UPLOAD_DIR):
            try:
                shutil.rmtree(UPLOAD_DIR)
            except OSError as e:
                st.sidebar.error(f"Error clearing old uploaded files: {e}")
        
        ensure_directories()

        # Save uploaded files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.sidebar.success("Files uploaded. Starting ingestion...")
        
        # Perform ingestion
        st.session_state.vectorstore, st.session_state.rag_chain = perform_ingestion_and_setup_rag(UPLOAD_DIR, CHROMA_DB_DIR)
        
        if st.session_state.rag_chain:
            st.success("Documents ingested and RAG system ready!")
        else:
            st.error("Document ingestion failed. Please check logs for errors.")
        
        st.rerun()
    else:
        st.sidebar.warning("Please upload files first before clicking 'Ingest Uploaded Documents'.")

# Clear All Sources Button
if st.sidebar.button("Clear All Sources"):
    st.sidebar.info("Clearing all sources and knowledge base...")
    
    # Clear cache and session state
    perform_ingestion_and_setup_rag.clear()
    st.session_state.rag_chain = None
    st.session_state.vectorstore = None
    st.session_state.messages = []

    time.sleep(0.5)

    # Clear directories
    if os.path.exists(UPLOAD_DIR):
        try:
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            st.sidebar.success("Uploaded files cleared.")
        except OSError as e:
            st.sidebar.error(f"Error clearing uploaded files: {e}")
    
    if os.path.exists(CHROMA_DB_DIR):
        try:
            shutil.rmtree(CHROMA_DB_DIR)
            st.sidebar.success("ChromaDB knowledge base cleared.")
        except OSError as e:
            st.sidebar.error(f"Error clearing ChromaDB: {e}")
    
    st.success("All sources and knowledge base cleared. Please upload new documents for RAG to work.")
    st.info("Page will refresh automatically.")
    st.rerun()

# Display current ingested documents
st.sidebar.subheader("Currently Ingested Files:")
if os.path.exists(UPLOAD_DIR) and os.listdir(UPLOAD_DIR):
    for f_name in os.listdir(UPLOAD_DIR):
        st.sidebar.text(f_name)
else:
    st.sidebar.text("No files ingested yet.")
    if not st.session_state.rag_chain:
        st.sidebar.warning("Upload documents to start chatting!")

# --- Chat Interface ---
st.subheader("Ask Your Study Assistant:")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents...", disabled=not st.session_state.rag_chain):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        # Process the query using the main module
        response_data = process_user_query(st.session_state.rag_chain, prompt, LLM_MODEL_NAME)
        
        if not response_data["success"]:
            # Handle errors
            st.error(response_data["message"])
            st.session_state.messages.append({"role": "assistant", "content": response_data["message"]})
        else:
            # Handle successful responses
            with st.chat_message("assistant"):
                if response_data["is_fallback"]:
                    # Fallback response (no sources found)
                    st.markdown(response_data["message"])
                    st.markdown(response_data["answer"])
                    with st.expander("**Sources:** No specific sources from your documents were found for this query."):
                        st.markdown("This answer is based on the LLM's general training data.")
                    
                    assistant_content = f"{response_data['message']}\n\n{response_data['answer']}\n\n**Sources:** No specific sources from your documents were found for this query."
                    st.session_state.messages.append({"role": "assistant", "content": assistant_content})
                else:
                    # Regular response with sources
                    st.markdown(response_data["answer"])
                    
                    if response_data["source_docs"]:
                        with st.expander("**Sources:**"):
                            formatted_sources = format_source_documents(response_data["source_docs"])
                            for source in formatted_sources:
                                st.markdown(f"- {source}")
                        
                        assistant_content = response_data["answer"] + "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in formatted_sources])
                    else:
                        with st.expander("**Sources:** No specific sources from your documents were found for this query."):
                            st.markdown("The LLM generated this answer without specific context from your uploaded documents.")
                        
                        assistant_content = response_data["answer"] + "\n\n**Sources:** No specific sources from your documents were found for this query."
                    
                    st.session_state.messages.append({"role": "assistant", "content": assistant_content})
