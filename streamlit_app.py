import streamlit as st
import os
import shutil
from app import ( # Import functions directly from app.py for now
    ingest_documents_from_dir,
    initialize_chroma_db,
    setup_rag_chain,
    UPLOAD_DIR,
    CHROMA_DB_DIR,
    LLM_MODEL_NAME
)
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Prob.lm Study Assistant", layout="wide")
# UPDATED: Logo and Title
st.title("📚 Prob.lm Study Assistant")

# --- Cached function for ingestion and RAG setup ---
# This function will only run once per session unless its inputs change or cache is cleared.
@st.cache_resource(show_spinner="Initializing knowledge base... This may take a moment.")
def perform_ingestion_and_setup_rag(upload_dir, chroma_db_dir):
    # Ensure directories exist
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(chroma_db_dir, exist_ok=True)

    # Removed granular "Loading documents from disk for ingestion..."
    all_loaded_documents = ingest_documents_from_dir(upload_dir)
    
    if not all_loaded_documents:
        st.sidebar.warning("No supported documents found in 'uploaded_docs' folder. Please upload or place files.")
        return None, None # Return None if no documents

    # Removed granular "Loaded X documents. Starting chunking..."
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    text_chunks = text_splitter.split_documents(all_loaded_documents)
    # Removed granular "Split into X total chunks."

    # Removed granular "Initializing ChromaDB and adding chunks..."
    vectorstore = initialize_chroma_db()
    
    # Attempt to clear the collection if it exists and has data
    try:
        if vectorstore._collection.count() > 0:
            vectorstore._collection.delete(ids=vectorstore._collection.get(limit=None)['ids'])
            # Removed granular "Cleared existing ChromaDB collection for fresh ingestion."
    except Exception as e:
        # Keep warning for errors, but not for empty collection on first run
        if "Collection not found" not in str(e): # Filter out common "collection not found" on first run
            st.sidebar.warning(f"Could not explicitly clear ChromaDB collection (may be empty or error on first run): {e}")

    for i, chunk in enumerate(text_chunks):
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "unknown_source_file"
        chunk.metadata["chunk_id"] = i
    
    vectorstore.add_documents(text_chunks)
    # Removed granular "Successfully added X chunks to knowledge base."
    
    # Removed granular "Setting up RAG chain..."
    rag_chain = setup_rag_chain(vectorstore, text_chunks)
    # Removed granular "Knowledge base ready for queries!"
    
    return vectorstore, rag_chain

# --- Initialize Session State and Auto-Ingest on Load ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Auto-ingest if documents are already in UPLOAD_DIR on first load
if not st.session_state.rag_chain and os.path.exists(UPLOAD_DIR) and os.listdir(UPLOAD_DIR):
    st.session_state.vectorstore, st.session_state.rag_chain = perform_ingestion_and_setup_rag(UPLOAD_DIR, CHROMA_DB_DIR)
    if st.session_state.rag_chain:
        st.success("Knowledge base loaded automatically from existing files in 'uploaded_docs'. Ready to chat!")
    else:
        st.error("Automatic knowledge base loading failed. Please check logs or try uploading new documents.")


# --- Sidebar for Document Management ---
st.sidebar.header("Document Management")

# File Uploader (for new uploads)
uploaded_files = st.sidebar.file_uploader(
    "Upload new study documents (PDF, DOCX, PPTX, JPG, PNG)",
    type=["pdf", "docx", "pptx", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="file_uploader" # Added a key to prevent warning
)

# Ingest Documents Button (for newly uploaded files)
if st.sidebar.button("Ingest Uploaded Documents"):
    if uploaded_files:
        st.sidebar.info("Preparing for fresh ingestion...") # More concise
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
        os.makedirs(UPLOAD_DIR, exist_ok=True) # Recreate empty directory
        # Removed "Old data cleared. Saving new uploads..."

        # Save newly uploaded files to UPLOAD_DIR
        # Removed "Saving uploaded files..." and individual file saves
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Removed individual file saved messages
        st.sidebar.success("Files uploaded. Starting ingestion...") # More concise
        

        # Clear cache and re-run ingestion
        perform_ingestion_and_setup_rag.clear() # Clear the cache so it re-runs
        st.session_state.vectorstore, st.session_state.rag_chain = perform_ingestion_and_setup_rag(UPLOAD_DIR, CHROMA_DB_DIR)
        
        if st.session_state.rag_chain:
            st.success("Documents ingested and RAG system ready!") # More concise success
        else:
            st.error("Document ingestion failed. Please check logs for errors.")
        st.session_state.messages = [] # Clear chat history on new ingestion
        st.experimental_rerun() # Rerun to update file list and clear uploader
    else:
        st.sidebar.warning("Please upload files first before clicking 'Ingest Uploaded Documents'.")

# NEW: Clear All Sources Button
if st.sidebar.button("Clear All Sources"):
    st.sidebar.info("Clearing all sources and knowledge base...")
    if os.path.exists(UPLOAD_DIR):
        try:
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR)
            # Removed "Uploaded files cleared."
        except OSError as e:
            st.sidebar.error(f"Error clearing uploaded files: {e}")
    
    if os.path.exists(CHROMA_DB_DIR):
        try:
            shutil.rmtree(CHROMA_DB_DIR)
            # Removed "ChromaDB knowledge base cleared."
        except OSError as e:
            st.sidebar.error(f"Error clearing ChromaDB: {e}")
    
    # Clear session state and cache
    perform_ingestion_and_setup_rag.clear() # Clear the cache
    st.session_state.rag_chain = None
    st.session_state.vectorstore = None
    st.session_state.messages = []
    st.success("All sources and knowledge base cleared. Please upload new documents to begin.") # More concise
    st.experimental_rerun() # Rerun to update UI


# Display current ingested documents
st.sidebar.subheader("Currently Ingested Files:")
if os.path.exists(UPLOAD_DIR) and os.listdir(UPLOAD_DIR):
    for f_name in os.listdir(UPLOAD_DIR):
        st.sidebar.text(f_name)
else:
    st.sidebar.text("No files ingested yet.")
    if not st.session_state.rag_chain: # Only show this warning if RAG isn't active
        st.sidebar.warning("Upload documents to start chatting!")


# --- Chat Interface ---
st.subheader("Ask Your Study Assistant:")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents...", disabled=not st.session_state.rag_chain):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.rag_chain:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke({"query": prompt})
                
                answer = response["result"]
                source_docs = response["source_documents"]

                # Handle LLM response for no sources / general answer
                if not source_docs and "I don't have enough information" in answer:
                    with st.chat_message("assistant"):
                        st.markdown("I don't have enough information in my current knowledge base to answer that directly. Here is a general answer from my training data:")
                        # Directly ask the LLM without context for a general answer
                        general_llm_response = Ollama(model=LLM_MODEL_NAME).invoke(prompt)
                        st.markdown(general_llm_response)
                        # UPDATED: Sources dropdown for general answer
                        with st.expander("**Sources:** No specific sources from your documents were found for this query."):
                            st.markdown("This answer is based on the LLM's general training data.")
                    st.session_state.messages.append({"role": "assistant", "content": f"I don't have enough information in my current knowledge base to answer that directly. Here is a general answer from my training data:\n\n{general_llm_response}\n\n**Sources:** No specific sources from your documents were found for this query."})
                else:
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if source_docs:
                            # UPDATED: Sources dropdown
                            with st.expander("**Sources:**"):
                                unique_sources = set()
                                for doc in source_docs:
                                    source_info = f"Source: {doc.metadata.get('source', 'N/A')}"
                                    if doc.metadata.get('page') is not None:
                                        source_info += f", Page: {doc.metadata.get('page', 'N/A')}"
                                    if doc.metadata.get('slide_number') is not None:
                                        source_info += f", Slide: {doc.metadata.get('slide_number', 'N/A')}"
                                    unique_sources.add(source_info)
                                
                                for source in sorted(list(unique_sources)):
                                    st.markdown(f"- {source}")
                        else: # This else block is for when source_docs is empty but answer is not generic "I don't have enough info"
                            # UPDATED: Sources dropdown for no sources
                            with st.expander("**Sources:** No specific sources from your documents were found for this query."):
                                st.markdown("The LLM generated this answer without specific context from your uploaded documents.")
                    
                    # Add assistant message to chat history
                    assistant_response_content = answer
                    if source_docs:
                        assistant_response_content += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sorted(list(unique_sources))])
                    else:
                        assistant_response_content += "\n\n**Sources:** No specific sources from your documents were found for this query."
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response_content})

            except Exception as e:
                st.error(f"An error occurred during query: {e}")
                st.warning("Please ensure your Ollama server is running and the Llama 3.2:1b model is available.")
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}. Please ensure Ollama server is running and Llama 3.2:1b model is available."})
    else:
        with st.chat_message("assistant"):
            st.warning("Please ingest documents first to activate the RAG system.")
        st.session_state.messages.append({"role": "assistant", "content": "Please ingest documents first to activate the RAG system."})

