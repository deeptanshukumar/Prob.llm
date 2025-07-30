# LLM interaction using ollama and RAG chain setup functions.

from langchain_ollama import OllamaLLM
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import Dict, Any, Optional, List
from langchain_core.documents import Document
from .retrieval import get_hybrid_retriever


def create_llm(model_name):
    # creates an ollama llm instance and returns it.
    return OllamaLLM(model=model_name)


def create_prompt_template(template=None):
    #make a template and return it
    if template is None:
        with open("prompt.txt", "r") as f:
            template = f.read()
    
    return PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )


def setup_rag_chain(vectorstore, 
                    llm_model_name, 
                    documents: Optional[List[Document]] = None,
                    prompt_template=None): 
    """setting up the RAG chain with hybrid retrieval (BM25 + Semantic + RRF).
    args:
        vectorstore: Chroma vectorstore instance
        llm_model_name: Name of the Ollama LLM model
        documents: List of Document objects for BM25 retrieval
        prompt_template: Custom prompt template string
    returns:
        RetrievalQA chain instance
    """
    # Create LLM
    llm = create_llm(llm_model_name)
    # Create prompt template
    prompt = create_prompt_template(prompt_template)
    
    # Create hybrid retriever (BM25 + Semantic with RRF) - optimized for speed
    if documents:
        retriever = get_hybrid_retriever(documents, vectorstore)
        print("Using hybrid retriever (BM25 + Semantic with RRF) - optimized with k=10")
    else:
        # Fall back to semantic only if no documents provided
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})  
        print("Using semantic retriever only")
    
    # Create the retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", #stuff is a simple chain that concatenates all documents and widely used method
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain


def query_rag_chain(rag_chain, query):
    """Query the RAG chain and return results.   
    args:
        rag_chain: RetrievalQA chain instance
        query: User query string
    returns:
        Dictionary containing result and source documents
    """
    try:
        response = rag_chain.invoke({"query": query})
        return {
            "success": True,
            "result": response["result"],
            "source_documents": response["source_documents"],
            "error": None
        }
    except Exception as e: # for any error, just return some default values
        return {
            "success": False,
            "result": None,
            "source_documents": [],
            "error": str(e)
        }


def generate_fallback_response(llm_model_name, query):
    """Generate a fallback response using the LLM without RAG context.
        return a LLM-generated response string
    """
    try:
        llm = create_llm(llm_model_name)
        response = llm.invoke(query)
        return response
    except Exception as e:
        return f"Error generating fallback response: {str(e)}"
