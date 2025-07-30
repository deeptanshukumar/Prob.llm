# Retrieval strategies using BM25 and Semantic retrieval with RRF via EnsembleRetriever.

from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


def create_bm25_retriever(documents, k=10):
    """ create a BM25 retriever from documents"""
    bm25_retriever = BM25Retriever.from_documents(documents, k=k)
    return bm25_retriever


def create_semantic_retriever(vectorstore, k=10):
    """Create a semantic retriever from vectorstore"""
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return semantic_retriever


def create_hybrid_retriever(
    documents, 
    vectorstore, 
    bm25_weight=0.5, 
    semantic_weight=0.5,
    k=10):
    """
    creates a hybrid retriever combining BM25 and semantic search with equal weights.
    Uses EnsembleRetriever with RRF (Reciprocal Rank Fusion) for combining results.
    """
    # Create individual retrievers
    bm25_retriever = create_bm25_retriever(documents, k)
    semantic_retriever = create_semantic_retriever(vectorstore, k)
    
    # Create hybrid retriever with RRF
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[bm25_weight, semantic_weight]
    )
    
    return hybrid_retriever



# this is the fucntion that makes the hybrid retriever from the above function by 
# calling create_hybrid_retriever which creates the hybrid_retriever and returns it 
def get_hybrid_retriever(documents, vectorstore):
    return create_hybrid_retriever(
        documents=documents,
        vectorstore=vectorstore,
        bm25_weight=0.5,
        semantic_weight=0.5,
        k=10 
    )
