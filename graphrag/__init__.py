"""
GraphRAG Package

Graph-enhanced retrieval-augmented generation system for FactBlock investment knowledge.
"""

from .smart_router import SmartGraphRAGRouter, QueryType, PerformanceMode
from .cypher_retriever import TextToCypherRetriever
from .vector_retriever import GraphVectorRetriever
from .embeddings import FactBlockEmbeddings
from .simple_embeddings import SimpleFactBlockEmbeddings

__version__ = "0.1.0"
__all__ = [
    "SmartGraphRAGRouter",
    "QueryType", 
    "PerformanceMode",
    "TextToCypherRetriever",
    "GraphVectorRetriever",
    "FactBlockEmbeddings",
    "SimpleFactBlockEmbeddings"
]