"""
Retrieval module for GraphRAG system.
"""

from typing import Dict, Any, Optional
from .base import BaseRetriever, RetrievalQuery, RetrievalResult
from .graph_rag import GraphRAGRetriever


class RetrievalModule:
    """
    Main API interface for the retrieval module.
    
    This class provides a clean, high-level interface for initializing and using
    the retrieval system. It abstracts away the underlying retrieval implementation
    details and provides a consistent API for other services.
    """
    
    def __init__(self, retriever_type: str = 'graphrag'):
        """
        Initialize the retrieval module.
        
        Args:
            retriever_type: Type of retriever to use ('graphrag' is currently supported)
        """
        self.retriever_type = retriever_type
        self.retriever = None
        self._initialized = False
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the retrieval module with configuration.
        
        Args:
            config: Configuration dictionary containing necessary parameters
                   For GraphRAG: {'neo4j_uri', 'neo4j_user', 'neo4j_password'}
                   
        Raises:
            ValueError: If retriever_type is not supported
            RuntimeError: If initialization fails
        """
        if self.retriever_type == 'graphrag':
            self.retriever = GraphRAGRetriever()
        else:
            raise ValueError(f"Unknown retriever type: {self.retriever_type}")
            
        self.retriever.initialize(config)
        self._initialized = True
        
    def retrieve(self, query_text: str, filters: Optional[Dict] = None, limit: int = 10):
        """
        Main retrieval method.
        
        Args:
            query_text: Text query to search for
            filters: Optional filters to apply (e.g., {'category': 'data_privacy'})
            limit: Maximum number of results to return
            
        Returns:
            List of RetrievalResult objects ranked by relevance
            
        Raises:
            RuntimeError: If module is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Module not initialized. Call initialize() first.")
            
        query = RetrievalQuery(
            query_text=query_text,
            filters=filters,
            limit=limit
        )
        return self.retriever.retrieve(query)
        
    def is_initialized(self) -> bool:
        """Check if the module has been initialized."""
        return self._initialized
        
    def get_retriever_type(self) -> str:
        """Get the current retriever type."""
        return self.retriever_type


# Export main classes
__all__ = [
    'RetrievalModule', 
    'RetrievalQuery', 
    'RetrievalResult',
    'BaseRetriever',
    'GraphRAGRetriever'
]