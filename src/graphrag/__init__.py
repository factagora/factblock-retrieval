"""
Enhanced GraphRAG Module

This module provides relationship-aware fact-checking capabilities that leverage
knowledge graph connections invisible to vector embeddings.
"""

from .relationship_aware_fact_checker import RelationshipAwareFactChecker, FactCheckResult
from .enhanced_graphrag import EnhancedGraphRAG, EnhancedFactCheckResult

__all__ = [
    'RelationshipAwareFactChecker', 
    'FactCheckResult',
    'EnhancedGraphRAG', 
    'EnhancedFactCheckResult'
]