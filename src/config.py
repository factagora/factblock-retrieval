"""
Configuration management for GraphRAG retrieval system.

This module provides configuration classes for managing Neo4j connections,
retrieval parameters, and other system settings using Pydantic BaseSettings
for automatic environment variable loading and validation.
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Dict, Any


class Neo4jConfig(BaseSettings):
    """
    Neo4j database configuration.
    
    Environment variables can be prefixed with NEO4J_ to override defaults:
    - NEO4J_URI: Database connection URI
    - NEO4J_USER: Database username  
    - NEO4J_PASSWORD: Database password
    """
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j database URI")
    user: str = Field(default="neo4j", description="Database username")
    password: str = Field(default="password123", description="Database password")
    
    class Config:
        env_prefix = "NEO4J_"
        case_sensitive = False


class RetrievalConfig(BaseSettings):
    """
    Retrieval system configuration.
    
    Environment variables can be prefixed with RETRIEVAL_ to override defaults:
    - RETRIEVAL_DEFAULT_LIMIT: Default number of results to return
    - RETRIEVAL_EMBEDDING_MODEL: Embedding model to use
    - RETRIEVAL_SCORE_THRESHOLD: Minimum relevance score threshold
    - RETRIEVAL_EXPAND_HOPS: Number of graph hops for context expansion
    """
    default_limit: int = Field(default=10, description="Default result limit", ge=1, le=100)
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for text similarity"
    )
    score_threshold: float = Field(
        default=0.7, 
        description="Minimum relevance score threshold",
        ge=0.0,
        le=1.0
    )
    expand_hops: int = Field(
        default=2, 
        description="Number of graph hops for context expansion",
        ge=1,
        le=5
    )
    
    class Config:
        env_prefix = "RETRIEVAL_"
        case_sensitive = False


class AppConfig:
    """
    Main application configuration that combines all config sections.
    
    This class provides a centralized way to access all configuration
    settings and convert them to formats expected by different components.
    """
    
    def __init__(self):
        """Initialize configuration by loading settings from environment."""
        self.neo4j = Neo4jConfig()
        self.retrieval = RetrievalConfig()
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format for component initialization.
        
        Returns:
            Dictionary with configuration values suitable for passing to
            retrieval module and other components
        """
        return {
            'neo4j_uri': self.neo4j.uri,
            'neo4j_user': self.neo4j.user,
            'neo4j_password': self.neo4j.password,
            'retrieval_config': self.retrieval.model_dump()
        }
        
    def get_neo4j_config(self) -> Dict[str, str]:
        """
        Get Neo4j-specific configuration.
        
        Returns:
            Dictionary with Neo4j connection parameters
        """
        return {
            'neo4j_uri': self.neo4j.uri,
            'neo4j_user': self.neo4j.user,
            'neo4j_password': self.neo4j.password
        }
        
    def get_retrieval_config(self) -> Dict[str, Any]:
        """
        Get retrieval-specific configuration.
        
        Returns:
            Dictionary with retrieval system parameters
        """
        return self.retrieval.model_dump()
        
    def validate_config(self) -> bool:
        """
        Validate that all required configuration is present and valid.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate Neo4j configuration
            if not self.neo4j.uri or not self.neo4j.user:
                return False
                
            # Validate retrieval configuration
            if self.retrieval.default_limit <= 0:
                return False
                
            if not (0.0 <= self.retrieval.score_threshold <= 1.0):
                return False
                
            return True
            
        except Exception:
            return False
            
    def __repr__(self) -> str:
        """String representation of configuration (sanitized)."""
        return (
            f"AppConfig("
            f"neo4j_uri='{self.neo4j.uri}', "
            f"neo4j_user='{self.neo4j.user}', "
            f"default_limit={self.retrieval.default_limit}, "
            f"score_threshold={self.retrieval.score_threshold}"
            f")"
        )


def load_config() -> AppConfig:
    """
    Convenience function to load application configuration.
    
    Returns:
        AppConfig instance with settings loaded from environment
    """
    return AppConfig()