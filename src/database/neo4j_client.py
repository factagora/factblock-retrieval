"""
Neo4j database client for GraphRAG retrieval system.
"""

from neo4j import GraphDatabase
from typing import Optional, Dict, Any, List
import os
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j database client with connection pooling and error handling.
    """
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j client with connection parameters.
        
        Args:
            uri: Neo4j database URI (e.g., bolt://localhost:7687)
            user: Database username
            password: Database password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def verify_connectivity(self) -> bool:
        """
        Verify database connectivity by running a simple query.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            logger.error(f"Connectivity check failed: {e}")
            return False
    
    @contextmanager
    def session(self):
        """Context manager for Neo4j sessions."""
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        with self.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def execute_write_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a write query (CREATE, MERGE, DELETE, etc.).
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        def _write_transaction(tx):
            result = tx.run(query, parameters or {})
            return [record.data() for record in result]
            
        with self.session() as session:
            return session.write_transaction(_write_transaction)
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get basic database information.
        
        Returns:
            Dictionary with database metadata
        """
        queries = {
            "node_count": "MATCH (n) RETURN count(n) as count",
            "relationship_count": "MATCH ()-[r]->() RETURN count(r) as count",
            "database_name": "CALL db.info() YIELD name RETURN name"
        }
        
        info = {}
        for key, query in queries.items():
            try:
                result = self.execute_query(query)
                if result:
                    info[key] = result[0].get("count") or result[0].get("name")
            except Exception as e:
                logger.warning(f"Failed to get {key}: {e}")
                info[key] = "unknown"
        
        return info
    
    @classmethod
    def from_env(cls) -> 'Neo4jClient':
        """
        Create Neo4j client from environment variables.
        
        Expected environment variables:
        - NEO4J_URI: Database URI
        - NEO4J_USER: Database username
        - NEO4J_PASSWORD: Database password
        
        Returns:
            Neo4jClient instance
        """
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        return cls(uri, user, password)