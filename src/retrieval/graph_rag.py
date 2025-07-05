from src.retrieval.base import BaseRetriever, RetrievalQuery, RetrievalResult
from src.database.neo4j_client import Neo4jClient
from typing import List, Dict, Any
import hashlib
import logging

logger = logging.getLogger(__name__)


class GraphRAGRetriever(BaseRetriever):
    """
    GraphRAG retrieval implementation using Neo4j for graph-based information retrieval.
    """
    
    def __init__(self):
        self.client = None
        self.embeddings_cache = {}
        self._initialized = False
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize retriever with Neo4j configuration."""
        self.client = Neo4jClient(
            uri=config['neo4j_uri'],
            user=config['neo4j_user'],
            password=config['neo4j_password']
        )
        
        # Verify connectivity
        if not self.client.verify_connectivity():
            raise RuntimeError("Failed to establish Neo4j connection")
            
        self._initialized = True
        logger.info("GraphRAG retriever initialized successfully")
        
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve relevant information using graph traversal and scoring.
        
        Args:
            query: RetrievalQuery containing query text, filters, and limit
            
        Returns:
            List of RetrievalResult objects ranked by relevance
        """
        if not self._initialized:
            raise RuntimeError("Retriever not initialized. Call initialize() first.")
            
        # 1. Generate query embedding (simplified for MVP - not used in current implementation)
        # query_embedding = self._generate_embedding(query.query_text)
        
        # 2. Find relevant nodes using graph traversal
        relevant_nodes = self._graph_search(query.query_text, query.filters)
        
        # 3. Expand context using graph relationships
        expanded_context = self._expand_context(relevant_nodes)
        
        # 4. Score and rank results
        scored_results = self._score_results(expanded_context, query.query_text)
        
        # 5. Format and return top results
        return self._format_results(scored_results[:query.limit])
        
    def _generate_embedding(self, text: str) -> str:
        """
        Generate a simple text embedding (hash-based for MVP).
        In production, this would use a proper embedding model.
        """
        return hashlib.md5(text.lower().encode()).hexdigest()
        
    def _graph_search(self, query_text: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant nodes in the graph based on query text and filters.
        """
        # Build base query
        cypher_query = """
        MATCH (n)
        WHERE n.description CONTAINS $search_text
           OR n.name CONTAINS $search_text
           OR n.title CONTAINS $search_text
           OR n.summary CONTAINS $search_text
        """
        
        parameters = {"search_text": query_text}
        
        # Apply category filter if provided
        if filters and "category" in filters:
            cypher_query += " AND n.category = $category"
            parameters["category"] = filters["category"]
            
        # Return nodes with their properties
        cypher_query += """
        RETURN n, labels(n) as node_type, id(n) as node_id
        LIMIT 20
        """
        
        try:
            results = self.client.execute_query(cypher_query, parameters)
            return results
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
            
    def _expand_context(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Expand context by finding related nodes through graph relationships.
        """
        if not nodes:
            return []
            
        # Extract node IDs for expansion
        node_ids = [node.get("node_id") for node in nodes if node.get("node_id")]
        
        if not node_ids:
            return nodes
            
        # Find connected nodes
        expansion_query = """
        MATCH (n)-[r]-(connected)
        WHERE id(n) IN $node_ids
        RETURN n, connected, type(r) as relationship_type, 
               labels(connected) as connected_type, id(connected) as connected_id
        LIMIT 50
        """
        
        try:
            expansion_results = self.client.execute_query(
                expansion_query, 
                {"node_ids": node_ids}
            )
            
            # Combine original nodes with expanded context
            expanded = nodes.copy()
            for result in expansion_results:
                if result.get("connected") and result.get("connected_id"):
                    expanded.append({
                        "n": result["connected"],
                        "node_type": result["connected_type"],
                        "node_id": result["connected_id"],
                        "relationship_type": result["relationship_type"],
                        "is_expansion": True
                    })
                    
            return expanded
            
        except Exception as e:
            logger.error(f"Context expansion failed: {e}")
            return nodes
            
    def _score_results(self, nodes: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
        """
        Score and rank nodes based on relevance to the query.
        """
        scored_nodes = []
        query_terms = set(query_text.lower().split())
        
        for node_data in nodes:
            node = node_data.get("n", {})
            score = 0.0
            
            # Base score for direct matches
            if not node_data.get("is_expansion", False):
                score += 1.0
                
            # Score based on text similarity
            text_fields = ["description", "name", "title", "summary"]
            for field in text_fields:
                if field in node:
                    field_text = str(node[field]).lower()
                    field_terms = set(field_text.split())
                    
                    # Simple term overlap scoring
                    overlap = len(query_terms.intersection(field_terms))
                    if overlap > 0:
                        score += (overlap / len(query_terms)) * 0.5
                        
            # Boost score for specific node types
            node_type = node_data.get("node_type", [])
            if "FederalRegulation" in node_type:
                score += 0.2
            elif "AgencyGuidance" in node_type:
                score += 0.15
            elif "EnforcementAction" in node_type:
                score += 0.1
                
            node_data["score"] = score
            scored_nodes.append(node_data)
            
        # Sort by score descending
        return sorted(scored_nodes, key=lambda x: x.get("score", 0), reverse=True)
        
    def _format_results(self, scored_nodes: List[Dict[str, Any]]) -> List[RetrievalResult]:
        """
        Format scored nodes into RetrievalResult objects.
        """
        results = []
        
        for node_data in scored_nodes:
            node = node_data.get("n", {})
            node_type = node_data.get("node_type", [])
            score = node_data.get("score", 0.0)
            
            # Extract content from various fields
            content_fields = ["description", "summary", "name", "title"]
            content_parts = []
            
            for field in content_fields:
                if field in node and node[field]:
                    content_parts.append(f"{field.title()}: {node[field]}")
                    
            content = " | ".join(content_parts) if content_parts else "No content available"
            
            # Build metadata
            metadata = {
                "node_id": node_data.get("node_id"),
                "node_type": node_type,
                "is_expansion": node_data.get("is_expansion", False),
                "relationship_type": node_data.get("relationship_type"),
            }
            
            # Add all node properties to metadata
            for key, value in node.items():
                if key not in ["description", "summary", "name", "title"]:
                    metadata[key] = value
                    
            # Determine source type
            source_type = node_type[0] if node_type else "Unknown"
            
            results.append(RetrievalResult(
                content=content,
                metadata=metadata,
                score=score,
                source_type=source_type
            ))
            
        return results