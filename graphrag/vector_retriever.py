#!/usr/bin/env python3
"""
Graph-Enhanced Vector Retriever

Combines vector similarity search with graph structure traversal
for enhanced FactBlock retrieval from Neo4j knowledge graph.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import logging
from neo4j import GraphDatabase, Driver

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.simple_embeddings import SimpleFactBlockEmbeddings

logger = logging.getLogger(__name__)

class GraphVectorRetriever:
    """Graph-Enhanced Vector Retriever for FactBlocks"""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_username: str = "neo4j", 
                 neo4j_password: str = "password",
                 neo4j_database: str = "neo4j"):
        """
        Initialize retriever with Neo4j connection and embeddings
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.driver: Optional[Driver] = None
        
        # Initialize embeddings
        self.embeddings = SimpleFactBlockEmbeddings()
        
        self._connect_neo4j()
        self._initialize_embeddings()
    
    def _connect_neo4j(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password)
            )
            
            # Test connection
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                
            if test_value == 1:
                logger.info(f"✅ Connected to Neo4j: {self.neo4j_uri}")
            else:
                raise Exception("Connection test failed")
                
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize embeddings from Neo4j data"""
        # For now, use the dataset file - later we can pull directly from Neo4j
        dataset_path = "data/processed/enhanced_knowledge_graph_dataset.json"
        
        if self.embeddings.initialize_from_dataset(dataset_path):
            logger.info("✅ Embeddings initialized from dataset")
        else:
            logger.error("❌ Failed to initialize embeddings")
    
    def get_factblock_from_neo4j(self, factblock_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full FactBlock data from Neo4j by ID"""
        query = """
        MATCH (f:FactBlock {id: $factblock_id})
        RETURN f
        """
        
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run(query, factblock_id=factblock_id)
                record = result.single()
                
                if record:
                    return dict(record["f"])
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Failed to get FactBlock {factblock_id}: {e}")
            return None
    
    def get_related_factblocks(self, factblock_id: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Get FactBlocks related through graph relationships
        
        Args:
            factblock_id: Starting FactBlock ID
            max_hops: Maximum relationship hops to traverse
            
        Returns:
            List of related FactBlock data with relationship info
        """
        query = f"""
        MATCH (start:FactBlock {{id: $factblock_id}})
        MATCH path = (start)-[r:RELATES_TO*1..{max_hops}]-(related:FactBlock)
        RETURN related, r, length(path) as distance
        ORDER BY distance, r[0].strength DESC
        LIMIT 20
        """
        
        related_factblocks = []
        
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run(query, factblock_id=factblock_id)
                
                for record in result:
                    related_fb = dict(record["related"])
                    distance = record["distance"]
                    relationships = record["r"]
                    
                    # Add graph context
                    related_fb["_graph_distance"] = distance
                    related_fb["_relationship_path"] = [
                        {
                            "type": rel.type,
                            "strength": rel.get("strength", 0),
                            "confidence": rel.get("confidence", 0),
                            "investment_insight": rel.get("investment_insight", "")
                        }
                        for rel in relationships
                    ]
                    
                    related_factblocks.append(related_fb)
            
            logger.info(f"🔗 Found {len(related_factblocks)} related FactBlocks")
            return related_factblocks
            
        except Exception as e:
            logger.error(f"❌ Failed to get related FactBlocks: {e}")
            return []
    
    def search(self, query: str, k: int = 5, include_graph_expansion: bool = True, 
               graph_expansion_factor: float = 0.3) -> List[Dict[str, Any]]:
        """
        Graph-enhanced vector search
        
        Args:
            query: Search query string
            k: Number of results to return
            include_graph_expansion: Whether to include graph-related FactBlocks
            graph_expansion_factor: Weight for graph-expanded results (0-1)
            
        Returns:
            List of enhanced FactBlock results with scores and graph context
        """
        # Step 1: Vector similarity search
        vector_results = self.embeddings.search(query, k=k*2)  # Get more for filtering
        
        enhanced_results = []
        seen_ids = set()
        
        # Step 2: Enhance with Neo4j data and graph expansion
        for idx, similarity_score, factblock_data in vector_results:
            factblock_id = factblock_data.get('id')
            
            if factblock_id in seen_ids:
                continue
            seen_ids.add(factblock_id)
            
            # Get full FactBlock from Neo4j
            neo4j_factblock = self.get_factblock_from_neo4j(factblock_id)
            if not neo4j_factblock:
                continue
            
            # Create enhanced result
            enhanced_result = {
                "factblock": neo4j_factblock,
                "vector_similarity": similarity_score,
                "graph_distance": 0,
                "combined_score": similarity_score,
                "source": "vector_search"
            }
            
            enhanced_results.append(enhanced_result)
            
            # Step 3: Graph expansion
            if include_graph_expansion and len(enhanced_results) < k:
                related_factblocks = self.get_related_factblocks(factblock_id, max_hops=1)
                
                for related_fb in related_factblocks:
                    if related_fb.get('id') in seen_ids:
                        continue
                    seen_ids.add(related_fb.get('id'))
                    
                    # Calculate combined score (vector + graph)
                    graph_score = 1.0 / (related_fb["_graph_distance"] + 1)  # Inverse distance
                    combined_score = (similarity_score * (1 - graph_expansion_factor) + 
                                    graph_score * graph_expansion_factor)
                    
                    enhanced_result = {
                        "factblock": related_fb,
                        "vector_similarity": 0.0,  # Not directly matched
                        "graph_distance": related_fb["_graph_distance"],
                        "combined_score": combined_score,
                        "source": "graph_expansion",
                        "relationship_path": related_fb.get("_relationship_path", [])
                    }
                    
                    enhanced_results.append(enhanced_result)
        
        # Sort by combined score and return top k
        enhanced_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return enhanced_results[:k]
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("🔐 Neo4j connection closed")


def demo_graph_vector_retriever():
    """Demo the graph-enhanced vector retriever"""
    
    print("🧪 Testing Graph-Enhanced Vector Retriever...")
    
    try:
        # Initialize retriever
        retriever = GraphVectorRetriever()
        
        # Test queries
        test_queries = [
            "OPEC oil production decisions",
            "inflation and monetary policy",
            "energy sector investment impact"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results = retriever.search(query, k=5, include_graph_expansion=True)
            
            for i, result in enumerate(results):
                factblock = result["factblock"]
                print(f"   {i+1}. Score: {result['combined_score']:.3f} "
                      f"({result['source']}) - {factblock['claim'][:60]}...")
                
                if result.get("relationship_path"):
                    path_types = [rel["type"] for rel in result["relationship_path"]]
                    print(f"      🔗 Graph path: {' → '.join(path_types)}")
        
        retriever.close()
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    demo_graph_vector_retriever()