#!/usr/bin/env python3
"""
TextToCypherRetriever

Converts natural language queries to Cypher queries for FactBlock investment
knowledge graph retrieval from Neo4j.
"""

import os
import sys
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from neo4j import GraphDatabase, Driver

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class TextToCypherRetriever:
    """Convert natural language to Cypher queries for FactBlock search"""
    
    def __init__(self, neo4j_uri: str = None,
                 neo4j_username: str = None, 
                 neo4j_password: str = None,
                 neo4j_database: str = None):
        """
        Initialize with Neo4j connection
        
        Args:
            neo4j_uri: Neo4j connection URI (defaults to env NEO4J_URI or localhost)
            neo4j_username: Neo4j username (defaults to env NEO4J_USER or neo4j)
            neo4j_password: Neo4j password (defaults to env NEO4J_PASSWORD or password)
            neo4j_database: Neo4j database name (defaults to neo4j)
        """
        # Use environment variables if parameters not provided
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver: Optional[Driver] = None
        
        # Initialize query patterns and entity mappings
        self._setup_query_patterns()
        self._setup_entity_mappings()
        
        self._connect_neo4j()
    
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
                logger.info(f"✅ TextToCypher connected to Neo4j: {self.neo4j_uri}")
            else:
                raise Exception("Connection test failed")
                
        except Exception as e:
            logger.error(f"❌ TextToCypher Neo4j connection failed: {e}")
            raise
    
    def _setup_query_patterns(self):
        """Setup natural language query patterns"""
        self.query_patterns = [
            # Entity-based patterns - Fixed for our FactBlock schema with Korean support
            {
                "pattern": r"(?:find|show|get).*(?:factblocks?|claims?).*(?:about|mention|regarding)\s+([A-Za-z가-힣\s]+)",
                "type": "entity_search",
                "template": """
                MATCH (f:FactBlock)
                WHERE toLower(f.claim) CONTAINS toLower($entity)
                   OR toLower(f.evidence) CONTAINS toLower($entity)
                   OR toLower(f.summary) CONTAINS toLower($entity)
                RETURN f
                ORDER BY f.confidence_score DESC
                LIMIT $limit
                """
            },
            
            
            # Sector-based patterns
            {
                "pattern": r"(?:find|show|get).*(?:factblocks?|claims?).*(?:in|from|about)\s+(energy|financial|banking|transportation|technology)\s+sector",
                "type": "sector_search", 
                "template": """
                MATCH (f:FactBlock)
                WHERE $sector IN f.affected_sectors
                RETURN f
                ORDER BY f.confidence_score DESC
                LIMIT $limit
                """
            },
            
            # Impact level patterns
            {
                "pattern": r"(?:find|show|get).*(?:high|critical|medium|low)\s+impact.*(?:factblocks?|claims?)",
                "type": "impact_search",
                "template": """
                MATCH (f:FactBlock)
                WHERE f.impact_level = $impact_level
                RETURN f
                ORDER BY f.confidence_score DESC
                LIMIT $limit
                """
            },
            
            # Relationship patterns
            {
                "pattern": r"(?:what|which|show).*(?:factblocks?|claims?).*(?:related|connected).*to\s+([A-Za-z\s]+)",
                "type": "relationship_search",
                "template": """
                MATCH (f1:FactBlock)-[r:RELATES_TO]-(f2:FactBlock)
                WHERE toLower(f1.claim) CONTAINS toLower($term) 
                   OR toLower(f2.claim) CONTAINS toLower($term)
                RETURN f1, f2, r
                ORDER BY r.strength DESC
                LIMIT $limit
                """
            },
            
            # Investment theme patterns
            {
                "pattern": r"(?:find|show|get).*(?:factblocks?|claims?).*(?:investment|theme).*([A-Za-z_]+)",
                "type": "theme_search",
                "template": """
                MATCH (f:FactBlock)
                WHERE ANY(theme IN f.investment_themes WHERE toLower(theme) CONTAINS toLower($theme))
                RETURN f
                ORDER BY f.confidence_score DESC
                LIMIT $limit
                """
            },
            
            # Confidence-based patterns
            {
                "pattern": r"(?:find|show|get).*(?:most|high).*(?:confident|reliable).*(?:factblocks?|claims?)",
                "type": "confidence_search",
                "template": """
                MATCH (f:FactBlock)
                WHERE f.confidence_score > 0.8
                RETURN f
                ORDER BY f.confidence_score DESC
                LIMIT $limit
                """
            },
            
            # General search patterns
            {
                "pattern": r"(?:find|search|get).*(?:factblocks?|claims?).*(?:with|containing)\s+([A-Za-z\s]+)",
                "type": "text_search",
                "template": """
                MATCH (f:FactBlock)
                WHERE toLower(f.claim) CONTAINS toLower($text) 
                   OR toLower(f.evidence) CONTAINS toLower($text)
                RETURN f
                ORDER BY f.confidence_score DESC
                LIMIT $limit
                """
            }
        ]
    
    def _setup_entity_mappings(self):
        """Setup entity name mappings for better recognition"""
        self.entity_mappings = {
            "opec": "OPEC",
            "federal reserve": "Federal Reserve", 
            "fed": "Federal Reserve",
            "중앙은행": "Federal Reserve",
            "연준": "Federal Reserve",
            "미국 연준": "Federal Reserve",
            "항공사": "Airlines",
            "airline": "Airlines",
            "airlines": "Airlines",
            "은행": "Banks",
            "bank": "Banks", 
            "banks": "Banks"
        }
    
    def parse_natural_language(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query into structured format
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary with query type, parameters, and Cypher template
        """
        query_lower = query.lower().strip()
        
        for pattern_info in self.query_patterns:
            match = re.search(pattern_info["pattern"], query_lower)
            if match:
                # Extract parameters based on pattern type
                params = {"limit": 10}  # Default limit
                
                if pattern_info["type"] == "entity_search":
                    entity = match.group(1).strip()
                    # Map to known entities
                    mapped_entity = self.entity_mappings.get(entity, entity)
                    params["entity"] = mapped_entity
                
                
                elif pattern_info["type"] == "sector_search":
                    sector = match.group(1).strip()
                    params["sector"] = sector
                
                elif pattern_info["type"] == "impact_search":
                    # Extract impact level from the query
                    if "high" in query_lower:
                        params["impact_level"] = "high"
                    elif "critical" in query_lower:
                        params["impact_level"] = "critical"
                    elif "medium" in query_lower:
                        params["impact_level"] = "medium"
                    elif "low" in query_lower:
                        params["impact_level"] = "low"
                
                elif pattern_info["type"] == "relationship_search":
                    term = match.group(1).strip()
                    params["term"] = term
                
                elif pattern_info["type"] == "theme_search":
                    theme = match.group(1).strip()
                    params["theme"] = theme
                
                elif pattern_info["type"] == "text_search":
                    text = match.group(1).strip()
                    params["text"] = text
                
                return {
                    "type": pattern_info["type"],
                    "cypher_template": pattern_info["template"],
                    "parameters": params,
                    "original_query": query
                }
        
        # Fallback: general text search that works for any language
        # Extract meaningful words from the query for better matching
        
        # Split query into words, removing common stop words and short words
        words = re.findall(r'\b\w{2,}\b', query)  # Extract words with 2+ characters
        stop_words = {'은', '는', '이', '가', '을', '를', '에', '의', 'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for'}
        meaningful_words = [word for word in words if word.lower() not in stop_words]
        
        # Use the full query for primary search, and individual words for secondary search
        return {
            "type": "fallback_search", 
            "cypher_template": """
            MATCH (f:FactBlock)
            WHERE toLower(f.claim) CONTAINS toLower($query)
               OR toLower(f.evidence) CONTAINS toLower($query)
               OR toLower(f.summary) CONTAINS toLower($query)
               OR ANY(word IN $meaningful_words WHERE (
                   toLower(f.claim) CONTAINS toLower(word)
                   OR toLower(f.evidence) CONTAINS toLower(word)
                   OR toLower(f.summary) CONTAINS toLower(word)
               ))
            RETURN f
            ORDER BY f.confidence_score DESC
            LIMIT $limit
            """,
            "parameters": {"query": query, "meaningful_words": meaningful_words, "limit": 10},
            "original_query": query
        }
    
    def execute_cypher(self, cypher_query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute Cypher query against Neo4j
        
        Args:
            cypher_query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of query results
        """
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run(cypher_query, parameters)
                
                records = []
                for record in result:
                    # Convert Neo4j record to dictionary
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        if hasattr(value, '_properties'):  # Neo4j node/relationship
                            record_dict[key] = dict(value)
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                
                logger.info(f"✅ Cypher query returned {len(records)} results")
                return records
                
        except Exception as e:
            logger.error(f"❌ Cypher query execution failed: {e}")
            return []
    
    def search(self, natural_query: str) -> Dict[str, Any]:
        """
        Main search method: natural language to results
        
        Args:
            natural_query: Natural language query
            
        Returns:
            Dictionary with query info, Cypher query, and results
        """
        # Parse natural language
        parsed_query = self.parse_natural_language(natural_query)
        
        # Execute Cypher query
        results = self.execute_cypher(
            parsed_query["cypher_template"],
            parsed_query["parameters"]
        )
        
        return {
            "original_query": natural_query,
            "query_type": parsed_query["type"], 
            "cypher_query": parsed_query["cypher_template"],
            "parameters": parsed_query["parameters"],
            "results": results,
            "result_count": len(results)
        }
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("🔐 TextToCypher Neo4j connection closed")


def demo_text_to_cypher():
    """Demo the text-to-cypher functionality"""
    
    print("🧪 Testing Text-to-Cypher Retriever...")
    
    try:
        # Initialize retriever
        retriever = TextToCypherRetriever()
        
        # Test queries covering different patterns
        test_queries = [
            "Find FactBlocks about OPEC",
            "Show me claims in the energy sector", 
            "Get high impact FactBlocks",
            "What FactBlocks are related to oil production?",
            "Find FactBlocks with investment themes",
            "Show me the most confident claims",
            "Find FactBlocks containing inflation"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"🔍 Query: '{query}'")
            print(f"{'='*60}")
            
            search_result = retriever.search(query)
            
            print(f"📊 Query Type: {search_result['query_type']}")
            print(f"🔢 Results: {search_result['result_count']}")
            print(f"📝 Parameters: {search_result['parameters']}")
            
            print(f"\n💻 Generated Cypher:")
            cypher_lines = search_result['cypher_query'].strip().split('\n')
            for line in cypher_lines:
                if line.strip():
                    print(f"   {line.strip()}")
            
            if search_result['results']:
                print(f"\n📋 Sample Results:")
                for i, result in enumerate(search_result['results'][:2]):  # Show first 2
                    if 'f' in result:  # FactBlock result
                        factblock = result['f']
                        print(f"   {i+1}. {factblock.get('claim', 'No claim')[:70]}...")
                        if factblock.get('impact_level'):
                            print(f"      📈 Impact: {factblock['impact_level']}")
                    elif 'f1' in result and 'f2' in result:  # Relationship result
                        print(f"   {i+1}. Relationship between:")
                        print(f"      A: {result['f1'].get('claim', '')[:50]}...")
                        print(f"      B: {result['f2'].get('claim', '')[:50]}...")
            else:
                print("   ❌ No results found")
        
        retriever.close()
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    demo_text_to_cypher()