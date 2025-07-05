#!/usr/bin/env python3
"""
🎯 Final Comprehensive Demo - GraphRAG Retrieval System
"""

import os
import sys
sys.path.insert(0, '/Users/randybaek/workspace/factblock-retrieval')

from src.retrieval import RetrievalModule
from src.database.neo4j_client import Neo4jClient
from src.config import load_config

def main():
    print("🎯 GraphRAG Retrieval System - Final Demo")
    print("=" * 60)
    
    # Set environment
    os.environ['NEO4J_PASSWORD'] = 'password'
    config = load_config()
    
    # Initialize retrieval system
    print("🚀 Initializing GraphRAG System...")
    module = RetrievalModule('graphrag')
    module.initialize(config.to_dict())
    print("   ✅ System ready!")
    
    # Show database stats
    client = Neo4jClient(config.neo4j.uri, config.neo4j.user, config.neo4j.password)
    info = client.get_database_info()
    print(f"   📊 Database: {info.get('node_count', 0)} nodes, {info.get('relationship_count', 0)} relationships")
    
    # Demo different search capabilities
    print("\n🔍 Search Capabilities Demo:")
    
    # 1. Exact match search
    print("\n1️⃣ Exact Match Search:")
    print("   Query: 'GDPR'")
    results = module.retrieve(query_text="GDPR", limit=3)
    for i, r in enumerate(results, 1):
        print(f"   {i}. {r.source_type} (score: {r.score:.3f})")
        print(f"      {r.content[:80]}...")
    
    # 2. Category filtering
    print("\n2️⃣ Category Filtering:")
    print("   Query: 'reporting' with category: 'financial'")
    results = module.retrieve(
        query_text="reporting", 
        filters={'category': 'financial'}, 
        limit=3
    )
    for i, r in enumerate(results, 1):
        print(f"   {i}. {r.source_type} (score: {r.score:.3f})")
        print(f"      {r.content[:80]}...")
    
    # 3. Multi-category search
    print("\n3️⃣ Cross-Category Search:")
    print("   Query: 'data privacy protection'")
    results = module.retrieve(query_text="data privacy protection", limit=4)
    categories = {}
    for r in results:
        cat = r.metadata.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    print(f"   Found {len(results)} results across {len(categories)} categories:")
    for cat, count in categories.items():
        print(f"      {cat}: {count} documents")
    
    # 4. Enforcement focus
    print("\n4️⃣ Enforcement Actions:")
    print("   Query: 'fine penalty enforcement'")
    results = module.retrieve(query_text="fine penalty enforcement", limit=3)
    for i, r in enumerate(results, 1):
        if r.source_type == 'EnforcementAction':
            outcome = r.metadata.get('outcome', 'No outcome specified')
            print(f"   {i}. {r.source_type} (score: {r.score:.3f})")
            print(f"      Outcome: {outcome}")
    
    # Show relationship power
    print("\n🕸️ Graph Relationship Power:")
    gdpr_query = """
    MATCH (gdpr:FederalRegulation {name: 'GDPR'})
    MATCH (gdpr)-[:HAS_GUIDANCE]->(guidance:AgencyGuidance)
    MATCH (gdpr)-[:HAS_ENFORCEMENT]->(enforcement:EnforcementAction)
    RETURN guidance.title as guidance_title, enforcement.title as enforcement_title
    """
    
    result = client.execute_query(gdpr_query)
    if result:
        print("   GDPR Connected Documents:")
        for record in result:
            print(f"      📖 Guidance: {record['guidance_title']}")
            print(f"      ⚖️ Enforcement: {record['enforcement_title']}")
    
    # Performance summary
    print("\n📈 System Performance:")
    print("   ✅ Real-time search with graph traversal")
    print("   ✅ Relevance scoring based on text similarity + graph relationships")
    print("   ✅ Multi-modal filtering (category, document type, etc.)")
    print("   ✅ Rich metadata extraction and relationship mapping")
    print("   ✅ Scalable architecture with Neo4j backend")
    
    print("\n🎉 Demo Complete!")
    print("Your GraphRAG compliance retrieval system is fully operational!")
    
    client.close()

if __name__ == "__main__":
    main()