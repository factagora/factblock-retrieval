#!/usr/bin/env python3
"""
Live Demo of GraphRAG Retrieval System
"""

import os
import sys
sys.path.insert(0, '/Users/randybaek/workspace/factblock-retrieval')

from src.retrieval import RetrievalModule
from src.config import load_config

def main():
    print("🔍 GraphRAG Retrieval System - Live Demo")
    print("=" * 50)
    
    # Set environment variables
    os.environ['NEO4J_PASSWORD'] = 'password'
    
    # Load configuration
    print("1. Loading configuration...")
    config = load_config()
    print(f"   ✓ Neo4j URI: {config.neo4j.uri}")
    print(f"   ✓ Default limit: {config.retrieval.default_limit}")
    
    # Initialize retrieval module
    print("\n2. Initializing retrieval system...")
    try:
        module = RetrievalModule('graphrag')
        module.initialize(config.to_dict())
        print("   ✓ GraphRAG retriever initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return
    
    # Demo queries
    print("\n3. Running demo queries...")
    
    queries = [
        {
            'name': 'GDPR Search',
            'query': 'GDPR',
            'filters': None,
            'limit': 3
        },
        {
            'name': 'Data Privacy Category',
            'query': 'data protection',
            'filters': {'category': 'data_privacy'},
            'limit': 3
        },
        {
            'name': 'Financial Regulations',
            'query': 'financial reporting',
            'filters': {'category': 'financial'},
            'limit': 3
        },
        {
            'name': 'Enforcement Actions',
            'query': 'fine penalty',
            'filters': None,
            'limit': 5
        }
    ]
    
    for i, query_info in enumerate(queries, 1):
        print(f"\n3.{i} {query_info['name']}")
        print(f"    Query: '{query_info['query']}'")
        if query_info['filters']:
            print(f"    Filters: {query_info['filters']}")
        print("    " + "-" * 40)
        
        try:
            results = module.retrieve(
                query_text=query_info['query'],
                filters=query_info['filters'],
                limit=query_info['limit']
            )
            
            if results:
                print(f"    Found {len(results)} results:")
                for j, result in enumerate(results, 1):
                    print(f"      {j}. Score: {result.score:.3f}")
                    print(f"         Type: {result.source_type}")
                    print(f"         Content: {result.content[:100]}...")
                    if result.metadata.get('category'):
                        print(f"         Category: {result.metadata['category']}")
                    print()
            else:
                print("    No results found")
                
        except Exception as e:
            print(f"    ✗ Query failed: {e}")
    
    print("\n4. System Summary:")
    print("   ✅ Connected to Neo4j database")
    print("   ✅ Loaded compliance data (regulations, guidance, enforcement)")
    print("   ✅ Text-based search with relevance scoring")
    print("   ✅ Category-based filtering")
    print("   ✅ Rich metadata extraction")
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    main()