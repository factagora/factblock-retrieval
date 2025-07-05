#!/usr/bin/env python3
"""
Simple test to debug GraphRAG API issues
"""

import sys
import os

# Set environment
os.environ['NEO4J_PASSWORD'] = 'password'

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.retrieval import RetrievalModule
from src.config import load_config

def test_retrieval_directly():
    """Test retrieval module directly"""
    print("Testing retrieval module directly...")
    
    try:
        config = load_config()
        module = RetrievalModule('graphrag')
        module.initialize(config.to_dict())
        
        # Test query
        results = module.retrieve(
            query_text="GDPR",
            filters={'category': 'data_privacy'},
            limit=3
        )
        
        print(f"Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.3f}, Type: {result.source_type}")
            print(f"   Content: {result.content[:100]}...")
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_retrieval_directly()