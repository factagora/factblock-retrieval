#!/usr/bin/env python3
"""
Basic GraphRAG Test Script

Test the GraphRAG system with simple components first.
"""

import os
import sys
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all GraphRAG components can be imported"""
    print("🧪 Testing GraphRAG imports...")
    
    try:
        from graphrag.simple_embeddings import SimpleFactBlockEmbeddings
        print("✅ SimpleFactBlockEmbeddings imported successfully")
        
        from graphrag.cypher_retriever import TextToCypherRetriever
        print("✅ TextToCypherRetriever imported successfully")
        
        from graphrag.vector_retriever import GraphVectorRetriever
        print("✅ GraphVectorRetriever imported successfully")
        
        from graphrag.smart_router import SmartGraphRAGRouter
        print("✅ SmartGraphRAGRouter imported successfully")
        
        from exporters.neo4j_loader import Neo4jDataLoader
        print("✅ Neo4jDataLoader imported successfully")
        
        from exporters.to_neo4j import Neo4jExporter
        print("✅ Neo4jExporter imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_embeddings():
    """Test the simple embeddings system"""
    print("\n🧪 Testing Simple Embeddings System...")
    
    try:
        from graphrag.simple_embeddings import SimpleFactBlockEmbeddings
        
        # Initialize embeddings
        embedder = SimpleFactBlockEmbeddings()
        
        # Test with sample data
        sample_factblocks = [
            {
                "id": "test_1",
                "claim": "OPEC agreed to reduce oil production by 2 million barrels per day",
                "evidence": "Major oil-producing countries in OPEC agreed to significant production cuts",
                "financial_metadata": {
                    "market_impact": {
                        "affected_sectors": ["energy", "transportation"],
                        "impact_level": "high"
                    },
                    "investment_themes": [
                        {"theme_name": "energy_investment"}
                    ]
                }
            },
            {
                "id": "test_2", 
                "claim": "Federal Reserve raises interest rates by 0.75 basis points",
                "evidence": "The Fed increased rates to combat inflation",
                "financial_metadata": {
                    "market_impact": {
                        "affected_sectors": ["financial", "real_estate"],
                        "impact_level": "high"
                    }
                }
            }
        ]
        
        # Build vocabulary and vectors
        embedder.build_vocabulary(sample_factblocks)
        embedder.build_factblock_vectors(sample_factblocks)
        
        # Test search
        query = "oil production energy markets"
        results = embedder.search(query, k=2)
        
        print(f"✅ Search completed: {len(results)} results")
        for i, (idx, score, factblock) in enumerate(results):
            print(f"   {i+1}. Score: {score:.3f} - {factblock['claim'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple embeddings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_smart_router_patterns():
    """Test the smart router query analysis"""
    print("\n🧪 Testing Smart Router Query Analysis...")
    
    try:
        from graphrag.smart_router import SmartGraphRAGRouter, QueryType, PerformanceMode
        
        # Initialize router (without connecting to Neo4j)
        router = SmartGraphRAGRouter(performance_mode=PerformanceMode.BALANCED)
        
        # Test query analysis
        test_queries = [
            "Find FactBlocks about OPEC",
            "What's the impact of inflation on markets?",
            "Get high impact energy sector claims",
            "Show me interesting investment trends"
        ]
        
        for query in test_queries:
            query_type, confidence = router.analyze_query(query)
            strategy = router.choose_strategy(query)
            
            print(f"   Query: '{query}'")
            print(f"   Type: {query_type.value} (confidence: {confidence:.2f})")
            print(f"   Strategy: Vector={strategy['use_vector']}, Cypher={strategy['use_cypher']}")
            print()
        
        print("✅ Smart router pattern analysis successful!")
        return True
        
    except Exception as e:
        print(f"❌ Smart router test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cypher_patterns():
    """Test the Cypher pattern matching"""
    print("\n🧪 Testing Cypher Pattern Matching...")
    
    try:
        from graphrag.cypher_retriever import TextToCypherRetriever
        
        # Initialize without Neo4j connection
        retriever = TextToCypherRetriever.__new__(TextToCypherRetriever)
        retriever._setup_query_patterns()
        retriever._setup_entity_mappings()
        
        # Test query parsing
        test_queries = [
            "Find FactBlocks about OPEC",
            "Show me claims in the energy sector",
            "Get high impact FactBlocks",
            "What FactBlocks are related to oil production?"
        ]
        
        for query in test_queries:
            parsed = retriever.parse_natural_language(query)
            print(f"   Query: '{query}'")
            print(f"   Type: {parsed['type']}")
            print(f"   Parameters: {parsed['parameters']}")
            print()
        
        print("✅ Cypher pattern matching successful!")
        return True
        
    except Exception as e:
        print(f"❌ Cypher patterns test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n🧪 Testing Configuration Loading...")
    
    try:
        import json
        with open("config/database.json", "r") as f:
            config = json.load(f)
        
        print(f"✅ Neo4j config loaded:")
        print(f"   URI: {config['neo4j']['uri']}")
        print(f"   Database: {config['neo4j']['database']}")
        print(f"   Batch size: {config['batch_settings']['batch_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading test failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("🚀 GraphRAG Basic Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Simple Embeddings", test_simple_embeddings),
        ("Smart Router Patterns", test_smart_router_patterns),
        ("Cypher Patterns", test_cypher_patterns),
        ("Configuration Loading", test_config_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All basic tests passed! GraphRAG system is ready.")
        print("\nNext steps:")
        print("1. Set up Neo4j database")
        print("2. Load your FactBlock data")
        print("3. Run the full GraphRAG test suite")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)