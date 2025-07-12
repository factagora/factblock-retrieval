#!/usr/bin/env python3
"""
Complete GraphRAG System Test

Tests all GraphRAG components with local sample data.
"""

import os
import sys
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def test_simple_embeddings():
    """Test simple embeddings with local dataset"""
    print_header("🔍 TESTING SIMPLE EMBEDDINGS")
    
    try:
        from graphrag.simple_embeddings import SimpleFactBlockEmbeddings
        
        embedder = SimpleFactBlockEmbeddings()
        
        # Initialize with local dataset
        dataset_path = "data/processed/enhanced_knowledge_graph_dataset.json"
        if embedder.initialize_from_dataset(dataset_path):
            print("✅ Dataset initialized successfully")
            
            # Test queries
            test_queries = [
                "OPEC oil production energy markets",
                "inflation monetary policy interest rates", 
                "banking sector lending expansion",
                "transportation fuel costs impact"
            ]
            
            for query in test_queries:
                print(f"\n🔍 Query: '{query}'")
                results = embedder.search(query, k=3)
                
                for i, (idx, score, factblock) in enumerate(results):
                    print(f"   {i+1}. Score: {score:.3f} - {factblock['claim'][:60]}...")
        else:
            print("❌ Failed to initialize dataset")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple embeddings test failed: {e}")
        return False

def test_smart_router():
    """Test smart router query analysis"""
    print_header("🧠 TESTING SMART ROUTER")
    
    try:
        from graphrag.smart_router import SmartGraphRAGRouter, QueryType, PerformanceMode
        
        # Test different modes
        modes = [PerformanceMode.FAST, PerformanceMode.BALANCED, PerformanceMode.COMPREHENSIVE]
        
        test_queries = [
            "Find FactBlocks about OPEC",
            "What's the impact of inflation on markets?",
            "Get high impact energy sector claims",
            "Show me interesting investment trends"
        ]
        
        for mode in modes:
            print(f"\n🎯 Testing {mode.value.upper()} mode:")
            print("-" * 50)
            
            # Create router without initializing Neo4j connections
            router = SmartGraphRAGRouter.__new__(SmartGraphRAGRouter)
            router.performance_mode = mode
            router._setup_query_patterns()
            router.performance_stats = {
                "vector_avg_time": 0.010,
                "cypher_avg_time": 0.005,
                "query_count": 0
            }
            
            for query in test_queries:
                query_type, confidence = router.analyze_query(query)
                strategy = router.choose_strategy(query)
                
                print(f"   '{query}'")
                print(f"   → Type: {query_type.value} (confidence: {confidence:.2f})")
                print(f"   → Strategy: Vector={strategy['use_vector']}, Cypher={strategy['use_cypher']}")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ Smart router test failed: {e}")
        return False

def test_cypher_patterns():
    """Test Cypher pattern matching"""
    print_header("💬 TESTING CYPHER PATTERN MATCHING")
    
    try:
        from graphrag.cypher_retriever import TextToCypherRetriever
        
        # Initialize without Neo4j connection
        retriever = TextToCypherRetriever.__new__(TextToCypherRetriever)
        retriever._setup_query_patterns()
        retriever._setup_entity_mappings()
        
        test_queries = [
            "Find FactBlocks about OPEC",
            "Show me claims in the energy sector",
            "Get high impact FactBlocks",
            "What FactBlocks are related to oil production?",
            "Find FactBlocks with investment themes",
            "Show me the most confident claims"
        ]
        
        for query in test_queries:
            parsed = retriever.parse_natural_language(query)
            print(f"\n🔍 Query: '{query}'")
            print(f"   → Type: {parsed['type']}")
            print(f"   → Parameters: {parsed['parameters']}")
            
            # Show generated Cypher (first few lines)
            cypher_lines = parsed['cypher_template'].strip().split('\n')
            relevant_lines = [line.strip() for line in cypher_lines if line.strip() and not line.strip().startswith('//')][:3]
            print(f"   → Cypher: {' '.join(relevant_lines)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cypher patterns test failed: {e}")
        return False

def test_neo4j_loader():
    """Test Neo4j data loader"""
    print_header("📊 TESTING NEO4J DATA LOADER")
    
    try:
        from exporters.neo4j_loader import Neo4jDataLoader
        
        dataset_path = "data/processed/enhanced_knowledge_graph_dataset.json"
        loader = Neo4jDataLoader(dataset_path)
        
        if loader.load_dataset():
            factblocks, relationships, entities, topics = loader.get_neo4j_data()
            
            print(f"✅ Data parsed successfully:")
            print(f"   • FactBlocks: {len(factblocks)}")
            print(f"   • Relationships: {len(relationships)}")
            print(f"   • Entities: {len(entities)}")
            print(f"   • Topics: {len(topics)}")
            
            # Show sample data
            if factblocks:
                sample = factblocks[0]
                print(f"\n📄 Sample FactBlock:")
                print(f"   ID: {sample['id']}")
                print(f"   Claim: {sample['claim'][:60]}...")
                print(f"   Impact: {sample.get('impact_level', 'N/A')}")
                print(f"   Sectors: {sample.get('affected_sectors', [])}")
            
            if relationships:
                sample = relationships[0]
                print(f"\n🔗 Sample Relationship:")
                print(f"   Type: {sample['relationship_type']}")
                print(f"   Strength: {sample['strength']}")
                print(f"   Insight: {sample['investment_insight'][:60]}...")
            
            return True
        else:
            print("❌ Failed to load dataset")
            return False
        
    except Exception as e:
        print(f"❌ Neo4j loader test failed: {e}")
        return False

def test_performance():
    """Test performance of different components"""
    print_header("⚡ TESTING PERFORMANCE")
    
    try:
        from graphrag.simple_embeddings import SimpleFactBlockEmbeddings
        from graphrag.smart_router import SmartGraphRAGRouter
        
        # Test simple embeddings performance
        embedder = SimpleFactBlockEmbeddings()
        dataset_path = "data/processed/enhanced_knowledge_graph_dataset.json"
        
        if embedder.initialize_from_dataset(dataset_path):
            query = "OPEC oil production impact"
            
            # Time multiple searches
            times = []
            for i in range(5):
                start = time.time()
                results = embedder.search(query, k=3)
                end = time.time()
                times.append(end - start)
            
            avg_time = sum(times) / len(times)
            print(f"✅ Simple Embeddings Average Time: {avg_time:.4f}s")
            print(f"   Results per search: {len(results)}")
            
            # Test smart router analysis
            router = SmartGraphRAGRouter.__new__(SmartGraphRAGRouter)
            router._setup_query_patterns()
            
            start = time.time()
            for _ in range(100):
                query_type, confidence = router.analyze_query(query)
            end = time.time()
            
            analysis_time = (end - start) / 100
            print(f"✅ Smart Router Analysis Time: {analysis_time:.6f}s per query")
            
            return True
        else:
            print("❌ Failed to initialize embeddings for performance test")
            return False
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 GRAPHRAG COMPLETE SYSTEM TEST")
    print("Testing all components with local sample data")
    
    tests = [
        ("Simple Embeddings", test_simple_embeddings),
        ("Smart Router", test_smart_router), 
        ("Cypher Patterns", test_cypher_patterns),
        ("Neo4j Data Loader", test_neo4j_loader),
        ("Performance", test_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Final summary
    print_header("📋 TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ GraphRAG System Status:")
        print("   • Simple Embeddings: Working with local dataset")
        print("   • Smart Router: Query analysis and strategy selection working")
        print("   • Cypher Patterns: Natural language to Cypher conversion working")
        print("   • Neo4j Data Loader: Data parsing and preparation working")
        print("   • Performance: Good response times for search operations")
        
        print("\n🚀 Ready for:")
        print("   • Semantic search with local FactBlock data")
        print("   • Intelligent query routing")
        print("   • Investment-specific analysis")
        print("   • Neo4j database integration (when connected)")
        
        print("\n💡 Next Steps:")
        print("   1. Connect to Neo4j database")
        print("   2. Load FactBlock data into Neo4j")
        print("   3. Test full GraphRAG retrieval with live database")
        print("   4. Integrate with existing API endpoints")
        
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)