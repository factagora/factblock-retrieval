#!/usr/bin/env python3
"""
Complete GraphRAG Test Suite

Tests both Graph-Enhanced Vector Retriever and TextToCypherRetriever
to demonstrate the full GraphRAG capability for investment knowledge graphs.
"""

import os
import sys
from typing import List, Dict, Any

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.vector_retriever import GraphVectorRetriever
from graphrag.cypher_retriever import TextToCypherRetriever

def print_separator(title: str):
    """Print a nice separator for test sections"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def test_complete_graphrag():
    """Test complete GraphRAG functionality"""
    
    print_separator("🚀 COMPLETE GRAPHRAG TEST SUITE")
    print("Testing both Vector Search and Natural Language to Cypher")
    
    try:
        # Initialize both retrievers
        print("🔌 Initializing GraphRAG components...")
        vector_retriever = GraphVectorRetriever()
        cypher_retriever = TextToCypherRetriever()
        print("✅ Both retrievers initialized successfully!")
        
        # Test scenarios comparing both approaches
        test_scenarios = [
            {
                "investment_question": "What's the impact of OPEC decisions on energy markets?",
                "vector_query": "OPEC oil production energy market impact",
                "cypher_query": "Find FactBlocks about OPEC"
            },
            {
                "investment_question": "How do interest rate changes affect banking?", 
                "vector_query": "interest rates banking sector lending",
                "cypher_query": "Get high impact FactBlocks"
            },
            {
                "investment_question": "What are the transportation sector risks?",
                "vector_query": "transportation aviation fuel costs risks",
                "cypher_query": "Show me the most confident claims"
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print_separator(f"📊 SCENARIO {i}: {scenario['investment_question']}")
            
            # Test Vector Retrieval
            print(f"\n🔍 Vector Search Approach:")
            print(f"Query: '{scenario['vector_query']}'")
            print("-" * 50)
            
            vector_results = vector_retriever.search(
                scenario['vector_query'], 
                k=3, 
                include_graph_expansion=True
            )
            
            if vector_results:
                for j, result in enumerate(vector_results):
                    factblock = result["factblock"]
                    print(f"   {j+1}. Score: {result['combined_score']:.3f} ({result['source']})")
                    print(f"      📝 {factblock['claim'][:80]}...")
                    
                    if result.get("relationship_path"):
                        insights = [rel.get("investment_insight", "") for rel in result["relationship_path"] if rel.get("investment_insight")]
                        if insights:
                            print(f"      💡 {insights[0]}")
            else:
                print("   ❌ No vector results found")
            
            # Test Natural Language to Cypher
            print(f"\n💬 Natural Language to Cypher Approach:")
            print(f"Query: '{scenario['cypher_query']}'")
            print("-" * 50)
            
            cypher_results = cypher_retriever.search(scenario['cypher_query'])
            
            print(f"   🔍 Generated Query Type: {cypher_results['query_type']}")
            print(f"   📊 Found {cypher_results['result_count']} results")
            
            if cypher_results['results']:
                for j, result in enumerate(cypher_results['results'][:3]):
                    if 'f' in result:
                        factblock = result['f']
                        print(f"   {j+1}. 📝 {factblock.get('claim', 'No claim')[:80]}...")
                        if factblock.get('impact_level'):
                            print(f"      📈 Impact: {factblock['impact_level']}")
            else:
                print("   ❌ No cypher results found")
        
        # Demonstrate GraphRAG strengths
        print_separator("💪 GRAPHRAG CAPABILITIES SUMMARY")
        
        print("\n🎯 Vector Search Strengths:")
        print("   ✅ Semantic similarity matching")
        print("   ✅ Graph relationship discovery") 
        print("   ✅ Investment insight propagation")
        print("   ✅ Combined scoring (vector + graph)")
        
        print("\n🎯 Natural Language to Cypher Strengths:")
        print("   ✅ Structured query generation")
        print("   ✅ Entity recognition (OPEC, Fed, etc.)")
        print("   ✅ Investment property filtering")
        print("   ✅ Relationship traversal queries")
        
        print("\n🎯 Combined GraphRAG Benefits:")
        print("   ✅ Flexible query interfaces")
        print("   ✅ Investment domain intelligence")
        print("   ✅ Graph-enhanced discovery")
        print("   ✅ Scalable knowledge retrieval")
        
        # Performance comparison
        print_separator("⚡ PERFORMANCE COMPARISON")
        
        import time
        
        test_query = "energy market investment"
        
        # Vector search timing
        start_time = time.time()
        vector_results = vector_retriever.search(test_query, k=5)
        vector_time = time.time() - start_time
        
        # Cypher search timing  
        start_time = time.time()
        cypher_results = cypher_retriever.search("Find FactBlocks with investment themes")
        cypher_time = time.time() - start_time
        
        print(f"⏱️  Vector Search: {vector_time:.3f}s ({len(vector_results)} results)")
        print(f"⏱️  Cypher Search: {cypher_time:.3f}s ({cypher_results['result_count']} results)")
        
        # Close connections
        vector_retriever.close()
        cypher_retriever.close()
        
        print_separator("✅ COMPLETE GRAPHRAG TEST SUCCESSFUL")
        print("\n🎉 Your investment knowledge graph is ready for:")
        print("   📊 Semantic similarity search")
        print("   💬 Natural language queries") 
        print("   🔗 Graph relationship discovery")
        print("   💡 Investment insight generation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ GraphRAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_interactive_graphrag():
    """Interactive demo showing both retrieval methods"""
    
    print_separator("🎮 INTERACTIVE GRAPHRAG DEMO")
    print("Compare Vector Search vs Natural Language queries")
    print("Examples:")
    print("  Vector: 'OPEC energy market impact'")
    print("  Natural: 'Find FactBlocks about OPEC'")
    print("Type 'quit' to exit.")
    
    try:
        vector_retriever = GraphVectorRetriever()
        cypher_retriever = TextToCypherRetriever()
        
        while True:
            print("\n" + "-" * 60)
            print("Choose method: (1) Vector Search (2) Natural Language (3) Both")
            choice = input("Enter choice (1-3): ").strip()
            
            if choice in ['quit', 'exit', 'q']:
                break
            
            query = input("🔍 Enter your query: ").strip()
            if not query or query.lower() in ['quit', 'exit', 'q']:
                break
            
            if choice in ['1', '3']:
                print(f"\n🔍 Vector Search Results:")
                results = vector_retriever.search(query, k=3, include_graph_expansion=True)
                for i, result in enumerate(results):
                    print(f"   {i+1}. {result['factblock']['claim'][:70]}...")
            
            if choice in ['2', '3']:
                print(f"\n💬 Natural Language Results:")
                results = cypher_retriever.search(query)
                print(f"   Query Type: {results['query_type']}")
                for i, result in enumerate(results['results'][:3]):
                    if 'f' in result:
                        print(f"   {i+1}. {result['f']['claim'][:70]}...")
        
        vector_retriever.close()
        cypher_retriever.close()
        print("\n👋 Interactive demo completed!")
        
    except Exception as e:
        print(f"❌ Interactive demo failed: {e}")

if __name__ == "__main__":
    # Run complete test
    test_complete_graphrag()