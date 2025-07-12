#!/usr/bin/env python3
"""
Test OPEC retrieval specifically to debug why GraphRAG isn't finding the data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.dirname(__file__))

from graphrag.smart_router import SmartGraphRAGRouter, PerformanceMode
from graphrag.cypher_retriever import TextToCypherRetriever
from database.neo4j_client import Neo4jClient
import json

def test_direct_neo4j_queries():
    """Test direct Neo4j queries to find OPEC data"""
    print("=== Testing Direct Neo4j Queries ===\n")
    
    client = Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    
    test_queries = [
        {
            "name": "Direct OPEC search in claim",
            "query": """
            MATCH (f:FactBlock)
            WHERE toLower(f.claim) CONTAINS 'opec'
            RETURN f.id, f.claim, f.evidence, f.confidence_score
            ORDER BY f.confidence_score DESC
            LIMIT 5
            """
        },
        {
            "name": "Korean text search for 감산",
            "query": """
            MATCH (f:FactBlock)
            WHERE f.claim CONTAINS '감산'
            RETURN f.id, f.claim, f.evidence, f.confidence_score
            ORDER BY f.confidence_score DESC
            LIMIT 5
            """
        },
        {
            "name": "Search for exact OPEC phrase",
            "query": """
            MATCH (f:FactBlock)
            WHERE f.claim CONTAINS 'OPEC이 감산 합의에 도달했다'
            RETURN f.id, f.claim, f.evidence, f.confidence_score
            ORDER BY f.confidence_score DESC
            LIMIT 5
            """
        },
        {
            "name": "Search in both claim and evidence",
            "query": """
            MATCH (f:FactBlock)
            WHERE f.claim CONTAINS 'OPEC' 
               OR f.evidence CONTAINS 'OPEC'
               OR f.claim CONTAINS '감산'
               OR f.evidence CONTAINS '감산'
            RETURN f.id, f.claim, f.evidence, f.confidence_score
            ORDER BY f.confidence_score DESC
            LIMIT 5
            """
        }
    ]
    
    for test in test_queries:
        print(f"Testing: {test['name']}")
        try:
            results = client.execute_query(test['query'])
            print(f"Found {len(results)} results")
            
            for i, result in enumerate(results[:2]):
                print(f"  Result {i+1}:")
                print(f"    ID: {result.get('f.id', 'N/A')}")
                print(f"    Claim: {result.get('f.claim', 'N/A')}")
                print(f"    Evidence: {result.get('f.evidence', 'N/A')}")
                print(f"    Confidence: {result.get('f.confidence_score', 'N/A')}")
                print()
        except Exception as e:
            print(f"Error: {e}")
        print()
    
    client.close()

def test_cypher_retriever():
    """Test the Cypher retriever with OPEC queries"""
    print("=== Testing Cypher Retriever ===\n")
    
    try:
        retriever = TextToCypherRetriever()
        
        test_queries = [
            "Find FactBlocks about OPEC",
            "Show me claims containing OPEC",
            "OPEC이 감산 합의에 도달했으며, 주요 산유국들이 원유 생산량을 일일 200만 배럴 감축하기로 합의했다",
            "Find FactBlocks containing oil production",
            "Get energy sector claims"
        ]
        
        for query in test_queries:
            print(f"Query: '{query}'")
            
            # First show how it parses the query
            parsed = retriever.parse_natural_language(query)
            print(f"  Parsed type: {parsed['type']}")
            print(f"  Parameters: {parsed['parameters']}")
            
            # Then show the search results
            result = retriever.search(query)
            print(f"  Results found: {result['result_count']}")
            
            if result['results']:
                for i, res in enumerate(result['results'][:2]):
                    if 'f' in res:
                        factblock = res['f']
                        print(f"    {i+1}. {factblock.get('claim', 'No claim')[:80]}...")
                        print(f"       Evidence: {factblock.get('evidence', 'No evidence')[:80]}...")
            print()
        
        retriever.close()
        
    except Exception as e:
        print(f"Error testing Cypher retriever: {e}")
        import traceback
        traceback.print_exc()

def test_smart_router():
    """Test the Smart Router with OPEC queries"""
    print("=== Testing Smart Router ===\n")
    
    try:
        router = SmartGraphRAGRouter(performance_mode=PerformanceMode.COMPREHENSIVE)
        
        opec_query = "OPEC이 감산 합의에 도달했으며, 주요 산유국들이 원유 생산량을 일일 200만 배럴 감축하기로 합의했다"
        
        print(f"Query: '{opec_query}'")
        
        # Test strategy selection
        strategy = router.choose_strategy(opec_query)
        print(f"Strategy:")
        print(f"  Type: {strategy['detected_type']}")
        print(f"  Confidence: {strategy['confidence']}")
        print(f"  Use Vector: {strategy['use_vector']}")
        print(f"  Use Cypher: {strategy['use_cypher']}")
        print(f"  Reasoning: {'; '.join(strategy['reasoning'])}")
        print()
        
        # Test actual search
        print("Executing search...")
        search_results = router.search(opec_query, max_results=5)
        
        print(f"Execution time: {search_results['execution_time']:.3f}s")
        print(f"Combined results: {search_results['result_count']}")
        
        # Show vector results
        if search_results.get('vector_results'):
            print(f"\nVector results: {len(search_results['vector_results'])}")
            for i, result in enumerate(search_results['vector_results'][:2]):
                print(f"  {i+1}. Score: {result.get('combined_score', 0):.3f}")
                factblock = result.get('factblock', {})
                print(f"     Claim: {factblock.get('claim', 'No claim')[:80]}...")
        
        # Show cypher results
        if search_results.get('cypher_results', {}).get('results'):
            cypher_results = search_results['cypher_results']['results']
            print(f"\nCypher results: {len(cypher_results)}")
            for i, result in enumerate(cypher_results[:2]):
                if 'f' in result:
                    factblock = result['f']
                    print(f"  {i+1}. Confidence: {factblock.get('confidence_score', 0):.3f}")
                    print(f"     Claim: {factblock.get('claim', 'No claim')[:80]}...")
        
        # Show combined results
        if search_results.get('combined_results'):
            print(f"\nCombined results: {len(search_results['combined_results'])}")
            for i, result in enumerate(search_results['combined_results']):
                factblock = result.get('factblock', {})
                print(f"  {i+1}. Score: {result.get('score', 0):.3f} (Source: {result.get('source', 'unknown')})")
                print(f"     Claim: {factblock.get('claim', 'No claim')[:80]}...")
                print(f"     Evidence: {factblock.get('evidence', 'No evidence')[:80]}...")
        
        router.close()
        
    except Exception as e:
        print(f"Error testing Smart Router: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("🔍 Debugging OPEC Retrieval Issues\n")
    
    test_direct_neo4j_queries()
    test_cypher_retriever()
    test_smart_router()
    
    print("✅ All tests completed!")

if __name__ == "__main__":
    main()