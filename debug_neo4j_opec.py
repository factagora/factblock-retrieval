#!/usr/bin/env python3
"""
Debug script for investigating OPEC FactBlock retrieval issues in Neo4j.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.neo4j_client import Neo4jClient
import json
from typing import List, Dict, Any

def main():
    """Main debugging function"""
    print("=== Neo4j OPEC FactBlock Debugging ===\n")
    
    # Connect to Neo4j
    print("1. Connecting to Neo4j...")
    try:
        client = Neo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j", 
            password="password"
        )
        
        if client.verify_connectivity():
            print("✓ Successfully connected to Neo4j")
        else:
            print("✗ Failed to verify Neo4j connectivity")
            return
            
    except Exception as e:
        print(f"✗ Failed to connect to Neo4j: {e}")
        return
    
    # Get database info
    print("\n2. Database Information:")
    try:
        info = client.get_database_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"   Error getting database info: {e}")
    
    # Check Neo4j version first
    print("\n3. Neo4j Version:")
    try:
        query = "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition"
        results = client.execute_query(query)
        for result in results:
            print(f"   {result['name']}: {result['versions']} ({result['edition']})")
    except Exception as e:
        print(f"   Error getting version: {e}")
    
    # Check what node types exist
    print("\n4. Node Types in Database:")
    try:
        query = "CALL db.labels() YIELD label RETURN label ORDER BY label"
        results = client.execute_query(query)
        for result in results:
            print(f"   - {result['label']}")
    except Exception as e:
        print(f"   Error getting node types: {e}")
    
    # Check FactBlock nodes
    print("\n5. FactBlock Node Analysis:")
    try:
        # Count FactBlocks
        query = "MATCH (f:FactBlock) RETURN count(f) as count"
        result = client.execute_query(query)
        factblock_count = result[0]['count'] if result else 0
        print(f"   Total FactBlocks: {factblock_count}")
        
        if factblock_count > 0:
            # Get sample FactBlock structure
            query = """
            MATCH (f:FactBlock) 
            RETURN f 
            LIMIT 1
            """
            result = client.execute_query(query)
            if result:
                sample_factblock = result[0]['f']
                print(f"   Sample FactBlock properties: {list(sample_factblock.keys())}")
        
    except Exception as e:
        print(f"   Error analyzing FactBlocks: {e}")
    
    # Search for OPEC-related content
    print("\n6. OPEC Content Search:")
    opec_query = "OPEC이 감산 합의에 도달했으며, 주요 산유국들이 원유 생산량을 일일 200만 배럴 감축하기로 합의했다"
    
    # Test different search approaches
    search_tests = [
        {
            "name": "Direct text search in all properties",
            "query": """
            MATCH (f:FactBlock)
            WHERE toLower(toString(f.content)) CONTAINS toLower('OPEC')
               OR toLower(toString(f.title)) CONTAINS toLower('OPEC')
               OR toLower(toString(f.summary)) CONTAINS toLower('OPEC')
               OR toLower(toString(f.claim)) CONTAINS toLower('OPEC')
               OR toLower(toString(f.evidence)) CONTAINS toLower('OPEC')
            RETURN f.id as id, f.title as title, f.content as content, f.claim as claim, f.evidence as evidence
            LIMIT 10
            """
        },
        {
            "name": "Korean text search for 감산 (production cut)",
            "query": """
            MATCH (f:FactBlock)
            WHERE toString(f.content) CONTAINS '감산'
               OR toString(f.title) CONTAINS '감산'
               OR toString(f.summary) CONTAINS '감산'
               OR toString(f.claim) CONTAINS '감산'
               OR toString(f.evidence) CONTAINS '감산'
            RETURN f.id as id, f.title as title, f.content as content, f.claim as claim, f.evidence as evidence
            LIMIT 10
            """
        },
        {
            "name": "Korean text search for 원유 (crude oil)",
            "query": """
            MATCH (f:FactBlock)
            WHERE toString(f.content) CONTAINS '원유'
               OR toString(f.title) CONTAINS '원유'
               OR toString(f.summary) CONTAINS '원유'
               OR toString(f.claim) CONTAINS '원유'
               OR toString(f.evidence) CONTAINS '원유'
            RETURN f.id as id, f.title as title, f.content as content, f.claim as claim, f.evidence as evidence
            LIMIT 10
            """
        },
        {
            "name": "Search for production-related keywords",
            "query": """
            MATCH (f:FactBlock)
            WHERE toLower(toString(f.content)) CONTAINS 'production'
               OR toLower(toString(f.content)) CONTAINS 'barrel'
               OR toLower(toString(f.content)) CONTAINS 'oil'
               OR toLower(toString(f.title)) CONTAINS 'production'
               OR toLower(toString(f.title)) CONTAINS 'barrel'
               OR toLower(toString(f.title)) CONTAINS 'oil'
            RETURN f.id as id, f.title as title, f.content as content, f.claim as claim, f.evidence as evidence
            LIMIT 10
            """
        },
        {
            "name": "All FactBlocks sample (first 5)",
            "query": """
            MATCH (f:FactBlock)
            RETURN f.id as id, f.title as title, f.content as content, f.claim as claim, f.evidence as evidence
            LIMIT 5
            """
        }
    ]
    
    for test in search_tests:
        print(f"\n   Testing: {test['name']}")
        try:
            results = client.execute_query(test['query'])
            print(f"   Found {len(results)} results")
            
            for i, result in enumerate(results[:3]):  # Show first 3 results
                print(f"   Result {i+1}:")
                print(f"     ID: {result.get('id', 'N/A')}")
                title = result.get('title', 'N/A')
                content = result.get('content', 'N/A') 
                claim = result.get('claim', 'N/A')
                evidence = result.get('evidence', 'N/A')
                
                print(f"     Title: {str(title)[:100] if title else 'N/A'}...")
                print(f"     Content: {str(content)[:100] if content else 'N/A'}...")
                print(f"     Claim: {str(claim)[:100] if claim else 'N/A'}...")
                print(f"     Evidence: {str(evidence)[:100] if evidence else 'N/A'}...")
                print()
                
        except Exception as e:
            print(f"   Error in search test: {e}")
    
    # Check indices (using Neo4j 4.x compatible commands)
    print("\n7. Index Information:")
    try:
        # Try the older syntax first
        query = "SHOW INDEXES"
        results = client.execute_query(query)
        if results:
            print(f"   Found {len(results)} indices:")
            for result in results:
                print(f"   Index: {result}")
                print()
        else:
            print("   No indices found")
    except Exception as e:
        print(f"   Error getting index info: {e}")
        # Try alternative approach
        try:
            query = "CALL db.schema.visualization()"
            results = client.execute_query(query)
            print("   Schema visualization available")
        except:
            print("   No index information available")
    
    # Check constraints
    print("\n8. Constraint Information:")
    try:
        query = "SHOW CONSTRAINTS"
        results = client.execute_query(query)
        if results:
            print(f"   Found {len(results)} constraints:")
            for result in results:
                print(f"   Constraint: {result}")
                print()
        else:
            print("   No constraints found")
    except Exception as e:
        print(f"   Error getting constraint info: {e}")
    
    # Test full-text search if available
    print("\n9. Full-text Search Test:")
    try:
        # Check if full-text indices exist
        query = "SHOW INDEXES WHERE type = 'FULLTEXT'"
        results = client.execute_query(query)
        
        if results:
            print(f"   Found {len(results)} full-text indices:")
            for result in results:
                print(f"     - {result}")
                
        else:
            print("   No full-text indices found")
            
    except Exception as e:
        print(f"   Error testing full-text search: {e}")
    
    # Test vector search if available
    print("\n10. Vector Search Test:")
    try:
        # Check for vector indices
        query = "SHOW INDEXES WHERE type = 'VECTOR'"
        results = client.execute_query(query)
        
        if results:
            print(f"   Found {len(results)} vector indices:")
            for result in results:
                print(f"     - {result}")
        else:
            print("   No vector indices found")
            
    except Exception as e:
        print(f"   Error checking vector indices: {e}")
    
    # Close connection
    client.close()
    print("\n=== Debugging Complete ===")

if __name__ == "__main__":
    main()