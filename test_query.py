#!/usr/bin/env python3
"""
Test direct Neo4j queries
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

def test_queries():
    """Test queries directly"""
    
    load_dotenv('.env.production')
    
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI', 'bolt://20.81.43.138:7687'),
        auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'password'))
    )
    
    entities = ['EU', '쿠키']
    
    print(f"Testing with entities: {entities}")
    
    with driver.session(database='neo4j') as session:
        # Test 1: Simple search
        print("\n🔍 Test 1: Simple text search")
        result = session.run("""
            MATCH (f:FactBlock)
            WHERE f.text CONTAINS 'EU' OR f.text CONTAINS '쿠키'
            RETURN f.id, f.text
            LIMIT 5
        """)
        
        count = 0
        for record in result:
            count += 1
            print(f"  {record['f.id']}: {record['f.text'][:100]}...")
        print(f"Found {count} FactBlocks")
        
        # Test 2: Using ANY with entities
        print("\n🔍 Test 2: Using ANY with entities")
        result = session.run("""
            MATCH (f:FactBlock)
            WHERE ANY(entity IN $entities WHERE 
                toLower(f.text) CONTAINS toLower(entity)
            )
            RETURN f.id, f.text
            LIMIT 5
        """, entities=entities)
        
        count = 0
        for record in result:
            count += 1
            print(f"  {record['f.id']}: {record['f.text'][:100]}...")
        print(f"Found {count} FactBlocks")
        
        # Test 3: Check if FactBlocks have text field
        print("\n🔍 Test 3: Check FactBlock structure")
        result = session.run("""
            MATCH (f:FactBlock)
            WHERE f.id STARTS WITH 'cascade_'
            RETURN f.id, f.text, f.title, keys(f) as properties
            LIMIT 3
        """)
        
        for record in result:
            print(f"  ID: {record['f.id']}")
            print(f"  Text: {record['f.text']}")
            print(f"  Title: {record['f.title']}")
            print(f"  Properties: {record['properties']}")
            print()
    
    driver.close()

if __name__ == "__main__":
    test_queries()