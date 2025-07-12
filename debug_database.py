#!/usr/bin/env python3
"""
Debug Database Contents

Check what FactBlocks exist in the database and their structure.
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

def debug_database():
    """Debug database contents"""
    
    # Load production config
    if os.path.exists('.env.production'):
        load_dotenv('.env.production')
        print("✅ Loaded .env.production")
    else:
        load_dotenv()
        print("⚠️ Using default .env")
    
    uri = os.getenv('NEO4J_URI', 'bolt://20.81.43.138:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'password')
    database = os.getenv('NEO4J_DATABASE', 'neo4j')
    
    print(f"🔗 Connecting to: {uri}")
    print(f"   User: {user}")
    print(f"   Database: {database}")
    print()
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session(database=database) as session:
        # Check total FactBlocks
        result = session.run("MATCH (f:FactBlock) RETURN count(f) as count")
        total_factblocks = result.single()['count']
        print(f"📊 Total FactBlocks: {total_factblocks}")
        
        # Check cascade FactBlocks
        print("\n🔍 Cascade FactBlocks in database:")
        result = session.run("""
            MATCH (f:FactBlock)
            WHERE f.id STARTS WITH 'cascade_'
            RETURN f.id, f.title, f.text, f.category
            LIMIT 10
        """)
        
        cascade_count = 0
        for record in result:
            cascade_count += 1
            print(f"  ID: {record['f.id']}")
            print(f"  Title: {record['f.title']}")
            print(f"  Text: {record['f.text'][:100]}...")
            print(f"  Category: {record['f.category']}")
            print()
        
        print(f"Found {cascade_count} cascade FactBlocks")
        
        # Check relationships
        print("\n🔗 Cascade Relationships:")
        result = session.run("""
            MATCH (f1:FactBlock)-[r:RELATES_TO]->(f2:FactBlock)
            WHERE f1.id STARTS WITH 'cascade_' AND f2.id STARTS WITH 'cascade_'
            RETURN f1.text as source_text, r.semantic_type, f2.text as target_text, r.explanation
            LIMIT 5
        """)
        
        rel_count = 0
        for record in result:
            rel_count += 1
            print(f"  Source: {record['source_text'][:50]}...")
            print(f"  Type: {record['r.semantic_type']}")
            print(f"  Target: {record['target_text'][:50]}...")
            print(f"  Explanation: {record['r.explanation']}")
            print()
        
        print(f"Found {rel_count} cascade relationships")
        
        # Test specific query for EU cookies
        print("\n🎯 Testing specific query for EU cookies:")
        result = session.run("""
            MATCH (f:FactBlock)
            WHERE f.text CONTAINS '쿠키' OR f.text CONTAINS 'EU' OR f.text CONTAINS '개인정보'
            RETURN f.id, f.text, f.category
            LIMIT 5
        """)
        
        cookie_count = 0
        for record in result:
            cookie_count += 1
            print(f"  ID: {record['f.id']}")
            print(f"  Text: {record['f.text']}")
            print(f"  Category: {record['f.category']}")
            print()
        
        print(f"Found {cookie_count} FactBlocks mentioning cookies/EU/privacy")
        
        # Check all FactBlock categories
        print("\n📂 FactBlock categories:")
        result = session.run("""
            MATCH (f:FactBlock)
            RETURN DISTINCT f.category as category, count(f) as count
            ORDER BY count DESC
        """)
        
        for record in result:
            print(f"  {record['category']}: {record['count']}")
        
        # Test if our specific texts exist
        print("\n🔍 Checking for specific regulatory cascade texts:")
        test_texts = [
            '쿠키 규제',
            '광고 타겟팅',
            '개인정보보호법'
        ]
        
        for text in test_texts:
            result = session.run("""
                MATCH (f:FactBlock)
                WHERE f.text CONTAINS $text
                RETURN count(f) as count
            """, text=text)
            count = result.single()['count']
            print(f"  '{text}': {count} FactBlocks")
    
    driver.close()

if __name__ == "__main__":
    debug_database()