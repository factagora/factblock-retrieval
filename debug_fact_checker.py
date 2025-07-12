#!/usr/bin/env python3
"""
Debug Fact Checker Directly
"""

import os
import logging
from dotenv import load_dotenv

# Enable logging
logging.basicConfig(level=logging.INFO)

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.graphrag.relationship_aware_fact_checker import RelationshipAwareFactChecker

def debug_fact_checker():
    """Debug fact checker directly"""
    
    load_dotenv('.env.production')
    
    # Initialize fact checker
    fact_checker = RelationshipAwareFactChecker(
        neo4j_uri=os.getenv('NEO4J_URI', 'bolt://20.81.43.138:7687'),
        neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
        neo4j_password=os.getenv('NEO4J_PASSWORD', 'password')
    )
    
    print("🔍 Testing RelationshipAwareFactChecker directly")
    
    # Test entity extraction
    claim = "EU가 쿠키 사용에 대한 명시적 동의를 의무화했다"
    entities = fact_checker._extract_claim_entities(claim)
    print(f"Entities: {entities}")
    
    # Test direct evidence finding
    evidence = fact_checker._find_direct_evidence(entities)
    print(f"Direct evidence count: {len(evidence)}")
    
    for ev in evidence[:3]:
        print(f"  - {ev.explanation}")
    
    # Test the exact query manually
    print("\n🧪 Testing exact query manually:")
    with fact_checker.driver.session(database='neo4j') as session:
        result = session.run("""
            MATCH (f:FactBlock)
            WHERE ANY(entity IN $entities WHERE 
                toLower(f.text) CONTAINS toLower(entity) OR
                toLower(f.title) CONTAINS toLower(entity)
            )
            RETURN f.id as id, f.title as title, f.text as text,
                   f.credibility_score as credibility, f.verdict as verdict,
                   f.investment_insight as insight
            ORDER BY f.credibility_score DESC
            LIMIT 5
        """, entities=entities)
        
        count = 0
        for record in result:
            count += 1
            print(f"  ID: {record['id']}")
            print(f"  Text: {record['text'][:100]}...")
            print(f"  Credibility: {record['credibility']}")
            print()
        
        print(f"Manual query found {count} FactBlocks")
    
    fact_checker.close()

if __name__ == "__main__":
    debug_fact_checker()