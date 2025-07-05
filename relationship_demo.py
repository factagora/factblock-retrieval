#!/usr/bin/env python3
"""
Demo showing Graph Relationships in the GraphRAG System
"""

import os
import sys
sys.path.insert(0, '/Users/randybaek/workspace/factblock-retrieval')

from src.database.neo4j_client import Neo4jClient
from src.config import load_config

def main():
    print("🕸️  GraphRAG Relationship Demo")
    print("=" * 50)
    
    # Set environment and connect
    os.environ['NEO4J_PASSWORD'] = 'password'
    config = load_config()
    client = Neo4jClient(config.neo4j.uri, config.neo4j.user, config.neo4j.password)
    
    print("1. Exploring GDPR and its related documents...")
    
    # Find GDPR and its relationships
    gdpr_query = """
    MATCH (gdpr:FederalRegulation {name: 'GDPR'})
    OPTIONAL MATCH (gdpr)-[r1:HAS_GUIDANCE]->(guidance:AgencyGuidance)
    OPTIONAL MATCH (gdpr)-[r2:HAS_ENFORCEMENT]->(enforcement:EnforcementAction)
    OPTIONAL MATCH (topic:ComplianceTopic)-[r3:COVERS]->(gdpr)
    RETURN gdpr, guidance, enforcement, topic
    """
    
    result = client.execute_query(gdpr_query)
    
    if result:
        print("   📋 GDPR Regulation found with connections:")
        for record in result:
            gdpr = record['gdpr']
            print(f"      🏛️  Regulation: {gdpr['name']}")
            print(f"          Citation: {gdpr['citation']}")
            print(f"          Category: {gdpr['category']}")
            
            guidance = record['guidance']
            if guidance:
                print(f"      📖 Related Guidance: {guidance['title']}")
                print(f"          Agency: {guidance['agency']}")
            
            enforcement = record['enforcement']
            if enforcement:
                print(f"      ⚖️  Enforcement Action: {enforcement['title']}")
                print(f"          Outcome: {enforcement['outcome']}")
            
            topic = record['topic']
            if topic:
                print(f"      🏷️  Compliance Topic: {topic['name']}")
    
    print("\n2. Showing all relationship types in the database...")
    
    # Show all relationship types
    rel_query = """
    MATCH ()-[r]->()
    RETURN type(r) as relationship_type, count(r) as count
    ORDER BY count DESC
    """
    
    relationships = client.execute_query(rel_query)
    print("   📊 Relationship Types:")
    for rel in relationships:
        print(f"      {rel['relationship_type']}: {rel['count']} connections")
    
    print("\n3. Finding enforcement actions and their related regulations...")
    
    # Find enforcement actions and what they relate to
    enforcement_query = """
    MATCH (reg:FederalRegulation)-[:HAS_ENFORCEMENT]->(enf:EnforcementAction)
    RETURN reg.name as regulation, enf.title as enforcement_title, enf.outcome as outcome
    """
    
    enforcements = client.execute_query(enforcement_query)
    print("   ⚖️  Enforcement Actions:")
    for enf in enforcements:
        print(f"      {enf['regulation']} → {enf['enforcement_title']}")
        print(f"          Outcome: {enf['outcome']}")
    
    print("\n4. Cross-category analysis...")
    
    # Count by category
    category_query = """
    MATCH (n)
    WHERE exists(n.category)
    RETURN n.category as category, count(n) as count
    ORDER BY count DESC
    """
    
    categories = client.execute_query(category_query)
    print("   📊 Documents by Category:")
    for cat in categories:
        print(f"      {cat['category']}: {cat['count']} documents")
    
    print("\n✅ Graph relationship demo completed!")
    print("\n💡 Key Insights:")
    print("   • Documents are connected through meaningful relationships")
    print("   • Regulations link to their guidance and enforcement actions")
    print("   • Topics cover multiple related regulations")
    print("   • The graph structure enables rich contextual retrieval")
    
    client.close()

if __name__ == "__main__":
    main()