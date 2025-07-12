#!/usr/bin/env python3
"""
Explore Local Database Contents

Check what FactBlocks exist in the local Neo4j database and their structure.
"""

from neo4j import GraphDatabase

def explore_local_database():
    """Explore local database contents"""
    
    # Use the exact credentials specified by the user
    uri = 'bolt://localhost:7687'
    user = 'neo4j'
    password = 'password'
    database = 'factblock'  # Try the factblock database first
    
    print(f"🔗 Connecting to local database: {uri}")
    print(f"   User: {user}")
    print(f"   Database: {database}")
    print()
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # First, let's list all databases
        with driver.session() as session:
            try:
                result = session.run("SHOW DATABASES")
                print("📂 Available databases:")
                for record in result:
                    print(f"  - {record['name']} (status: {record.get('currentStatus', 'unknown')})")
                print()
            except Exception as e:
                print(f"Could not list databases: {e}")
        
        # Try connecting to the factblock database
        try:
            with driver.session(database=database) as session:
                # Check total FactBlocks
                result = session.run("MATCH (f:FactBlock) RETURN count(f) as count")
                total_factblocks = result.single()['count']
                print(f"📊 Total FactBlocks in '{database}' database: {total_factblocks}")
                
                if total_factblocks > 0:
                    explore_factblocks(session)
                else:
                    print(f"No FactBlocks found in '{database}' database")
        
        except Exception as e:
            print(f"Failed to connect to '{database}' database: {e}")
            print("Trying default 'neo4j' database...")
            
            # Try the default neo4j database
            with driver.session(database='neo4j') as session:
                result = session.run("MATCH (f:FactBlock) RETURN count(f) as count")
                total_factblocks = result.single()['count']
                print(f"📊 Total FactBlocks in 'neo4j' database: {total_factblocks}")
                
                if total_factblocks > 0:
                    explore_factblocks(session)
                else:
                    print("No FactBlocks found in 'neo4j' database either")
                    
                    # Let's see what nodes we do have
                    print("\n🔍 Checking for any nodes in the database:")
                    result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
                    for record in result:
                        print(f"  {record['labels']}: {record['count']} nodes")
        
        driver.close()
        
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        print("Make sure Neo4j is running on localhost:7687 with credentials neo4j/password")

def explore_factblocks(session):
    """Explore FactBlock structure and content"""
    
    # Check FactBlock properties
    print("\n🔍 FactBlock properties:")
    result = session.run("""
        MATCH (f:FactBlock)
        RETURN keys(f) as properties
        LIMIT 1
    """)
    
    for record in result:
        print(f"  Properties: {record['properties']}")
    
    # Sample FactBlocks
    print("\n📋 Sample FactBlocks:")
    result = session.run("""
        MATCH (f:FactBlock)
        RETURN f
        LIMIT 10
    """)
    
    sample_count = 0
    for record in result:
        sample_count += 1
        fb = record['f']
        print(f"  ID: {fb.get('id', 'N/A')}")
        print(f"  Claim: {str(fb.get('claim', 'N/A'))[:100]}...")
        print(f"  Summary: {str(fb.get('summary', 'N/A'))[:100]}...")
        print(f"  Source Type: {fb.get('source_type', 'N/A')}")
        print(f"  Publication: {fb.get('publication', 'N/A')}")
        print(f"  Verdict: {fb.get('verdict', 'N/A')}")
        print(f"  Impact Level: {fb.get('impact_level', 'N/A')}")
        print(f"  Affected Sectors: {fb.get('affected_sectors', 'N/A')}")
        print()
    
    print(f"Showed {sample_count} sample FactBlocks")
    
    # Check source types
    print("\n📂 FactBlock source types:")
    result = session.run("""
        MATCH (f:FactBlock)
        WHERE f.source_type IS NOT NULL
        RETURN f.source_type as source_type, count(f) as count
        ORDER BY count DESC
    """)
    
    for record in result:
        print(f"  {record['source_type']}: {record['count']}")
    
    # Check publications
    print("\n📰 FactBlock publications:")
    result = session.run("""
        MATCH (f:FactBlock)
        WHERE f.publication IS NOT NULL
        RETURN f.publication as publication, count(f) as count
        ORDER BY count DESC
        LIMIT 20
    """)
    
    for record in result:
        print(f"  {record['publication']}: {record['count']}")
    
    # Check verdicts
    print("\n⚖️ FactBlock verdicts:")
    result = session.run("""
        MATCH (f:FactBlock)
        WHERE f.verdict IS NOT NULL
        RETURN f.verdict as verdict, count(f) as count
        ORDER BY count DESC
    """)
    
    for record in result:
        print(f"  {record['verdict']}: {record['count']}")
    
    # Check affected sectors
    print("\n🏢 Affected sectors:")
    result = session.run("""
        MATCH (f:FactBlock)
        WHERE f.affected_sectors IS NOT NULL
        RETURN f.affected_sectors as affected_sectors, count(f) as count
        ORDER BY count DESC
        LIMIT 20
    """)
    
    for record in result:
        print(f"  {record['affected_sectors']}: {record['count']}")
    
    # Check relationships
    print("\n🔗 Relationship types:")
    result = session.run("""
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY count DESC
    """)
    
    for record in result:
        print(f"  {record['rel_type']}: {record['count']}")
    
    # Sample relationships
    print("\n🔗 Sample relationships:")
    result = session.run("""
        MATCH (f1:FactBlock)-[r]->(f2:FactBlock)
        RETURN f1.claim as source_claim, type(r) as rel_type, f2.claim as target_claim, 
               r.semantic_type as semantic_type, r.explanation as explanation
        LIMIT 5
    """)
    
    rel_count = 0
    for record in result:
        rel_count += 1
        source_claim = str(record.get('source_claim', 'N/A'))
        target_claim = str(record.get('target_claim', 'N/A'))
        print(f"  Source: {source_claim[:50]}...")
        print(f"  Relationship: {record['rel_type']}")
        print(f"  Semantic Type: {record.get('semantic_type', 'N/A')}")
        print(f"  Target: {target_claim[:50]}...")
        if record.get('explanation'):
            print(f"  Explanation: {record['explanation']}")
        print()
    
    print(f"Found {rel_count} sample relationships")
    
    # Look for regulatory cascade patterns
    print("\n📊 Looking for regulatory cascade patterns:")
    result = session.run("""
        MATCH (reg:FactBlock)-[r1]->(comp:FactBlock)-[r2]->(impact:FactBlock)
        WHERE reg.claim CONTAINS 'regulation' OR reg.claim CONTAINS 'law' OR reg.claim CONTAINS 'policy'
            OR reg.summary CONTAINS 'regulation' OR reg.summary CONTAINS 'law' OR reg.summary CONTAINS 'policy'
        RETURN reg.claim as regulation, comp.claim as compliance, impact.claim as business_impact
        LIMIT 5
    """)
    
    cascade_count = 0
    for record in result:
        cascade_count += 1
        regulation = str(record.get('regulation', 'N/A'))
        compliance = str(record.get('compliance', 'N/A'))
        business_impact = str(record.get('business_impact', 'N/A'))
        print(f"  Regulation: {regulation[:60]}...")
        print(f"  Compliance: {compliance[:60]}...")
        print(f"  Impact: {business_impact[:60]}...")
        print()
    
    print(f"Found {cascade_count} potential regulatory cascades")
    
    # Look for specific content themes
    print("\n🎯 Content themes analysis:")
    themes = ['AI', 'privacy', 'regulation', 'compliance', 'technology', 'data', 'security', 'financial', 'environment']
    
    for theme in themes:
        result = session.run("""
            MATCH (f:FactBlock)
            WHERE f.claim CONTAINS $theme OR f.summary CONTAINS $theme
            RETURN count(f) as count
        """, theme=theme)
        count = result.single()['count']
        if count > 0:
            print(f"  {theme}: {count} FactBlocks")

if __name__ == "__main__":
    explore_local_database()