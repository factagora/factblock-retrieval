#!/usr/bin/env python3
"""
Script to analyze actual claims in Neo4j FactBlock database
to create realistic example texts for GraphRAG testing.
"""

import os
import sys
import json
from typing import Dict, List, Any
from collections import defaultdict, Counter
import re

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.neo4j_client import Neo4jClient

def analyze_factblock_database():
    """Analyze the Neo4j FactBlock database to extract real claims and data."""
    
    # Connect to local Neo4j database
    client = Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j", 
        password="password"
    )
    
    try:
        # Verify connection
        if not client.verify_connectivity():
            print("Failed to connect to Neo4j database")
            return
            
        print("✓ Connected to Neo4j database successfully")
        
        # Get basic database info
        db_info = client.get_database_info()
        print(f"\nDatabase Info:")
        print(f"  Nodes: {db_info.get('node_count', 'unknown')}")
        print(f"  Relationships: {db_info.get('relationship_count', 'unknown')}")
        
        # Get all node labels and relationship types
        labels_query = "CALL db.labels() YIELD label RETURN label"
        rel_types_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        
        labels = [r['label'] for r in client.execute_query(labels_query)]
        rel_types = [r['relationshipType'] for r in client.execute_query(rel_types_query)]
        
        print(f"\nNode Labels: {labels}")
        print(f"Relationship Types: {rel_types}")
        
        # Analyze FactBlocks
        print("\n" + "="*50)
        print("ANALYZING FACTBLOCKS")
        print("="*50)
        
        analyze_factblocks(client)
        
        # Analyze Claims
        print("\n" + "="*50)
        print("ANALYZING CLAIMS")
        print("="*50)
        
        analyze_claims(client)
        
        # Analyze Companies and Financial Data
        print("\n" + "="*50)
        print("ANALYZING COMPANIES & FINANCIAL DATA")
        print("="*50)
        
        analyze_companies_and_financials(client)
        
        # Analyze Relationships and Cross-References
        print("\n" + "="*50)
        print("ANALYZING RELATIONSHIPS")
        print("="*50)
        
        analyze_relationships(client)
        
        # Generate example texts based on real data
        print("\n" + "="*50)
        print("GENERATING REALISTIC EXAMPLES")
        print("="*50)
        
        generate_realistic_examples(client)
        
    except Exception as e:
        print(f"Error analyzing database: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

def analyze_factblocks(client: Neo4jClient):
    """Analyze FactBlock nodes to understand structure and content."""
    
    # Get FactBlock count
    factblock_count_query = "MATCH (f:FactBlock) RETURN count(f) as count"
    count_result = client.execute_query(factblock_count_query)
    factblock_count = count_result[0]['count'] if count_result else 0
    
    print(f"Total FactBlocks: {factblock_count}")
    
    if factblock_count == 0:
        print("No FactBlocks found in database")
        return
    
    # Get sample FactBlocks with their properties
    sample_query = """
    MATCH (f:FactBlock)
    RETURN f
    LIMIT 10
    """
    
    sample_factblocks = client.execute_query(sample_query)
    
    if sample_factblocks:
        print(f"\nSample FactBlock properties:")
        for i, record in enumerate(sample_factblocks[:3]):
            fb = record['f']
            print(f"\nFactBlock {i+1}:")
            for key, value in fb.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {key}: {value}")
    
    # Get FactBlock property keys
    props_query = """
    MATCH (f:FactBlock)
    UNWIND keys(f) as key
    RETURN DISTINCT key
    ORDER BY key
    """
    
    props = [r['key'] for r in client.execute_query(props_query)]
    print(f"\nFactBlock properties: {props}")

def analyze_claims(client: Neo4jClient):
    """Analyze Claims to find concrete factual statements."""
    
    # Check if Claims exist
    claims_count_query = "MATCH (c:Claim) RETURN count(c) as count"
    count_result = client.execute_query(claims_count_query)
    claims_count = count_result[0]['count'] if count_result else 0
    
    print(f"Total Claims: {claims_count}")
    
    if claims_count == 0:
        print("No Claims found in database")
        return
    
    # Get sample claims with text content
    sample_claims_query = """
    MATCH (c:Claim)
    WHERE c.text IS NOT NULL AND c.text <> ""
    RETURN c.text as claim_text, c
    LIMIT 20
    """
    
    sample_claims = client.execute_query(sample_claims_query)
    
    if sample_claims:
        print(f"\nSample Claims:")
        for i, record in enumerate(sample_claims[:10]):
            claim_text = record['claim_text']
            print(f"\n{i+1}. {claim_text}")
    
    # Find claims with numbers/percentages
    numeric_claims_query = """
    MATCH (c:Claim)
    WHERE c.text IS NOT NULL 
    AND (c.text =~ '.*\\d+.*%.*' OR c.text =~ '.*\\$\\d+.*' OR c.text =~ '.*\\d+\\.\\d+.*')
    RETURN c.text as claim_text
    LIMIT 15
    """
    
    numeric_claims = client.execute_query(numeric_claims_query)
    
    if numeric_claims:
        print(f"\nClaims with Financial/Numeric Data:")
        for i, record in enumerate(numeric_claims):
            print(f"{i+1}. {record['claim_text']}")

def analyze_companies_and_financials(client: Neo4jClient):
    """Analyze company-related nodes and financial data."""
    
    # Check for different types of company-related nodes
    company_labels = ['Company', 'Organization', 'Entity']
    
    for label in company_labels:
        count_query = f"MATCH (n:{label}) RETURN count(n) as count"
        try:
            count_result = client.execute_query(count_query)
            count = count_result[0]['count'] if count_result else 0
            print(f"Total {label} nodes: {count}")
            
            if count > 0:
                # Get sample companies
                sample_query = f"""
                MATCH (n:{label})
                RETURN n
                LIMIT 10
                """
                
                companies = client.execute_query(sample_query)
                print(f"\nSample {label} nodes:")
                for i, record in enumerate(companies[:5]):
                    company = record['n']
                    name = company.get('name', company.get('title', 'Unknown'))
                    print(f"  {i+1}. {name}")
                    
        except Exception as e:
            print(f"No {label} nodes found or error: {e}")
    
    # Look for financial terms in text content
    financial_terms_query = """
    MATCH (n)
    WHERE ANY(prop IN keys(n) WHERE 
        n[prop] IS NOT NULL AND 
        toString(n[prop]) =~ '.*(?i)(revenue|profit|earnings|million|billion|trillion|\\$\\d+).*'
    )
    RETURN labels(n) as node_type, n
    LIMIT 10
    """
    
    try:
        financial_nodes = client.execute_query(financial_terms_query)
        
        if financial_nodes:
            print(f"\nNodes with Financial Terms:")
            for i, record in enumerate(financial_nodes):
                node_type = record['node_type']
                node = record['n']
                print(f"{i+1}. {node_type}: {dict(node)}")
                
    except Exception as e:
        print(f"Error searching for financial terms: {e}")

def analyze_relationships(client: Neo4jClient):
    """Analyze relationships between nodes to understand connections."""
    
    # Get relationship type counts
    rel_count_query = """
    MATCH ()-[r]->()
    RETURN type(r) as rel_type, count(r) as count
    ORDER BY count DESC
    """
    
    rel_counts = client.execute_query(rel_count_query)
    
    if rel_counts:
        print("Relationship types and counts:")
        for record in rel_counts:
            print(f"  {record['rel_type']}: {record['count']}")
    
    # Find nodes with the most connections
    connected_nodes_query = """
    MATCH (n)
    WITH n, 
         COUNT {(n)-[]-()} as degree
    WHERE degree > 0
    RETURN labels(n) as node_type, n, degree
    ORDER BY degree DESC
    LIMIT 10
    """
    
    connected_nodes = client.execute_query(connected_nodes_query)
    
    if connected_nodes:
        print(f"\nMost connected nodes:")
        for i, record in enumerate(connected_nodes):
            node_type = record['node_type']
            node = record['n']
            degree = record['degree']
            name = node.get('name', node.get('title', str(node.get('id', 'Unknown'))))
            print(f"  {i+1}. {node_type} '{name}': {degree} connections")

def extract_concrete_facts(text: str) -> List[str]:
    """Extract concrete facts from text using regex patterns."""
    facts = []
    
    # Financial patterns
    money_pattern = r'\$[\d,.]+(?: (?:million|billion|trillion))?'
    percentage_pattern = r'\d+(?:\.\d+)?%'
    number_pattern = r'\d+(?:,\d{3})*(?:\.\d+)?'
    
    money_matches = re.findall(money_pattern, text, re.IGNORECASE)
    percentage_matches = re.findall(percentage_pattern, text)
    
    facts.extend([f"Financial amount: {m}" for m in money_matches])
    facts.extend([f"Percentage: {p}" for p in percentage_matches])
    
    # Date patterns
    date_patterns = [
        r'\b\d{4}\b',  # Years
        r'\b(?:Q[1-4]|quarter [1-4])\s+\d{4}\b',  # Quarters
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        facts.extend([f"Date reference: {m}" for m in matches])
    
    return facts

def generate_realistic_examples(client: Neo4jClient):
    """Generate realistic example texts based on actual database content."""
    
    print("Generating realistic example texts based on actual claims...")
    
    # Get diverse claims for examples
    diverse_claims_query = """
    MATCH (c:Claim)
    WHERE c.text IS NOT NULL AND length(c.text) > 20
    WITH c, rand() as r
    ORDER BY r
    RETURN c.text as claim_text
    LIMIT 20
    """
    
    try:
        claims = client.execute_query(diverse_claims_query)
        
        examples = []
        
        for i, record in enumerate(claims[:10]):
            claim_text = record['claim_text']
            
            # Extract concrete facts
            facts = extract_concrete_facts(claim_text)
            
            # Create example based on this claim
            example = {
                "id": f"real_example_{i+1}",
                "text": claim_text,
                "extracted_facts": facts,
                "source": "actual_database_claim"
            }
            
            examples.append(example)
        
        # Save examples to file
        with open('realistic_factcheck_examples.json', 'w') as f:
            json.dump(examples, f, indent=2)
        
        print(f"\nGenerated {len(examples)} realistic examples:")
        for example in examples[:5]:
            print(f"\nExample: {example['text'][:100]}...")
            if example['extracted_facts']:
                print(f"Facts found: {example['extracted_facts']}")
        
        print(f"\nAll examples saved to: realistic_factcheck_examples.json")
        
    except Exception as e:
        print(f"Error generating examples: {e}")

if __name__ == "__main__":
    analyze_factblock_database()