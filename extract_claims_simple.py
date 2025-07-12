#!/usr/bin/env python3
"""
Simple script to extract concrete claims from Neo4j FactBlock database
"""

import os
import sys
import json
import re
from typing import Dict, List, Any

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.neo4j_client import Neo4jClient

def extract_factblock_claims():
    """Extract concrete claims from FactBlocks for realistic examples."""
    
    # Connect to local Neo4j database
    client = Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j", 
        password="password"
    )
    
    try:
        print("✓ Connected to Neo4j database successfully")
        
        # Get all FactBlocks with their claims
        factblocks_query = """
        MATCH (f:FactBlock)
        RETURN f.claim as claim, 
               f.evidence as evidence,
               f.summary as summary,
               f.impact_level as impact_level,
               f.affected_sectors as sectors,
               f.confidence_score as confidence,
               f.publication as publication,
               f.author as author,
               f.published_date as date,
               f.credibility_score as credibility
        ORDER BY f.confidence_score DESC
        """
        
        factblocks = client.execute_query(factblocks_query)
        
        print(f"Found {len(factblocks)} FactBlocks")
        
        # Extract concrete financial/numerical claims
        concrete_claims = []
        
        for fb in factblocks:
            claim = fb.get('claim', '')
            evidence = fb.get('evidence', '')
            
            # Look for claims with specific numbers, percentages, financial data
            if has_concrete_data(claim) or has_concrete_data(evidence):
                concrete_claims.append({
                    'claim': claim,
                    'evidence': evidence,
                    'summary': fb.get('summary', ''),
                    'sectors': fb.get('sectors', []),
                    'confidence': fb.get('confidence', 0),
                    'publication': fb.get('publication', ''),
                    'impact_level': fb.get('impact_level', ''),
                    'credibility': fb.get('credibility', 0)
                })
        
        print(f"\nFound {len(concrete_claims)} FactBlocks with concrete data:")
        
        # Display the most concrete claims
        for i, claim_data in enumerate(concrete_claims[:15]):
            print(f"\n{i+1}. CLAIM: {claim_data['claim']}")
            print(f"   EVIDENCE: {claim_data['evidence']}")
            print(f"   SECTORS: {claim_data['sectors']}")
            print(f"   CONFIDENCE: {claim_data['confidence']:.2f}")
            print(f"   IMPACT: {claim_data['impact_level']}")
        
        # Get Entity information for companies/organizations
        entities_query = """
        MATCH (e:Entity)
        RETURN e.name as name
        ORDER BY e.name
        """
        
        entities = client.execute_query(entities_query)
        entity_names = [e['name'] for e in entities if e.get('name')]
        
        print(f"\nEntities in database: {entity_names}")
        
        # Get sectors and themes
        sectors_query = """
        MATCH (f:FactBlock)
        UNWIND f.affected_sectors as sector
        RETURN sector, count(*) as count
        ORDER BY count DESC
        """
        
        sectors = client.execute_query(sectors_query)
        
        print(f"\nMost common sectors:")
        for sector_data in sectors:
            print(f"  {sector_data['sector']}: {sector_data['count']} FactBlocks")
        
        # Generate realistic examples based on actual data
        generate_examples(concrete_claims, entity_names, sectors)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

def has_concrete_data(text: str) -> bool:
    """Check if text contains concrete numerical or financial data."""
    if not text:
        return False
    
    patterns = [
        r'\d+%',  # Percentages
        r'\d+차례',  # Korean: X times
        r'\d+배럴',  # Barrels
        r'\d+만',  # Korean: tens of thousands
        r'\d{4}년',  # Years in Korean
        r'\$[\d,]+',  # Dollar amounts
        r'\d+억',  # Korean: hundreds of millions
        r'\d+조',  # Korean: trillions
        r'15%',  # Specific percentages
        r'200만',  # Specific large numbers
        r'7차례',  # Specific frequencies
        r'2022년',  # Specific years
    ]
    
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    
    return False

def generate_examples(concrete_claims: List[Dict], entities: List[str], sectors: List[Dict]):
    """Generate realistic fact-check examples based on actual data."""
    
    examples = []
    
    # Example 1: Based on OPEC oil production claim
    examples.append({
        "example_id": "opec_oil_production",
        "text": "OPEC이 감산 합의에 도달했으며, 주요 산유국들이 원유 생산량을 일일 200만 배럴 감축하기로 합의했다",
        "source_claim": "OPEC이 감산 합의에 도달했다",
        "expected_evidence": "주요 산유국들이 원유 생산량을 일일 200만 배럴 감축하기로 합의했다",
        "sectors": ["energy"],
        "concrete_facts": ["200만 배럴", "일일 감축"],
        "entities_mentioned": ["OPEC"]
    })
    
    # Example 2: Based on airline fuel cost claim  
    examples.append({
        "example_id": "airline_fuel_costs",
        "text": "글로벌 항공사의 연료비가 15% 상승했으며, 원유 공급 감소로 인한 유가 상승이 항공유 가격을 직접적으로 끌어올렸다",
        "source_claim": "글로벌 항공사의 연료비가 15% 상승했다",
        "expected_evidence": "원유 공급 감소로 인한 유가 상승이 항공유 가격을 직접적으로 끌어올렸다",
        "sectors": ["transportation"],
        "concrete_facts": ["15% 상승"],
        "entities_mentioned": ["글로벌 항공사"]
    })
    
    # Example 3: Based on Fed interest rate claim
    examples.append({
        "example_id": "fed_interest_rates",
        "text": "미국 연준이 2022년 기준금리를 7차례 인상했으며, 연방준비제도가 인플레이션 억제를 위해 공격적인 통화긴축 정책을 실시했다",
        "source_claim": "미국 연준이 2022년 기준금리를 7차례 인상했다",
        "expected_evidence": "연방준비제도가 인플레이션 억제를 위해 공격적인 통화긴축 정책을 실시했다",
        "sectors": ["financials"],
        "concrete_facts": ["2022년", "7차례", "기준금리 인상"],
        "entities_mentioned": ["연준", "연방준비제도"]
    })
    
    # Generate additional examples based on most common patterns
    for i, claim_data in enumerate(concrete_claims[:5]):
        if i < 3:  # Skip first 3 as we already created specific examples
            continue
            
        examples.append({
            "example_id": f"factblock_claim_{i+1}",
            "text": f"{claim_data['claim']} {claim_data['evidence']}",
            "source_claim": claim_data['claim'],
            "expected_evidence": claim_data['evidence'],
            "sectors": claim_data['sectors'],
            "confidence": claim_data['confidence'],
            "impact_level": claim_data['impact_level']
        })
    
    # Save examples
    with open('realistic_factcheck_examples.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nGenerated {len(examples)} realistic examples based on actual database claims:")
    print("="*60)
    
    for example in examples:
        print(f"\nExample ID: {example['example_id']}")
        print(f"Text: {example['text']}")
        if 'concrete_facts' in example:
            print(f"Concrete Facts: {example['concrete_facts']}")
        if 'entities_mentioned' in example:
            print(f"Entities: {example['entities_mentioned']}")
        print(f"Sectors: {example.get('sectors', [])}")
        print("-" * 40)
    
    print(f"\nAll examples saved to: realistic_factcheck_examples.json")
    
    # Summary statistics
    print(f"\nSUMMARY:")
    print(f"- Total FactBlocks analyzed: {len(concrete_claims)}")
    print(f"- Entities in database: {len(entities)}")
    print(f"- Most common sectors: {[s['sector'] for s in sectors[:5]]}")
    print(f"- Generated examples: {len(examples)}")

if __name__ == "__main__":
    extract_factblock_claims()