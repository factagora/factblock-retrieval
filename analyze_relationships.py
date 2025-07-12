#!/usr/bin/env python3
"""
Analyze relationships between FactBlocks to understand cross-references
"""

import os
import sys
import json

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.neo4j_client import Neo4jClient

def analyze_factblock_relationships():
    """Analyze relationships between FactBlocks."""
    
    client = Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j", 
        password="password"
    )
    
    try:
        print("✓ Connected to Neo4j database successfully")
        
        # Get FactBlocks that are connected to each other
        related_factblocks_query = """
        MATCH (f1:FactBlock)-[r:RELATES_TO]->(f2:FactBlock)
        RETURN f1.claim as claim1, 
               f1.evidence as evidence1,
               f1.affected_sectors as sectors1,
               f2.claim as claim2,
               f2.evidence as evidence2,  
               f2.affected_sectors as sectors2,
               type(r) as relationship_type
        LIMIT 20
        """
        
        related_factblocks = client.execute_query(related_factblocks_query)
        
        print(f"Found {len(related_factblocks)} FactBlock relationships:")
        
        for i, rel in enumerate(related_factblocks):
            print(f"\n{i+1}. RELATIONSHIP:")
            print(f"   Claim 1: {rel['claim1']}")
            print(f"   Evidence 1: {rel['evidence1']}")
            print(f"   Sectors 1: {rel['sectors1']}")
            print(f"   --> {rel['relationship_type']} -->")
            print(f"   Claim 2: {rel['claim2']}")
            print(f"   Evidence 2: {rel['evidence2']}")
            print(f"   Sectors 2: {rel['sectors2']}")
        
        # Find FactBlocks that mention the same entities
        entity_connections_query = """
        MATCH (f:FactBlock)-[m:MENTIONS]->(e:Entity)<-[m2:MENTIONS]-(f2:FactBlock)
        WHERE f <> f2
        RETURN f.claim as claim1,
               f2.claim as claim2,
               e.name as entity_name,
               f.affected_sectors as sectors1,
               f2.affected_sectors as sectors2
        LIMIT 15
        """
        
        entity_connections = client.execute_query(entity_connections_query)
        
        print(f"\n\nFound {len(entity_connections)} FactBlocks connected through shared entities:")
        
        for i, conn in enumerate(entity_connections):
            print(f"\n{i+1}. ENTITY CONNECTION - {conn['entity_name']}:")
            print(f"   Claim 1: {conn['claim1']}")
            print(f"   Sectors 1: {conn['sectors1']}")
            print(f"   Claim 2: {conn['claim2']}")
            print(f"   Sectors 2: {conn['sectors2']}")
        
        # Find FactBlocks in the same sectors that could be related
        sector_connections_query = """
        MATCH (f1:FactBlock), (f2:FactBlock)
        WHERE f1 <> f2 
        AND ANY(sector IN f1.affected_sectors WHERE sector IN f2.affected_sectors)
        WITH f1, f2, 
             [sector IN f1.affected_sectors WHERE sector IN f2.affected_sectors] as shared_sectors
        RETURN f1.claim as claim1,
               f2.claim as claim2,
               shared_sectors,
               f1.impact_level as impact1,
               f2.impact_level as impact2
        LIMIT 10
        """
        
        sector_connections = client.execute_query(sector_connections_query)
        
        print(f"\n\nFound {len(sector_connections)} FactBlocks in shared sectors:")
        
        sector_patterns = {}
        for i, conn in enumerate(sector_connections):
            shared = conn['shared_sectors']
            for sector in shared:
                if sector not in sector_patterns:
                    sector_patterns[sector] = []
                sector_patterns[sector].append({
                    'claim1': conn['claim1'],
                    'claim2': conn['claim2'],
                    'impact1': conn['impact1'],
                    'impact2': conn['impact2']
                })
        
        # Show patterns by sector
        for sector, connections in sector_patterns.items():
            print(f"\n{sector.upper()} SECTOR CONNECTIONS:")
            for j, conn in enumerate(connections[:3]):  # Show first 3 per sector
                print(f"   {j+1}. {conn['claim1']} ({conn['impact1']})")
                print(f"      <-> {conn['claim2']} ({conn['impact2']})")
        
        # Generate cross-reference examples
        generate_cross_reference_examples(related_factblocks, entity_connections, sector_patterns)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

def generate_cross_reference_examples(related_factblocks, entity_connections, sector_patterns):
    """Generate examples that span multiple FactBlocks."""
    
    cross_ref_examples = []
    
    # Example 1: Energy sector chain reaction
    if 'energy' in sector_patterns:
        energy_claims = sector_patterns['energy']
        if len(energy_claims) > 0:
            cross_ref_examples.append({
                "example_id": "energy_sector_chain",
                "text": "원유 생산량 감축으로 인한 에너지 시장 충격이 항공 업계에도 영향을 미쳤다. OPEC의 감산 합의 이후 항공사들의 연료비가 크게 상승했다.",
                "cross_references": [
                    "OPEC이 감산 합의에 도달했다",
                    "글로벌 항공사의 연료비가 15% 상승했다"
                ],
                "sectors": ["energy", "transportation"],
                "relationship_type": "causal_chain"
            })
    
    # Example 2: Semiconductor supply chain impact
    if 'semiconductors' in sector_patterns and 'automobiles' in sector_patterns:
        cross_ref_examples.append({
            "example_id": "semiconductor_auto_chain",
            "text": "반도체 부족 현상이 자동차 산업에 직접적인 타격을 주었다. 현대자동차가 생산량을 15% 감축한 것도 반도체 수급 문제 때문이다.",
            "cross_references": [
                "반도체 부족 현상이 2024년까지 지속될 전망이다",
                "현대자동차는 차량 생산량을 15% 감축한다고 발표했다"
            ],
            "sectors": ["semiconductors", "automobiles"],
            "relationship_type": "supply_chain_impact"
        })
    
    # Example 3: Fed policy ripple effects
    cross_ref_examples.append({
        "example_id": "fed_policy_ripples",
        "text": "미국 연준의 공격적인 금리 인상이 글로벌 시장에 광범위한 영향을 미쳤다. 2022년 7차례 금리 인상은 다양한 산업 부문의 투자 심리를 위축시켰다.",
        "cross_references": [
            "미국 연준이 2022년 기준금리를 7차례 인상했다"
        ],
        "sectors": ["financials"],
        "relationship_type": "policy_impact"
    })
    
    # Example based on actual entity connections
    if entity_connections:
        conn = entity_connections[0]
        cross_ref_examples.append({
            "example_id": "entity_multi_mention",
            "text": f"{conn['claim1']} 또한 {conn['claim2']}",
            "cross_references": [conn['claim1'], conn['claim2']],
            "shared_entity": conn['entity_name'],
            "sectors": list(set(conn['sectors1'] + conn['sectors2'])),
            "relationship_type": "shared_entity"
        })
    
    # Save cross-reference examples
    with open('cross_reference_examples.json', 'w', encoding='utf-8') as f:
        json.dump(cross_ref_examples, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nGenerated {len(cross_ref_examples)} cross-reference examples:")
    print("="*60)
    
    for example in cross_ref_examples:
        print(f"\nExample ID: {example['example_id']}")
        print(f"Text: {example['text']}")
        print(f"Cross-references: {example['cross_references']}")
        print(f"Sectors: {example['sectors']}")
        print(f"Relationship: {example['relationship_type']}")
        print("-" * 40)
    
    print(f"\nCross-reference examples saved to: cross_reference_examples.json")

if __name__ == "__main__":
    analyze_factblock_relationships()