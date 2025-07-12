#!/usr/bin/env python3
"""
Test Enhanced Fact-Checking System

This script tests the relationship-aware fact-checking capabilities with
our regulatory cascade examples.
"""

import os
import sys
import json
import logging
from dotenv import load_dotenv

# Enable logging
logging.basicConfig(level=logging.INFO)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.graphrag.enhanced_graphrag import EnhancedGraphRAG

def test_enhanced_fact_checking():
    """Test enhanced fact-checking with regulatory cascade examples"""
    
    load_dotenv()
    
    # Initialize Enhanced GraphRAG
    enhanced_graphrag = EnhancedGraphRAG(
        neo4j_uri=os.getenv('NEO4J_URI', 'bolt://20.81.43.138:7687'),
        neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
        neo4j_password=os.getenv('NEO4J_PASSWORD', 'password'),
        enable_parallel_processing=True
    )
    
    # Test claims based on our exact regulatory cascade examples
    test_claims = [
        "EU가 쿠키 사용에 대한 명시적 동의를 의무화했다",
        "미국 캘리포니아주가 아동 온라인 개인정보 수집을 전면 금지했다",
        "중국이 탄소배출권 거래제를 화학업계로 확대 적용했다"
    ]
    
    print("🧪 Testing Enhanced Fact-Checking System")
    print("=" * 60)
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\n📋 Test {i}: {claim}")
        print("-" * 50)
        
        try:
            result = enhanced_graphrag.fact_check_enhanced(
                claim=claim,
                max_evidence=3,
                max_relationship_depth=3,
                min_confidence_threshold=0.5,
                enable_regulatory_cascades=True
            )
            
            print(f"✅ Final Verdict: {result.final_verdict}")
            print(f"📊 Final Confidence: {result.final_confidence:.2f}")
            print(f"🔗 Regulatory Cascades Found: {len(result.regulatory_cascades)}")
            print(f"💡 Unique Insights: {len(result.unique_relationship_insights)}")
            print(f"⏱️ Total Time: {result.total_time_ms:.1f}ms")
            
            if result.unique_relationship_insights:
                print("\n🎯 Unique Relationship Insights:")
                for insight in result.unique_relationship_insights:
                    print(f"  • {insight}")
            
            if result.regulatory_cascades:
                print(f"\n🏛️ Regulatory Cascades Detected:")
                for cascade in result.regulatory_cascades[:2]:  # Show first 2
                    print(f"  • {cascade.get('explanation', 'Unknown cascade')}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    enhanced_graphrag.close()
    print("\n✅ Enhanced fact-checking tests completed!")

if __name__ == "__main__":
    test_enhanced_fact_checking()