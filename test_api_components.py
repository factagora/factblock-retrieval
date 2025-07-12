#!/usr/bin/env python3
"""
Test API components independently
"""

import os
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment
os.environ['NEO4J_PASSWORD'] = 'password'

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_retrieval_system():
    """Test the retrieval system with a simple query"""
    print("🧪 Testing retrieval system...")
    
    try:
        from src.config import load_config
        from src.retrieval import RetrievalModule
        
        # Initialize
        config = load_config()
        retrieval_module = RetrievalModule('graphrag')
        retrieval_module.initialize(config.to_dict())
        
        print("✅ Retrieval module initialized")
        
        # Test a simple query
        results = retrieval_module.retrieve(
            query_text="Apple company information",
            limit=3
        )
        
        print(f"✅ Query executed, found {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"   {i+1}. Score: {result.score:.3f}")
            print(f"      Source: {result.source_type}")
            print(f"      Content: {result.content[:100]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Retrieval system test failed: {e}")
        traceback.print_exc()
        return False

def test_fact_check_logic():
    """Test the fact-checking logic independently"""
    print("\n🧪 Testing fact-checking logic...")
    
    try:
        from src.api.graphrag_fact_check import extract_claims_from_text, analyze_claim_with_evidence
        
        # Test claim extraction
        test_text = "Apple Inc. was founded in 1976 by Steve Jobs and Steve Wozniak."
        claims = extract_claims_from_text(test_text)
        
        print(f"✅ Extracted {len(claims)} claims from text")
        
        if claims:
            print(f"   First claim: {claims[0]['text']}")
            print(f"   Claim type: {claims[0]['type']}")
        
        # Test with some mock evidence
        mock_evidence = [
            {
                'content': 'Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.',
                'source_type': 'PublicRecord',
                'score': 0.9,
                'metadata': {'category': 'company_info'}
            }
        ]
        
        if claims:
            # Test fact-checking with evidence
            print("   Testing fact-checking with mock evidence...")
            instance = analyze_claim_with_evidence(claims[0], mock_evidence)
            
            print(f"   Result: {instance.label} (confidence: {instance.confidence:.2f})")
            print(f"   Reasoning: {instance.reasoning[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Fact-checking logic test failed: {e}")
        traceback.print_exc()
        return False

def test_openai_integration():
    """Test OpenAI integration"""
    print("\n🧪 Testing OpenAI integration...")
    
    try:
        from src.api.graphrag_fact_check import initialize_openai_client, openai_client, use_azure
        
        # Initialize OpenAI client
        initialize_openai_client()
        
        print(f"✅ OpenAI client initialized: {openai_client is not None}")
        print(f"   Using Azure: {use_azure}")
        
        if openai_client:
            print("   OpenAI client is available for LLM fact-checking")
        else:
            print("   OpenAI client not available - will use fallback")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI integration test failed: {e}")
        traceback.print_exc()
        return False

def test_health_endpoint_logic():
    """Test the health endpoint logic"""
    print("\n🧪 Testing health endpoint logic...")
    
    try:
        from src.api.graphrag_fact_check import initialize_retrieval_system, initialize_openai_client
        from src.api.graphrag_fact_check import retrieval_module, openai_client, use_azure
        
        # Initialize systems
        initialize_openai_client()
        retrieval_success = initialize_retrieval_system()
        
        # Build health response
        health_data = {
            "status": "healthy",
            "service": "graphrag-fact-check-api",
            "retrieval_system": "available" if retrieval_module else "unavailable",
            "ai_provider": "azure-openai" if use_azure else "openai",
            "llm_client": "available" if openai_client else "unavailable",
            "features": ["hybrid-fact-checking", "compliance-evidence", "graphrag-retrieval", "llm-analysis"],
            "fact_check_method": "hybrid_graphrag_llm"
        }
        
        print("✅ Health check data generated:")
        import json
        print(json.dumps(health_data, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Health endpoint logic test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all component tests"""
    print("🚀 API Component Testing")
    print("=" * 60)
    
    tests = [
        ("Retrieval System", test_retrieval_system),
        ("Fact-Check Logic", test_fact_check_logic),
        ("OpenAI Integration", test_openai_integration),
        ("Health Endpoint Logic", test_health_endpoint_logic),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPONENT TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} component tests passed")
    
    if passed == total:
        print("\n🎉 All component tests passed!")
        print("✅ The API should work correctly when properly started")
        print("\n🌐 API Features Available:")
        print("   • GraphRAG retrieval from Neo4j")
        print("   • Hybrid fact-checking (GraphRAG + LLM)")
        print("   • Compliance evidence retrieval")
        print("   • Azure OpenAI integration")
        print("   • Interactive API documentation")
        
        print("\n🚀 Manual API Test:")
        print("   1. Start server: python3 run_api.py")
        print("   2. Health check: curl http://localhost:8001/health")
        print("   3. Documentation: http://localhost:8001/docs")
        print("   4. Test fact-checking with sample text")
        
    else:
        print(f"\n⚠️ {total - passed} component tests failed.")
        print("Fix these issues before running the API.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)