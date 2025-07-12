#!/usr/bin/env python3
"""
Test the local API endpoint
"""

import requests
import json
import os
from dotenv import load_dotenv

def test_api():
    """Test the local API"""
    
    # Load environment
    load_dotenv()
    
    base_url = "http://localhost:8002"
    
    print("🧪 Testing Local Enhanced GraphRAG API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n📊 Health Check:")
    try:
        response = requests.get(f"{base_url}/health")
        health = response.json()
        print(f"  Status: {health['status']}")
        print(f"  Enhanced GraphRAG: {health['enhanced_graphrag']}")
        print(f"  Features: {len(health['features'])} features available")
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        return
    
    # Test 2: Enhanced fact-checking with exact data
    print("\n🔍 Enhanced Fact-Checking Test:")
    test_data = {
        "text": "EU가 쿠키 사용에 대한 명시적 동의를 의무화하는 GDPR 개정안을 발표했다",
        "max_evidence": 3,
        "max_relationship_depth": 2,
        "min_confidence_threshold": 0.5,
        "enable_regulatory_cascades": True
    }
    
    try:
        response = requests.post(
            f"{base_url}/fact-check-enhanced",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ Final Verdict: {result['final_verdict']}")
            print(f"  📊 Final Confidence: {result['final_confidence']:.2f}")
            print(f"  🔗 Regulatory Cascades: {len(result['regulatory_cascades'])}")
            print(f"  💡 Unique Insights: {len(result['unique_relationship_insights'])}")
            print(f"  ⏱️ Total Time: {result['total_time_ms']:.1f}ms")
            
            if result['unique_relationship_insights']:
                print("  🎯 Insights:")
                for insight in result['unique_relationship_insights']:
                    print(f"    • {insight}")
        else:
            print(f"  ❌ API Error: {response.status_code}")
            print(f"  Response: {response.text}")
            
    except Exception as e:
        print(f"  ❌ Request failed: {e}")
    
    # Test 3: Traditional fact-checking for comparison
    print("\n🔍 Traditional Fact-Checking Test (for comparison):")
    traditional_data = {
        "text": "EU가 쿠키 사용에 대한 명시적 동의를 의무화하는 GDPR 개정안을 발표했다",
        "max_evidence": 3
    }
    
    try:
        response = requests.post(
            f"{base_url}/fact-check-graphrag",
            json=traditional_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ Total Claims: {result['total_claims']}")
            print(f"  📊 Evidence Summary: {result['evidence_summary']['total_documents']} documents")
            print(f"  ⏱️ Processing Time: {result['processing_time']}s")
        else:
            print(f"  ❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Request failed: {e}")
    
    # Test 4: Root endpoint
    print("\n🏠 Root Endpoint:")
    try:
        response = requests.get(f"{base_url}/")
        root = response.json()
        print(f"  Service: {root['service']}")
        print(f"  Version: {root['version']}")
        print(f"  Available Endpoints: {len(root['endpoints'])}")
    except Exception as e:
        print(f"  ❌ Root request failed: {e}")
    
    print(f"\n✅ API testing completed!")
    print(f"🌐 Access API documentation at: {base_url}/docs")

if __name__ == "__main__":
    test_api()