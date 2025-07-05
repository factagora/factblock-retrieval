#!/usr/bin/env python3
"""
Test script for GraphRAG Fact Check API
"""

import requests
import json
import time
import os

def test_graphrag_fact_check_api():
    """Test the GraphRAG fact-check API"""
    
    # API endpoint
    base_url = "http://localhost:8001"
    
    print("🧪 Testing GraphRAG Fact Check API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✓ Health check passed")
            print(f"   ✓ Service: {health_data.get('service')}")
            print(f"   ✓ Retrieval system: {health_data.get('retrieval_system')}")
            print(f"   ✓ AI provider: {health_data.get('ai_provider')}")
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Cannot connect to API: {e}")
        print("   ℹ  Make sure the API is running: python src/api/graphrag_fact_check.py")
        return False
    
    # Test 2: Basic fact-check
    print("\n2. Testing basic fact-check...")
    test_text = "GDPR requires companies to obtain explicit consent for data processing and allows users to request deletion of their personal data."
    
    try:
        response = requests.post(
            f"{base_url}/fact-check-graphrag",
            json={"text": test_text},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Fact-check completed")
            print(f"   ✓ Total claims: {result['total_claims']}")
            print(f"   ✓ Processing time: {result['processing_time']:.3f}s")
            print(f"   ✓ Evidence documents: {result['evidence_summary']['total_documents']}")
            
            # Show first instance
            if result['instances']:
                instance = result['instances'][0]
                print(f"   ✓ First claim label: {instance['label']}")
                print(f"   ✓ Confidence: {instance['confidence']:.3f}")
                print(f"   ✓ Evidence count: {len(instance.get('compliance_evidence', []))}")
        else:
            print(f"   ✗ Fact-check failed: {response.status_code}")
            print(f"   ✗ Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Request failed: {e}")
        return False
    
    # Test 3: Compliance-focused fact-check
    print("\n3. Testing compliance-focused fact-check...")
    test_text = "Financial institutions must report suspicious transactions to FinCEN within 30 days and maintain records for at least 5 years."
    
    try:
        response = requests.post(
            f"{base_url}/fact-check-graphrag",
            json={
                "text": test_text,
                "compliance_focus": ["financial"],
                "max_evidence": 3
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Compliance-focused fact-check completed")
            print(f"   ✓ Compliance coverage: {result['compliance_coverage']}")
            print(f"   ✓ Source types: {result['evidence_summary']['source_types']}")
        else:
            print(f"   ✗ Compliance fact-check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Request failed: {e}")
        return False
    
    # Test 4: Complex multi-claim text
    print("\n4. Testing complex multi-claim text...")
    complex_text = """
    The European Union's General Data Protection Regulation (GDPR) came into effect on May 25, 2018.
    Under GDPR, companies can be fined up to 4% of their annual global turnover or €20 million, whichever is higher.
    Organizations must implement data protection by design and by default.
    Data subjects have the right to be forgotten and can request deletion of their personal data.
    """
    
    try:
        response = requests.post(
            f"{base_url}/fact-check-graphrag",
            json={
                "text": complex_text,
                "compliance_focus": ["data_privacy"],
                "max_evidence": 5
            },
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Complex text analysis completed")
            print(f"   ✓ Total claims analyzed: {result['total_claims']}")
            print(f"   ✓ Average relevance score: {result['evidence_summary']['avg_relevance_score']:.3f}")
            
            # Show summary of labels
            labels = [instance['label'] for instance in result['instances']]
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            print(f"   ✓ Label distribution: {label_counts}")
            
        else:
            print(f"   ✗ Complex text analysis failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Request failed: {e}")
        return False
    
    print("\n✅ All GraphRAG API tests completed successfully!")
    return True

def compare_with_original_api():
    """Compare GraphRAG API with original fact-check API"""
    
    print("\n🔄 Comparing GraphRAG API with Original API")
    print("=" * 50)
    
    test_text = "GDPR requires explicit consent for data processing and includes the right to be forgotten."
    
    # Test original API (if available)
    print("\n1. Testing original fact-check API...")
    try:
        response = requests.post(
            "http://localhost:8000/fact-check",
            json={"text": test_text},
            timeout=30
        )
        
        if response.status_code == 200:
            original_result = response.json()
            print(f"   ✓ Original API result:")
            print(f"     - Total claims: {original_result['total_claims']}")
            print(f"     - Processing time: {original_result['processing_time']:.3f}s")
            if original_result['instances']:
                print(f"     - First label: {original_result['instances'][0]['label']}")
                print(f"     - First confidence: {original_result['instances'][0]['confidence']:.3f}")
        else:
            print(f"   ⚠ Original API not available: {response.status_code}")
            original_result = None
    except:
        print("   ⚠ Original API not running")
        original_result = None
    
    # Test GraphRAG API
    print("\n2. Testing GraphRAG fact-check API...")
    try:
        response = requests.post(
            "http://localhost:8001/fact-check-graphrag",
            json={
                "text": test_text,
                "compliance_focus": ["data_privacy"]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            graphrag_result = response.json()
            print(f"   ✓ GraphRAG API result:")
            print(f"     - Total claims: {graphrag_result['total_claims']}")
            print(f"     - Processing time: {graphrag_result['processing_time']:.3f}s")
            print(f"     - Evidence documents: {graphrag_result['evidence_summary']['total_documents']}")
            print(f"     - Compliance coverage: {graphrag_result['compliance_coverage']}")
            if graphrag_result['instances']:
                instance = graphrag_result['instances'][0]
                print(f"     - First label: {instance['label']}")
                print(f"     - First confidence: {instance['confidence']:.3f}")
                print(f"     - Source reliability: {instance.get('source_reliability', 'N/A')}")
                
            print("\n3. Key Advantages of GraphRAG API:")
            print("   ✓ Compliance-specific evidence retrieval")
            print("   ✓ Source reliability scoring")
            print("   ✓ Category-based filtering")
            print("   ✓ Rich metadata from regulatory sources")
            print("   ✓ Structured compliance coverage analysis")
            
        else:
            print(f"   ✗ GraphRAG API failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ✗ GraphRAG API error: {e}")

if __name__ == "__main__":
    # Set environment for testing
    os.environ['NEO4J_PASSWORD'] = 'password'
    
    success = test_graphrag_fact_check_api()
    
    if success:
        compare_with_original_api()
    else:
        print("\n❌ GraphRAG API tests failed")