#!/usr/bin/env python3
"""
Test the running API
"""

import requests
import json
import time

def test_api_endpoints():
    """Test the API endpoints"""
    base_url = "http://localhost:8001"
    
    print("🧪 Testing API endpoints...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ Root endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    # Test example texts endpoint
    try:
        response = requests.get(f"{base_url}/example-texts", timeout=5)
        print(f"✅ Example texts endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Found {data['total_count']} examples")
            print(f"   Categories: {data['categories']}")
        
    except Exception as e:
        print(f"❌ Example texts endpoint failed: {e}")
    
    # Test fact-check endpoint with simple text
    try:
        test_data = {
            "text": "Apple Inc. was founded in 1976 by Steve Jobs and Steve Wozniak.",
            "max_evidence": 3
        }
        
        response = requests.post(
            f"{base_url}/fact-check-graphrag",
            json=test_data,
            timeout=30
        )
        
        print(f"✅ Fact-check endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Processed {data['total_claims']} claims")
            print(f"   Processing time: {data['processing_time']}s")
            print(f"   Evidence documents: {data['evidence_summary']['total_documents']}")
            
            # Show first result
            if data['instances']:
                instance = data['instances'][0]
                print(f"   First result: {instance['label']} (confidence: {instance['confidence']:.2f})")
                print(f"   Reasoning: {instance['reasoning'][:100]}...")
        
    except Exception as e:
        print(f"❌ Fact-check endpoint failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing running API...")
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    test_api_endpoints()
    
    print("\n✅ API testing complete!")
    print("🌐 API is running at: http://localhost:8001")
    print("📚 Documentation: http://localhost:8001/docs")
    print("🔍 Interactive docs: http://localhost:8001/redoc")