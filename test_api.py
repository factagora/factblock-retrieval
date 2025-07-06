#!/usr/bin/env python3
"""
Basic API tests for GraphRAG Fact-Check API
Used by GitHub Actions CI/CD pipeline
"""

import sys
import os
sys.path.insert(0, 'src')

from fastapi.testclient import TestClient
from api.graphrag_fact_check import app

def test_endpoints():
    """Test basic API endpoints without external dependencies"""
    client = TestClient(app)
    
    print("🧪 Testing API endpoints...")
    
    # Test health endpoint
    response = client.get('/health')
    assert response.status_code == 200, f"Health check failed: {response.status_code}"
    data = response.json()
    assert 'status' in data
    print('✅ Health endpoint passed')
    
    # Test example texts endpoint
    response = client.get('/example-texts')
    assert response.status_code == 200, f"Example texts failed: {response.status_code}"
    data = response.json()
    assert 'examples' in data
    assert 'total_count' in data
    assert data['total_count'] > 0
    assert len(data['examples']) > 0
    
    # Validate example structure
    example = data['examples'][0]
    required_fields = ['id', 'title', 'text', 'category', 'complexity', 'why_graphrag_better']
    for field in required_fields:
        assert field in example, f"Missing field in example: {field}"
    
    print(f'✅ Example texts endpoint passed ({data["total_count"]} examples)')
    
    # Test root endpoint
    response = client.get('/')
    assert response.status_code == 200, f"Root endpoint failed: {response.status_code}"
    data = response.json()
    assert 'service' in data
    assert 'endpoints' in data
    print('✅ Root endpoint passed')
    
    # Test debug endpoint
    response = client.get('/debug')
    assert response.status_code == 200, f"Debug endpoint failed: {response.status_code}"
    print('✅ Debug endpoint passed')
    
    print('🎉 All API tests passed!')

if __name__ == '__main__':
    test_endpoints()