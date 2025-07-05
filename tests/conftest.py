"""
Test configuration and fixtures for GraphRAG retrieval system tests.
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.config import AppConfig
from src.retrieval import RetrievalModule


@pytest.fixture
def mock_neo4j_client():
    """Mock Neo4j client for testing without database dependency."""
    mock_client = Mock()
    mock_client.verify_connectivity.return_value = True
    mock_client.execute_query.return_value = []
    return mock_client


@pytest.fixture
def test_config():
    """Test configuration with safe defaults."""
    return {
        'neo4j_uri': 'bolt://localhost:7687',
        'neo4j_user': 'test_user',
        'neo4j_password': 'test_password'
    }


@pytest.fixture
def app_config():
    """Application configuration for testing."""
    return AppConfig()


@pytest.fixture
def retrieval_module_mock(test_config, monkeypatch):
    """RetrievalModule with mocked Neo4j client."""
    # Mock the Neo4jClient to avoid actual database connections
    mock_client_class = Mock()
    mock_client_instance = Mock()
    mock_client_instance.verify_connectivity.return_value = True
    mock_client_instance.execute_query.return_value = []
    mock_client_class.return_value = mock_client_instance
    
    monkeypatch.setattr('src.retrieval.graph_rag.Neo4jClient', mock_client_class)
    
    module = RetrievalModule('graphrag')
    module.initialize(test_config)
    return module


@pytest.fixture
def sample_query_results():
    """Sample query results for testing."""
    return [
        {
            'n': {
                'name': 'GDPR',
                'description': 'General Data Protection Regulation',
                'category': 'data_privacy'
            },
            'node_type': ['FederalRegulation'],
            'node_id': 1
        },
        {
            'n': {
                'title': 'Data Protection Guidelines',
                'summary': 'Guidelines for data protection compliance',
                'category': 'data_privacy'
            },
            'node_type': ['AgencyGuidance'],
            'node_id': 2
        }
    ]