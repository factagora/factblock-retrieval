"""
Integration tests for GraphRAG retrieval system.

These tests require a running Neo4j instance and are marked with @pytest.mark.integration.
Run with: pytest -m integration
"""

import pytest
import os
from src.retrieval import RetrievalModule
from src.database.neo4j_client import Neo4jClient
from src.config import AppConfig
from src.models import FederalRegulation, Category
from datetime import date


@pytest.mark.integration
class TestNeo4jIntegration:
    """Integration tests with actual Neo4j database."""
    
    @pytest.fixture(scope="class")
    def neo4j_client(self):
        """Create Neo4j client for integration testing."""
        # Use test database configuration
        uri = os.getenv("NEO4J_TEST_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_TEST_USER", "neo4j")
        password = os.getenv("NEO4J_TEST_PASSWORD", "password123")
        
        client = Neo4jClient(uri, user, password)
        
        # Verify connection
        if not client.verify_connectivity():
            pytest.skip("Neo4j database not available for integration tests")
        
        yield client
        
        # Cleanup after tests
        client.close()
    
    def test_database_connectivity(self, neo4j_client):
        """Test basic database connectivity."""
        assert neo4j_client.verify_connectivity()
        
        # Test basic query
        result = neo4j_client.execute_query("RETURN 1 as test")
        assert len(result) == 1
        assert result[0]["test"] == 1
    
    def test_database_info(self, neo4j_client):
        """Test database information retrieval."""
        info = neo4j_client.get_database_info()
        
        assert "node_count" in info
        assert "relationship_count" in info
        assert isinstance(info["node_count"], (int, str))
    
    @pytest.fixture
    def clean_database(self, neo4j_client):
        """Clean database before and after each test."""
        # Clean before test
        neo4j_client.execute_query("MATCH (n) DETACH DELETE n")
        
        yield
        
        # Clean after test
        neo4j_client.execute_query("MATCH (n) DETACH DELETE n")
    
    def test_create_and_query_nodes(self, neo4j_client, clean_database):
        """Test creating and querying nodes."""
        # Create test data
        create_query = """
        CREATE (r:FederalRegulation {
            name: $name,
            description: $description,
            category: $category
        })
        """
        
        neo4j_client.execute_write_query(create_query, {
            "name": "Test Regulation",
            "description": "A test regulation for data privacy",
            "category": "data_privacy"
        })
        
        # Query the data
        search_query = """
        MATCH (r:FederalRegulation)
        WHERE r.description CONTAINS $search_text
        RETURN r
        """
        
        results = neo4j_client.execute_query(search_query, {
            "search_text": "data privacy"
        })
        
        assert len(results) == 1
        assert results[0]["r"]["name"] == "Test Regulation"


@pytest.mark.integration
class TestRetrievalIntegration:
    """Integration tests for the full retrieval system."""
    
    @pytest.fixture(scope="class")
    def retrieval_module(self):
        """Create configured retrieval module."""
        config = AppConfig()
        module = RetrievalModule('graphrag')
        
        try:
            module.initialize(config.to_dict())
        except RuntimeError:
            pytest.skip("Neo4j database not available for integration tests")
        
        return module
    
    @pytest.fixture
    def sample_data(self, retrieval_module):
        """Load sample data for testing."""
        # Clean database
        client = retrieval_module.retriever.client
        client.execute_query("MATCH (n) DETACH DELETE n")
        
        # Create sample regulations
        regulations = [
            {
                "name": "GDPR",
                "description": "General Data Protection Regulation for privacy",
                "category": "data_privacy",
                "citation": "Regulation (EU) 2016/679"
            },
            {
                "name": "CCPA",
                "description": "California Consumer Privacy Act",
                "category": "data_privacy",
                "citation": "Cal. Civ. Code § 1798.100"
            },
            {
                "name": "SOX",
                "description": "Sarbanes-Oxley Act for financial reporting",
                "category": "financial",
                "citation": "Public Law 107-204"
            }
        ]
        
        create_query = """
        CREATE (r:FederalRegulation {
            name: $name,
            description: $description,
            category: $category,
            citation: $citation
        })
        """
        
        for reg in regulations:
            client.execute_write_query(create_query, reg)
        
        # Create some agency guidance
        guidance_query = """
        CREATE (g:AgencyGuidance {
            title: $title,
            summary: $summary,
            category: $category,
            agency: $agency
        })
        """
        
        client.execute_write_query(guidance_query, {
            "title": "Data Protection Guidelines",
            "summary": "Guidelines for implementing data privacy measures",
            "category": "data_privacy",
            "agency": "Data Protection Authority"
        })
        
        yield
        
        # Cleanup
        client.execute_query("MATCH (n) DETACH DELETE n")
    
    def test_basic_retrieval(self, retrieval_module, sample_data):
        """Test basic retrieval functionality."""
        results = retrieval_module.retrieve(
            "data privacy regulations",
            limit=10
        )
        
        assert len(results) > 0
        assert all(result.score >= 0 for result in results)
        
        # Should find GDPR and CCPA
        content_texts = [result.content.lower() for result in results]
        assert any("gdpr" in text for text in content_texts)
    
    def test_filtered_retrieval(self, retrieval_module, sample_data):
        """Test retrieval with category filters."""
        results = retrieval_module.retrieve(
            "regulations",
            filters={"category": "data_privacy"},
            limit=10
        )
        
        assert len(results) > 0
        
        # All results should be data privacy related
        for result in results:
            metadata = result.metadata
            if "category" in metadata:
                assert metadata["category"] == "data_privacy"
    
    def test_result_ranking(self, retrieval_module, sample_data):
        """Test that results are properly ranked by relevance."""
        results = retrieval_module.retrieve(
            "GDPR data protection",
            limit=5
        )
        
        assert len(results) > 0
        
        # Results should be sorted by score (descending)
        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)
        
        # GDPR should be the top result for this query
        top_result = results[0]
        assert "gdpr" in top_result.content.lower()
    
    def test_limit_parameter(self, retrieval_module, sample_data):
        """Test that limit parameter is respected."""
        results = retrieval_module.retrieve(
            "regulations",
            limit=2
        )
        
        assert len(results) <= 2
    
    def test_empty_query_handling(self, retrieval_module, sample_data):
        """Test handling of queries that return no results."""
        results = retrieval_module.retrieve(
            "nonexistent regulation about aliens",
            limit=10
        )
        
        # Should return empty list, not raise an error
        assert isinstance(results, list)
    
    def test_graph_expansion(self, retrieval_module, sample_data):
        """Test that graph relationships are used for context expansion."""
        # First, create a relationship between regulations
        client = retrieval_module.retriever.client
        
        relationship_query = """
        MATCH (gdpr:FederalRegulation {name: "GDPR"})
        MATCH (guidance:AgencyGuidance)
        CREATE (gdpr)-[:RELATED_TO]->(guidance)
        """
        
        client.execute_write_query(relationship_query)
        
        results = retrieval_module.retrieve(
            "GDPR",
            limit=10
        )
        
        # Should include both the GDPR regulation and related guidance
        assert len(results) >= 2
        
        # Check if expansion results are marked
        expansion_results = [r for r in results if r.metadata.get("is_expansion", False)]
        assert len(expansion_results) > 0


@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error handling scenarios."""
    
    def test_invalid_database_connection(self):
        """Test handling of invalid database connections."""
        config = {
            'neo4j_uri': 'bolt://invalid:7687',
            'neo4j_user': 'invalid',
            'neo4j_password': 'invalid'
        }
        
        module = RetrievalModule('graphrag')
        
        with pytest.raises(Exception):  # Could be various connection errors
            module.initialize(config)
    
    def test_database_query_error(self, monkeypatch):
        """Test handling of database query errors."""
        # This would require mocking database failures
        # Implementation depends on specific error scenarios
        pass


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-m", "integration", "-v"])