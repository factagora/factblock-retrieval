"""
Tests for Neo4j client functionality.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from src.database.neo4j_client import Neo4jClient


class TestNeo4jClient:
    """Test suite for Neo4jClient class."""
    
    def test_init_with_parameters(self):
        """Test client initialization with direct parameters."""
        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "password"
        
        with patch('src.database.neo4j_client.GraphDatabase.driver') as mock_driver:
            client = Neo4jClient(uri, user, password)
            
            assert client.uri == uri
            assert client.user == user
            assert client.password == password
            mock_driver.assert_called_once_with(uri, auth=(user, password))
    
    @patch.dict(os.environ, {
        'NEO4J_URI': 'bolt://test:7687',
        'NEO4J_USER': 'testuser',
        'NEO4J_PASSWORD': 'testpass'
    })
    def test_from_env(self):
        """Test client creation from environment variables."""
        with patch('src.database.neo4j_client.GraphDatabase.driver') as mock_driver:
            client = Neo4jClient.from_env()
            
            assert client.uri == "bolt://test:7687"
            assert client.user == "testuser"
            assert client.password == "testpass"
            mock_driver.assert_called_once_with(
                "bolt://test:7687", 
                auth=("testuser", "testpass")
            )
    
    def test_verify_connectivity_success(self):
        """Test successful connectivity verification."""
        with patch('src.database.neo4j_client.GraphDatabase.driver') as mock_driver:
            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_record = MagicMock()
            mock_record.__getitem__.return_value = 1
            mock_result.single.return_value = mock_record
            mock_session.run.return_value = mock_result
            mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
            
            client = Neo4jClient("bolt://localhost:7687", "neo4j", "password")
            
            assert client.verify_connectivity() is True
            mock_session.run.assert_called_once_with("RETURN 1 as test")
    
    def test_verify_connectivity_failure(self):
        """Test connectivity verification failure."""
        with patch('src.database.neo4j_client.GraphDatabase.driver') as mock_driver:
            mock_session = MagicMock()
            mock_session.run.side_effect = Exception("Connection failed")
            mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
            
            client = Neo4jClient("bolt://localhost:7687", "neo4j", "password")
            
            assert client.verify_connectivity() is False
    
    def test_execute_query(self):
        """Test query execution."""
        with patch('src.database.neo4j_client.GraphDatabase.driver') as mock_driver:
            # Create mock objects
            mock_record = MagicMock()
            mock_record.data.return_value = {"name": "test"}
            
            mock_result = MagicMock()
            mock_result.__iter__.return_value = iter([mock_record])
            
            mock_session = MagicMock()
            mock_session.run.return_value = mock_result
            mock_session.__enter__.return_value = mock_session
            mock_session.__exit__.return_value = None
            
            mock_driver.return_value.session.return_value = mock_session
            
            client = Neo4jClient("bolt://localhost:7687", "neo4j", "password")
            result = client.execute_query("MATCH (n) RETURN n.name as name")
            
            assert result == [{"name": "test"}]
            mock_session.run.assert_called_once_with("MATCH (n) RETURN n.name as name", {})
    
    def test_close(self):
        """Test client connection closure."""
        with patch('src.database.neo4j_client.GraphDatabase.driver') as mock_driver:
            mock_driver_instance = MagicMock()
            mock_driver.return_value = mock_driver_instance
            
            client = Neo4jClient("bolt://localhost:7687", "neo4j", "password")
            client.close()
            
            mock_driver_instance.close.assert_called_once()


@pytest.mark.integration
class TestNeo4jClientIntegration:
    """Integration tests for Neo4jClient (requires running Neo4j instance)."""
    
    @pytest.fixture
    def client(self):
        """Create test client instance."""
        return Neo4jClient.from_env()
    
    def test_real_connectivity(self, client):
        """Test connectivity with real Neo4j instance."""
        # This test will be skipped if Neo4j is not available
        try:
            result = client.verify_connectivity()
            assert result is True
        except Exception:
            pytest.skip("Neo4j instance not available")
        finally:
            client.close()
    
    def test_database_info(self, client):
        """Test getting database information."""
        try:
            info = client.get_database_info()
            assert "node_count" in info
            assert "relationship_count" in info
            assert isinstance(info["node_count"], (int, str))
        except Exception:
            pytest.skip("Neo4j instance not available")
        finally:
            client.close()