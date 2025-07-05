"""
Tests for configuration management module.
"""

import os
import pytest
from src.config import Neo4jConfig, RetrievalConfig, AppConfig, load_config


class TestNeo4jConfig:
    """Test Neo4j configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = Neo4jConfig()
        assert config.uri == "bolt://localhost:7687"
        assert config.user == "neo4j"
        assert config.password == "password123"
    
    def test_environment_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("NEO4J_URI", "bolt://custom:7687")
        monkeypatch.setenv("NEO4J_USER", "custom_user")
        monkeypatch.setenv("NEO4J_PASSWORD", "custom_pass")
        
        config = Neo4jConfig()
        assert config.uri == "bolt://custom:7687"
        assert config.user == "custom_user"
        assert config.password == "custom_pass"


class TestRetrievalConfig:
    """Test retrieval configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = RetrievalConfig()
        assert config.default_limit == 10
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.score_threshold == 0.7
        assert config.expand_hops == 2
    
    def test_environment_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("RETRIEVAL_DEFAULT_LIMIT", "20")
        monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.8")
        monkeypatch.setenv("RETRIEVAL_EXPAND_HOPS", "3")
        
        config = RetrievalConfig()
        assert config.default_limit == 20
        assert config.score_threshold == 0.8
        assert config.expand_hops == 3
    
    def test_validation_constraints(self):
        """Test validation constraints."""
        with pytest.raises(ValueError):
            RetrievalConfig(default_limit=0)  # Should be >= 1
        
        with pytest.raises(ValueError):
            RetrievalConfig(score_threshold=1.5)  # Should be <= 1.0
        
        with pytest.raises(ValueError):
            RetrievalConfig(expand_hops=0)  # Should be >= 1


class TestAppConfig:
    """Test main application configuration."""
    
    def test_initialization(self):
        """Test configuration initialization."""
        config = AppConfig()
        assert isinstance(config.neo4j, Neo4jConfig)
        assert isinstance(config.retrieval, RetrievalConfig)
    
    def test_to_dict(self):
        """Test configuration dictionary conversion."""
        config = AppConfig()
        config_dict = config.to_dict()
        
        expected_keys = {'neo4j_uri', 'neo4j_user', 'neo4j_password', 'retrieval_config'}
        assert set(config_dict.keys()) == expected_keys
        
        assert config_dict['neo4j_uri'] == config.neo4j.uri
        assert config_dict['neo4j_user'] == config.neo4j.user
        assert config_dict['neo4j_password'] == config.neo4j.password
        assert isinstance(config_dict['retrieval_config'], dict)
    
    def test_get_neo4j_config(self):
        """Test Neo4j configuration extraction."""
        config = AppConfig()
        neo4j_config = config.get_neo4j_config()
        
        expected_keys = {'neo4j_uri', 'neo4j_user', 'neo4j_password'}
        assert set(neo4j_config.keys()) == expected_keys
        assert neo4j_config['neo4j_uri'] == config.neo4j.uri
    
    def test_get_retrieval_config(self):
        """Test retrieval configuration extraction."""
        config = AppConfig()
        retrieval_config = config.get_retrieval_config()
        
        expected_keys = {'default_limit', 'embedding_model', 'score_threshold', 'expand_hops'}
        assert set(retrieval_config.keys()) == expected_keys
        assert retrieval_config['default_limit'] == config.retrieval.default_limit
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        config = AppConfig()
        assert config.validate_config() is True
    
    def test_validate_config_invalid(self, monkeypatch):
        """Test configuration validation with invalid config."""
        # Test with invalid URI
        monkeypatch.setenv("NEO4J_URI", "")
        config = AppConfig()
        assert config.validate_config() is False
    
    def test_repr(self):
        """Test string representation."""
        config = AppConfig()
        repr_str = repr(config)
        assert "AppConfig(" in repr_str
        assert "neo4j_uri=" in repr_str
        assert "neo4j_user=" in repr_str
        assert "default_limit=" in repr_str
        assert "score_threshold=" in repr_str


def test_load_config():
    """Test convenience function for loading configuration."""
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.validate_config() is True