"""
Tests for retrieval module components.
"""

import pytest
from unittest.mock import Mock, patch
from src.retrieval import RetrievalModule, RetrievalQuery, RetrievalResult, GraphRAGRetriever
from src.retrieval.base import BaseRetriever


class TestRetrievalQuery:
    """Test RetrievalQuery dataclass."""
    
    def test_creation_with_defaults(self):
        """Test query creation with default values."""
        query = RetrievalQuery("test query")
        assert query.query_text == "test query"
        assert query.filters is None
        assert query.limit == 10
    
    def test_creation_with_all_params(self):
        """Test query creation with all parameters."""
        filters = {"category": "data_privacy"}
        query = RetrievalQuery("test query", filters, 5)
        assert query.query_text == "test query"
        assert query.filters == filters
        assert query.limit == 5


class TestRetrievalResult:
    """Test RetrievalResult dataclass."""
    
    def test_creation(self):
        """Test result creation."""
        metadata = {"node_id": 1, "node_type": ["FederalRegulation"]}
        result = RetrievalResult("test content", metadata, 0.85, "regulation")
        
        assert result.content == "test content"
        assert result.metadata == metadata
        assert result.score == 0.85
        assert result.source_type == "regulation"


class TestRetrievalModule:
    """Test main RetrievalModule interface."""
    
    def test_initialization_default(self):
        """Test module initialization with defaults."""
        module = RetrievalModule()
        assert module.get_retriever_type() == "graphrag"
        assert not module.is_initialized()
    
    def test_initialization_custom_type(self):
        """Test module initialization with custom retriever type."""
        module = RetrievalModule("custom")
        assert module.get_retriever_type() == "custom"
    
    def test_initialize_graphrag(self, test_config, monkeypatch):
        """Test initialization with GraphRAG retriever."""
        # Mock Neo4jClient to avoid database connection
        mock_client_class = Mock()
        mock_client_instance = Mock()
        mock_client_instance.verify_connectivity.return_value = True
        mock_client_class.return_value = mock_client_instance
        
        monkeypatch.setattr('src.retrieval.graph_rag.Neo4jClient', mock_client_class)
        
        module = RetrievalModule("graphrag")
        module.initialize(test_config)
        
        assert module.is_initialized()
        assert isinstance(module.retriever, GraphRAGRetriever)
    
    def test_initialize_unknown_type(self, test_config):
        """Test initialization with unknown retriever type."""
        module = RetrievalModule("unknown")
        with pytest.raises(ValueError, match="Unknown retriever type: unknown"):
            module.initialize(test_config)
    
    def test_retrieve_uninitialized(self):
        """Test retrieval without initialization."""
        module = RetrievalModule()
        with pytest.raises(RuntimeError, match="Module not initialized"):
            module.retrieve("test query")
    
    def test_retrieve_success(self, retrieval_module_mock):
        """Test successful retrieval."""
        # Mock the retrieve method to return sample results
        mock_results = [
            RetrievalResult("content", {"id": 1}, 0.9, "regulation")
        ]
        retrieval_module_mock.retriever.retrieve = Mock(return_value=mock_results)
        
        results = retrieval_module_mock.retrieve("test query", {"category": "data_privacy"}, 5)
        
        assert len(results) == 1
        assert results[0].content == "content"
        assert results[0].score == 0.9
        
        # Verify the retrieve method was called with correct parameters
        retrieval_module_mock.retriever.retrieve.assert_called_once()
        call_args = retrieval_module_mock.retriever.retrieve.call_args[0][0]
        assert call_args.query_text == "test query"
        assert call_args.filters == {"category": "data_privacy"}
        assert call_args.limit == 5


class TestGraphRAGRetriever:
    """Test GraphRAGRetriever implementation."""
    
    def test_initialization(self):
        """Test retriever initialization."""
        retriever = GraphRAGRetriever()
        assert retriever.client is None
        assert not retriever._initialized
    
    def test_initialize_success(self, test_config, monkeypatch):
        """Test successful initialization."""
        mock_client_class = Mock()
        mock_client_instance = Mock()
        mock_client_instance.verify_connectivity.return_value = True
        mock_client_class.return_value = mock_client_instance
        
        monkeypatch.setattr('src.retrieval.graph_rag.Neo4jClient', mock_client_class)
        
        retriever = GraphRAGRetriever()
        retriever.initialize(test_config)
        
        assert retriever._initialized
        assert retriever.client is not None
    
    def test_initialize_connection_failure(self, test_config, monkeypatch):
        """Test initialization with connection failure."""
        mock_client_class = Mock()
        mock_client_instance = Mock()
        mock_client_instance.verify_connectivity.return_value = False
        mock_client_class.return_value = mock_client_instance
        
        monkeypatch.setattr('src.retrieval.graph_rag.Neo4jClient', mock_client_class)
        
        retriever = GraphRAGRetriever()
        with pytest.raises(RuntimeError, match="Failed to establish Neo4j connection"):
            retriever.initialize(test_config)
    
    def test_retrieve_uninitialized(self):
        """Test retrieval without initialization."""
        retriever = GraphRAGRetriever()
        query = RetrievalQuery("test")
        
        with pytest.raises(RuntimeError, match="Retriever not initialized"):
            retriever.retrieve(query)
    
    def test_graph_search(self, sample_query_results, monkeypatch):
        """Test graph search functionality."""
        mock_client_class = Mock()
        mock_client_instance = Mock()
        mock_client_instance.verify_connectivity.return_value = True
        mock_client_instance.execute_query.return_value = sample_query_results
        mock_client_class.return_value = mock_client_instance
        
        monkeypatch.setattr('src.retrieval.graph_rag.Neo4jClient', mock_client_class)
        
        retriever = GraphRAGRetriever()
        retriever.initialize({'neo4j_uri': 'test', 'neo4j_user': 'test', 'neo4j_password': 'test'})
        
        results = retriever._graph_search("data privacy", {"category": "data_privacy"})
        assert len(results) == 2
        assert results[0]['n']['name'] == 'GDPR'
    
    def test_score_results(self):
        """Test result scoring functionality."""
        retriever = GraphRAGRetriever()
        
        nodes = [
            {
                'n': {'description': 'data privacy regulation', 'name': 'GDPR'},
                'node_type': ['FederalRegulation'],
                'is_expansion': False
            },
            {
                'n': {'summary': 'guidance on privacy'},
                'node_type': ['AgencyGuidance'],
                'is_expansion': True
            }
        ]
        
        scored_results = retriever._score_results(nodes, "data privacy")
        
        assert len(scored_results) == 2
        assert all('score' in result for result in scored_results)
        # Direct match should have higher score than expansion
        assert scored_results[0]['score'] > scored_results[1]['score']
    
    def test_format_results(self):
        """Test result formatting."""
        retriever = GraphRAGRetriever()
        
        scored_nodes = [
            {
                'n': {'name': 'GDPR', 'description': 'Data protection regulation'},
                'node_type': ['FederalRegulation'],
                'node_id': 1,
                'score': 0.9
            }
        ]
        
        results = retriever._format_results(scored_nodes)
        
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert results[0].score == 0.9
        assert results[0].source_type == 'FederalRegulation'
        assert 'Name: GDPR' in results[0].content
        assert 'Description: Data protection regulation' in results[0].content
    
    def test_generate_embedding(self):
        """Test embedding generation (simple hash-based)."""
        retriever = GraphRAGRetriever()
        
        embedding1 = retriever._generate_embedding("test text")
        embedding2 = retriever._generate_embedding("test text")
        embedding3 = retriever._generate_embedding("different text")
        
        assert embedding1 == embedding2  # Same text should produce same embedding
        assert embedding1 != embedding3  # Different text should produce different embedding
        assert isinstance(embedding1, str)


class TestBaseRetriever:
    """Test abstract base retriever."""
    
    def test_cannot_instantiate(self):
        """Test that BaseRetriever cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseRetriever()
    
    def test_subclass_must_implement_methods(self):
        """Test that subclasses must implement abstract methods."""
        class IncompleteRetriever(BaseRetriever):
            pass
        
        with pytest.raises(TypeError):
            IncompleteRetriever()
    
    def test_complete_subclass_works(self):
        """Test that complete subclass implementation works."""
        class CompleteRetriever(BaseRetriever):
            def retrieve(self, query):
                return []
            
            def initialize(self, config):
                pass
        
        # Should not raise an error
        retriever = CompleteRetriever()
        assert isinstance(retriever, BaseRetriever)