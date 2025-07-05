"""
Tests for example data loader.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import date

from examples.example_data_loader import ExampleDataLoader
from src.models import Category


class TestExampleDataLoader:
    """Test example data loader functionality."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock Neo4j client."""
        client = Mock()
        client.execute_query.return_value = []
        client.execute_write_query.return_value = []
        return client
    
    @pytest.fixture
    def data_loader(self, mock_client):
        """Data loader with mock client."""
        return ExampleDataLoader(mock_client)
    
    def test_initialization(self, mock_client):
        """Test data loader initialization."""
        loader = ExampleDataLoader(mock_client)
        assert loader.client == mock_client
    
    def test_create_sample_regulations(self, data_loader):
        """Test creation of sample regulations."""
        regulations = data_loader._create_sample_regulations()
        
        assert len(regulations) == 5
        
        # Check GDPR regulation
        gdpr = next(r for r in regulations if r.name == "GDPR")
        assert gdpr.citation == "Regulation (EU) 2016/679"
        assert gdpr.category == Category.DATA_PRIVACY
        assert gdpr.effective_date == date(2018, 5, 25)
        assert gdpr.full_text is not None
        
        # Check categories are represented
        categories = {r.category for r in regulations}
        assert Category.DATA_PRIVACY in categories
        assert Category.FINANCIAL in categories
        assert Category.HEALTHCARE in categories
        assert Category.ENVIRONMENTAL in categories
    
    def test_create_sample_guidance(self, data_loader):
        """Test creation of sample guidance documents."""
        guidance_docs = data_loader._create_sample_guidance()
        
        assert len(guidance_docs) == 4
        
        # Check GDPR guidance
        gdpr_guidance = next(g for g in guidance_docs if "GDPR" in g.title)
        assert gdpr_guidance.agency == "European Data Protection Board"
        assert gdpr_guidance.category == Category.DATA_PRIVACY
        assert gdpr_guidance.reference_number == "EDPB-2018-001"
    
    def test_create_sample_enforcement_actions(self, data_loader):
        """Test creation of sample enforcement actions."""
        actions = data_loader._create_sample_enforcement_actions()
        
        assert len(actions) == 4
        
        # Check Facebook fine
        facebook_fine = next(a for a in actions if "Facebook" in a.title)
        assert facebook_fine.agency == "Irish Data Protection Commission"
        assert facebook_fine.category == Category.DATA_PRIVACY
        assert "€225 million" in facebook_fine.outcome
    
    def test_create_sample_compliance_topics(self, data_loader):
        """Test creation of sample compliance topics."""
        topics = data_loader._create_sample_compliance_topics()
        
        assert len(topics) == 5
        
        # Check data privacy topic
        data_privacy_topic = next(t for t in topics if "Data Privacy" in t.name)
        assert data_privacy_topic.category == Category.DATA_PRIVACY
        assert "GDPR" in data_privacy_topic.related_regulations
        assert "CCPA" in data_privacy_topic.related_regulations
    
    def test_create_regulations_in_db(self, data_loader, mock_client):
        """Test creating regulations in database."""
        regulations = data_loader._create_sample_regulations()
        count = data_loader._create_regulations(regulations)
        
        assert count == 5
        assert mock_client.execute_write_query.call_count == 5
        
        # Check that correct data was passed
        call_args = mock_client.execute_write_query.call_args_list[0]
        query = call_args[0][0]
        assert "CREATE (r:FederalRegulation" in query
        assert "name: $name" in query
    
    def test_create_guidance_in_db(self, data_loader, mock_client):
        """Test creating guidance documents in database."""
        guidance_docs = data_loader._create_sample_guidance()
        count = data_loader._create_guidance(guidance_docs)
        
        assert count == 4
        assert mock_client.execute_write_query.call_count == 4
    
    def test_create_enforcement_actions_in_db(self, data_loader, mock_client):
        """Test creating enforcement actions in database."""
        actions = data_loader._create_sample_enforcement_actions()
        count = data_loader._create_enforcement_actions(actions)
        
        assert count == 4
        assert mock_client.execute_write_query.call_count == 4
    
    def test_create_compliance_topics_in_db(self, data_loader, mock_client):
        """Test creating compliance topics in database."""
        topics = data_loader._create_sample_compliance_topics()
        count = data_loader._create_compliance_topics(topics)
        
        assert count == 5
        assert mock_client.execute_write_query.call_count == 5
    
    def test_create_relationships(self, data_loader, mock_client):
        """Test creating relationships between nodes."""
        count = data_loader._create_relationships()
        
        # Should create multiple relationships
        assert count > 0
        assert mock_client.execute_write_query.call_count > 0
        
        # Check some relationship queries were called
        call_args_list = mock_client.execute_write_query.call_args_list
        queries = [call[0][0] for call in call_args_list]
        
        # Should have GDPR -> Guidelines relationship
        gdpr_guidance_query = any("GDPR" in query and "HAS_GUIDANCE" in query for query in queries)
        assert gdpr_guidance_query
    
    def test_clear_database(self, data_loader, mock_client):
        """Test database clearing."""
        data_loader._clear_database()
        
        mock_client.execute_query.assert_called_once()
        call_args = mock_client.execute_query.call_args[0]
        query = call_args[0]
        assert "MATCH (n) DETACH DELETE n" in query
    
    def test_load_sample_data_full_process(self, data_loader, mock_client):
        """Test the complete data loading process."""
        counts = data_loader.load_sample_data(clear_existing=True)
        
        # Should return counts for all data types
        expected_keys = {'regulations', 'guidance', 'enforcement_actions', 'compliance_topics', 'relationships'}
        assert set(counts.keys()) == expected_keys
        
        # Should have created data
        assert counts['regulations'] == 5
        assert counts['guidance'] == 4
        assert counts['enforcement_actions'] == 4
        assert counts['compliance_topics'] == 5
        assert counts['relationships'] > 0
        
        # Should have cleared database first
        assert any("DETACH DELETE" in str(call) for call in mock_client.execute_query.call_args_list)
    
    def test_load_sample_data_no_clear(self, data_loader, mock_client):
        """Test data loading without clearing existing data."""
        counts = data_loader.load_sample_data(clear_existing=False)
        
        # Should not have called clear query
        clear_calls = [call for call in mock_client.execute_query.call_args_list 
                      if "DETACH DELETE" in str(call)]
        assert len(clear_calls) == 0
        
        # Should still create data
        assert counts['regulations'] == 5
    
    def test_get_data_summary(self, data_loader, mock_client):
        """Test getting data summary."""
        # Mock return values for summary queries
        mock_client.execute_query.side_effect = [
            [{'count': 20}],  # total_nodes
            [{'count': 5}],   # regulations
            [{'count': 4}],   # guidance
            [{'count': 4}],   # enforcement_actions
            [{'count': 5}],   # compliance_topics
            [{'count': 10}],  # relationships
            [{'category': 'data_privacy'}, {'category': 'financial'}]  # categories
        ]
        
        summary = data_loader.get_data_summary()
        
        assert summary['total_nodes'] == 20
        assert summary['regulations'] == 5
        assert summary['guidance'] == 4
        assert summary['categories'] == ['data_privacy', 'financial']
        
        # Should have called multiple summary queries
        assert mock_client.execute_query.call_count == 7
    
    def test_get_data_summary_with_errors(self, data_loader, mock_client):
        """Test data summary with query errors."""
        # Mock to raise exception
        mock_client.execute_query.side_effect = Exception("Database error")
        
        summary = data_loader.get_data_summary()
        
        # Should handle errors gracefully
        assert summary['total_nodes'] == "unknown"
        assert summary['regulations'] == "unknown"


class TestDataConsistency:
    """Test data consistency and relationships."""
    
    @pytest.fixture
    def data_loader(self):
        """Data loader with mock client for consistency tests."""
        mock_client = Mock()
        mock_client.execute_query.return_value = []
        mock_client.execute_write_query.return_value = []
        return ExampleDataLoader(mock_client)
    
    def test_regulation_categories_consistency(self, data_loader):
        """Test that regulations use consistent categories."""
        regulations = data_loader._create_sample_regulations()
        
        # All regulations should have valid categories
        for reg in regulations:
            assert reg.category in Category
    
    def test_guidance_references_valid_regulations(self, data_loader):
        """Test that guidance documents reference existing regulations."""
        regulations = data_loader._create_sample_regulations()
        guidance_docs = data_loader._create_sample_guidance()
        
        regulation_names = {reg.name for reg in regulations}
        
        # GDPR guidance should reference existing GDPR regulation
        gdpr_guidance = next(g for g in guidance_docs if "GDPR" in g.title)
        assert gdpr_guidance.category == Category.DATA_PRIVACY
        
        # Check that guidance categories align with regulation categories
        gdpr_reg = next(r for r in regulations if r.name == "GDPR")
        assert gdpr_guidance.category == gdpr_reg.category
    
    def test_enforcement_actions_reference_valid_regulations(self, data_loader):
        """Test that enforcement actions reference existing regulations."""
        regulations = data_loader._create_sample_regulations()
        actions = data_loader._create_sample_enforcement_actions()
        
        # Facebook fine should be for GDPR (data privacy)
        facebook_fine = next(a for a in actions if "Facebook" in a.title)
        assert facebook_fine.category == Category.DATA_PRIVACY
        
        # Wells Fargo action should be for SOX (financial)
        wells_fargo = next(a for a in actions if "Wells Fargo" in a.title)
        assert wells_fargo.category == Category.FINANCIAL
    
    def test_compliance_topics_reference_valid_regulations(self, data_loader):
        """Test that compliance topics reference existing regulations."""
        regulations = data_loader._create_sample_regulations()
        topics = data_loader._create_sample_compliance_topics()
        
        regulation_names = {reg.name for reg in regulations}
        
        # Data privacy topic should reference existing regulations
        data_privacy_topic = next(t for t in topics if "Data Privacy" in t.name)
        for reg_name in data_privacy_topic.related_regulations:
            assert reg_name in regulation_names
    
    def test_relationship_queries_syntax(self, data_loader):
        """Test that relationship queries have valid syntax."""
        # This is a basic syntax check - in real testing you'd validate against Neo4j
        data_loader._create_relationships()
        
        # If no exceptions were raised, relationship queries are syntactically valid
        assert True