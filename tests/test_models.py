"""
Tests for data models module.
"""

import pytest
from datetime import date
from pydantic import ValidationError
from src.models import (
    Category,
    FederalRegulation,
    AgencyGuidance,
    EnforcementAction,
    ComplianceTopic
)


class TestCategory:
    """Test Category enum."""
    
    def test_category_values(self):
        """Test all category enum values."""
        assert Category.FINANCIAL == "financial"
        assert Category.HEALTHCARE == "healthcare"
        assert Category.ENVIRONMENTAL == "environmental"
        assert Category.DATA_PRIVACY == "data_privacy"
        assert Category.OTHER == "other"
    
    def test_category_iteration(self):
        """Test iterating over category values."""
        categories = list(Category)
        assert len(categories) == 5
        assert Category.DATA_PRIVACY in categories


class TestFederalRegulation:
    """Test FederalRegulation model."""
    
    def test_valid_creation(self):
        """Test creating a valid federal regulation."""
        regulation = FederalRegulation(
            name="GDPR",
            citation="Regulation (EU) 2016/679",
            description="General Data Protection Regulation",
            effective_date=date(2018, 5, 25),
            category=Category.DATA_PRIVACY
        )
        
        assert regulation.name == "GDPR"
        assert regulation.citation == "Regulation (EU) 2016/679"
        assert regulation.description == "General Data Protection Regulation"
        assert regulation.effective_date == date(2018, 5, 25)
        assert regulation.category == Category.DATA_PRIVACY
        assert regulation.full_text is None  # Optional field
    
    def test_with_full_text(self):
        """Test creating regulation with full text."""
        full_text = "Article 1: This regulation applies to..."
        regulation = FederalRegulation(
            name="Test Regulation",
            citation="Test 123",
            description="Test description",
            effective_date=date(2020, 1, 1),
            category=Category.FINANCIAL,
            full_text=full_text
        )
        
        assert regulation.full_text == full_text
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        with pytest.raises(ValidationError):
            FederalRegulation(
                # Missing name
                citation="Test 123",
                description="Test description",
                effective_date=date(2020, 1, 1),
                category=Category.FINANCIAL
            )
    
    def test_invalid_category(self):
        """Test validation with invalid category."""
        with pytest.raises(ValidationError):
            FederalRegulation(
                name="Test",
                citation="Test 123",
                description="Test description",
                effective_date=date(2020, 1, 1),
                category="invalid_category"  # Should be Category enum
            )
    
    def test_serialization(self):
        """Test model serialization."""
        regulation = FederalRegulation(
            name="Test",
            citation="Test 123",
            description="Test description",
            effective_date=date(2020, 1, 1),
            category=Category.FINANCIAL
        )
        
        data = regulation.model_dump()
        assert data["name"] == "Test"
        assert data["category"] == "financial"
        assert data["effective_date"] == date(2020, 1, 1)


class TestAgencyGuidance:
    """Test AgencyGuidance model."""
    
    def test_valid_creation(self):
        """Test creating valid agency guidance."""
        guidance = AgencyGuidance(
            title="Data Protection Guidelines",
            agency="Data Protection Authority",
            date_issued=date(2021, 3, 15),
            summary="Guidelines for implementing data protection measures",
            reference_number="DPA-2021-001",
            category=Category.DATA_PRIVACY
        )
        
        assert guidance.title == "Data Protection Guidelines"
        assert guidance.agency == "Data Protection Authority"
        assert guidance.date_issued == date(2021, 3, 15)
        assert guidance.summary == "Guidelines for implementing data protection measures"
        assert guidance.reference_number == "DPA-2021-001"
        assert guidance.category == Category.DATA_PRIVACY
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        with pytest.raises(ValidationError):
            AgencyGuidance(
                title="Test Guidelines",
                # Missing agency
                date_issued=date(2021, 1, 1),
                summary="Test summary",
                reference_number="TEST-001",
                category=Category.OTHER
            )


class TestEnforcementAction:
    """Test EnforcementAction model."""
    
    def test_valid_creation(self):
        """Test creating valid enforcement action."""
        action = EnforcementAction(
            title="Privacy Violation Penalty",
            agency="Federal Trade Commission",
            date=date(2022, 6, 10),
            summary="Company fined for privacy violations",
            docket_number="FTC-2022-0123",
            outcome="$500,000 fine imposed",
            category=Category.DATA_PRIVACY
        )
        
        assert action.title == "Privacy Violation Penalty"
        assert action.agency == "Federal Trade Commission"
        assert action.date == date(2022, 6, 10)
        assert action.summary == "Company fined for privacy violations"
        assert action.docket_number == "FTC-2022-0123"
        assert action.outcome == "$500,000 fine imposed"
        assert action.category == Category.DATA_PRIVACY
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        with pytest.raises(ValidationError):
            EnforcementAction(
                title="Test Action",
                agency="Test Agency",
                date=date(2022, 1, 1),
                summary="Test summary",
                # Missing docket_number
                outcome="Test outcome",
                category=Category.OTHER
            )


class TestComplianceTopic:
    """Test ComplianceTopic model."""
    
    def test_valid_creation(self):
        """Test creating valid compliance topic."""
        topic = ComplianceTopic(
            name="Data Privacy Compliance",
            description="Requirements and best practices for data privacy",
            related_regulations=["GDPR", "CCPA", "PIPEDA"],
            category=Category.DATA_PRIVACY
        )
        
        assert topic.name == "Data Privacy Compliance"
        assert topic.description == "Requirements and best practices for data privacy"
        assert topic.related_regulations == ["GDPR", "CCPA", "PIPEDA"]
        assert topic.category == Category.DATA_PRIVACY
    
    def test_empty_related_regulations(self):
        """Test topic with empty related regulations list."""
        topic = ComplianceTopic(
            name="General Compliance",
            description="General compliance requirements",
            related_regulations=[],
            category=Category.OTHER
        )
        
        assert topic.related_regulations == []
    
    def test_invalid_related_regulations_type(self):
        """Test validation with invalid related regulations type."""
        with pytest.raises(ValidationError):
            ComplianceTopic(
                name="Test Topic",
                description="Test description",
                related_regulations="not a list",  # Should be list
                category=Category.OTHER
            )


class TestModelInteroperability:
    """Test model interoperability and integration."""
    
    def test_all_models_use_same_categories(self):
        """Test that all models use the same category enum."""
        # Create instances of each model with the same category
        category = Category.HEALTHCARE
        
        regulation = FederalRegulation(
            name="HIPAA",
            citation="45 CFR 160",
            description="Health Insurance Portability Act",
            effective_date=date(1996, 8, 21),
            category=category
        )
        
        guidance = AgencyGuidance(
            title="HIPAA Compliance Guide",
            agency="HHS",
            date_issued=date(2020, 1, 1),
            summary="Guide for HIPAA compliance",
            reference_number="HHS-2020-001",
            category=category
        )
        
        action = EnforcementAction(
            title="HIPAA Violation",
            agency="HHS",
            date=date(2021, 1, 1),
            summary="Healthcare provider fined",
            docket_number="HHS-2021-001",
            outcome="Fine imposed",
            category=category
        )
        
        topic = ComplianceTopic(
            name="Healthcare Privacy",
            description="Healthcare privacy requirements",
            related_regulations=["HIPAA"],
            category=category
        )
        
        # All should have the same category
        assert regulation.category == guidance.category == action.category == topic.category
        assert regulation.category == Category.HEALTHCARE
    
    def test_model_serialization_consistency(self):
        """Test that all models serialize categories consistently."""
        models = [
            FederalRegulation(
                name="Test", citation="Test", description="Test",
                effective_date=date(2020, 1, 1), category=Category.FINANCIAL
            ),
            AgencyGuidance(
                title="Test", agency="Test", date_issued=date(2020, 1, 1),
                summary="Test", reference_number="Test", category=Category.FINANCIAL
            ),
            EnforcementAction(
                title="Test", agency="Test", date=date(2020, 1, 1),
                summary="Test", docket_number="Test", outcome="Test",
                category=Category.FINANCIAL
            ),
            ComplianceTopic(
                name="Test", description="Test", related_regulations=[],
                category=Category.FINANCIAL
            )
        ]
        
        for model in models:
            data = model.model_dump()
            assert data["category"] == "financial"