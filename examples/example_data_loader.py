"""
Example data loader for populating Neo4j with sample compliance data.

This module provides functionality to load sample regulatory data into Neo4j
for testing and demonstration purposes. It creates nodes for different types
of compliance documents and establishes relationships between them.
"""

from datetime import date
from typing import List, Dict, Any
import logging

from src.models import (
    FederalRegulation,
    AgencyGuidance,
    EnforcementAction,
    ComplianceTopic,
    Category
)
from src.database.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class ExampleDataLoader:
    """
    Loads sample compliance data into Neo4j database.
    
    This class provides methods to populate the database with example
    regulations, guidance documents, enforcement actions, and compliance
    topics along with their relationships.
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize the data loader.
        
        Args:
            neo4j_client: Connected Neo4j client instance
        """
        self.client = neo4j_client
        
    def load_sample_data(self, clear_existing: bool = True) -> Dict[str, int]:
        """
        Load all sample data into the database.
        
        Args:
            clear_existing: Whether to clear existing data before loading
            
        Returns:
            Dictionary with counts of created nodes by type
        """
        logger.info("Starting sample data loading process")
        
        if clear_existing:
            self._clear_database()
            
        # Create sample data
        regulations = self._create_sample_regulations()
        guidance_docs = self._create_sample_guidance()
        enforcement_actions = self._create_sample_enforcement_actions()
        compliance_topics = self._create_sample_compliance_topics()
        
        # Load data into database
        reg_count = self._create_regulations(regulations)
        guidance_count = self._create_guidance(guidance_docs)
        enforcement_count = self._create_enforcement_actions(enforcement_actions)
        topic_count = self._create_compliance_topics(compliance_topics)
        
        # Create relationships between nodes
        relationship_count = self._create_relationships()
        
        counts = {
            'regulations': reg_count,
            'guidance': guidance_count,
            'enforcement_actions': enforcement_count,
            'compliance_topics': topic_count,
            'relationships': relationship_count
        }
        
        logger.info(f"Sample data loading completed: {counts}")
        return counts
        
    def _clear_database(self) -> None:
        """Clear all existing data from the database."""
        logger.info("Clearing existing database data")
        
        clear_query = "MATCH (n) DETACH DELETE n"
        self.client.execute_query(clear_query)
        
        logger.info("Database cleared successfully")
        
    def _create_sample_regulations(self) -> List[FederalRegulation]:
        """Create sample federal regulations."""
        return [
            FederalRegulation(
                name="GDPR",
                citation="Regulation (EU) 2016/679",
                description="General Data Protection Regulation establishing comprehensive data protection rules for the European Union",
                effective_date=date(2018, 5, 25),
                category=Category.DATA_PRIVACY,
                full_text="Article 1: This regulation lays down rules relating to the protection of natural persons..."
            ),
            FederalRegulation(
                name="CCPA",
                citation="Cal. Civ. Code § 1798.100",
                description="California Consumer Privacy Act providing privacy rights to California residents",
                effective_date=date(2020, 1, 1),
                category=Category.DATA_PRIVACY
            ),
            FederalRegulation(
                name="SOX",
                citation="Public Law 107-204",
                description="Sarbanes-Oxley Act establishing financial reporting and corporate governance standards",
                effective_date=date(2002, 7, 30),
                category=Category.FINANCIAL
            ),
            FederalRegulation(
                name="HIPAA",
                citation="45 CFR 160",
                description="Health Insurance Portability and Accountability Act protecting health information privacy",
                effective_date=date(1996, 8, 21),
                category=Category.HEALTHCARE
            ),
            FederalRegulation(
                name="Clean Air Act",
                citation="42 U.S.C. § 7401",
                description="Federal law regulating air emissions from stationary and mobile sources",
                effective_date=date(1970, 12, 31),
                category=Category.ENVIRONMENTAL
            )
        ]
        
    def _create_sample_guidance(self) -> List[AgencyGuidance]:
        """Create sample agency guidance documents."""
        return [
            AgencyGuidance(
                title="GDPR Implementation Guidelines",
                agency="European Data Protection Board",
                date_issued=date(2018, 5, 25),
                summary="Comprehensive guidelines for implementing GDPR compliance measures in organizations",
                reference_number="EDPB-2018-001",
                category=Category.DATA_PRIVACY
            ),
            AgencyGuidance(
                title="SOX Section 404 Compliance Guide",
                agency="Securities and Exchange Commission",
                date_issued=date(2003, 6, 5),
                summary="Guidance on internal control requirements under Section 404 of Sarbanes-Oxley Act",
                reference_number="SEC-2003-404",
                category=Category.FINANCIAL
            ),
            AgencyGuidance(
                title="HIPAA Security Rule Guidance",
                agency="Department of Health and Human Services",
                date_issued=date(2003, 4, 21),
                summary="Implementation guidance for HIPAA Security Rule requirements",
                reference_number="HHS-2003-SEC",
                category=Category.HEALTHCARE
            ),
            AgencyGuidance(
                title="Data Breach Notification Guidelines",
                agency="Federal Trade Commission",
                date_issued=date(2019, 7, 15),
                summary="Best practices for data breach notification and response procedures",
                reference_number="FTC-2019-DBN",
                category=Category.DATA_PRIVACY
            )
        ]
        
    def _create_sample_enforcement_actions(self) -> List[EnforcementAction]:
        """Create sample enforcement actions."""
        return [
            EnforcementAction(
                title="Facebook GDPR Fine",
                agency="Irish Data Protection Commission",
                date=date(2021, 9, 2),
                summary="Record €225 million fine for GDPR violations related to transparency and data processing",
                docket_number="DPC-2021-FB-001",
                outcome="€225 million fine imposed, compliance monitoring required",
                category=Category.DATA_PRIVACY
            ),
            EnforcementAction(
                title="Wells Fargo SOX Violation",
                agency="Securities and Exchange Commission",
                date=date(2020, 2, 21),
                summary="Enforcement action for inadequate internal controls and financial reporting deficiencies",
                docket_number="SEC-2020-WF-404",
                outcome="$3 billion civil penalty, remedial measures required",
                category=Category.FINANCIAL
            ),
            EnforcementAction(
                title="Anthem HIPAA Breach",
                agency="Department of Health and Human Services",
                date=date(2018, 10, 15),
                summary="Largest HIPAA settlement for data breach affecting 78.8 million individuals",
                docket_number="HHS-2018-ANT",
                outcome="$16 million settlement, corrective action plan",
                category=Category.HEALTHCARE
            ),
            EnforcementAction(
                title="Equifax Data Breach Settlement",
                agency="Federal Trade Commission",
                date=date(2019, 7, 22),
                summary="Settlement for massive data breach exposing personal information of 147 million consumers",
                docket_number="FTC-2019-EFX",
                outcome="$700 million settlement fund, enhanced security requirements",
                category=Category.DATA_PRIVACY
            )
        ]
        
    def _create_sample_compliance_topics(self) -> List[ComplianceTopic]:
        """Create sample compliance topics."""
        return [
            ComplianceTopic(
                name="Data Privacy and Protection",
                description="Comprehensive framework for protecting personal data and ensuring privacy rights",
                related_regulations=["GDPR", "CCPA", "HIPAA"],
                category=Category.DATA_PRIVACY
            ),
            ComplianceTopic(
                name="Financial Reporting and Governance",
                description="Requirements for accurate financial reporting and corporate governance practices",
                related_regulations=["SOX"],
                category=Category.FINANCIAL
            ),
            ComplianceTopic(
                name="Healthcare Information Security",
                description="Protecting health information and ensuring healthcare data security",
                related_regulations=["HIPAA"],
                category=Category.HEALTHCARE
            ),
            ComplianceTopic(
                name="Environmental Compliance",
                description="Meeting environmental protection and pollution control requirements",
                related_regulations=["Clean Air Act"],
                category=Category.ENVIRONMENTAL
            ),
            ComplianceTopic(
                name="Data Breach Response",
                description="Procedures and requirements for responding to data security incidents",
                related_regulations=["GDPR", "CCPA"],
                category=Category.DATA_PRIVACY
            )
        ]
        
    def _create_regulations(self, regulations: List[FederalRegulation]) -> int:
        """Create federal regulation nodes in the database."""
        create_query = """
        CREATE (r:FederalRegulation {
            name: $name,
            citation: $citation,
            description: $description,
            effective_date: $effective_date,
            category: $category,
            full_text: $full_text
        })
        """
        
        count = 0
        for regulation in regulations:
            self.client.execute_write_query(create_query, regulation.model_dump())
            count += 1
            
        logger.info(f"Created {count} federal regulation nodes")
        return count
        
    def _create_guidance(self, guidance_docs: List[AgencyGuidance]) -> int:
        """Create agency guidance nodes in the database."""
        create_query = """
        CREATE (g:AgencyGuidance {
            title: $title,
            agency: $agency,
            date_issued: $date_issued,
            summary: $summary,
            reference_number: $reference_number,
            category: $category
        })
        """
        
        count = 0
        for guidance in guidance_docs:
            self.client.execute_write_query(create_query, guidance.model_dump())
            count += 1
            
        logger.info(f"Created {count} agency guidance nodes")
        return count
        
    def _create_enforcement_actions(self, actions: List[EnforcementAction]) -> int:
        """Create enforcement action nodes in the database."""
        create_query = """
        CREATE (e:EnforcementAction {
            title: $title,
            agency: $agency,
            date: $date,
            summary: $summary,
            docket_number: $docket_number,
            outcome: $outcome,
            category: $category
        })
        """
        
        count = 0
        for action in actions:
            self.client.execute_write_query(create_query, action.model_dump())
            count += 1
            
        logger.info(f"Created {count} enforcement action nodes")
        return count
        
    def _create_compliance_topics(self, topics: List[ComplianceTopic]) -> int:
        """Create compliance topic nodes in the database."""
        create_query = """
        CREATE (t:ComplianceTopic {
            name: $name,
            description: $description,
            related_regulations: $related_regulations,
            category: $category
        })
        """
        
        count = 0
        for topic in topics:
            self.client.execute_write_query(create_query, topic.model_dump())
            count += 1
            
        logger.info(f"Created {count} compliance topic nodes")
        return count
        
    def _create_relationships(self) -> int:
        """Create relationships between nodes."""
        relationships = [
            # GDPR relationships
            {
                'query': """
                MATCH (r:FederalRegulation {name: "GDPR"})
                MATCH (g:AgencyGuidance {reference_number: "EDPB-2018-001"})
                CREATE (r)-[:HAS_GUIDANCE]->(g)
                """,
                'description': 'GDPR -> Implementation Guidelines'
            },
            {
                'query': """
                MATCH (r:FederalRegulation {name: "GDPR"})
                MATCH (e:EnforcementAction {docket_number: "DPC-2021-FB-001"})
                CREATE (r)-[:HAS_ENFORCEMENT]->(e)
                """,
                'description': 'GDPR -> Facebook Fine'
            },
            {
                'query': """
                MATCH (t:ComplianceTopic {name: "Data Privacy and Protection"})
                MATCH (r:FederalRegulation {name: "GDPR"})
                CREATE (t)-[:COVERS]->(r)
                """,
                'description': 'Data Privacy Topic -> GDPR'
            },
            
            # SOX relationships
            {
                'query': """
                MATCH (r:FederalRegulation {name: "SOX"})
                MATCH (g:AgencyGuidance {reference_number: "SEC-2003-404"})
                CREATE (r)-[:HAS_GUIDANCE]->(g)
                """,
                'description': 'SOX -> Section 404 Guidance'
            },
            {
                'query': """
                MATCH (r:FederalRegulation {name: "SOX"})
                MATCH (e:EnforcementAction {docket_number: "SEC-2020-WF-404"})
                CREATE (r)-[:HAS_ENFORCEMENT]->(e)
                """,
                'description': 'SOX -> Wells Fargo Enforcement'
            },
            
            # HIPAA relationships
            {
                'query': """
                MATCH (r:FederalRegulation {name: "HIPAA"})
                MATCH (g:AgencyGuidance {reference_number: "HHS-2003-SEC"})
                CREATE (r)-[:HAS_GUIDANCE]->(g)
                """,
                'description': 'HIPAA -> Security Rule Guidance'
            },
            {
                'query': """
                MATCH (r:FederalRegulation {name: "HIPAA"})
                MATCH (e:EnforcementAction {docket_number: "HHS-2018-ANT"})
                CREATE (r)-[:HAS_ENFORCEMENT]->(e)
                """,
                'description': 'HIPAA -> Anthem Breach'
            },
            
            # Data Privacy topic relationships
            {
                'query': """
                MATCH (t:ComplianceTopic {name: "Data Privacy and Protection"})
                MATCH (r:FederalRegulation {name: "CCPA"})
                CREATE (t)-[:COVERS]->(r)
                """,
                'description': 'Data Privacy Topic -> CCPA'
            },
            {
                'query': """
                MATCH (t:ComplianceTopic {name: "Data Breach Response"})
                MATCH (g:AgencyGuidance {reference_number: "FTC-2019-DBN"})
                CREATE (t)-[:PROVIDES_GUIDANCE]->(g)
                """,
                'description': 'Data Breach Topic -> FTC Guidance'
            },
            
            # Cross-category relationships
            {
                'query': """
                MATCH (g:AgencyGuidance {reference_number: "FTC-2019-DBN"})
                MATCH (e:EnforcementAction {docket_number: "FTC-2019-EFX"})
                CREATE (g)-[:RELATES_TO]->(e)
                """,
                'description': 'Breach Guidance -> Equifax Settlement'
            }
        ]
        
        count = 0
        for rel in relationships:
            try:
                self.client.execute_write_query(rel['query'])
                count += 1
                logger.debug(f"Created relationship: {rel['description']}")
            except Exception as e:
                logger.warning(f"Failed to create relationship {rel['description']}: {e}")
                
        logger.info(f"Created {count} relationships")
        return count
        
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        summary_queries = {
            'total_nodes': "MATCH (n) RETURN count(n) as count",
            'regulations': "MATCH (r:FederalRegulation) RETURN count(r) as count",
            'guidance': "MATCH (g:AgencyGuidance) RETURN count(g) as count",
            'enforcement_actions': "MATCH (e:EnforcementAction) RETURN count(e) as count",
            'compliance_topics': "MATCH (t:ComplianceTopic) RETURN count(t) as count",
            'relationships': "MATCH ()-[r]->() RETURN count(r) as count",
            'categories': "MATCH (n) WHERE exists(n.category) RETURN DISTINCT n.category as category ORDER BY category"
        }
        
        summary = {}
        for key, query in summary_queries.items():
            try:
                result = self.client.execute_query(query)
                if key == 'categories':
                    summary[key] = [r['category'] for r in result]
                else:
                    summary[key] = result[0]['count'] if result else 0
            except Exception as e:
                logger.warning(f"Failed to get {key} summary: {e}")
                summary[key] = "unknown"
                
        return summary


def main():
    """Example usage of the data loader."""
    from src.config import load_config
    
    # Load configuration
    config = load_config()
    
    # Create Neo4j client
    client = Neo4jClient(
        uri=config.neo4j.uri,
        user=config.neo4j.user,
        password=config.neo4j.password
    )
    
    try:
        # Verify connection
        if not client.verify_connectivity():
            print("Failed to connect to Neo4j database")
            return
            
        # Load sample data
        loader = ExampleDataLoader(client)
        counts = loader.load_sample_data()
        
        print("Sample data loaded successfully!")
        print(f"Created: {counts}")
        
        # Get data summary
        summary = loader.get_data_summary()
        print(f"\nData summary: {summary}")
        
    finally:
        client.close()


if __name__ == "__main__":
    main()