"""
Demo of GraphRAG Retrieval System without requiring Neo4j database.

This demo shows how the system works using mocked data, so you can see
the functionality without setting up a database.
"""

import logging
from unittest.mock import Mock
from datetime import date
from typing import List, Dict, Any

from src.retrieval import RetrievalModule, RetrievalQuery, RetrievalResult
from src.retrieval.graph_rag import GraphRAGRetriever
from src.models import FederalRegulation, AgencyGuidance, EnforcementAction, Category
from src.config import AppConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MockNeo4jClient:
    """Mock Neo4j client that simulates database responses."""
    
    def __init__(self, *args, **kwargs):
        # Sample data that would normally be in Neo4j
        self.sample_data = self._create_sample_data()
        
    def verify_connectivity(self):
        return True
    
    def execute_query(self, query: str, parameters: Dict = None):
        """Simulate Neo4j query execution with sample data."""
        if parameters is None:
            parameters = {}
            
        search_text = parameters.get('search_text', '').lower()
        category_filter = parameters.get('category')
        
        # Filter sample data based on query
        results = []
        for item in self.sample_data:
            # Check if search text matches
            item_text = ' '.join([
                item.get('name', ''),
                item.get('title', ''),
                item.get('description', ''),
                item.get('summary', '')
            ]).lower()
            
            if search_text in item_text:
                # Apply category filter if specified
                if category_filter is None or item.get('category') == category_filter:
                    results.append({
                        'n': item,
                        'node_type': [item['node_type']],
                        'node_id': item['id']
                    })
        
        return results[:20]  # Limit results
    
    def execute_write_query(self, query: str, parameters: Dict = None):
        """Mock write query - just return empty result."""
        return []
    
    def close(self):
        pass
        
    def _create_sample_data(self):
        """Create sample compliance data."""
        return [
            # Federal Regulations
            {
                'id': 1,
                'node_type': 'FederalRegulation',
                'name': 'GDPR',
                'description': 'General Data Protection Regulation for privacy and data protection in the European Union',
                'citation': 'Regulation (EU) 2016/679',
                'category': 'data_privacy',
                'effective_date': '2018-05-25'
            },
            {
                'id': 2,
                'node_type': 'FederalRegulation',
                'name': 'CCPA',
                'description': 'California Consumer Privacy Act providing privacy rights to California residents',
                'citation': 'Cal. Civ. Code § 1798.100',
                'category': 'data_privacy',
                'effective_date': '2020-01-01'
            },
            {
                'id': 3,
                'node_type': 'FederalRegulation',
                'name': 'SOX',
                'description': 'Sarbanes-Oxley Act establishing financial reporting and corporate governance standards',
                'citation': 'Public Law 107-204',
                'category': 'financial',
                'effective_date': '2002-07-30'
            },
            {
                'id': 4,
                'node_type': 'FederalRegulation',
                'name': 'HIPAA',
                'description': 'Health Insurance Portability and Accountability Act protecting health information privacy',
                'citation': '45 CFR 160',
                'category': 'healthcare',
                'effective_date': '1996-08-21'
            },
            # Agency Guidance
            {
                'id': 5,
                'node_type': 'AgencyGuidance',
                'title': 'GDPR Implementation Guidelines',
                'summary': 'Comprehensive guidelines for implementing GDPR compliance measures in organizations',
                'agency': 'European Data Protection Board',
                'category': 'data_privacy',
                'reference_number': 'EDPB-2018-001'
            },
            {
                'id': 6,
                'node_type': 'AgencyGuidance',
                'title': 'SOX Section 404 Compliance Guide',
                'summary': 'Guidance on internal control requirements under Section 404 of Sarbanes-Oxley Act',
                'agency': 'Securities and Exchange Commission',
                'category': 'financial',
                'reference_number': 'SEC-2003-404'
            },
            # Enforcement Actions
            {
                'id': 7,
                'node_type': 'EnforcementAction',
                'title': 'Facebook GDPR Fine',
                'summary': 'Record €225 million fine for GDPR violations related to transparency and data processing',
                'agency': 'Irish Data Protection Commission',
                'category': 'data_privacy',
                'outcome': '€225 million fine imposed, compliance monitoring required'
            },
            {
                'id': 8,
                'node_type': 'EnforcementAction',
                'title': 'Wells Fargo SOX Violation',
                'summary': 'Enforcement action for inadequate internal controls and financial reporting deficiencies',
                'agency': 'Securities and Exchange Commission',
                'category': 'financial',
                'outcome': '$3 billion civil penalty, remedial measures required'
            }
        ]

def demonstrate_basic_functionality():
    """Demonstrate basic retrieval functionality."""
    
    print("=" * 60)
    print("🔍 GraphRAG Retrieval System - Live Demo")
    print("=" * 60)
    print()
    
    # Patch the Neo4j client to use our mock
    import src.retrieval.graph_rag
    original_client = src.retrieval.graph_rag.Neo4jClient
    src.retrieval.graph_rag.Neo4jClient = MockNeo4jClient
    
    try:
        # 1. Configuration
        print("1️⃣  Setting up configuration...")
        config = AppConfig()
        print(f"   ✓ Neo4j URI: {config.neo4j.uri}")
        print(f"   ✓ Default limit: {config.retrieval.default_limit}")
        print()
        
        # 2. Initialize retrieval module
        print("2️⃣  Initializing retrieval module...")
        module = RetrievalModule('graphrag')
        module.initialize(config.to_dict())
        print("   ✓ GraphRAG retriever initialized")
        print("   ✓ Mock database connected")
        print()
        
        # 3. Demonstrate different types of queries
        queries = [
            {
                'name': 'Data Privacy Query',
                'text': 'GDPR data protection requirements',
                'filters': {'category': 'data_privacy'},
                'limit': 3
            },
            {
                'name': 'Financial Compliance Query',
                'text': 'financial reporting standards',
                'filters': {'category': 'financial'},
                'limit': 3
            },
            {
                'name': 'General Enforcement Query',
                'text': 'violation penalties fines',
                'filters': None,
                'limit': 5
            },
            {
                'name': 'Healthcare Privacy Query',
                'text': 'healthcare privacy protection',
                'filters': {'category': 'healthcare'},
                'limit': 3
            }
        ]
        
        for i, query_info in enumerate(queries, 1):
            print(f"3️⃣.{i} {query_info['name']}")
            print(f"    Query: '{query_info['text']}'")
            if query_info['filters']:
                print(f"    Filters: {query_info['filters']}")
            print("    " + "-" * 50)
            
            results = module.retrieve(
                query_text=query_info['text'],
                filters=query_info['filters'],
                limit=query_info['limit']
            )
            
            if results:
                for j, result in enumerate(results, 1):
                    print(f"    Result {j}:")
                    print(f"      📊 Score: {result.score:.3f}")
                    print(f"      📂 Source: {result.source_type}")
                    print(f"      🏷️  Category: {result.metadata.get('category', 'N/A')}")
                    
                    # Extract title/name from content
                    content_lines = result.content.split(' | ')
                    title_line = content_lines[0] if content_lines else "Unknown"
                    print(f"      📄 Title: {title_line}")
                    
                    if result.metadata.get('citation'):
                        print(f"      📚 Citation: {result.metadata['citation']}")
                    if result.metadata.get('agency'):
                        print(f"      🏛️  Agency: {result.metadata['agency']}")
                    
                    print()
            else:
                print("    ❌ No results found")
                print()
        
        # 4. Show system capabilities
        print("4️⃣  System Capabilities Demonstrated:")
        print("   ✅ Text-based search across multiple document types")
        print("   ✅ Category-based filtering (data_privacy, financial, healthcare)")
        print("   ✅ Relevance scoring and ranking")
        print("   ✅ Rich metadata extraction (citations, agencies, outcomes)")
        print("   ✅ Structured result format with consistent API")
        print("   ✅ Configurable result limits and thresholds")
        print()
        
        # 5. Show data model examples
        print("5️⃣  Data Model Examples:")
        print()
        
        # Create sample objects to show the data models
        sample_regulation = FederalRegulation(
            name="Sample Regulation",
            citation="Example 123",
            description="Sample regulatory text",
            effective_date=date(2024, 1, 1),
            category=Category.DATA_PRIVACY
        )
        
        sample_guidance = AgencyGuidance(
            title="Sample Guidance",
            agency="Sample Agency",
            date_issued=date(2024, 1, 1),
            summary="Sample guidance summary",
            reference_number="SAMPLE-001",
            category=Category.DATA_PRIVACY
        )
        
        print("   📋 FederalRegulation Model:")
        print(f"      {sample_regulation}")
        print()
        
        print("   📋 AgencyGuidance Model:")
        print(f"      {sample_guidance}")
        print()
        
        print("6️⃣  Next Steps to Run with Real Database:")
        print("   1. Install Docker and run: docker-compose up -d")
        print("   2. Load sample data: python examples/example_data_loader.py")
        print("   3. Run full examples: python examples/basic_usage.py")
        print("   4. Try advanced features: python examples/advanced_usage.py")
        print()
        
    finally:
        # Restore original client
        src.retrieval.graph_rag.Neo4jClient = original_client
    
    print("=" * 60)
    print("✅ Demo completed successfully!")
    print("=" * 60)

def show_configuration_examples():
    """Show different configuration options."""
    
    print("\n" + "=" * 60)
    print("⚙️  Configuration Examples")
    print("=" * 60)
    print()
    
    # Example 1: Default configuration
    print("1️⃣  Default Configuration:")
    config = AppConfig()
    print(f"   Neo4j URI: {config.neo4j.uri}")
    print(f"   Default limit: {config.retrieval.default_limit}")
    print(f"   Score threshold: {config.retrieval.score_threshold}")
    print(f"   Expand hops: {config.retrieval.expand_hops}")
    print()
    
    # Example 2: Environment variable configuration
    print("2️⃣  Environment Variable Configuration:")
    print("   Set these environment variables:")
    print("   export NEO4J_URI='bolt://localhost:7687'")
    print("   export NEO4J_USER='neo4j'")
    print("   export NEO4J_PASSWORD='your-password'")
    print("   export RETRIEVAL_DEFAULT_LIMIT=15")
    print("   export RETRIEVAL_SCORE_THRESHOLD=0.8")
    print()
    
    # Example 3: Programmatic configuration
    print("3️⃣  Programmatic Configuration:")
    print("""
   from src.config import AppConfig
   
   config = AppConfig()
   config.neo4j.uri = "bolt://production:7687"
   config.neo4j.user = "prod_user"
   config.retrieval.default_limit = 20
   config.retrieval.score_threshold = 0.8
   
   # Use with retrieval module
   module = RetrievalModule('graphrag')
   module.initialize(config.to_dict())
   """)

def show_api_examples():
    """Show API usage examples."""
    
    print("\n" + "=" * 60)
    print("🔧 API Usage Examples")
    print("=" * 60)
    print()
    
    print("1️⃣  Basic Usage Pattern:")
    print("""
   from src.retrieval import RetrievalModule
   from src.config import load_config
   
   # Initialize
   config = load_config()
   module = RetrievalModule('graphrag')
   module.initialize(config.to_dict())
   
   # Query
   results = module.retrieve(
       query_text="compliance requirements",
       filters={'category': 'data_privacy'},
       limit=10
   )
   
   # Process results
   for result in results:
       print(f"Score: {result.score}")
       print(f"Content: {result.content}")
       print(f"Source: {result.source_type}")
   """)
   
    print("2️⃣  Advanced Usage Pattern:")
    print("""
   # Custom configuration
   config = {
       'neo4j_uri': 'bolt://localhost:7687',
       'neo4j_user': 'neo4j',
       'neo4j_password': 'password123',
       'retrieval_config': {
           'default_limit': 15,
           'score_threshold': 0.8
       }
   }
   
   # Batch processing
   queries = [
       "GDPR compliance",
       "financial reporting",
       "healthcare privacy"
   ]
   
   all_results = []
   for query in queries:
       results = module.retrieve(query, limit=5)
       all_results.extend(results)
   """)

if __name__ == "__main__":
    try:
        demonstrate_basic_functionality()
        show_configuration_examples()
        show_api_examples()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user.")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        print("This is a mock demo - for full functionality, set up Neo4j database.")