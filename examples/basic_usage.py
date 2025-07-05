"""
Basic usage example for GraphRAG Retrieval System.

This example demonstrates the core functionality of the retrieval system
including initialization, data loading, and performing queries.
"""

import logging
from src.retrieval import RetrievalModule
from src.config import load_config
from src.database.neo4j_client import Neo4jClient
from examples.example_data_loader import ExampleDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate basic usage of the GraphRAG retrieval system."""
    
    print("=== GraphRAG Retrieval System - Basic Usage Example ===\n")
    
    # Step 1: Load configuration
    print("1. Loading configuration...")
    config = load_config()
    print(f"   Neo4j URI: {config.neo4j.uri}")
    print(f"   Default limit: {config.retrieval.default_limit}")
    print()
    
    # Step 2: Initialize retrieval module
    print("2. Initializing retrieval module...")
    try:
        module = RetrievalModule('graphrag')
        module.initialize(config.to_dict())
        print("   ✓ Module initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize module: {e}")
        print("   Make sure Neo4j is running on bolt://localhost:7687")
        return
    print()
    
    # Step 3: Load sample data (optional - check if data exists first)
    print("3. Checking for existing data...")
    client = Neo4jClient(
        uri=config.neo4j.uri,
        user=config.neo4j.user,
        password=config.neo4j.password
    )
    
    try:
        # Check if we have any data
        result = client.execute_query("MATCH (n) RETURN count(n) as count")
        node_count = result[0]['count'] if result else 0
        
        if node_count == 0:
            print("   No data found. Loading sample data...")
            loader = ExampleDataLoader(client)
            counts = loader.load_sample_data()
            print(f"   ✓ Loaded sample data: {counts}")
        else:
            print(f"   ✓ Found {node_count} existing nodes")
    except Exception as e:
        print(f"   ✗ Error checking/loading data: {e}")
        return
    finally:
        client.close()
    print()
    
    # Step 4: Perform sample queries
    print("4. Performing sample queries...\n")
    
    # Query 1: Data privacy regulations
    print("Query 1: 'GDPR data protection requirements'")
    print("-" * 50)
    results = module.retrieve(
        query_text="GDPR data protection requirements",
        filters={'category': 'data_privacy'},
        limit=3
    )
    
    if results:
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Score: {result.score:.3f}")
            print(f"  Source: {result.source_type}")
            print(f"  Content: {result.content[:150]}...")
            if result.metadata.get('citation'):
                print(f"  Citation: {result.metadata['citation']}")
            print()
    else:
        print("No results found for this query.\n")
    
    # Query 2: Financial regulations
    print("Query 2: 'financial reporting requirements'")
    print("-" * 50)
    results = module.retrieve(
        query_text="financial reporting requirements",
        filters={'category': 'financial'},
        limit=3
    )
    
    if results:
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Score: {result.score:.3f}")
            print(f"  Source: {result.source_type}")
            print(f"  Content: {result.content[:150]}...")
            print()
    else:
        print("No results found for this query.\n")
    
    # Query 3: General compliance search (no filters)
    print("Query 3: 'compliance enforcement penalties'")
    print("-" * 50)
    results = module.retrieve(
        query_text="compliance enforcement penalties",
        limit=5
    )
    
    if results:
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Score: {result.score:.3f}")
            print(f"  Source: {result.source_type}")
            print(f"  Category: {result.metadata.get('category', 'unknown')}")
            print(f"  Content: {result.content[:100]}...")
            print()
    else:
        print("No results found for this query.\n")
    
    print("=== Example completed successfully! ===")


def demonstrate_configuration():
    """Demonstrate different configuration options."""
    
    print("\n=== Configuration Examples ===\n")
    
    # Example 1: Environment variable configuration
    print("1. Environment Variable Configuration:")
    print("   Set these environment variables:")
    print("   export NEO4J_URI='bolt://localhost:7687'")
    print("   export NEO4J_USER='neo4j'")
    print("   export NEO4J_PASSWORD='your-password'")
    print("   export RETRIEVAL_DEFAULT_LIMIT=15")
    print()
    
    # Example 2: Programmatic configuration
    print("2. Programmatic Configuration:")
    print("""
from src.config import AppConfig

config = AppConfig()
config.neo4j.uri = "bolt://production:7687"
config.neo4j.user = "production_user"
config.neo4j.password = "secure_password"
config.retrieval.default_limit = 20
config.retrieval.score_threshold = 0.8

# Use with retrieval module
module = RetrievalModule('graphrag')
module.initialize(config.to_dict())
""")
    
    # Example 3: Configuration validation
    print("3. Configuration Validation:")
    config = load_config()
    is_valid = config.validate_config()
    print(f"   Current configuration is valid: {is_valid}")
    print(f"   Configuration summary: {config}")


def demonstrate_advanced_features():
    """Demonstrate advanced features of the retrieval system."""
    
    print("\n=== Advanced Features ===\n")
    
    # Load configuration
    config = load_config()
    
    try:
        module = RetrievalModule('graphrag')
        module.initialize(config.to_dict())
    except Exception as e:
        print(f"Cannot demonstrate advanced features: {e}")
        return
    
    # Feature 1: Score threshold filtering
    print("1. Score Threshold Filtering:")
    results = module.retrieve(
        query_text="data privacy",
        limit=10
    )
    
    high_score_results = [r for r in results if r.score > 0.5]
    print(f"   Total results: {len(results)}")
    print(f"   High-score results (>0.5): {len(high_score_results)}")
    print()
    
    # Feature 2: Category-based filtering
    print("2. Category-based Filtering:")
    categories = ['data_privacy', 'financial', 'healthcare']
    
    for category in categories:
        results = module.retrieve(
            query_text="compliance requirements",
            filters={'category': category},
            limit=5
        )
        print(f"   {category.replace('_', ' ').title()}: {len(results)} results")
    print()
    
    # Feature 3: Metadata analysis
    print("3. Metadata Analysis:")
    results = module.retrieve(
        query_text="regulations",
        limit=10
    )
    
    if results:
        source_types = {}
        for result in results:
            source_type = result.source_type
            source_types[source_type] = source_types.get(source_type, 0) + 1
        
        print("   Source type distribution:")
        for source_type, count in source_types.items():
            print(f"     {source_type}: {count}")
    print()


if __name__ == "__main__":
    try:
        main()
        demonstrate_configuration()
        demonstrate_advanced_features()
    except KeyboardInterrupt:
        print("\nExample interrupted by user.")
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        print(f"\nExample failed: {e}")
        print("Please check that Neo4j is running and accessible.")