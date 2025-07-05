"""
Advanced usage examples for GraphRAG Retrieval System.

This example demonstrates advanced features including custom retrievers,
batch processing, performance monitoring, and integration patterns.
"""

import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager

from src.retrieval import RetrievalModule, RetrievalQuery, RetrievalResult
from src.retrieval.base import BaseRetriever
from src.config import load_config
from src.database.neo4j_client import Neo4jClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryBenchmark:
    """Benchmark results for a query."""
    query: str
    execution_time: float
    result_count: int
    avg_score: float
    top_score: float


class PerformanceMonitor:
    """Monitor and analyze retrieval performance."""
    
    def __init__(self):
        self.benchmarks: List[QueryBenchmark] = []
    
    @contextmanager
    def measure_query(self, query: str):
        """Context manager to measure query performance."""
        start_time = time.time()
        results = None
        
        def set_results(query_results):
            nonlocal results
            results = query_results
        
        yield set_results
        
        execution_time = time.time() - start_time
        
        if results:
            avg_score = sum(r.score for r in results) / len(results)
            top_score = max(r.score for r in results) if results else 0.0
        else:
            avg_score = 0.0
            top_score = 0.0
        
        benchmark = QueryBenchmark(
            query=query,
            execution_time=execution_time,
            result_count=len(results) if results else 0,
            avg_score=avg_score,
            top_score=top_score
        )
        
        self.benchmarks.append(benchmark)
        logger.info(f"Query '{query[:50]}...' took {execution_time:.3f}s, {benchmark.result_count} results")
    
    def print_summary(self):
        """Print performance summary."""
        if not self.benchmarks:
            print("No benchmarks recorded.")
            return
        
        print("\n=== Performance Summary ===")
        print(f"Total queries: {len(self.benchmarks)}")
        
        avg_time = sum(b.execution_time for b in self.benchmarks) / len(self.benchmarks)
        print(f"Average execution time: {avg_time:.3f}s")
        
        total_results = sum(b.result_count for b in self.benchmarks)
        print(f"Total results returned: {total_results}")
        
        if total_results > 0:
            avg_score = sum(b.avg_score * b.result_count for b in self.benchmarks) / total_results
            print(f"Average result score: {avg_score:.3f}")
        
        print("\nSlowest queries:")
        sorted_benchmarks = sorted(self.benchmarks, key=lambda x: x.execution_time, reverse=True)
        for i, benchmark in enumerate(sorted_benchmarks[:3], 1):
            print(f"{i}. '{benchmark.query[:50]}...' - {benchmark.execution_time:.3f}s")


class CustomScoreRetriever(BaseRetriever):
    """Custom retriever that applies additional scoring logic."""
    
    def __init__(self, base_retriever: BaseRetriever):
        self.base_retriever = base_retriever
        self._initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the base retriever."""
        self.base_retriever.initialize(config)
        self._initialized = True
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve results with custom scoring."""
        if not self._initialized:
            raise RuntimeError("Retriever not initialized")
        
        # Get base results
        results = self.base_retriever.retrieve(query)
        
        # Apply custom scoring
        enhanced_results = []
        for result in results:
            # Boost scores for recent documents
            custom_score = result.score
            
            # Boost enforcement actions
            if result.source_type == "EnforcementAction":
                custom_score *= 1.2
            
            # Boost exact matches in content
            query_terms = set(query.query_text.lower().split())
            content_terms = set(result.content.lower().split())
            exact_matches = len(query_terms.intersection(content_terms))
            if exact_matches > 0:
                custom_score *= (1.0 + exact_matches * 0.1)
            
            # Create new result with updated score
            enhanced_result = RetrievalResult(
                content=result.content,
                metadata={**result.metadata, "original_score": result.score},
                score=custom_score,
                source_type=result.source_type
            )
            enhanced_results.append(enhanced_result)
        
        # Re-sort by new scores
        return sorted(enhanced_results, key=lambda x: x.score, reverse=True)


def demonstrate_custom_retriever():
    """Demonstrate custom retriever implementation."""
    
    print("=== Custom Retriever Example ===\n")
    
    config = load_config()
    
    # Initialize standard retriever
    standard_module = RetrievalModule('graphrag')
    try:
        standard_module.initialize(config.to_dict())
    except Exception as e:
        print(f"Cannot initialize retriever: {e}")
        return
    
    # Create custom retriever
    custom_retriever = CustomScoreRetriever(standard_module.retriever)
    custom_retriever.initialize(config.to_dict())
    
    # Compare results
    query = RetrievalQuery("enforcement action penalties", limit=5)
    
    print("Standard retriever results:")
    standard_results = standard_module.retriever.retrieve(query)
    for i, result in enumerate(standard_results[:3], 1):
        print(f"{i}. Score: {result.score:.3f} - {result.source_type}")
    
    print("\nCustom retriever results:")
    custom_results = custom_retriever.retrieve(query)
    for i, result in enumerate(custom_results[:3], 1):
        original_score = result.metadata.get("original_score", "N/A")
        print(f"{i}. Score: {result.score:.3f} (was {original_score}) - {result.source_type}")
    
    print()


def demonstrate_batch_processing():
    """Demonstrate batch processing of multiple queries."""
    
    print("=== Batch Processing Example ===\n")
    
    config = load_config()
    module = RetrievalModule('graphrag')
    
    try:
        module.initialize(config.to_dict())
    except Exception as e:
        print(f"Cannot initialize retriever: {e}")
        return
    
    # Define batch of queries
    queries = [
        {"text": "GDPR data protection", "category": "data_privacy"},
        {"text": "financial reporting requirements", "category": "financial"},
        {"text": "healthcare privacy violations", "category": "healthcare"},
        {"text": "environmental compliance penalties", "category": "environmental"},
        {"text": "enforcement action fines", "category": None},
    ]
    
    print("Processing batch of queries...")
    
    all_results = []
    monitor = PerformanceMonitor()
    
    for i, query_info in enumerate(queries, 1):
        query_text = query_info["text"]
        category = query_info["category"]
        
        with monitor.measure_query(query_text) as set_results:
            filters = {"category": category} if category else None
            results = module.retrieve(
                query_text=query_text,
                filters=filters,
                limit=3
            )
            set_results(results)
            all_results.extend(results)
        
        print(f"Query {i}: '{query_text}' -> {len(results)} results")
        if results:
            top_result = results[0]
            print(f"  Top result: {top_result.source_type} (score: {top_result.score:.3f})")
        print()
    
    # Print summary
    print(f"Batch processing complete. Total results: {len(all_results)}")
    monitor.print_summary()


def demonstrate_result_analysis():
    """Demonstrate analysis of retrieval results."""
    
    print("\n=== Result Analysis Example ===\n")
    
    config = load_config()
    module = RetrievalModule('graphrag')
    
    try:
        module.initialize(config.to_dict())
    except Exception as e:
        print(f"Cannot initialize retriever: {e}")
        return
    
    # Perform comprehensive search
    results = module.retrieve(
        query_text="compliance",
        limit=20
    )
    
    if not results:
        print("No results found for analysis.")
        return
    
    # Analyze source types
    source_distribution = {}
    for result in results:
        source_type = result.source_type
        source_distribution[source_type] = source_distribution.get(source_type, 0) + 1
    
    print("Source Type Distribution:")
    for source_type, count in sorted(source_distribution.items()):
        percentage = (count / len(results)) * 100
        print(f"  {source_type}: {count} ({percentage:.1f}%)")
    print()
    
    # Analyze score distribution
    scores = [result.score for result in results]
    print("Score Statistics:")
    print(f"  Minimum: {min(scores):.3f}")
    print(f"  Maximum: {max(scores):.3f}")
    print(f"  Average: {sum(scores) / len(scores):.3f}")
    print(f"  Median: {sorted(scores)[len(scores) // 2]:.3f}")
    print()
    
    # Analyze categories
    categories = {}
    for result in results:
        category = result.metadata.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
    
    print("Category Distribution:")
    for category, count in sorted(categories.items()):
        percentage = (count / len(results)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    print()
    
    # Find highest scoring results by type
    print("Highest Scoring Results by Type:")
    type_best = {}
    for result in results:
        source_type = result.source_type
        if source_type not in type_best or result.score > type_best[source_type].score:
            type_best[source_type] = result
    
    for source_type, result in type_best.items():
        print(f"  {source_type}: {result.score:.3f}")
        print(f"    Content: {result.content[:100]}...")
    print()


def demonstrate_integration_patterns():
    """Demonstrate common integration patterns."""
    
    print("=== Integration Patterns Example ===\n")
    
    # Pattern 1: Service wrapper
    class ComplianceSearchService:
        """Service wrapper for compliance searches."""
        
        def __init__(self, config_dict: Dict[str, Any]):
            self.module = RetrievalModule('graphrag')
            self.module.initialize(config_dict)
        
        def search_regulations(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
            """Search for regulations only."""
            results = self.module.retrieve(
                query_text=query,
                filters={'source_type': 'FederalRegulation'},
                limit=limit
            )
            
            return [
                {
                    'title': self._extract_title(result),
                    'citation': result.metadata.get('citation', 'N/A'),
                    'relevance': result.score,
                    'summary': result.content[:200]
                }
                for result in results
            ]
        
        def search_enforcement(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
            """Search for enforcement actions only."""
            results = self.module.retrieve(
                query_text=query,
                filters={'source_type': 'EnforcementAction'},
                limit=limit
            )
            
            return [
                {
                    'title': self._extract_title(result),
                    'agency': result.metadata.get('agency', 'N/A'),
                    'outcome': result.metadata.get('outcome', 'N/A'),
                    'relevance': result.score
                }
                for result in results
            ]
        
        def _extract_title(self, result: RetrievalResult) -> str:
            """Extract title from result content."""
            lines = result.content.split('|')
            for line in lines:
                if line.strip().startswith(('Name:', 'Title:')):
                    return line.split(':', 1)[1].strip()
            return result.source_type
    
    # Demonstrate service usage
    config = load_config()
    
    try:
        service = ComplianceSearchService(config.to_dict())
        
        print("1. Regulation Search Service:")
        regulations = service.search_regulations("data privacy", limit=3)
        for reg in regulations:
            print(f"   {reg['title']} (Score: {reg['relevance']:.3f})")
            print(f"   Citation: {reg['citation']}")
        print()
        
        print("2. Enforcement Search Service:")
        enforcement = service.search_enforcement("privacy violation", limit=3)
        for action in enforcement:
            print(f"   {action['title']} (Score: {action['relevance']:.3f})")
            print(f"   Agency: {action['agency']}")
        print()
        
    except Exception as e:
        print(f"Service demonstration failed: {e}")
    
    # Pattern 2: Configuration examples
    print("3. Configuration Pattern Examples:")
    print("""
# Development configuration
dev_config = {
    'neo4j_uri': 'bolt://localhost:7687',
    'neo4j_user': 'neo4j',
    'neo4j_password': 'dev_password',
    'retrieval_config': {
        'default_limit': 10,
        'score_threshold': 0.5
    }
}

# Production configuration
prod_config = {
    'neo4j_uri': 'bolt://prod-cluster:7687',
    'neo4j_user': 'prod_user',
    'neo4j_password': os.getenv('NEO4J_PASSWORD'),
    'retrieval_config': {
        'default_limit': 20,
        'score_threshold': 0.7
    }
}
""")


def main():
    """Run all advanced examples."""
    
    print("=== GraphRAG Retrieval System - Advanced Usage Examples ===\n")
    
    try:
        demonstrate_custom_retriever()
        demonstrate_batch_processing()
        demonstrate_result_analysis()
        demonstrate_integration_patterns()
        
        print("=== All advanced examples completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Advanced examples failed: {e}", exc_info=True)
        print(f"Examples failed: {e}")
        print("Please ensure Neo4j is running with sample data loaded.")


if __name__ == "__main__":
    main()