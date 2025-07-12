#!/usr/bin/env python3
"""
Smart GraphRAG Router

Intelligently routes queries to the best retrieval method based on
query analysis and performance requirements.
"""

import os
import sys
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.vector_retriever import GraphVectorRetriever
from graphrag.cypher_retriever import TextToCypherRetriever

logger = logging.getLogger(__name__)

class QueryType(Enum):
    SEMANTIC = "semantic"           # Conceptual/similarity search
    STRUCTURED = "structured"       # Precise filtering/analytics
    ENTITY_SPECIFIC = "entity"      # Known entity queries
    EXPLORATORY = "exploratory"     # Discovery/browsing
    HYBRID = "hybrid"              # Benefits from both approaches

class PerformanceMode(Enum):
    FAST = "fast"           # < 50ms, single method
    BALANCED = "balanced"   # < 200ms, smart routing
    COMPREHENSIVE = "comprehensive"  # < 500ms, hybrid when beneficial

class SmartGraphRAGRouter:
    """Smart router that chooses optimal retrieval strategy"""
    
    def __init__(self, performance_mode: PerformanceMode = PerformanceMode.BALANCED):
        """
        Initialize the smart router
        
        Args:
            performance_mode: Performance requirements mode
        """
        self.performance_mode = performance_mode
        self.vector_retriever: Optional[GraphVectorRetriever] = None
        self.cypher_retriever: Optional[TextToCypherRetriever] = None
        
        # Query pattern analysis
        self._setup_query_patterns()
        
        # Performance tracking
        self.performance_stats = {
            "vector_avg_time": 0.010,  # Default estimates
            "cypher_avg_time": 0.005,
            "query_count": 0
        }
        
        self._initialize_retrievers()
    
    def _initialize_retrievers(self):
        """Initialize retrievers lazily"""
        try:
            self.vector_retriever = GraphVectorRetriever()
            self.cypher_retriever = TextToCypherRetriever()
            logger.info("✅ Smart router initialized both retrievers")
        except Exception as e:
            logger.error(f"❌ Router initialization failed: {e}")
            raise
    
    def _setup_query_patterns(self):
        """Setup patterns for query classification"""
        self.patterns = {
            QueryType.ENTITY_SPECIFIC: [
                r'\b(OPEC|오펙|Tesla|Apple|Microsoft|Google|Amazon|Fed|Federal Reserve|연준|현대자동차|반도체)\b',
                r'\babout\s+[A-Z][a-z]+\b',
                r'\bmentioning?\s+[A-Z][a-z]+\b',
                r'(감산|원유|배럴|산유국|항공사|연료비|금리)',  # Korean terms
            ],
            
            QueryType.STRUCTURED: [
                r'\b(find|get|show)\s+(all|high|low|critical)\b',
                r'\b(confidence|impact)\s+(>|<|=)\s*\d+',
                r'\b(in|from)\s+(energy|banking|tech|financial)\s+sector\b',
                r'\b(before|after|since|during)\s+\d{4}\b',
                r'\bcount\b|\btop\s+\d+\b|\blimit\s+\d+\b',
            ],
            
            QueryType.SEMANTIC: [
                r'\b(similar|like|related)\s+to\b',
                r'\b(impact|effect|influence)\s+of\b',
                r'\b(risk|opportunity|trend)\b',
                r'\b(what|how|why)\s+(happens|affects|causes)\b',
                r'\bimplications?\s+of\b',
            ],
            
            QueryType.EXPLORATORY: [
                r'\b(explore|discover|browse)\b',
                r'\b(what\'s|what\s+are)\s+(the|some)\b',
                r'\b(overview|summary)\s+of\b',
                r'\b(tell me about|explain)\b',
                r'\b(interesting|notable)\b',
            ]
        }
        
        # Known entities for quick recognition
        self.known_entities = {
            'opec', '오펙', 'tesla', 'apple', 'microsoft', 'google', 'amazon', 
            'fed', 'federal reserve', '연준', '중앙은행', 'nvidia',
            'meta', 'facebook', 'bitcoin', 'ethereum', '현대자동차',
            '감산', '원유', '배럴', '산유국', '항공사', '연료비', '금리', '반도체'
        }
        
        # Investment domain terms
        self.investment_terms = {
            'ipo', 'earnings', 'dividend', 'merger', 'acquisition',
            'volatility', 'correlation', 'beta', 'alpha', 'sharpe',
            'esg', 'sustainability', 'climate', 'regulation'
        }
    
    def analyze_query(self, query: str) -> Tuple[QueryType, float]:
        """
        Analyze query to determine best approach
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (QueryType, confidence_score)
        """
        query_lower = query.lower().strip()
        scores = {query_type: 0.0 for query_type in QueryType}
        
        # Pattern matching
        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[query_type] += 1.0
        
        # Entity recognition boost
        for entity in self.known_entities:
            if entity in query_lower:
                scores[QueryType.ENTITY_SPECIFIC] += 2.0
        
        # Investment term boost
        for term in self.investment_terms:
            if term in query_lower:
                scores[QueryType.SEMANTIC] += 0.5
        
        # Query structure analysis
        if '?' in query:
            scores[QueryType.SEMANTIC] += 0.5
            scores[QueryType.EXPLORATORY] += 0.5
        
        if any(word in query_lower for word in ['find', 'get', 'show', 'list']):
            scores[QueryType.STRUCTURED] += 1.0
        
        if len(query.split()) > 10:  # Long queries tend to be semantic
            scores[QueryType.SEMANTIC] += 0.5
        
        # Determine if hybrid approach would be beneficial
        top_scores = sorted(scores.values(), reverse=True)
        if len(top_scores) >= 2 and top_scores[0] - top_scores[1] < 1.0:
            scores[QueryType.HYBRID] = (top_scores[0] + top_scores[1]) / 2
        
        # Get best match
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type] / max(sum(scores.values()), 1.0)
        
        return best_type, confidence
    
    def estimate_performance(self, query_type: QueryType, use_hybrid: bool = False) -> float:
        """Estimate query execution time"""
        if use_hybrid:
            return self.performance_stats["vector_avg_time"] + self.performance_stats["cypher_avg_time"]
        elif query_type in [QueryType.SEMANTIC, QueryType.EXPLORATORY]:
            return self.performance_stats["vector_avg_time"]
        else:
            return self.performance_stats["cypher_avg_time"]
    
    def choose_strategy(self, query: str) -> Dict[str, Any]:
        """
        Choose optimal retrieval strategy
        
        Args:
            query: User query string
            
        Returns:
            Strategy decision with reasoning
        """
        query_type, confidence = self.analyze_query(query)
        
        strategy = {
            "query": query,
            "detected_type": query_type.value,
            "confidence": confidence,
            "use_vector": False,
            "use_cypher": False,
            "reasoning": []
        }
        
        # Performance mode considerations
        if self.performance_mode == PerformanceMode.FAST:
            # Always use fastest single method
            if query_type == QueryType.STRUCTURED:
                strategy["use_cypher"] = True
                strategy["reasoning"].append("Fast mode: Cypher for structured query")
            else:
                strategy["use_vector"] = True
                strategy["reasoning"].append("Fast mode: Vector for semantic query")
        
        elif self.performance_mode == PerformanceMode.COMPREHENSIVE:
            # Use hybrid when beneficial
            if query_type == QueryType.HYBRID or confidence < 0.7:
                strategy["use_vector"] = True
                strategy["use_cypher"] = True
                strategy["reasoning"].append("Comprehensive mode: Hybrid for better coverage")
            elif query_type in [QueryType.STRUCTURED, QueryType.ENTITY_SPECIFIC]:
                strategy["use_cypher"] = True
                strategy["reasoning"].append("Comprehensive mode: Cypher for precise results")
            else:
                strategy["use_vector"] = True
                strategy["reasoning"].append("Comprehensive mode: Vector for semantic search")
        
        else:  # BALANCED mode
            # Smart routing based on query analysis
            if query_type == QueryType.ENTITY_SPECIFIC and confidence > 0.8:
                strategy["use_cypher"] = True
                strategy["reasoning"].append("High confidence entity query → Cypher")
            elif query_type == QueryType.STRUCTURED and confidence > 0.7:
                strategy["use_cypher"] = True
                strategy["reasoning"].append("High confidence structured query → Cypher")
            elif query_type == QueryType.SEMANTIC:
                strategy["use_vector"] = True
                strategy["reasoning"].append("Semantic query → Vector search")
            elif query_type == QueryType.HYBRID or confidence < 0.5:
                strategy["use_vector"] = True
                strategy["use_cypher"] = True
                strategy["reasoning"].append("Low confidence → Hybrid approach")
            else:
                strategy["use_vector"] = True
                strategy["reasoning"].append("Default → Vector search")
        
        return strategy
    
    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Execute smart search with optimal strategy
        
        Args:
            query: User query string
            max_results: Maximum results to return
            
        Returns:
            Combined search results with metadata
        """
        start_time = time.time()
        
        # Choose strategy
        strategy = self.choose_strategy(query)
        
        results = {
            "query": query,
            "strategy": strategy,
            "vector_results": [],
            "cypher_results": {},
            "combined_results": [],
            "execution_time": 0.0,
            "result_count": 0
        }
        
        try:
            # Execute vector search if needed
            if strategy["use_vector"]:
                vector_start = time.time()
                vector_results = self.vector_retriever.search(
                    query, k=max_results, include_graph_expansion=True
                )
                vector_time = time.time() - vector_start
                
                results["vector_results"] = vector_results
                results["vector_time"] = vector_time
                
                # Update performance stats
                self._update_performance_stats("vector", vector_time)
            
            # Execute cypher search if needed
            if strategy["use_cypher"]:
                cypher_start = time.time()
                cypher_result = self.cypher_retriever.search(query)
                cypher_time = time.time() - cypher_start
                
                results["cypher_results"] = cypher_result
                results["cypher_time"] = cypher_time
                
                # Update performance stats
                self._update_performance_stats("cypher", cypher_time)
                
                # Fallback: If Cypher returns no results and we haven't used vector search, try it
                if (not cypher_result.get("results") and not strategy["use_vector"]):
                    logger.info("🔄 Cypher returned no results, falling back to vector search")
                    vector_start = time.time()
                    vector_results = self.vector_retriever.search(
                        query, k=max_results, include_graph_expansion=True
                    )
                    vector_time = time.time() - vector_start
                    
                    results["vector_results"] = vector_results
                    results["vector_time"] = vector_time
                    results["fallback_used"] = True
                    strategy["reasoning"].append("Cypher fallback → Vector search")
                    
                    # Update performance stats
                    self._update_performance_stats("vector", vector_time)
            
            # Combine results intelligently
            results["combined_results"] = self._combine_results(
                results["vector_results"], 
                results["cypher_results"], 
                max_results
            )
            
            results["execution_time"] = time.time() - start_time
            results["result_count"] = len(results["combined_results"])
            
            logger.info(f"✅ Smart search completed in {results['execution_time']:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Smart search failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def _combine_results(self, vector_results: List[Dict], cypher_results: Dict, max_results: int) -> List[Dict]:
        """Intelligently combine results from both sources"""
        combined = []
        seen_ids = set()
        
        # Add vector results first (they have similarity scores)
        if vector_results:  # Check if vector_results is not empty
            for result in vector_results:
                factblock = result["factblock"]
                fb_id = factblock.get("id")
                
                if fb_id not in seen_ids:
                    seen_ids.add(fb_id)
                    combined.append({
                        "factblock": factblock,
                        "score": result["combined_score"],
                        "source": "vector",
                        "metadata": result
                    })
        
        # Add unique cypher results
        if cypher_results.get("results"):
            for result in cypher_results["results"]:
                if "f" in result:  # FactBlock result
                    factblock = result["f"]
                    fb_id = factblock.get("id")
                    
                    if fb_id not in seen_ids:
                        seen_ids.add(fb_id)
                        combined.append({
                            "factblock": factblock,
                            "score": factblock.get("confidence_score", 0.5),  # Use confidence as score
                            "source": "cypher",
                            "metadata": {"cypher_query_type": cypher_results.get("query_type")}
                        })
        
        # Sort by score and limit
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:max_results]
    
    def _update_performance_stats(self, method: str, execution_time: float):
        """Update performance statistics for future optimization"""
        key = f"{method}_avg_time"
        current_avg = self.performance_stats[key]
        count = self.performance_stats["query_count"]
        
        # Exponential moving average
        alpha = 0.1  # Learning rate
        self.performance_stats[key] = (1 - alpha) * current_avg + alpha * execution_time
        self.performance_stats["query_count"] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.performance_stats.copy()
    
    def close(self):
        """Close all connections"""
        if self.vector_retriever:
            self.vector_retriever.close()
        if self.cypher_retriever:
            self.cypher_retriever.close()
        logger.info("🔐 Smart router connections closed")


def demo_smart_router():
    """Demo the smart routing functionality"""
    
    print("🧠 Testing Smart GraphRAG Router...")
    print("=" * 60)
    
    try:
        # Test different performance modes
        modes = [PerformanceMode.FAST, PerformanceMode.BALANCED, PerformanceMode.COMPREHENSIVE]
        
        test_queries = [
            "Find FactBlocks about OPEC",  # Entity-specific
            "What's the impact of inflation on markets?",  # Semantic
            "Get high impact energy sector claims",  # Structured
            "Show me interesting investment trends",  # Exploratory
            "How do interest rates affect banking and what are the specific mechanisms?"  # Hybrid
        ]
        
        for mode in modes:
            print(f"\n🎯 Testing {mode.value.upper()} mode:")
            print("-" * 40)
            
            router = SmartGraphRAGRouter(performance_mode=mode)
            
            for query in test_queries:
                print(f"\n🔍 Query: '{query}'")
                
                # Show strategy decision
                strategy = router.choose_strategy(query)
                print(f"   📊 Type: {strategy['detected_type']} (confidence: {strategy['confidence']:.2f})")
                print(f"   🎯 Strategy: Vector={strategy['use_vector']}, Cypher={strategy['use_cypher']}")
                print(f"   💡 Reasoning: {'; '.join(strategy['reasoning'])}")
                
                # Show estimated performance
                estimated_time = router.estimate_performance(
                    QueryType(strategy['detected_type']), 
                    strategy['use_vector'] and strategy['use_cypher']
                )
                print(f"   ⏱️  Estimated time: {estimated_time:.3f}s")
            
            router.close()
        
        print(f"\n✅ Smart routing demo completed!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    demo_smart_router()