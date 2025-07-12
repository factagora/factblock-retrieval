#!/usr/bin/env python3
"""
Enhanced GraphRAG with Relationship-Aware Fact Checking

This module integrates the relationship-aware fact checker with the existing
GraphRAG system to provide superior fact-checking capabilities that leverage
both vector similarity and knowledge graph relationships.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import relationship checker with proper handling
try:
    from .relationship_aware_fact_checker import RelationshipAwareFactChecker, FactCheckResult
except ImportError:
    try:
        from src.graphrag.relationship_aware_fact_checker import RelationshipAwareFactChecker, FactCheckResult
    except ImportError:
        print("⚠️ RelationshipAwareFactChecker not available")
        RelationshipAwareFactChecker = None
        FactCheckResult = None

# Import basic retrieval as fallback for traditional GraphRAG
try:
    from src.retrieval.graph_rag import GraphRAG
except ImportError:
    # Fallback implementation
    class GraphRAG:
        def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
            self.neo4j_uri = neo4j_uri
            self.neo4j_user = neo4j_user  
            self.neo4j_password = neo4j_password
            
        def fact_check(self, claim, max_evidence=5):
            return {
                'verdict': 'NEEDS_VERIFICATION',
                'confidence': 0.5,
                'evidence': [],
                'explanation': 'Traditional GraphRAG not available'
            }
        
        def close(self):
            pass


@dataclass
class EnhancedFactCheckResult:
    """Enhanced fact-check result combining vector and relationship evidence"""
    claim: str
    
    # Traditional RAG results
    vector_verdict: str
    vector_confidence: float
    vector_evidence: List[Dict]
    
    # Relationship-aware results
    relationship_verdict: str
    relationship_confidence: float
    relationship_evidence: List[Dict]
    regulatory_cascades: List[Dict]
    
    # Combined results
    final_verdict: str
    final_confidence: float
    explanation: str
    unique_relationship_insights: List[str]
    
    # Performance metrics
    vector_time_ms: float
    relationship_time_ms: float
    total_time_ms: float


class EnhancedGraphRAG:
    """
    Enhanced GraphRAG that combines vector similarity with relationship-aware fact checking
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        embedding_model = None,
        enable_parallel_processing: bool = True
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.enable_parallel = enable_parallel_processing
        
        # Initialize components
        self.traditional_rag = GraphRAG(neo4j_uri, neo4j_user, neo4j_password)
        
        if RelationshipAwareFactChecker is not None:
            self.relationship_checker = RelationshipAwareFactChecker(
                neo4j_uri, neo4j_user, neo4j_password
            )
        else:
            self.relationship_checker = None
            self.logger.warning("RelationshipAwareFactChecker not available")
        self.embedding_model = embedding_model
        
        self.logger = logging.getLogger(__name__)
        
    def fact_check_enhanced(
        self,
        claim: str,
        max_evidence: int = 5,
        max_relationship_depth: int = 3,
        min_confidence_threshold: float = 0.6,
        enable_regulatory_cascades: bool = True
    ) -> EnhancedFactCheckResult:
        """
        Perform enhanced fact-checking using both vector and relationship approaches
        
        Args:
            claim: The claim to fact-check
            max_evidence: Maximum number of evidence pieces to return
            max_relationship_depth: Maximum relationship traversal depth
            min_confidence_threshold: Minimum confidence threshold
            enable_regulatory_cascades: Whether to detect regulatory cascades
            
        Returns:
            Enhanced fact-check result with combined insights
        """
        start_time = time.time()
        
        if self.enable_parallel:
            return self._fact_check_parallel(
                claim, max_evidence, max_relationship_depth, 
                min_confidence_threshold, enable_regulatory_cascades
            )
        else:
            return self._fact_check_sequential(
                claim, max_evidence, max_relationship_depth,
                min_confidence_threshold, enable_regulatory_cascades
            )
    
    def _fact_check_parallel(
        self,
        claim: str,
        max_evidence: int,
        max_relationship_depth: int,
        min_confidence_threshold: float,
        enable_regulatory_cascades: bool
    ) -> EnhancedFactCheckResult:
        """Perform fact-checking with parallel processing"""
        
        start_time = time.time()
        
        # Run vector and relationship analysis in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit vector-based fact checking
            vector_future = executor.submit(
                self._run_vector_fact_check,
                claim, max_evidence
            )
            
            # Submit relationship-based fact checking
            relationship_future = executor.submit(
                self._run_relationship_fact_check,
                claim, max_relationship_depth, min_confidence_threshold
            )
            
            # Collect results
            vector_result = None
            relationship_result = None
            vector_time = 0
            relationship_time = 0
            
            for future in as_completed([vector_future, relationship_future]):
                if future == vector_future:
                    vector_result, vector_time = future.result()
                elif future == relationship_future:
                    relationship_result, relationship_time = future.result()
        
        # Combine results
        total_time = (time.time() - start_time) * 1000
        
        return self._combine_results(
            claim, vector_result, relationship_result,
            vector_time, relationship_time, total_time
        )
    
    def _fact_check_sequential(
        self,
        claim: str,
        max_evidence: int,
        max_relationship_depth: int,
        min_confidence_threshold: float,
        enable_regulatory_cascades: bool
    ) -> EnhancedFactCheckResult:
        """Perform fact-checking sequentially"""
        
        start_time = time.time()
        
        # Vector-based fact checking
        vector_result, vector_time = self._run_vector_fact_check(claim, max_evidence)
        
        # Relationship-based fact checking
        relationship_result, relationship_time = self._run_relationship_fact_check(
            claim, max_relationship_depth, min_confidence_threshold
        )
        
        # Combine results
        total_time = (time.time() - start_time) * 1000
        
        return self._combine_results(
            claim, vector_result, relationship_result,
            vector_time, relationship_time, total_time
        )
    
    def _run_vector_fact_check(self, claim: str, max_evidence: int) -> tuple:
        """Run traditional vector-based fact checking"""
        start_time = time.time()
        
        try:
            # Use existing GraphRAG for vector-based fact checking
            result = self.traditional_rag.fact_check(claim, max_evidence=max_evidence)
            
            vector_time = (time.time() - start_time) * 1000
            
            return result, vector_time
            
        except Exception as e:
            self.logger.error(f"Vector fact-check failed: {e}")
            vector_time = (time.time() - start_time) * 1000
            
            # Return fallback result
            return {
                'verdict': 'ERROR',
                'confidence': 0.0,
                'evidence': [],
                'explanation': f"Vector fact-check failed: {str(e)}"
            }, vector_time
    
    def _run_relationship_fact_check(
        self, 
        claim: str, 
        max_depth: int, 
        min_threshold: float
    ) -> tuple:
        """Run relationship-aware fact checking"""
        start_time = time.time()
        
        if self.relationship_checker is None:
            relationship_time = (time.time() - start_time) * 1000
            # Create a basic fallback result
            if FactCheckResult is not None:
                return FactCheckResult(
                    verdict='INSUFFICIENT_EVIDENCE',
                    confidence=0.0,
                    vector_evidence=[],
                    relationship_evidence=[],
                    regulatory_cascades=[],
                    explanation="Relationship checker not available",
                    unique_insights=[]
                ), relationship_time
            else:
                # Even more basic fallback
                return {
                    'verdict': 'INSUFFICIENT_EVIDENCE',
                    'confidence': 0.0,
                    'relationship_evidence': [],
                    'regulatory_cascades': [],
                    'explanation': "Relationship checker not available",
                    'unique_insights': []
                }, relationship_time
        
        try:
            result = self.relationship_checker.fact_check_with_relationships(
                claim=claim,
                max_relationship_depth=max_depth,
                min_confidence_threshold=min_threshold
            )
            
            relationship_time = (time.time() - start_time) * 1000
            
            return result, relationship_time
            
        except Exception as e:
            self.logger.error(f"Relationship fact-check failed: {e}")
            relationship_time = (time.time() - start_time) * 1000
            
            # Return fallback result
            if FactCheckResult is not None:
                return FactCheckResult(
                    verdict='ERROR',
                    confidence=0.0,
                    vector_evidence=[],
                    relationship_evidence=[],
                    regulatory_cascades=[],
                    explanation=f"Relationship fact-check failed: {str(e)}",
                    unique_insights=[]
                ), relationship_time
            else:
                return {
                    'verdict': 'ERROR',
                    'confidence': 0.0,
                    'relationship_evidence': [],
                    'regulatory_cascades': [],
                    'explanation': f"Relationship fact-check failed: {str(e)}",
                    'unique_insights': []
                }, relationship_time
    
    def _combine_results(
        self,
        claim: str,
        vector_result: Dict,
        relationship_result: FactCheckResult,
        vector_time: float,
        relationship_time: float,
        total_time: float
    ) -> EnhancedFactCheckResult:
        """Combine vector and relationship fact-check results"""
        
        # Extract vector results
        vector_verdict = vector_result.get('verdict', 'UNKNOWN')
        vector_confidence = vector_result.get('confidence', 0.0)
        vector_evidence = vector_result.get('evidence', [])
        
        # Extract relationship results
        relationship_verdict = relationship_result.verdict
        relationship_confidence = relationship_result.confidence
        relationship_evidence = [asdict(ev) for ev in relationship_result.relationship_evidence]
        regulatory_cascades = relationship_result.regulatory_cascades
        unique_insights = relationship_result.unique_insights
        
        # Combine verdicts and confidence
        final_verdict, final_confidence = self._merge_verdicts_and_confidence(
            vector_verdict, vector_confidence,
            relationship_verdict, relationship_confidence,
            regulatory_cascades
        )
        
        # Generate combined explanation
        explanation = self._generate_combined_explanation(
            claim, vector_result, relationship_result,
            final_verdict, final_confidence
        )
        
        return EnhancedFactCheckResult(
            claim=claim,
            vector_verdict=vector_verdict,
            vector_confidence=vector_confidence,
            vector_evidence=vector_evidence,
            relationship_verdict=relationship_verdict,
            relationship_confidence=relationship_confidence,
            relationship_evidence=relationship_evidence,
            regulatory_cascades=regulatory_cascades,
            final_verdict=final_verdict,
            final_confidence=final_confidence,
            explanation=explanation,
            unique_relationship_insights=unique_insights,
            vector_time_ms=vector_time,
            relationship_time_ms=relationship_time,
            total_time_ms=total_time
        )
    
    def _merge_verdicts_and_confidence(
        self,
        vector_verdict: str,
        vector_confidence: float,
        relationship_verdict: str,
        relationship_confidence: float,
        regulatory_cascades: List[Dict]
    ) -> tuple:
        """Merge verdicts and confidence scores from both approaches"""
        
        # Verdict priority: CONTRADICTED > SUPPORTED > PARTIALLY_SUPPORTED > INSUFFICIENT_EVIDENCE
        verdict_priority = {
            'CONTRADICTED': 4,
            'SUPPORTED': 3,
            'PARTIALLY_SUPPORTED': 2,
            'INSUFFICIENT_EVIDENCE': 1,
            'ERROR': 0,
            'UNKNOWN': 0
        }
        
        # Choose the higher priority verdict
        vector_priority = verdict_priority.get(vector_verdict, 0)
        relationship_priority = verdict_priority.get(relationship_verdict, 0)
        
        if relationship_priority > vector_priority:
            final_verdict = relationship_verdict
        elif vector_priority > relationship_priority:
            final_verdict = vector_verdict
        else:
            # Same priority - use relationship verdict if it found regulatory cascades
            if regulatory_cascades:
                final_verdict = relationship_verdict
            else:
                final_verdict = vector_verdict
        
        # Combine confidence scores
        # Give more weight to relationship confidence if regulatory cascades found
        if regulatory_cascades:
            cascade_boost = min(0.2, len(regulatory_cascades) * 0.05)
            final_confidence = (
                vector_confidence * 0.4 + 
                relationship_confidence * 0.6 + 
                cascade_boost
            )
        else:
            final_confidence = (vector_confidence * 0.6 + relationship_confidence * 0.4)
        
        final_confidence = min(1.0, max(0.0, final_confidence))
        
        return final_verdict, final_confidence
    
    def _generate_combined_explanation(
        self,
        claim: str,
        vector_result: Dict,
        relationship_result: FactCheckResult,
        final_verdict: str,
        final_confidence: float
    ) -> str:
        """Generate comprehensive explanation combining both approaches"""
        
        explanation_parts = [
            f"Enhanced Fact-Check Result for: '{claim}'",
            f"Final Verdict: {final_verdict} (Confidence: {final_confidence:.2f})",
            "",
            "=== VECTOR SIMILARITY ANALYSIS ===",
            f"Verdict: {vector_result.get('verdict', 'UNKNOWN')} "
            f"(Confidence: {vector_result.get('confidence', 0.0):.2f})",
            f"Evidence pieces: {len(vector_result.get('evidence', []))}"
        ]
        
        # Add vector evidence summary
        vector_evidence = vector_result.get('evidence', [])
        if vector_evidence:
            explanation_parts.append("Top vector evidence:")
            for i, ev in enumerate(vector_evidence[:3], 1):
                title = ev.get('title', 'Unknown')
                score = ev.get('similarity_score', 0.0)
                explanation_parts.append(f"  {i}. {title} (similarity: {score:.2f})")
        
        explanation_parts.extend([
            "",
            "=== RELATIONSHIP GRAPH ANALYSIS ===",
            f"Verdict: {relationship_result.verdict} "
            f"(Confidence: {relationship_result.confidence:.2f})",
            f"Relationship evidence: {len(relationship_result.relationship_evidence)}",
            f"Regulatory cascades: {len(relationship_result.regulatory_cascades)}"
        ])
        
        # Add relationship insights
        if relationship_result.unique_insights:
            explanation_parts.append("Unique relationship insights:")
            for insight in relationship_result.unique_insights:
                explanation_parts.append(f"  • {insight}")
        
        # Add regulatory cascade details
        if relationship_result.regulatory_cascades:
            explanation_parts.append("Regulatory cascades detected:")
            for i, cascade in enumerate(relationship_result.regulatory_cascades[:3], 1):
                explanation_parts.append(f"  {i}. {cascade.get('explanation', 'Unknown cascade')}")
        
        explanation_parts.extend([
            "",
            "=== COMBINED ANALYSIS ===",
            "The enhanced fact-checker leverages both vector similarity and knowledge graph relationships.",
            "Relationship analysis provides insights that vector embeddings cannot capture,",
            "particularly for regulatory cascades and multi-hop semantic connections."
        ])
        
        return "\n".join(explanation_parts)
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for both approaches"""
        # This would be implemented to track and return performance statistics
        pass
    
    def close(self):
        """Close all connections"""
        if hasattr(self.traditional_rag, 'close'):
            self.traditional_rag.close()
        self.relationship_checker.close()


# API Integration functions
def create_enhanced_fact_check_endpoint(enhanced_graphrag: EnhancedGraphRAG):
    """Create FastAPI endpoint for enhanced fact-checking"""
    
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
    
    router = APIRouter()
    
    class FactCheckRequest(BaseModel):
        text: str
        max_evidence: int = 5
        max_relationship_depth: int = 3
        min_confidence_threshold: float = 0.6
        enable_regulatory_cascades: bool = True
    
    @router.post("/fact-check-enhanced")
    async def fact_check_enhanced(request: FactCheckRequest):
        """Enhanced fact-checking endpoint"""
        try:
            result = enhanced_graphrag.fact_check_enhanced(
                claim=request.text,
                max_evidence=request.max_evidence,
                max_relationship_depth=request.max_relationship_depth,
                min_confidence_threshold=request.min_confidence_threshold,
                enable_regulatory_cascades=request.enable_regulatory_cascades
            )
            
            return asdict(result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return router