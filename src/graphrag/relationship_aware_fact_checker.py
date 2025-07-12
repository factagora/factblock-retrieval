#!/usr/bin/env python3
"""
Relationship-Aware Fact Checker

This module implements sophisticated fact-checking that leverages knowledge graph
relationships that vector embeddings cannot capture. It focuses on regulatory
cascades and multi-hop semantic connections.

Key innovations:
1. Regulatory Cascade Detection: regulation → compliance → business impact
2. Semantic Relationship Traversal: Goes beyond vector similarity
3. Credibility Propagation: Uses graph structure for evidence weighting
4. Contradiction Discovery: Finds conflicting facts through relationships
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from neo4j import GraphDatabase
import numpy as np
from datetime import datetime, timedelta


@dataclass
class RelationshipEvidence:
    """Evidence found through relationship traversal"""
    source_factblock_id: str
    target_factblock_id: str
    relationship_type: str
    semantic_type: str
    strength: float
    path_length: int
    credibility_score: float
    investment_insight: str
    explanation: str


@dataclass
class FactCheckResult:
    """Enhanced fact-check result with relationship insights"""
    verdict: str  # "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"
    confidence: float
    vector_evidence: List[Dict]  # Traditional RAG evidence
    relationship_evidence: List[RelationshipEvidence]  # Graph-based evidence
    regulatory_cascades: List[Dict]  # Detected regulatory chains
    explanation: str
    unique_insights: List[str]  # What graph found that vectors couldn't


class RelationshipAwareFactChecker:
    """
    Advanced fact-checker that combines vector similarity with graph relationships
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.neo4j_database = 'neo4j'  # Default database
        self.logger = logging.getLogger(__name__)
        
        # Relationship semantic types for regulatory cascades
        self.regulatory_cascade_types = {
            'triggers_compliance_change': 0.9,
            'forces_business_model_change': 0.85,
            'creates_market_pressure': 0.8,
            'influences_investment_decision': 0.75,
            'affects_operational_strategy': 0.7,
            'drives_technology_adoption': 0.65
        }
        
    def fact_check_with_relationships(
        self, 
        claim: str, 
        vector_evidence: List[Dict] = None,
        max_relationship_depth: int = 3,
        min_confidence_threshold: float = 0.6
    ) -> FactCheckResult:
        """
        Perform relationship-aware fact-checking
        
        Args:
            claim: The claim to fact-check
            vector_evidence: Traditional RAG evidence (if available)
            max_relationship_depth: Maximum hops to traverse
            min_confidence_threshold: Minimum confidence for evidence
            
        Returns:
            Enhanced fact-check result with relationship insights
        """
        
        # Step 1: Extract key entities from claim
        claim_entities = self._extract_claim_entities(claim)
        
        # Step 2: Find directly related FactBlocks
        direct_evidence = self._find_direct_evidence(claim_entities)
        
        # Step 3: Discover relationship-based evidence
        relationship_evidence = self._discover_relationship_evidence(
            claim_entities, 
            max_relationship_depth
        )
        
        # Step 4: Detect regulatory cascades
        regulatory_cascades = self._detect_regulatory_cascades(
            claim_entities,
            relationship_evidence
        )
        
        # Step 5: Analyze contradictions
        contradictions = self._find_contradictions(
            direct_evidence + relationship_evidence
        )
        
        # Step 6: Calculate relationship-aware confidence
        confidence = self._calculate_relationship_confidence(
            direct_evidence,
            relationship_evidence,
            regulatory_cascades,
            contradictions
        )
        
        # Step 7: Generate verdict
        verdict = self._generate_verdict(
            direct_evidence,
            relationship_evidence,
            contradictions,
            confidence,
            min_confidence_threshold
        )
        
        # Step 8: Identify unique insights
        unique_insights = self._identify_unique_insights(
            relationship_evidence,
            regulatory_cascades,
            vector_evidence or []
        )
        
        # Step 9: Generate explanation
        explanation = self._generate_explanation(
            verdict,
            confidence,
            direct_evidence,
            relationship_evidence,
            regulatory_cascades,
            unique_insights
        )
        
        return FactCheckResult(
            verdict=verdict,
            confidence=confidence,
            vector_evidence=vector_evidence or [],
            relationship_evidence=relationship_evidence,
            regulatory_cascades=regulatory_cascades,
            explanation=explanation,
            unique_insights=unique_insights
        )
    
    def _extract_claim_entities(self, claim: str) -> List[str]:
        """Extract key entities from the claim"""
        # Use simple entity extraction (APOC NLP not available)
        return self._simple_entity_extraction(claim)
    
    def _simple_entity_extraction(self, claim: str) -> List[str]:
        """Simple fallback entity extraction"""
        entities = []
        
        # Keywords that indicate important entities
        entity_keywords = [
            'EU', '미국', '일본', '중국', '한국',  # Countries/regions
            '개인정보보호법', '아동보호법', '탄소배출', '규제', '법률',  # Regulations
            '기업', '회사', '은행', '제조업체', '소셜미디어',  # Organizations
            '투자', '변경', '도입', '전환', '강화',  # Actions
            '쿠키', '광고', '연령인증', '생산공정', '타겟팅'  # Technologies/processes
        ]
        
        # Extract keywords present in claim
        for keyword in entity_keywords:
            if keyword in claim:
                entities.append(keyword)
        
        # Also look for known entities in the database
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower($claim) CONTAINS toLower(e.name)
                    RETURN e.name as entity
                    ORDER BY size(e.name) DESC
                    LIMIT 5
                """, claim=claim)
                
                db_entities = [record["entity"] for record in result]
                entities.extend(db_entities)
        except Exception as e:
            self.logger.warning(f"Failed to query database entities: {e}")
            # Continue without database entities
        
        # Remove duplicates and return
        unique_entities = list(set(entities))
        self.logger.info(f"Extracted entities: {unique_entities}")
        return unique_entities
    
    def _find_direct_evidence(self, entities: List[str]) -> List[RelationshipEvidence]:
        """Find FactBlocks directly mentioning the entities"""
        if not entities:
            self.logger.info("No entities provided for direct evidence search")
            return []
            
        self.logger.info(f"Searching for direct evidence with entities: {entities}")
        
        with self.driver.session(database=self.neo4j_database) as session:
            result = session.run("""
                MATCH (f:FactBlock)
                WHERE ANY(entity IN $entities WHERE 
                    toLower(f.text) CONTAINS toLower(entity) OR
                    toLower(f.title) CONTAINS toLower(entity)
                )
                RETURN f.id as id, f.title as title, f.text as text,
                       f.credibility_score as credibility, f.verdict as verdict,
                       f.investment_insight as insight
                ORDER BY f.credibility_score DESC
                LIMIT 20
            """, entities=entities)
            
            evidence = []
            count = 0
            for record in result:
                count += 1
                evidence.append(RelationshipEvidence(
                    source_factblock_id=record["id"],
                    target_factblock_id=record["id"],
                    relationship_type="DIRECT_MATCH",
                    semantic_type="direct_mention",
                    strength=1.0,
                    path_length=0,
                    credibility_score=record["credibility"] or 0.5,
                    investment_insight=record["insight"] or "",
                    explanation=f"Direct mention in: {record['title']}"
                ))
            
            self.logger.info(f"Found {count} direct evidence FactBlocks")
            return evidence
    
    def _discover_relationship_evidence(
        self, 
        entities: List[str], 
        max_depth: int
    ) -> List[RelationshipEvidence]:
        """Discover evidence through relationship traversal"""
        if not entities:
            return []
            
        evidence = []
        
        # Multi-hop relationship traversal
        for depth in range(1, max_depth + 1):
            depth_evidence = self._traverse_relationships_at_depth(entities, depth)
            evidence.extend(depth_evidence)
        
        # Sort by strength and credibility
        evidence.sort(key=lambda e: e.strength * e.credibility_score, reverse=True)
        
        return evidence[:50]  # Limit to top 50 evidence pieces
    
    def _traverse_relationships_at_depth(
        self, 
        entities: List[str], 
        depth: int
    ) -> List[RelationshipEvidence]:
        """Traverse relationships at a specific depth"""
        with self.driver.session(database=self.neo4j_database) as session:
            # Build variable-length path query
            path_pattern = "-[r:RELATES_TO]->(f)" * depth
            
            result = session.run(f"""
                MATCH (start:FactBlock)
                WHERE ANY(entity IN $entities WHERE 
                    toLower(start.text) CONTAINS toLower(entity) OR
                    toLower(start.title) CONTAINS toLower(entity)
                )
                MATCH path = (start){path_pattern}
                WHERE f <> start
                RETURN start.id as start_id, f.id as end_id,
                       relationships(path) as rels,
                       f.credibility_score as credibility,
                       f.investment_insight as insight,
                       f.title as title,
                       length(path) as path_length
                ORDER BY f.credibility_score DESC
                LIMIT 100
            """, entities=entities)
            
            evidence = []
            for record in result:
                # Calculate path strength
                path_strength = 1.0
                semantic_types = []
                
                for rel in record["rels"]:
                    rel_strength = rel.get("strength", 0.8)
                    path_strength *= rel_strength
                    
                    semantic_type = rel.get("semantic_type", "general_relation")
                    semantic_types.append(semantic_type)
                
                # Boost regulatory cascade relationships
                if any(st in self.regulatory_cascade_types for st in semantic_types):
                    cascade_boost = max(self.regulatory_cascade_types.get(st, 0.5) for st in semantic_types)
                    path_strength *= cascade_boost
                
                evidence.append(RelationshipEvidence(
                    source_factblock_id=record["start_id"],
                    target_factblock_id=record["end_id"],
                    relationship_type="MULTI_HOP",
                    semantic_type=" → ".join(semantic_types),
                    strength=path_strength,
                    path_length=record["path_length"],
                    credibility_score=record["credibility"] or 0.5,
                    investment_insight=record["insight"] or "",
                    explanation=f"Connected through {depth}-hop path: {record['title']}"
                ))
            
            return evidence
    
    def _detect_regulatory_cascades(
        self, 
        entities: List[str], 
        evidence: List[RelationshipEvidence]
    ) -> List[Dict]:
        """Detect regulatory cascade patterns"""
        cascades = []
        
        with self.driver.session(database=self.neo4j_database) as session:
            # Look for regulation → compliance → business impact chains
            result = session.run("""
                MATCH (reg:FactBlock)-[r1:RELATES_TO]->(comp:FactBlock)-[r2:RELATES_TO]->(bus:FactBlock)
                WHERE r1.semantic_type IN ['triggers_compliance_change', 'creates_regulatory_pressure']
                  AND r2.semantic_type IN ['forces_business_model_change', 'affects_operational_strategy']
                  AND ANY(entity IN $entities WHERE 
                      toLower(reg.text) CONTAINS toLower(entity) OR
                      toLower(comp.text) CONTAINS toLower(entity) OR
                      toLower(bus.text) CONTAINS toLower(entity)
                  )
                RETURN reg.id as regulation_id, reg.title as regulation_title,
                       comp.id as compliance_id, comp.title as compliance_title,
                       bus.id as business_id, bus.title as business_title,
                       r1.semantic_type as reg_type, r2.semantic_type as bus_type,
                       r1.strength * r2.strength as cascade_strength
                ORDER BY cascade_strength DESC
                LIMIT 20
            """, entities=entities)
            
            for record in result:
                cascades.append({
                    'type': 'regulatory_cascade',
                    'strength': record['cascade_strength'],
                    'regulation': {
                        'id': record['regulation_id'],
                        'title': record['regulation_title']
                    },
                    'compliance': {
                        'id': record['compliance_id'],
                        'title': record['compliance_title']
                    },
                    'business_impact': {
                        'id': record['business_id'],
                        'title': record['business_title']
                    },
                    'relationship_types': [record['reg_type'], record['bus_type']],
                    'explanation': f"Regulatory cascade: {record['regulation_title']} → {record['compliance_title']} → {record['business_title']}"
                })
        
        return cascades
    
    def _find_contradictions(self, evidence: List[RelationshipEvidence]) -> List[Dict]:
        """Find contradictory evidence through relationship analysis"""
        contradictions = []
        
        # Group evidence by FactBlock pairs
        factblock_pairs = {}
        for ev in evidence:
            pair_key = tuple(sorted([ev.source_factblock_id, ev.target_factblock_id]))
            if pair_key not in factblock_pairs:
                factblock_pairs[pair_key] = []
            factblock_pairs[pair_key].append(ev)
        
        # Look for contradictory verdicts or credibility scores
        with self.driver.session(database=self.neo4j_database) as session:
            for (fb1_id, fb2_id), evidences in factblock_pairs.items():
                if fb1_id == fb2_id:
                    continue
                    
                result = session.run("""
                    MATCH (f1:FactBlock {id: $fb1_id}), (f2:FactBlock {id: $fb2_id})
                    RETURN f1.verdict as verdict1, f1.credibility_score as cred1,
                           f2.verdict as verdict2, f2.credibility_score as cred2,
                           f1.title as title1, f2.title as title2
                """, fb1_id=fb1_id, fb2_id=fb2_id)
                
                record = result.single()
                if record:
                    v1, v2 = record["verdict1"], record["verdict2"]
                    c1, c2 = record["cred1"] or 0.5, record["cred2"] or 0.5
                    
                    # Check for verdict contradiction
                    if v1 and v2 and v1 != v2 and abs(c1 - c2) < 0.3:
                        contradictions.append({
                            'type': 'verdict_contradiction',
                            'factblock1': {'id': fb1_id, 'title': record['title1'], 'verdict': v1},
                            'factblock2': {'id': fb2_id, 'title': record['title2'], 'verdict': v2},
                            'severity': abs(c1 - c2),
                            'explanation': f"Contradictory verdicts: {v1} vs {v2}"
                        })
        
        return contradictions
    
    def _calculate_relationship_confidence(
        self,
        direct_evidence: List[RelationshipEvidence],
        relationship_evidence: List[RelationshipEvidence],
        regulatory_cascades: List[Dict],
        contradictions: List[Dict]
    ) -> float:
        """Calculate confidence based on relationship analysis"""
        
        # Base confidence from direct evidence
        direct_confidence = 0.0
        if direct_evidence:
            direct_scores = [ev.credibility_score for ev in direct_evidence]
            direct_confidence = np.mean(direct_scores) * 0.6
        
        # Relationship confidence
        relationship_confidence = 0.0
        if relationship_evidence:
            # Weight by strength and path length
            weighted_scores = []
            for ev in relationship_evidence:
                weight = ev.strength / (ev.path_length + 1)  # Penalize longer paths
                weighted_scores.append(ev.credibility_score * weight)
            
            relationship_confidence = np.mean(weighted_scores) * 0.3
        
        # Regulatory cascade boost
        cascade_boost = 0.0
        if regulatory_cascades:
            cascade_strengths = [c['strength'] for c in regulatory_cascades]
            cascade_boost = np.mean(cascade_strengths) * 0.1
        
        # Contradiction penalty
        contradiction_penalty = len(contradictions) * 0.05
        
        final_confidence = min(1.0, max(0.0, 
            direct_confidence + relationship_confidence + cascade_boost - contradiction_penalty
        ))
        
        return final_confidence
    
    def _generate_verdict(
        self,
        direct_evidence: List[RelationshipEvidence],
        relationship_evidence: List[RelationshipEvidence],
        contradictions: List[Dict],
        confidence: float,
        threshold: float
    ) -> str:
        """Generate verdict based on evidence analysis"""
        
        if contradictions:
            return "CONTRADICTED"
        
        if confidence >= threshold:
            return "SUPPORTED"
        elif confidence >= threshold * 0.7:
            return "PARTIALLY_SUPPORTED"
        else:
            return "INSUFFICIENT_EVIDENCE"
    
    def _identify_unique_insights(
        self,
        relationship_evidence: List[RelationshipEvidence],
        regulatory_cascades: List[Dict],
        vector_evidence: List[Dict]
    ) -> List[str]:
        """Identify insights only available through relationship analysis"""
        
        insights = []
        
        # Multi-hop insights
        multi_hop_evidence = [ev for ev in relationship_evidence if ev.path_length > 1]
        if multi_hop_evidence:
            insights.append(
                f"Found {len(multi_hop_evidence)} indirect connections "
                f"through relationship traversal that vector similarity would miss"
            )
        
        # Regulatory cascade insights
        if regulatory_cascades:
            insights.append(
                f"Detected {len(regulatory_cascades)} regulatory cascade patterns "
                f"showing cause-effect chains invisible to vector embeddings"
            )
        
        # Semantic relationship insights
        semantic_types = set()
        for ev in relationship_evidence:
            if ev.semantic_type and ev.semantic_type != "general_relation":
                semantic_types.add(ev.semantic_type)
        
        if semantic_types:
            insights.append(
                f"Leveraged semantic relationships ({', '.join(list(semantic_types)[:3])}) "
                f"that provide context beyond text similarity"
            )
        
        return insights
    
    def _generate_explanation(
        self,
        verdict: str,
        confidence: float,
        direct_evidence: List[RelationshipEvidence],
        relationship_evidence: List[RelationshipEvidence],
        regulatory_cascades: List[Dict],
        unique_insights: List[str]
    ) -> str:
        """Generate comprehensive explanation"""
        
        explanation_parts = [
            f"Verdict: {verdict} (Confidence: {confidence:.2f})"
        ]
        
        if direct_evidence:
            explanation_parts.append(
                f"Found {len(direct_evidence)} direct evidence pieces"
            )
        
        if relationship_evidence:
            explanation_parts.append(
                f"Discovered {len(relationship_evidence)} relationship-based evidence pieces"
            )
        
        if regulatory_cascades:
            explanation_parts.append(
                f"Identified {len(regulatory_cascades)} regulatory cascade patterns"
            )
        
        if unique_insights:
            explanation_parts.append("Unique relationship insights:")
            explanation_parts.extend([f"• {insight}" for insight in unique_insights])
        
        return "\n".join(explanation_parts)
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()