#!/usr/bin/env python3
"""
GraphRAG Fact Check Demo - Direct Integration
Demonstrates enhanced fact-checking using compliance-specific knowledge retrieval
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

# Set environment
os.environ['NEO4J_PASSWORD'] = 'password'

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.retrieval import RetrievalModule
from src.config import load_config

class GraphRAGFactChecker:
    """GraphRAG-powered fact checker for compliance domain"""
    
    def __init__(self):
        self.retrieval_module = None
        self.initialize()
    
    def initialize(self):
        """Initialize the retrieval system"""
        try:
            config = load_config()
            self.retrieval_module = RetrievalModule('graphrag')
            self.retrieval_module.initialize(config.to_dict())
            print("✓ GraphRAG retrieval system initialized")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize retrieval system: {e}")
            return False
    
    def extract_claims(self, text: str) -> List[str]:
        """Extract potential factual claims from text"""
        # Simple sentence splitting for demo
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
    
    def retrieve_evidence(self, claim: str, category: str = None, max_docs: int = 5) -> List[Dict[str, Any]]:
        """Retrieve compliance evidence for a claim"""
        if not self.retrieval_module:
            return []
        
        try:
            # Extract key terms from claim for better matching
            import re
            key_terms = []
            
            # Extract regulation names
            if 'gdpr' in claim.lower() or 'general data protection' in claim.lower():
                key_terms.append('GDPR')
            if 'sarbanes' in claim.lower() or 'sox' in claim.lower():
                key_terms.append('Sarbanes-Oxley')
            if 'data protection' in claim.lower() or 'privacy' in claim.lower():
                key_terms.extend(['data protection', 'privacy'])
            if 'financial' in claim.lower() or 'reporting' in claim.lower():
                key_terms.extend(['financial reporting', 'financial'])
            if 'consent' in claim.lower():
                key_terms.append('consent')
            if 'fine' in claim.lower() or 'penalty' in claim.lower():
                key_terms.extend(['fine', 'penalty'])
            
            # Use the most specific term or the claim itself
            query_text = key_terms[0] if key_terms else claim
            
            print(f"DEBUG: Searching for '{query_text}' (from claim: '{claim[:50]}...')")
            
            filters = {'category': category} if category else None
            results = self.retrieval_module.retrieve(
                query_text=query_text,
                filters=filters,
                limit=max_docs
            )
            
            evidence = []
            for result in results:
                evidence.append({
                    'content': result.content,
                    'source_type': result.source_type,
                    'score': result.score,
                    'category': result.metadata.get('category', 'unknown'),
                    'metadata': result.metadata
                })
            
            return evidence
            
        except Exception as e:
            print(f"Error retrieving evidence: {e}")
            return []
    
    def assess_claim(self, claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess a claim based on retrieved evidence"""
        
        if not evidence:
            return {
                'label': 'Needs Verification',
                'confidence': 0.3,
                'reasoning': 'No relevant compliance evidence found',
                'evidence_count': 0,
                'source_reliability': None
            }
        
        # Calculate evidence quality
        avg_score = sum(doc['score'] for doc in evidence) / len(evidence)
        
        # Weight by source type reliability
        source_weights = {
            'FederalRegulation': 1.0,
            'AgencyGuidance': 0.9,
            'EnforcementAction': 0.8,
            'ComplianceTopic': 0.7
        }
        
        total_weight = 0
        weighted_score = 0
        for doc in evidence:
            weight = source_weights.get(doc['source_type'], 0.6)
            total_weight += weight
            weighted_score += doc['score'] * weight
        
        source_reliability = weighted_score / total_weight if total_weight > 0 else 0.5
        
        # Determine assessment
        if avg_score > 1.5 and source_reliability > 0.7:
            label = 'Likely True'
            confidence = min(0.9, 0.7 + (avg_score - 1.5) * 0.1)
            reasoning = f"Strong support from {len(evidence)} high-quality compliance documents"
        elif avg_score > 0.8:
            label = 'Partially Supported'
            confidence = 0.6 + (avg_score - 0.8) * 0.2
            reasoning = f"Moderate support from {len(evidence)} compliance documents"
        elif evidence:
            label = 'Needs Verification'
            confidence = 0.4 + avg_score * 0.1
            reasoning = f"Limited support from available compliance evidence"
        else:
            label = 'Insufficient Evidence'
            confidence = 0.2
            reasoning = "No relevant compliance documentation found"
        
        return {
            'label': label,
            'confidence': round(confidence, 3),
            'reasoning': reasoning,
            'evidence_count': len(evidence),
            'source_reliability': round(source_reliability, 3),
            'avg_relevance': round(avg_score, 3)
        }
    
    def fact_check(self, text: str, compliance_focus: str = None) -> Dict[str, Any]:
        """Perform comprehensive fact-checking on text"""
        
        start_time = time.time()
        
        # Extract claims
        claims = self.extract_claims(text)
        if not claims:
            claims = [text]  # Treat entire text as one claim
        
        results = []
        all_evidence = []
        
        for claim in claims:
            # Retrieve evidence
            evidence = self.retrieve_evidence(claim, compliance_focus)
            all_evidence.extend(evidence)
            
            # Assess claim
            assessment = self.assess_claim(claim, evidence)
            
            results.append({
                'claim': claim,
                'assessment': assessment,
                'evidence': evidence[:3]  # Show top 3 pieces of evidence
            })
        
        processing_time = time.time() - start_time
        
        # Summary statistics
        evidence_summary = {
            'total_documents': len(all_evidence),
            'avg_relevance': round(sum(doc['score'] for doc in all_evidence) / len(all_evidence), 3) if all_evidence else 0,
            'source_types': list(set(doc['source_type'] for doc in all_evidence)),
            'categories': list(set(doc['category'] for doc in all_evidence))
        }
        
        return {
            'claims_analyzed': len(results),
            'processing_time': round(processing_time, 3),
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'evidence_summary': evidence_summary
        }

def demo_graphrag_fact_checking():
    """Demonstrate GraphRAG fact-checking capabilities"""
    
    print("🔍 GraphRAG Fact-Checking Demo")
    print("=" * 50)
    
    # Initialize fact checker
    checker = GraphRAGFactChecker()
    
    if not checker.retrieval_module:
        print("❌ Cannot proceed - retrieval system not available")
        return
    
    # Test cases
    test_cases = [
        {
            'name': 'GDPR Compliance Claims',
            'text': 'GDPR requires explicit consent for data processing. Companies can be fined up to 4% of annual revenue for violations. Users have the right to be forgotten.',
            'focus': 'data_privacy'
        },
        {
            'name': 'Financial Regulation Claims', 
            'text': 'Sarbanes-Oxley Act requires CEO and CFO certification of financial reports. Public companies must maintain adequate internal controls.',
            'focus': 'financial'
        },
        {
            'name': 'Mixed Claims',
            'text': 'Organizations must comply with data protection regulations and maintain accurate financial reporting standards.',
            'focus': None
        },
        {
            'name': 'Potentially False Claim',
            'text': 'Companies can freely transfer personal data to third countries without any restrictions under GDPR.',
            'focus': 'data_privacy'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"Text: \"{test_case['text']}\"")
        if test_case['focus']:
            print(f"Focus: {test_case['focus']}")
        print("-" * 60)
        
        # Perform fact-checking
        result = checker.fact_check(test_case['text'], test_case['focus'])
        
        print(f"✓ Analysis completed in {result['processing_time']}s")
        print(f"✓ Claims analyzed: {result['claims_analyzed']}")
        print(f"✓ Evidence documents: {result['evidence_summary']['total_documents']}")
        print(f"✓ Average relevance: {result['evidence_summary']['avg_relevance']}")
        print(f"✓ Source types: {', '.join(result['evidence_summary']['source_types'])}")
        
        print("\nClaim Assessments:")
        for j, claim_result in enumerate(result['results'], 1):
            assessment = claim_result['assessment']
            print(f"  {j}. \"{claim_result['claim'][:80]}{'...' if len(claim_result['claim']) > 80 else ''}\"")
            print(f"     → {assessment['label']} (confidence: {assessment['confidence']})")
            print(f"     → {assessment['reasoning']}")
            print(f"     → Evidence: {assessment['evidence_count']} docs, reliability: {assessment['source_reliability']}")
            
            if claim_result['evidence']:
                print("     → Top evidence:")
                for k, evidence in enumerate(claim_result['evidence'], 1):
                    print(f"       {k}. {evidence['source_type']} (score: {evidence['score']:.3f})")
                    print(f"          {evidence['content'][:120]}...")
    
    print(f"\n✅ GraphRAG Fact-Checking Demo Completed!")
    print("\nKey Advantages:")
    print("✓ Compliance-specific evidence retrieval")
    print("✓ Source reliability weighting")
    print("✓ Category-based filtering")
    print("✓ Rich regulatory metadata")
    print("✓ Transparency in evidence sources")

if __name__ == "__main__":
    demo_graphrag_fact_checking()