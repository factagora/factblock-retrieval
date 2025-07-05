#!/usr/bin/env python3
"""
GraphRAG-powered Fact Check API
Enhanced fact-checking using compliance-specific knowledge retrieval
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import sys
import json
import re
import time
from datetime import datetime
import openai

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.retrieval import RetrievalModule
from src.config import load_config

app = FastAPI(title="GraphRAG Fact Check API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client for AI analysis
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

if azure_endpoint and azure_api_key:
    openai.api_type = "azure"
    openai.api_base = azure_endpoint
    openai.api_key = azure_api_key
    openai.api_version = "2024-02-15-preview"
    use_azure = True
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    use_azure = False

# Global retrieval module - initialized on startup
retrieval_module = None

class GraphRAGFactCheckInstance(BaseModel):
    label: str = Field(..., description="Classification label")
    text: str = Field(..., description="Text being fact-checked")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Reasoning for the classification")
    start_index: Optional[int] = Field(None, description="Start position in original text")
    end_index: Optional[int] = Field(None, description="End position in original text")
    compliance_evidence: Optional[List[Dict[str, Any]]] = Field(None, description="Supporting compliance documents")
    source_reliability: Optional[float] = Field(None, description="Reliability score of sources")

class GraphRAGFactCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to fact-check")
    compliance_focus: Optional[List[str]] = Field(
        None, 
        description="Specific compliance areas to focus on (e.g., 'data_privacy', 'financial')"
    )
    max_evidence: Optional[int] = Field(5, ge=1, le=20, description="Maximum evidence documents to retrieve")

class GraphRAGFactCheckResponse(BaseModel):
    instances: List[GraphRAGFactCheckInstance]
    total_claims: int
    processing_time: float
    timestamp: str
    evidence_summary: Dict[str, Any]
    compliance_coverage: Dict[str, float]

def initialize_retrieval_system():
    """Initialize the GraphRAG retrieval system"""
    global retrieval_module
    
    # Set environment variables
    if not os.getenv('NEO4J_PASSWORD'):
        os.environ['NEO4J_PASSWORD'] = 'password'
    
    try:
        config = load_config()
        retrieval_module = RetrievalModule('graphrag')
        retrieval_module.initialize(config.to_dict())
        return True
    except Exception as e:
        print(f"Failed to initialize retrieval system: {e}")
        return False

def extract_claims_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract factual claims from text for analysis"""
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    claims = []
    current_pos = 0
    
    # Keywords that indicate factual claims
    claim_indicators = [
        r'\b(?:according to|research shows|studies indicate|data shows|reports state|statistics show)\b',
        r'\b(?:the company|organization|agency|government)\s+(?:reported|stated|announced|declared)\b',
        r'\b(?:\d+(?:\.\d+)?(?:%|percent|billion|million|thousand))\b',
        r'\b(?:increase|decrease|rise|fall|grew|declined|improved|worsened)\s+(?:by|to|from)\b',
        r'\b(?:comply|compliance|violate|violation|breach|adhere|adherence)\b',
        r'\b(?:regulation|policy|law|rule|guideline|standard|requirement)\b'
    ]
    
    for sentence in sentences:
        start_pos = text.find(sentence, current_pos)
        end_pos = start_pos + len(sentence)
        current_pos = end_pos
        
        # Check if sentence contains claim indicators
        has_claim = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in claim_indicators)
        
        if has_claim or len(sentence.split()) > 10:  # Long sentences likely contain claims
            claims.append({
                'text': sentence,
                'start_index': start_pos,
                'end_index': end_pos,
                'type': 'factual_claim'
            })
    
    return claims

def retrieve_compliance_evidence(claim_text: str, compliance_focus: Optional[List[str]] = None, max_evidence: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant compliance evidence for a claim"""
    if not retrieval_module:
        return []
    
    try:
        # Prepare filters based on compliance focus
        filters = {}
        if compliance_focus:
            # Map compliance focus to categories
            category_mapping = {
                'data_privacy': 'data_privacy',
                'financial': 'financial',
                'healthcare': 'healthcare',
                'environmental': 'environmental',
                'employment': 'employment'
            }
            
            categories = [category_mapping.get(focus, focus) for focus in compliance_focus if focus in category_mapping]
            if categories:
                filters['category'] = categories[0]  # Use first category for now
        
        # Retrieve relevant documents
        results = retrieval_module.retrieve(
            query_text=claim_text,
            filters=filters if filters else None,
            limit=max_evidence
        )
        
        evidence = []
        for result in results:
            evidence.append({
                'content': result.content,
                'source_type': result.source_type,
                'score': result.score,
                'metadata': result.metadata,
                'category': result.metadata.get('category', 'unknown')
            })
        
        return evidence
        
    except Exception as e:
        print(f"Error retrieving compliance evidence: {e}")
        return []

def analyze_claim_with_evidence(claim: Dict[str, Any], evidence: List[Dict[str, Any]]) -> GraphRAGFactCheckInstance:
    """Analyze a claim using retrieved compliance evidence and AI"""
    
    # Prepare evidence summary for AI analysis
    evidence_text = ""
    if evidence:
        evidence_text = "\n".join([
            f"- {doc['source_type']}: {doc['content'][:200]}... (Score: {doc['score']:.3f})"
            for doc in evidence[:3]  # Use top 3 pieces of evidence
        ])
    
    # Calculate source reliability
    source_reliability = None
    if evidence:
        # Weight by score and source type reliability
        source_weights = {
            'FederalRegulation': 1.0,
            'AgencyGuidance': 0.9,
            'EnforcementAction': 0.8,
            'other': 0.6
        }
        
        total_weight = 0
        weighted_score = 0
        for doc in evidence:
            weight = source_weights.get(doc['source_type'], 0.6)
            total_weight += weight
            weighted_score += doc['score'] * weight
        
        source_reliability = weighted_score / total_weight if total_weight > 0 else 0.5
    
    # Prepare AI prompt for analysis
    system_prompt = """You are a compliance fact-checking expert. Analyze the claim against the provided compliance evidence and determine:

1. Label: "Likely True", "Likely False", "Needs Verification", "Neutral", "Compliance Issue"
2. Confidence score (0.0 to 1.0)
3. Detailed reasoning

Consider:
- "Likely True": Claim is supported by compliance evidence
- "Likely False": Claim contradicts compliance evidence
- "Needs Verification": Insufficient evidence to determine
- "Neutral": No factual claims about compliance
- "Compliance Issue": Claim indicates potential compliance violation

Focus on regulatory accuracy, data consistency, and compliance alignment."""

    user_prompt = f"""Claim to analyze: "{claim['text']}"

Relevant Compliance Evidence:
{evidence_text if evidence_text else "No specific compliance evidence found."}

Provide analysis as JSON:
{{"label": "string", "confidence": 0.85, "reasoning": "detailed explanation"}}"""

    try:
        # Call AI for analysis
        if use_azure:
            response = openai.ChatCompletion.create(
                engine=azure_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
        else:
            response = openai.ChatCompletion.create(
                model="gpt-4" if os.getenv("OPENAI_API_KEY") else "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = content[json_start:json_end]
                ai_result = json.loads(json_str)
                
                return GraphRAGFactCheckInstance(
                    label=ai_result.get("label", "Neutral"),
                    text=claim['text'],
                    confidence=float(ai_result.get("confidence", 0.5)),
                    reasoning=ai_result.get("reasoning", "AI analysis completed"),
                    start_index=claim.get('start_index'),
                    end_index=claim.get('end_index'),
                    compliance_evidence=evidence,
                    source_reliability=source_reliability
                )
        except json.JSONDecodeError:
            pass
            
    except Exception as e:
        print(f"AI analysis error: {e}")
    
    # Fallback analysis based on evidence
    if evidence:
        # Simple heuristic based on evidence quality
        avg_score = sum(doc['score'] for doc in evidence) / len(evidence)
        
        if avg_score > 0.8:
            label = "Likely True"
            confidence = 0.8
            reasoning = f"Supported by {len(evidence)} high-quality compliance documents"
        elif avg_score > 0.5:
            label = "Needs Verification"
            confidence = 0.6
            reasoning = f"Partial support from {len(evidence)} compliance documents"
        else:
            label = "Needs Verification"
            confidence = 0.4
            reasoning = f"Limited support from available compliance evidence"
    else:
        label = "Needs Verification"
        confidence = 0.3
        reasoning = "No relevant compliance evidence found"
    
    return GraphRAGFactCheckInstance(
        label=label,
        text=claim['text'],
        confidence=confidence,
        reasoning=reasoning,
        start_index=claim.get('start_index'),
        end_index=claim.get('end_index'),
        compliance_evidence=evidence,
        source_reliability=source_reliability
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the retrieval system on startup"""
    success = initialize_retrieval_system()
    if not success:
        print("Warning: GraphRAG retrieval system failed to initialize")

@app.post("/fact-check-graphrag", response_model=GraphRAGFactCheckResponse)
async def fact_check_with_graphrag(request: GraphRAGFactCheckRequest):
    """Enhanced fact-checking using GraphRAG compliance retrieval"""
    
    if not retrieval_module:
        raise HTTPException(
            status_code=503,
            detail="GraphRAG retrieval system not available. Please check Neo4j connection."
        )
    
    start_time = time.time()
    
    try:
        # Extract claims from text
        claims = extract_claims_from_text(request.text)
        
        if not claims:
            # If no claims detected, analyze entire text as one claim
            claims = [{
                'text': request.text,
                'start_index': 0,
                'end_index': len(request.text),
                'type': 'general'
            }]
        
        instances = []
        all_evidence = []
        compliance_categories = {}
        
        # Analyze each claim
        for claim in claims:
            # Retrieve compliance evidence
            evidence = retrieve_compliance_evidence(
                claim['text'],
                request.compliance_focus,
                request.max_evidence
            )
            
            all_evidence.extend(evidence)
            
            # Track compliance categories
            for doc in evidence:
                category = doc.get('category', 'unknown')
                compliance_categories[category] = compliance_categories.get(category, 0) + 1
            
            # Analyze claim with evidence
            instance = analyze_claim_with_evidence(claim, evidence)
            instances.append(instance)
        
        processing_time = time.time() - start_time
        
        # Calculate compliance coverage
        total_evidence = len(all_evidence)
        compliance_coverage = {}
        if total_evidence > 0:
            for category, count in compliance_categories.items():
                compliance_coverage[category] = count / total_evidence
        
        # Evidence summary
        evidence_summary = {
            "total_documents": total_evidence,
            "avg_relevance_score": sum(doc['score'] for doc in all_evidence) / total_evidence if all_evidence else 0,
            "source_types": list(set(doc['source_type'] for doc in all_evidence)),
            "categories_covered": list(compliance_categories.keys())
        }
        
        return GraphRAGFactCheckResponse(
            instances=instances,
            total_claims=len(instances),
            processing_time=round(processing_time, 3),
            timestamp=datetime.now().isoformat(),
            evidence_summary=evidence_summary,
            compliance_coverage=compliance_coverage
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "graphrag-fact-check-api",
        "retrieval_system": "available" if retrieval_module else "unavailable",
        "ai_provider": "azure-openai" if use_azure else "openai",
        "features": ["compliance-evidence", "graphrag-retrieval", "ai-analysis"]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "GraphRAG Fact Check API",
        "version": "1.0.0",
        "description": "Enhanced fact-checking using compliance-specific knowledge retrieval",
        "endpoints": {
            "fact_check": "/fact-check-graphrag",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))  # Use different port from original API
    uvicorn.run(app, host="0.0.0.0", port=port)