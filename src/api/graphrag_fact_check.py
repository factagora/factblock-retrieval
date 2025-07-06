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
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# Initialize client variables
openai_client = None
use_azure = False

def test_azure_client():
    """Test Azure OpenAI client creation"""
    try:
        print("Testing Azure OpenAI client creation...")
        test_client = AzureOpenAI(
            api_key=azure_api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=azure_endpoint
        )
        print("✓ Azure OpenAI client created successfully")
        return test_client
    except Exception as e:
        print(f"❌ Azure OpenAI client creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def initialize_openai_client():
    """Initialize OpenAI client with proper error handling"""
    global openai_client, use_azure
    
    print(f"Initializing OpenAI client...")
    print(f"Azure endpoint: {azure_endpoint}")
    print(f"Azure API key present: {bool(azure_api_key)}")
    print(f"Azure deployment: {azure_deployment}")
    
    try:
        if azure_endpoint and azure_api_key:
            print("Attempting to initialize Azure OpenAI client...")
            
            # Try to create the client with more specific error handling
            try:
                openai_client = AzureOpenAI(
                    api_key=azure_api_key,
                    api_version="2024-02-15-preview",
                    azure_endpoint=azure_endpoint
                )
                use_azure = True
                print("✓ Successfully initialized Azure OpenAI client")
                
                # Test the client with a simple call
                print("Testing Azure OpenAI client with simple call...")
                # Note: We won't actually make a call during initialization to avoid costs
                
            except ImportError as e:
                print(f"❌ Import error with AzureOpenAI: {e}")
                openai_client = None
                use_azure = False
            except Exception as e:
                print(f"❌ Azure OpenAI initialization failed: {e}")
                print("Falling back to regular OpenAI...")
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai_api_key:
                    openai_client = OpenAI(api_key=openai_api_key)
                    use_azure = False
                    print("✓ Successfully initialized fallback OpenAI client")
                else:
                    print("❌ No fallback OpenAI API key available")
                    openai_client = None
                    use_azure = False
        else:
            print("Azure credentials not available, trying regular OpenAI...")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                openai_client = OpenAI(api_key=openai_api_key)
                use_azure = False
                print("✓ Successfully initialized OpenAI client")
            else:
                print("❌ Warning: No OpenAI API keys found")
                openai_client = None
                
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        import traceback
        traceback.print_exc()
        openai_client = None
        use_azure = False
    
    print(f"Final state: openai_client = {openai_client is not None}, use_azure = {use_azure}")

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
    
    # Hybrid fact-checking results
    graphrag_result: Optional[Dict[str, Any]] = Field(None, description="GraphRAG-based fact-check result")
    llm_result: Optional[Dict[str, Any]] = Field(None, description="LLM-based fact-check result")
    hybrid_confidence: Optional[float] = Field(None, description="Combined confidence from both methods")

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
    hybrid_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of hybrid fact-checking results")

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

def graphrag_fact_check(claim: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stage 1: GraphRAG-based fact-checking using retrieved evidence
    """
    if not evidence:
        return {
            'label': 'Needs Verification',
            'confidence': 0.3,
            'reasoning': 'No relevant compliance evidence found in GraphRAG',
            'evidence_count': 0,
            'avg_score': 0.0
        }
    
    # Calculate evidence-based assessment
    avg_score = sum(doc['score'] for doc in evidence) / len(evidence)
    evidence_count = len(evidence)
    
    # Source reliability weighting
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
    
    reliability = weighted_score / total_weight if total_weight > 0 else 0.5
    
    # GraphRAG logic for fact-checking
    if avg_score >= 0.8 and evidence_count >= 3:
        label = 'Likely True'
        confidence = min(0.9, reliability * 0.9)
        reasoning = f"Strong support from {evidence_count} high-quality compliance documents (avg score: {avg_score:.3f})"
    elif avg_score >= 0.6 and evidence_count >= 2:
        label = 'Likely True'
        confidence = min(0.8, reliability * 0.8)
        reasoning = f"Good support from {evidence_count} compliance documents (avg score: {avg_score:.3f})"
    elif avg_score >= 0.4:
        label = 'Needs Verification'
        confidence = 0.6
        reasoning = f"Partial support from {evidence_count} compliance documents (avg score: {avg_score:.3f})"
    else:
        label = 'Likely False'
        confidence = 0.4
        reasoning = f"Limited or contradictory evidence from {evidence_count} documents (avg score: {avg_score:.3f})"
    
    return {
        'label': label,
        'confidence': confidence,
        'reasoning': reasoning,
        'evidence_count': evidence_count,
        'avg_score': avg_score,
        'source_reliability': reliability
    }

def llm_fact_check(claim: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 2: LLM-based fact-checking using general knowledge
    """
    if not openai_client:
        return {
            'label': 'Needs Verification',
            'confidence': 0.3,
            'reasoning': 'LLM not available for independent fact-checking',
            'method': 'fallback'
        }
    
    system_prompt = """You are an expert fact-checker. Analyze the given claim using your general knowledge and reasoning abilities.

Provide a fact-check assessment with:
1. Label: "Likely True", "Likely False", "Needs Verification", "Neutral"
2. Confidence score (0.0 to 1.0)
3. Detailed reasoning based on your knowledge

Consider:
- "Likely True": Claim aligns with established facts and knowledge
- "Likely False": Claim contradicts established facts or contains errors
- "Needs Verification": Claim requires additional evidence or is uncertain
- "Neutral": Statement doesn't make factual claims

Focus on factual accuracy, logical consistency, and plausibility."""

    user_prompt = f"""Claim to fact-check: "{claim['text']}"

Provide your analysis as JSON:
{{"label": "string", "confidence": 0.85, "reasoning": "detailed explanation based on general knowledge"}}"""

    try:
        if use_azure:
            response = openai_client.chat.completions.create(
                model=azure_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=400
            )
        else:
            response = openai_client.chat.completions.create(
                model="gpt-4" if os.getenv("OPENAI_API_KEY") else "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=400
            )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = content[json_start:json_end]
                llm_result = json.loads(json_str)
                
                return {
                    'label': llm_result.get('label', 'Needs Verification'),
                    'confidence': float(llm_result.get('confidence', 0.5)),
                    'reasoning': llm_result.get('reasoning', 'LLM analysis completed'),
                    'method': 'llm'
                }
        except json.JSONDecodeError:
            pass
            
    except Exception as e:
        print(f"LLM fact-check error: {e}")
    
    # Fallback
    return {
        'label': 'Needs Verification',
        'confidence': 0.3,
        'reasoning': 'LLM analysis failed - unable to process claim',
        'method': 'fallback'
    }

def combine_hybrid_results(graphrag_result: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine GraphRAG and LLM results into a final hybrid assessment
    """
    # Label mapping for consistency
    label_weights = {
        'Likely True': 1.0,
        'Likely False': -1.0,
        'Needs Verification': 0.0,
        'Neutral': 0.0,
        'Compliance Issue': -0.5
    }
    
    graphrag_weight = label_weights.get(graphrag_result['label'], 0.0)
    llm_weight = label_weights.get(llm_result['label'], 0.0)
    
    # Weight GraphRAG higher for compliance-related claims
    graphrag_importance = 0.6
    llm_importance = 0.4
    
    # Calculate combined confidence
    combined_confidence = (
        graphrag_result['confidence'] * graphrag_importance +
        llm_result['confidence'] * llm_importance
    )
    
    # Determine final label based on weighted scores
    combined_weight = graphrag_weight * graphrag_importance + llm_weight * llm_importance
    
    if combined_weight > 0.3:
        final_label = 'Likely True'
    elif combined_weight < -0.3:
        final_label = 'Likely False'
    else:
        final_label = 'Needs Verification'
    
    # Adjust confidence based on agreement
    if graphrag_result['label'] == llm_result['label']:
        # Both methods agree - boost confidence
        combined_confidence = min(0.95, combined_confidence * 1.2)
        agreement = 'high'
    elif (graphrag_weight > 0) == (llm_weight > 0):
        # Same direction but different labels - moderate confidence
        combined_confidence = combined_confidence * 0.9
        agreement = 'moderate'
    else:
        # Disagreement - lower confidence
        combined_confidence = combined_confidence * 0.7
        agreement = 'low'
    
    reasoning = f"Hybrid analysis: GraphRAG says '{graphrag_result['label']}' (confidence: {graphrag_result['confidence']:.2f}), LLM says '{llm_result['label']}' (confidence: {llm_result['confidence']:.2f}). Agreement: {agreement}. {graphrag_result['reasoning']} | {llm_result['reasoning']}"
    
    return {
        'label': final_label,
        'confidence': combined_confidence,
        'reasoning': reasoning,
        'agreement': agreement,
        'graphrag_weight': graphrag_importance,
        'llm_weight': llm_importance
    }

def analyze_claim_with_evidence(claim: Dict[str, Any], evidence: List[Dict[str, Any]]) -> GraphRAGFactCheckInstance:
    """
    Hybrid fact-checking: Combine GraphRAG and LLM approaches
    """
    
    # Stage 1: GraphRAG-based fact-checking
    print(f"Stage 1: GraphRAG fact-checking for claim: {claim['text'][:50]}...")
    graphrag_result = graphrag_fact_check(claim, evidence)
    
    # Stage 2: LLM-based fact-checking  
    print(f"Stage 2: LLM fact-checking for claim: {claim['text'][:50]}...")
    llm_result = llm_fact_check(claim)
    
    # Stage 3: Combine results
    print(f"Stage 3: Combining hybrid results...")
    hybrid_result = combine_hybrid_results(graphrag_result, llm_result)
    
    # Calculate source reliability from evidence
    source_reliability = graphrag_result.get('source_reliability', 0.5)
    
    return GraphRAGFactCheckInstance(
        label=hybrid_result['label'],
        text=claim['text'],
        confidence=hybrid_result['confidence'],
        reasoning=hybrid_result['reasoning'],
        start_index=claim.get('start_index'),
        end_index=claim.get('end_index'),
        compliance_evidence=evidence,
        source_reliability=source_reliability,
        
        # Store individual results for transparency
        graphrag_result=graphrag_result,
        llm_result=llm_result,
        hybrid_confidence=hybrid_result['confidence']
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the retrieval system and OpenAI client on startup"""
    # Initialize OpenAI client
    initialize_openai_client()
    
    # Initialize retrieval system
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
        
        # Hybrid summary
        hybrid_summary = {
            "method": "hybrid_graphrag_llm",
            "total_claims_analyzed": len(instances),
            "graphrag_available": bool(retrieval_module),
            "llm_available": bool(openai_client),
            "avg_hybrid_confidence": sum(inst.hybrid_confidence for inst in instances if inst.hybrid_confidence) / len(instances) if instances else 0,
            "agreement_distribution": {
                "high": sum(1 for inst in instances if inst.graphrag_result and inst.llm_result and inst.graphrag_result.get('label') == inst.llm_result.get('label')),
                "moderate": sum(1 for inst in instances if inst.graphrag_result and inst.llm_result and inst.graphrag_result.get('label') != inst.llm_result.get('label')),
                "low": sum(1 for inst in instances if not inst.graphrag_result or not inst.llm_result)
            }
        }
        
        return GraphRAGFactCheckResponse(
            instances=instances,
            total_claims=len(instances),
            processing_time=round(processing_time, 3),
            timestamp=datetime.now().isoformat(),
            evidence_summary=evidence_summary,
            compliance_coverage=compliance_coverage,
            hybrid_summary=hybrid_summary
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
        "llm_client": "available" if openai_client else "unavailable",
        "features": ["hybrid-fact-checking", "compliance-evidence", "graphrag-retrieval", "llm-analysis"],
        "fact_check_method": "hybrid_graphrag_llm"
    }

@app.get("/debug")
async def debug_status():
    """Debug endpoint to check initialization status"""
    return {
        "openai_client_initialized": openai_client is not None,
        "use_azure": use_azure,
        "azure_endpoint": azure_endpoint,
        "azure_api_key_present": bool(azure_api_key),
        "azure_deployment": azure_deployment,
        "retrieval_module_initialized": retrieval_module is not None,
        "environment_variables": {
            "AZURE_OPENAI_ENDPOINT": bool(os.getenv("AZURE_OPENAI_ENDPOINT")),
            "AZURE_OPENAI_API_KEY": bool(os.getenv("AZURE_OPENAI_API_KEY")),
            "AZURE_OPENAI_DEPLOYMENT": bool(os.getenv("AZURE_OPENAI_DEPLOYMENT")),
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY"))
        }
    }

@app.post("/reinitialize-openai")
async def reinitialize_openai():
    """Manually reinitialize OpenAI client"""
    initialize_openai_client()
    return {
        "success": openai_client is not None,
        "use_azure": use_azure,
        "message": "OpenAI client reinitialized" if openai_client else "OpenAI client initialization failed"
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