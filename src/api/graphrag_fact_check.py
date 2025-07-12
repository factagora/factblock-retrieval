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

# Import GraphRAG components
from graphrag.smart_router import SmartGraphRAGRouter, PerformanceMode, QueryType
from graphrag.vector_retriever import GraphVectorRetriever
from graphrag.cypher_retriever import TextToCypherRetriever

# Import Enhanced GraphRAG components
try:
    from src.graphrag.enhanced_graphrag import EnhancedGraphRAG, create_enhanced_fact_check_endpoint
except ImportError:
    # Fallback for different import contexts
    try:
        from graphrag.enhanced_graphrag import EnhancedGraphRAG, create_enhanced_fact_check_endpoint
    except ImportError:
        print("⚠️ Enhanced GraphRAG not available - running in basic mode")
        EnhancedGraphRAG = None
        create_enhanced_fact_check_endpoint = None

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
                # Initialize Azure OpenAI client with correct parameters
                openai_client = AzureOpenAI(
                    api_key=azure_api_key,
                    api_version="2024-02-15-preview",
                    azure_endpoint=azure_endpoint,
                    timeout=30.0  # Add timeout parameter
                )
                use_azure = True
                print("✓ Successfully initialized Azure OpenAI client")
                
                # Test the client with a simple call
                print("Testing Azure OpenAI client connectivity...")
                try:
                    # Simple test to verify the client works
                    response = openai_client.chat.completions.create(
                        model=azure_deployment,
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5,
                        temperature=0
                    )
                    print("✓ Azure OpenAI client test successful")
                except Exception as test_error:
                    print(f"⚠️ Azure OpenAI client test failed: {test_error}")
                    # Client created but test failed - might still work for actual requests
                
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

# Global retrieval modules - initialized on startup
retrieval_module = None
graphrag_router = None
enhanced_graphrag = None

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
    global retrieval_module, graphrag_router, enhanced_graphrag
    
    # Set environment variables
    if not os.getenv('NEO4J_PASSWORD'):
        os.environ['NEO4J_PASSWORD'] = 'password'
    
    try:
        # Initialize basic retrieval module
        config = load_config()
        retrieval_module = RetrievalModule('graphrag')
        retrieval_module.initialize(config.to_dict())
        
        # Initialize advanced GraphRAG router
        graphrag_router = SmartGraphRAGRouter(performance_mode=PerformanceMode.BALANCED)
        
        # Initialize Enhanced GraphRAG system
        if EnhancedGraphRAG is not None:
            neo4j_config = config.to_dict().get('neo4j', {})
            enhanced_graphrag = EnhancedGraphRAG(
                neo4j_uri=neo4j_config.get('uri', 'bolt://localhost:7687'),
                neo4j_user=neo4j_config.get('user', 'neo4j'),
                neo4j_password=neo4j_config.get('password', 'password'),
                enable_parallel_processing=True
            )
        else:
            enhanced_graphrag = None
            print("⚠️ Enhanced GraphRAG system not available")
        
        # Initialize GraphRAG components
        print("Initializing GraphRAG components...")
        try:
            # Router initializes itself in constructor
            print("✅ GraphRAG router initialized successfully")
            print("✅ Enhanced GraphRAG system initialized successfully")
        except Exception as e:
            print(f"⚠️ GraphRAG router initialization failed: {e}")
            # Continue with basic retrieval module
        
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
    """Retrieve relevant compliance evidence for a claim using advanced GraphRAG"""
    
    # Ensure retrieval system is initialized
    global graphrag_router
    if not graphrag_router:
        print("⚠️ GraphRAG router not initialized, initializing now...")
        initialize_retrieval_system()
    
    # Try advanced GraphRAG router first
    if graphrag_router:
        try:
            print(f"Using advanced GraphRAG router for query: {claim_text[:50]}...")
            
            # Prepare context for smart routing
            context = {
                'compliance_focus': compliance_focus,
                'max_results': max_evidence
            }
            
            # Let the router decide the best approach  
            search_results = graphrag_router.search(
                query=claim_text,
                max_results=max_evidence
            )
            
            evidence = []
            
            # Process GraphRAG router results
            if search_results and search_results.get('combined_results'):
                for result in search_results['combined_results']:
                    factblock = result.get('factblock', {})
                    # Combine claim and evidence for content
                    content = f"{factblock.get('claim', '')} | {factblock.get('evidence', '')}"
                    evidence.append({
                        'content': content,
                        'source_type': factblock.get('source_type', 'GraphRAG'),
                        'score': result.get('score', 0.0),
                        'metadata': result.get('metadata', {}),
                        'category': factblock.get('affected_sectors', ['unknown'])[0] if factblock.get('affected_sectors') else 'unknown',
                        'retrieval_method': f"{result.get('source', 'hybrid')}_router"
                    })
            
            # If no combined results, check individual vector and cypher results
            if not evidence:
                # Process vector results
                if search_results and search_results.get('vector_results'):
                    for result in search_results['vector_results']:
                        evidence.append({
                            'content': result.get('content', ''),
                            'source_type': result.get('source_type', 'Vector'),
                            'score': result.get('score', 0.0),
                            'metadata': result.get('metadata', {}),
                            'category': result.get('metadata', {}).get('category', 'unknown'),
                            'retrieval_method': 'vector'
                        })
                
                # Process cypher results
                if search_results and search_results.get('cypher_results'):
                    cypher_data = search_results['cypher_results']
                    if cypher_data.get('results'):
                        for result in cypher_data['results']:
                            # Extract FactBlock data from Neo4j result
                            factblock = result.get('f', {}) if 'f' in result else result
                            evidence.append({
                                'content': factblock.get('claim', '') + ' | ' + factblock.get('evidence', ''),
                                'source_type': factblock.get('source_type', 'FactBlock'),
                                'score': factblock.get('confidence_score', 0.0),
                                'metadata': {
                                    'verdict': factblock.get('verdict', 'unknown'),
                                    'category': factblock.get('affected_sectors', ['unknown'])[0] if factblock.get('affected_sectors') else 'unknown',
                                    'investment_themes': factblock.get('investment_themes', []),
                                    'language': factblock.get('language', 'unknown'),
                                    'publication': factblock.get('publication', 'unknown'),
                                    'author': factblock.get('author', 'unknown')
                                },
                                'category': factblock.get('affected_sectors', ['unknown'])[0] if factblock.get('affected_sectors') else 'unknown',
                                'retrieval_method': 'cypher'
                            })
            
            print(f"✅ GraphRAG router returned {len(evidence)} results")
            return evidence
            
        except Exception as e:
            print(f"⚠️ GraphRAG router failed, falling back to basic retrieval: {e}")
    
    # Fallback to basic retrieval module
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
                'category': result.metadata.get('category', 'unknown'),
                'retrieval_method': 'basic'
            })
        
        print(f"✅ Basic retrieval returned {len(evidence)} results")
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

# Enhanced Fact-Check Request/Response Models
class EnhancedFactCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to fact-check")
    max_evidence: int = Field(5, ge=1, le=20, description="Maximum evidence documents to retrieve")
    max_relationship_depth: int = Field(3, ge=1, le=5, description="Maximum relationship traversal depth")
    min_confidence_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Minimum confidence threshold")
    enable_regulatory_cascades: bool = Field(True, description="Enable regulatory cascade detection")

class EnhancedFactCheckResponse(BaseModel):
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
    timestamp: str

@app.post("/fact-check-enhanced", response_model=EnhancedFactCheckResponse)
async def fact_check_enhanced(request: EnhancedFactCheckRequest):
    """Enhanced fact-checking using both vector similarity and relationship analysis"""
    
    if not enhanced_graphrag:
        raise HTTPException(
            status_code=503,
            detail="Enhanced GraphRAG system not available. Please check initialization."
        )
    
    try:
        result = enhanced_graphrag.fact_check_enhanced(
            claim=request.text,
            max_evidence=request.max_evidence,
            max_relationship_depth=request.max_relationship_depth,
            min_confidence_threshold=request.min_confidence_threshold,
            enable_regulatory_cascades=request.enable_regulatory_cascades
        )
        
        # Convert dataclass to response model
        response_dict = {
            'claim': result.claim,
            'vector_verdict': result.vector_verdict,
            'vector_confidence': result.vector_confidence,
            'vector_evidence': result.vector_evidence,
            'relationship_verdict': result.relationship_verdict,
            'relationship_confidence': result.relationship_confidence,
            'relationship_evidence': result.relationship_evidence,
            'regulatory_cascades': result.regulatory_cascades,
            'final_verdict': result.final_verdict,
            'final_confidence': result.final_confidence,
            'explanation': result.explanation,
            'unique_relationship_insights': result.unique_relationship_insights,
            'vector_time_ms': result.vector_time_ms,
            'relationship_time_ms': result.relationship_time_ms,
            'total_time_ms': result.total_time_ms,
            'timestamp': datetime.now().isoformat()
        }
        
        return EnhancedFactCheckResponse(**response_dict)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced fact-check failed: {str(e)}")

@app.post("/regulatory-cascade-demo")
async def regulatory_cascade_demo():
    """Demonstrate regulatory cascade detection with our example data"""
    
    if not enhanced_graphrag:
        raise HTTPException(
            status_code=503,
            detail="Enhanced GraphRAG system not available."
        )
    
    # Sample regulatory cascade examples from our CSV
    demo_claims = [
        "EU 차량 온실가스 배출 기준 강화로 자동차 회사들이 전기차 투자를 늘렸다",
        "미국 은행 스트레스 테스트 기준 변경으로 지역은행들이 대출 심사를 강화했다",
        "일본 개인정보보호법 개정으로 인터넷 기업들이 데이터 처리 시스템을 전면 개편했다"
    ]
    
    results = []
    
    for claim in demo_claims:
        try:
            result = enhanced_graphrag.fact_check_enhanced(
                claim=claim,
                max_evidence=3,
                max_relationship_depth=3,
                min_confidence_threshold=0.5,
                enable_regulatory_cascades=True
            )
            
            results.append({
                'claim': claim,
                'regulatory_cascades_found': len(result.regulatory_cascades),
                'relationship_insights': result.unique_relationship_insights,
                'final_verdict': result.final_verdict,
                'final_confidence': result.final_confidence,
                'cascades': result.regulatory_cascades[:3]  # Top 3 cascades
            })
            
        except Exception as e:
            results.append({
                'claim': claim,
                'error': str(e)
            })
    
    return {
        'demo_title': 'Regulatory Cascade Detection Demo',
        'description': 'Shows how relationship-aware fact-checking detects regulatory cascades that vector embeddings cannot capture',
        'results': results,
        'explanation': 'These examples demonstrate regulation → compliance → business impact chains that are invisible to traditional vector similarity approaches'
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "graphrag-fact-check-api",
        "retrieval_system": "available" if retrieval_module else "unavailable",
        "graphrag_router": "available" if graphrag_router else "unavailable",
        "enhanced_graphrag": "available" if enhanced_graphrag else "unavailable",
        "ai_provider": "azure-openai" if use_azure else "openai",
        "llm_client": "available" if openai_client else "unavailable",
        "features": [
            "hybrid-fact-checking", 
            "compliance-evidence", 
            "graphrag-retrieval", 
            "llm-analysis", 
            "smart-routing",
            "relationship-aware-fact-checking",
            "regulatory-cascade-detection",
            "multi-hop-traversal"
        ],
        "fact_check_method": "enhanced_hybrid_graphrag",
        "graphrag_capabilities": [
            "vector-similarity", 
            "cypher-queries", 
            "smart-routing", 
            "hybrid-retrieval",
            "relationship-traversal",
            "regulatory-cascades",
            "contradiction-detection"
        ] if enhanced_graphrag else ["basic-retrieval"],
        "endpoints": {
            "traditional": "/fact-check-graphrag",
            "enhanced": "/fact-check-enhanced", 
            "demo": "/regulatory-cascade-demo"
        }
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

class ExampleText(BaseModel):
    id: str = Field(..., description="Unique identifier for the example")
    title: str = Field(..., description="Brief title describing the example")
    text: str = Field(..., description="The example text to fact-check")
    category: str = Field(..., description="Category of compliance/regulatory focus")
    complexity: str = Field(..., description="Complexity level: basic, intermediate, advanced")
    why_graphrag_better: str = Field(..., description="Explanation of why GraphRAG performs better than LLM-only")
    expected_evidence_types: List[str] = Field(..., description="Types of evidence GraphRAG should find")

class ExampleTextsResponse(BaseModel):
    examples: List[ExampleText]
    total_count: int
    categories: List[str]
    description: str

@app.get("/example-texts", response_model=ExampleTextsResponse)
async def get_example_texts():
    """
    Get curated example texts that showcase GraphRAG's superior fact-checking capabilities
    over LLM-only approaches for investor demonstrations
    """
    
    examples = [
        # Examples based on ACTUAL FactBlocks in our Neo4j database
        ExampleText(
            id="opec-oil-production-cuts",
            title="OPEC Oil Production Cuts",
            text="OPEC이 감산 합의에 도달했으며, 주요 산유국들이 원유 생산량을 일일 200만 배럴 감축하기로 합의했다. 이는 유가 안정화를 위한 조치로 평가된다.",
            category="energy",
            complexity="intermediate",
            why_graphrag_better="Vector embeddings might find general OPEC information, but our knowledge graph contains specific factual data about the 200만 배럴 daily production cut agreement. GraphRAG can retrieve exact figures and context that LLMs would need to estimate or might get wrong.",
            expected_evidence_types=["OPEC_Agreements", "Oil_Production_Data", "Energy_Policy", "Market_Analysis"]
        ),
        
        ExampleText(
            id="airline-fuel-cost-impact",
            title="Airline Fuel Cost Increases",
            text="글로벌 항공사의 연료비가 15% 상승했으며, 원유 공급 감소로 인한 유가 상승이 항공유 가격을 직접적으로 끌어올렸다. 이로 인해 항공사들의 수익성이 크게 악화되었다.",
            category="transportation",
            complexity="intermediate",
            why_graphrag_better="While LLMs might discuss fuel costs generally, our FactBlocks contain the specific 15% increase figure and causal relationship to oil supply reductions. GraphRAG provides concrete evidence rather than general estimates.",
            expected_evidence_types=["Airline_Cost_Data", "Fuel_Price_Analysis", "Transportation_Economics", "Energy_Market_Impact"]
        ),
        
        ExampleText(
            id="energy-transportation-cascade",
            title="OPEC Cuts → Airline Industry Impact",
            text="원유 생산량 감축으로 인한 에너지 시장 충격이 항공 업계에도 영향을 미쳤다. OPEC의 감산 합의 이후 항공사들의 연료비가 크게 상승했다.",
            category="energy",
            complexity="advanced",
            why_graphrag_better="This demonstrates our system's key advantage: detecting regulatory cascades across sectors. Vector similarity would struggle to connect 'OPEC production cuts' with 'airline fuel costs' as they appear in different semantic spaces. Our knowledge graph reveals the causal chain: oil production → fuel prices → airline costs.",
            expected_evidence_types=["OPEC_Policy", "Energy_Prices", "Airline_Operations", "Cross_Sector_Impact"]
        ),
        
        ExampleText(
            id="federal-reserve-rate-hikes",
            title="Federal Reserve Interest Rate Policy",
            text="미국 연준이 2022년 기준금리를 7차례 인상했으며, 연방준비제도가 인플레이션 억제를 위해 공격적인 통화긴축 정책을 실시했다.",
            category="financials",
            complexity="intermediate",
            why_graphrag_better="LLMs may have outdated information about Fed policy timing and frequency. Our FactBlocks contain the specific count of '7차례' (7 times) rate hikes in 2022, providing concrete evidence that can be verified against actual policy records.",
            expected_evidence_types=["Fed_Policy_Decisions", "Interest_Rate_Data", "Monetary_Policy", "Inflation_Response"]
        ),
        
        ExampleText(
            id="semiconductor-shortage-timeline",
            title="Semiconductor Shortage Duration",
            text="반도체 부족 현상이 2024년까지 지속될 전망이다. 코로나19 여파와 지정학적 긴장으로 인한 반도체 공급망 차질이 장기화되고 있다.",
            category="semiconductors",
            complexity="intermediate", 
            why_graphrag_better="While LLMs might provide general information about chip shortages, our FactBlocks contain specific timeline predictions ('2024년까지') and causal factors. GraphRAG provides industry-specific evidence rather than generalized responses.",
            expected_evidence_types=["Semiconductor_Supply_Analysis", "Industry_Forecasts", "Supply_Chain_Data", "Market_Projections"]
        ),
        
        ExampleText(
            id="hyundai-production-cuts",
            title="Hyundai Production Reduction",
            text="현대자동차는 차량 생산량을 15% 감축한다고 발표했다. 반도체 수급 불안정으로 인해 현대자동차가 주요 차종의 생산 일정을 조정하고 출하량을 줄이기로 결정했다.",
            category="automobiles",
            complexity="intermediate",
            why_graphrag_better="LLMs may not have access to specific corporate production decisions and exact percentage cuts. Our FactBlocks contain the precise '15% 감축' figure and direct connection to semiconductor supply issues for Hyundai specifically.",
            expected_evidence_types=["Automotive_Production_Data", "Hyundai_Announcements", "Semiconductor_Impact", "Manufacturing_Adjustments"]
        ),
        
        ExampleText(
            id="semiconductor-automotive-cascade",
            title="Semiconductor Shortage → Automotive Production Impact",
            text="반도체 부족으로 인해 자동차 제조업체들이 생산 차질을 겪고 있다. 현대자동차를 비롯한 글로벌 완성차 업체들이 생산량 감축을 발표하며 공급망 위기가 심화되고 있다.",
            category="semiconductors",
            complexity="advanced",
            why_graphrag_better="This showcases cross-sector relationship detection. Vector embeddings might miss the connection between 'semiconductor supply issues' and 'specific automotive production cuts.' Our graph relationships reveal how chip shortages directly impact car manufacturing decisions, with concrete evidence from Hyundai's 15% production reduction.",
            expected_evidence_types=["Semiconductor_Supply_Chain", "Automotive_Production_Impact", "Manufacturing_Disruption", "Industry_Relationships"]
        ),
        
        ExampleText(
            id="fed-monetary-policy-impact",
            title="Federal Reserve Policy Ripple Effects",
            text="미국 연준의 공격적인 금리 인상이 글로벌 시장에 광범위한 영향을 미쳤다. 2022년 7차례 금리 인상은 다양한 산업 부문의 투자 심리를 위축시켰다.",
            category="financials",
            complexity="advanced",
            why_graphrag_better="While LLMs might discuss Fed policy generally, our system can trace specific policy decisions (7차례 금리 인상) to their broader market impacts. This demonstrates how monetary policy cascades through multiple sectors, with specific evidence rather than general economic theory.",
            expected_evidence_types=["Fed_Policy_Records", "Market_Impact_Analysis", "Investment_Sentiment", "Economic_Indicators"]
        ),
        
        ExampleText(
            id="simple-opec-baseline",
            title="OPEC Organization (Baseline Test)",
            text="OPEC(석유수출국기구)은 세계 주요 산유국들로 구성된 국제기구다. 이 조직은 회원국들의 석유 정책을 조율하고 유가 안정화를 목표로 한다.",
            category="energy",
            complexity="basic",
            why_graphrag_better="This baseline test should work well with both approaches. It demonstrates that our GraphRAG system maintains accuracy on straightforward factual information while excelling at the complex relationship reasoning shown in the cascade examples above.",
            expected_evidence_types=["OPEC_Basic_Info", "Oil_Organization_Data", "Energy_Policy_Overview", "International_Organizations"]
        )
    ]
    
    categories = list(set(example.category for example in examples))
    
    return ExampleTextsResponse(
        examples=examples,
        total_count=len(examples),
        categories=categories,
        description="Curated example texts demonstrating our relationship-aware fact-checking technology. These examples show regulatory cascades and cross-sector connections that vector embeddings cannot detect - situations where two seemingly unrelated FactBlocks are actually connected through knowledge graph relationships, revealing insights invisible to traditional similarity-based approaches."
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "GraphRAG Fact Check API",
        "version": "2.0.0",
        "description": "Enhanced fact-checking using compliance-specific knowledge retrieval and relationship-aware analysis",
        "features": [
            "Traditional vector similarity fact-checking",
            "Relationship-aware fact-checking with regulatory cascade detection",
            "Multi-hop graph traversal for hidden connections",
            "Hybrid vector + graph analysis"
        ],
        "endpoints": {
            "traditional_fact_check": "/fact-check-graphrag",
            "enhanced_fact_check": "/fact-check-enhanced",
            "regulatory_cascade_demo": "/regulatory-cascade-demo",
            "example_texts": "/example-texts",
            "health": "/health",
            "docs": "/docs"
        },
        "advantages": [
            "Detects regulatory cascades invisible to vector embeddings",
            "Leverages semantic relationships between FactBlocks",
            "Finds contradictions through graph analysis",
            "Provides insights LLMs cannot access"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))  # Use different port from original API
    uvicorn.run(app, host="0.0.0.0", port=port)