#!/usr/bin/env python3
"""
FactBlock Embeddings Module

Handles embedding generation and management for FactBlock content
using sentence-transformers for high-quality embeddings.
"""

import os
import sys
from typing import List, Dict, Any, Optional
import logging

# Simplified version - will use basic text similarity for now
# Can upgrade to sentence-transformers later when dependencies are ready

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class FactBlockEmbeddings:
    """Manage embeddings for FactBlock content"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with sentence transformer model
        
        Args:
            model_name: Hugging Face model name for embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"📥 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def embed_factblock(self, factblock: Dict[str, Any]) -> np.ndarray:
        """
        Generate embedding for a single FactBlock
        
        Args:
            factblock: FactBlock dictionary with claim, evidence, etc.
            
        Returns:
            numpy array of embeddings
        """
        # Combine claim and evidence for richer embeddings
        text_content = self._prepare_factblock_text(factblock)
        return self.model.encode(text_content)
    
    def embed_factblocks(self, factblocks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple FactBlocks
        
        Args:
            factblocks: List of FactBlock dictionaries
            
        Returns:
            List of numpy arrays containing embeddings
        """
        texts = [self._prepare_factblock_text(fb) for fb in factblocks]
        logger.info(f"🔄 Generating embeddings for {len(texts)} FactBlocks...")
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query
        
        Args:
            query: Search query string
            
        Returns:
            numpy array of query embedding
        """
        return self.model.encode(query)
    
    def _prepare_factblock_text(self, factblock: Dict[str, Any]) -> str:
        """
        Prepare FactBlock text content for embedding
        
        Combines claim, evidence, and investment context for rich embeddings
        """
        parts = []
        
        # Core content
        if factblock.get('claim'):
            parts.append(f"Claim: {factblock['claim']}")
        
        if factblock.get('evidence'):
            parts.append(f"Evidence: {factblock['evidence']}")
        
        if factblock.get('summary'):
            parts.append(f"Summary: {factblock['summary']}")
        
        # Investment context
        financial_metadata = factblock.get('financial_metadata', {})
        market_impact = financial_metadata.get('market_impact', {})
        
        if market_impact.get('affected_sectors'):
            sectors = ', '.join(market_impact['affected_sectors'])
            parts.append(f"Affected sectors: {sectors}")
        
        if market_impact.get('impact_level'):
            parts.append(f"Impact level: {market_impact['impact_level']}")
        
        # Investment themes
        investment_themes = financial_metadata.get('investment_themes', [])
        if investment_themes:
            themes = ', '.join([theme.get('theme_name', '') for theme in investment_themes])
            parts.append(f"Investment themes: {themes}")
        
        return " | ".join(parts)
    
    def similarity_search(self, query_embedding: np.ndarray, 
                         factblock_embeddings: List[np.ndarray], 
                         k: int = 5) -> List[int]:
        """
        Find most similar FactBlocks using cosine similarity
        
        Args:
            query_embedding: Query embedding vector
            factblock_embeddings: List of FactBlock embeddings
            k: Number of top results to return
            
        Returns:
            List of indices of most similar FactBlocks
        """
        similarities = []
        
        for i, fb_embedding in enumerate(factblock_embeddings):
            # Cosine similarity
            similarity = np.dot(query_embedding, fb_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(fb_embedding)
            )
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k indices
        return [idx for idx, _ in similarities[:k]]


def demo_embeddings():
    """Demo the embeddings functionality"""
    
    print("🧪 Testing FactBlock Embeddings...")
    
    # Sample FactBlock data
    sample_factblocks = [
        {
            "claim": "OPEC agreed to reduce oil production",
            "evidence": "Major oil-producing countries agreed to cut daily oil production by 2 million barrels",
            "summary": "OPEC production cuts impact global oil supply",
            "financial_metadata": {
                "market_impact": {
                    "affected_sectors": ["energy", "transportation"],
                    "impact_level": "high"
                },
                "investment_themes": [
                    {"theme_name": "energy_investment"}
                ]
            }
        },
        {
            "claim": "Federal Reserve raises interest rates",
            "evidence": "The Fed increased the federal funds rate by 0.75 basis points",
            "summary": "Monetary policy tightening to combat inflation",
            "financial_metadata": {
                "market_impact": {
                    "affected_sectors": ["financial", "real_estate"],
                    "impact_level": "high"
                }
            }
        }
    ]
    
    # Initialize embeddings
    embedder = FactBlockEmbeddings()
    
    # Generate embeddings
    embeddings = embedder.embed_factblocks(sample_factblocks)
    
    # Test query
    query = "oil production and energy markets"
    query_embedding = embedder.embed_query(query)
    
    # Find similar FactBlocks
    similar_indices = embedder.similarity_search(query_embedding, embeddings, k=2)
    
    print(f"🔍 Query: '{query}'")
    print(f"📊 Most similar FactBlocks:")
    for i, idx in enumerate(similar_indices):
        print(f"   {i+1}. {sample_factblocks[idx]['claim']}")
    
    return True


if __name__ == "__main__":
    demo_embeddings()