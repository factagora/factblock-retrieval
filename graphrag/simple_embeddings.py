#!/usr/bin/env python3
"""
Simple FactBlock Embeddings Module

Basic text similarity for FactBlock content without heavy ML dependencies.
This provides a working foundation that can be upgraded to sentence-transformers later.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from collections import Counter

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class SimpleFactBlockEmbeddings:
    """Simple embeddings using TF-IDF-like similarity for FactBlock content"""
    
    def __init__(self):
        """Initialize simple embeddings"""
        self.vocabulary = set()
        self.factblock_vectors = []
        self.factblocks = []
        logger.info("✅ Simple embeddings initialized")
    
    def load_factblocks_from_neo4j_data(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load FactBlocks from the enhanced knowledge graph dataset"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            factblocks = data.get('factblocks', [])
            logger.info(f"📥 Loaded {len(factblocks)} FactBlocks from dataset")
            return factblocks
            
        except Exception as e:
            logger.error(f"❌ Failed to load FactBlocks: {e}")
            return []
    
    def prepare_factblock_text(self, factblock: Dict[str, Any]) -> str:
        """Prepare FactBlock text content for processing"""
        parts = []
        
        # Core content
        if factblock.get('claim'):
            parts.append(factblock['claim'])
        
        if factblock.get('evidence'):
            parts.append(factblock['evidence'])
        
        if factblock.get('summary'):
            parts.append(factblock['summary'])
        
        # Investment context
        financial_metadata = factblock.get('financial_metadata', {})
        market_impact = financial_metadata.get('market_impact', {})
        
        if market_impact.get('affected_sectors'):
            sectors = ' '.join(market_impact['affected_sectors'])
            parts.append(sectors)
        
        if market_impact.get('impact_level'):
            parts.append(market_impact['impact_level'])
        
        # Investment themes
        investment_themes = financial_metadata.get('investment_themes', [])
        if investment_themes:
            themes = ' '.join([theme.get('theme_name', '') for theme in investment_themes])
            parts.append(themes)
        
        return ' '.join(parts).lower()
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.lower().split()
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        return tokens
    
    def build_vocabulary(self, factblocks: List[Dict[str, Any]]):
        """Build vocabulary from all FactBlocks"""
        all_tokens = []
        
        for fb in factblocks:
            text = self.prepare_factblock_text(fb)
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        
        # Keep only tokens that appear at least twice
        token_counts = Counter(all_tokens)
        self.vocabulary = {token for token, count in token_counts.items() if count >= 2}
        
        logger.info(f"📚 Built vocabulary with {len(self.vocabulary)} terms")
    
    def vectorize_text(self, text: str) -> Dict[str, float]:
        """Convert text to simple vector (term frequencies)"""
        tokens = self.tokenize(text)
        token_counts = Counter(tokens)
        
        # Create vector with vocabulary terms
        vector = {}
        total_tokens = len(tokens)
        
        for token in self.vocabulary:
            if token in token_counts:
                # Simple TF (term frequency)
                vector[token] = token_counts[token] / total_tokens
            else:
                vector[token] = 0.0
                
        return vector
    
    def build_factblock_vectors(self, factblocks: List[Dict[str, Any]]):
        """Build vectors for all FactBlocks"""
        self.factblocks = factblocks
        self.factblock_vectors = []
        
        for fb in factblocks:
            text = self.prepare_factblock_text(fb)
            vector = self.vectorize_text(text)
            self.factblock_vectors.append(vector)
        
        logger.info(f"🔄 Built vectors for {len(self.factblock_vectors)} FactBlocks")
    
    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors"""
        # Get common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        mag1 = sum(val ** 2 for val in vec1.values()) ** 0.5
        mag2 = sum(val ** 2 for val in vec2.values()) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search for most similar FactBlocks
        
        Args:
            query: Search query string
            k: Number of top results to return
            
        Returns:
            List of (index, similarity_score, factblock) tuples
        """
        if not self.factblock_vectors:
            logger.error("❌ No FactBlock vectors available. Run build_factblock_vectors first.")
            return []
        
        # Vectorize query
        query_vector = self.vectorize_text(query)
        
        # Calculate similarities
        similarities = []
        for i, fb_vector in enumerate(self.factblock_vectors):
            similarity = self.cosine_similarity(query_vector, fb_vector)
            similarities.append((i, similarity, self.factblocks[i]))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def initialize_from_dataset(self, dataset_path: str):
        """Initialize embeddings from enhanced knowledge graph dataset"""
        factblocks = self.load_factblocks_from_neo4j_data(dataset_path)
        if factblocks:
            self.build_vocabulary(factblocks)
            self.build_factblock_vectors(factblocks)
            logger.info("🎉 Embeddings initialized successfully")
            return True
        return False


def demo_simple_embeddings():
    """Demo the simple embeddings functionality"""
    
    print("🧪 Testing Simple FactBlock Embeddings...")
    
    # Initialize embeddings
    embedder = SimpleFactBlockEmbeddings()
    
    # Load from our actual dataset
    dataset_path = "data/processed/enhanced_knowledge_graph_dataset.json"
    
    if not embedder.initialize_from_dataset(dataset_path):
        print("❌ Failed to initialize embeddings")
        return False
    
    # Test queries
    test_queries = [
        "OPEC oil production energy markets",
        "inflation monetary policy interest rates",
        "commodity pressure supply chain",
        "investment themes energy sector"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = embedder.search(query, k=3)
        
        for i, (idx, score, factblock) in enumerate(results):
            print(f"   {i+1}. Score: {score:.3f} - {factblock['claim'][:80]}...")
    
    return True


if __name__ == "__main__":
    demo_simple_embeddings()