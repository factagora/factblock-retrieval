#!/usr/bin/env python3
"""
Script to run GraphRAG API in foreground with better error handling
"""

import os
import sys

# Set environment
os.environ['NEO4J_PASSWORD'] = 'password'

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Starting GraphRAG Fact Check API...")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:2]}")

try:
    from src.api.graphrag_fact_check import app
    import uvicorn
    
    print("✓ Imports successful")
    print("Starting server on http://localhost:8002")
    
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()