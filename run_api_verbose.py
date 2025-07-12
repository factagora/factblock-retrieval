#!/usr/bin/env python3
"""
Enhanced script to run GraphRAG API with better error handling and logging
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set environment
os.environ['NEO4J_PASSWORD'] = 'password'

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("🚀 Starting Enhanced GraphRAG Fact Check API...")
print(f"📁 Project root: {project_root}")
print(f"🐍 Python path: {sys.path[0]}")

# Check environment variables
print("\n🔧 Environment Check:")
neo4j_uri = os.getenv('NEO4J_URI', 'not set')
print(f"  NEO4J_URI: {neo4j_uri}")

try:
    print("\n📦 Loading modules...")
    from src.api.graphrag_fact_check import app
    import uvicorn
    
    print("✅ Imports successful")
    
    # Check if enhanced endpoints are available
    routes = [route.path for route in app.routes]
    enhanced_available = '/fact-check-enhanced' in routes
    print(f"🔧 Enhanced endpoint available: {enhanced_available}")
    print(f"📋 Available routes: {len(routes)} total")
    
    print(f"\n🌐 Starting server on http://localhost:8002")
    print("📋 Available endpoints:")
    print("  • http://localhost:8002/docs - Interactive API docs")
    print("  • http://localhost:8002/health - Health check")
    print("  • http://localhost:8002/fact-check-enhanced - Enhanced fact-checking")
    print("  • http://localhost:8002/regulatory-cascade-demo - Demo endpoint")
    print("\n⚡ Server starting...")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8002, 
        log_level="info",
        access_log=True
    )
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔍 Checking if required modules are available...")
    
    try:
        import src.api.graphrag_fact_check
        print("✅ Main API module found")
    except ImportError as ie:
        print(f"❌ Cannot import API module: {ie}")
    
    try:
        import uvicorn
        print("✅ Uvicorn found")
    except ImportError:
        print("❌ Uvicorn not found - install with: pip install uvicorn")
        
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n🔍 Debug Information:")
    print(f"  Current working directory: {os.getcwd()}")
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version: {sys.version}")