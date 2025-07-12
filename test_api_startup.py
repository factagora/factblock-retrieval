#!/usr/bin/env python3
"""
Test API startup directly
"""

import os
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment
os.environ['NEO4J_PASSWORD'] = 'password'

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("🚀 Testing API startup...")
print(f"Project root: {project_root}")

try:
    print("Step 1: Importing modules...")
    from src.api.graphrag_fact_check import app
    print("✅ FastAPI app imported successfully")
    
    print("Step 2: Testing app creation...")
    print(f"App type: {type(app)}")
    print(f"App routes: {len(app.routes)}")
    
    print("Step 3: Checking startup events...")
    # The app should have startup events
    print(f"Router: {app.router}")
    
    print("Step 4: Manual startup simulation...")
    # Try to manually trigger startup
    from src.api.graphrag_fact_check import initialize_openai_client, initialize_retrieval_system
    
    print("   Initializing OpenAI client...")
    initialize_openai_client()
    
    print("   Initializing retrieval system...")
    success = initialize_retrieval_system()
    print(f"   Retrieval system initialized: {success}")
    
    print("✅ API startup simulation successful!")
    
    print("\n🚀 Now starting server...")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
    
except Exception as e:
    print(f"❌ Error during startup: {e}")
    traceback.print_exc()
    sys.exit(1)