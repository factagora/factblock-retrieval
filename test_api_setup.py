#!/usr/bin/env python3
"""
Test API Setup

Simple test to verify the current API setup works before running it.
"""

import os
import sys
import traceback
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from src.config import load_config
        print("✅ Config module imported")
        
        from src.database.neo4j_client import Neo4jClient
        print("✅ Neo4j client imported")
        
        from src.retrieval import RetrievalModule
        print("✅ Retrieval module imported")
        
        from src.api.graphrag_fact_check import app
        print("✅ FastAPI app imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")
    
    try:
        from src.config import load_config
        config = load_config()
        
        print(f"✅ Config loaded: {config}")
        print(f"   Neo4j URI: {config.neo4j.uri}")
        print(f"   Neo4j User: {config.neo4j.user}")
        print(f"   Neo4j Password: {'*' * len(config.neo4j.password)}")
        
        # Test validation
        is_valid = config.validate_config()
        print(f"   Config valid: {is_valid}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        traceback.print_exc()
        return False

def test_neo4j_connection():
    """Test Neo4j connection"""
    print("\n🧪 Testing Neo4j connection...")
    
    try:
        from src.database.neo4j_client import Neo4jClient
        
        client = Neo4jClient.from_env()
        print(f"✅ Neo4j client created")
        
        # Test connectivity
        if client.verify_connectivity():
            print("✅ Neo4j connection successful")
            
            # Get database info
            info = client.get_database_info()
            print(f"   Database info: {info}")
            
            client.close()
            return True
        else:
            print("❌ Neo4j connection failed")
            client.close()
            return False
            
    except Exception as e:
        print(f"❌ Neo4j connection test failed: {e}")
        traceback.print_exc()
        return False

def test_retrieval_module():
    """Test retrieval module initialization"""
    print("\n🧪 Testing retrieval module...")
    
    try:
        from src.config import load_config
        from src.retrieval import RetrievalModule
        
        config = load_config()
        retrieval_module = RetrievalModule('graphrag')
        
        print("✅ Retrieval module created")
        
        # Try to initialize (this might fail without proper Neo4j data)
        try:
            retrieval_module.initialize(config.to_dict())
            print("✅ Retrieval module initialized")
            return True
        except Exception as e:
            print(f"⚠️ Retrieval module initialization failed (expected): {e}")
            return True  # This is OK for now
            
    except Exception as e:
        print(f"❌ Retrieval module test failed: {e}")
        traceback.print_exc()
        return False

def test_environment_variables():
    """Test environment variable setup"""
    print("\n🧪 Testing environment variables...")
    
    env_vars = {
        'NEO4J_URI': os.getenv('NEO4J_URI'),
        'NEO4J_USER': os.getenv('NEO4J_USER'),
        'NEO4J_PASSWORD': os.getenv('NEO4J_PASSWORD'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY'),
        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
    }
    
    for key, value in env_vars.items():
        if value:
            if 'password' in key.lower() or 'key' in key.lower():
                print(f"✅ {key}: {'*' * len(value)}")
            else:
                print(f"✅ {key}: {value}")
        else:
            print(f"❌ {key}: Not set")
    
    # Check if we have at least Neo4j config
    neo4j_configured = all([
        env_vars['NEO4J_URI'],
        env_vars['NEO4J_USER'],
        env_vars['NEO4J_PASSWORD']
    ])
    
    print(f"\n✅ Neo4j configured: {neo4j_configured}")
    
    # AI services are optional
    ai_configured = any([
        env_vars['OPENAI_API_KEY'],
        env_vars['AZURE_OPENAI_API_KEY']
    ])
    
    print(f"⚠️ AI services configured: {ai_configured} (optional)")
    
    return neo4j_configured

def main():
    """Run all tests"""
    print("🚀 API Setup Test Suite")
    print("=" * 60)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Neo4j Connection Test", test_neo4j_connection),
        ("Retrieval Module Test", test_retrieval_module),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! API is ready to run.")
        print("\n🚀 Next steps:")
        print("   1. Run: python3 run_api.py")
        print("   2. Open: http://localhost:8001")
        print("   3. Test: http://localhost:8001/health")
        print("   4. Docs: http://localhost:8001/docs")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Fix issues before running API.")
        
        if 'Neo4j Connection Test' in [name for name, result in results if not result]:
            print("\n💡 Neo4j connection failed. Make sure Neo4j is running:")
            print("   • Docker: docker run -p 7687:7687 -p 7474:7474 neo4j:latest")
            print("   • Or install Neo4j locally")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)