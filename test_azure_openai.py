#!/usr/bin/env python3
"""
Test Azure OpenAI client initialization
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Get Azure OpenAI credentials
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

print(f"Azure endpoint: {azure_endpoint}")
print(f"Azure API key present: {bool(azure_api_key)}")
print(f"Azure deployment: {azure_deployment}")

try:
    print("\nTesting Azure OpenAI client initialization...")
    
    # Try different initialization approaches
    print("Approach 1: Basic initialization")
    client = AzureOpenAI(
        api_key=azure_api_key,
        api_version="2024-02-15-preview",
        azure_endpoint=azure_endpoint
    )
    print("✓ Basic initialization successful")
    
    print("\nApproach 2: Test with minimal request")
    response = client.chat.completions.create(
        model=azure_deployment,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5,
        temperature=0
    )
    print("✓ Test request successful")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Try with different API version
    try:
        print("\nTrying with different API version...")
        client = AzureOpenAI(
            api_key=azure_api_key,
            api_version="2023-12-01-preview",
            azure_endpoint=azure_endpoint
        )
        print("✓ Alternative API version successful")
        
        response = client.chat.completions.create(
            model=azure_deployment,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
            temperature=0
        )
        print("✓ Alternative API version test successful")
        print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e2:
        print(f"❌ Alternative API version also failed: {e2}")