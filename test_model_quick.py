#!/usr/bin/env python3
"""
Quick test script with shorter timeouts and simpler test cases.
"""

import requests
import time

API_URL = "http://localhost:8000"

# Simple test cases
SIMPLE_TESTS = [
    "How do I reset my password?",
    "I can't log in",
    "Where is my order?",
]

print("=" * 80)
print("Quick Model Test")
print("=" * 80)
print()

# Check health
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        print("✓ API is healthy")
    else:
        print("❌ API health check failed")
        exit(1)
except Exception as e:
    print(f"❌ Cannot connect to API: {e}")
    exit(1)

print()

# Test each query
for i, query in enumerate(SIMPLE_TESTS, 1):
    print(f"Test {i}: {query}")
    print("-" * 80)
    
    start = time.time()
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={
                "messages": [{"role": "user", "content": query}],
                "max_new_tokens": 150,  # Shorter responses
                "temperature": 0.6,
                "top_p": 0.85,
                "repetition_penalty": 1.5,
            },
            timeout=90,  # Longer timeout for 7B model
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("content", "")
            print(f"Response ({elapsed:.1f}s):")
            print(f"  {content}")
            print()
            
            # Quick quality check
            if len(content) < 20:
                print("  ⚠ Response too short")
            elif len(content) > 400:
                print("  ⚠ Response very long")
            elif "error" in content.lower():
                print("  ⚠ Contains error language")
            else:
                print("  ✓ Response looks reasonable")
        else:
            print(f"  ❌ Error: {response.status_code}")
            print(f"  {response.text}")
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        print(f"  ❌ Request timed out after {elapsed:.1f}s")
        print("  The model may be too slow on CPU, or there's an issue")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ Error after {elapsed:.1f}s: {e}")
    
    print()

print("=" * 80)
print("Test complete!")
print("=" * 80)
