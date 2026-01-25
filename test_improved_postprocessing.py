#!/usr/bin/env python3
"""
Quick test script to evaluate improved post-processing.
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

# Test cases that previously had rambling issues
TEST_CASES = [
    {
        "query": "How do I reset my password?",
        "previous_issue": "Very long, rambling response"
    },
    {
        "query": "I can't log in",
        "previous_issue": "Missing keywords, incomplete"
    },
    {
        "query": "Where is my order?",
        "previous_issue": "Spacing issues, rambling"
    },
    {
        "query": "I want a refund",
        "previous_issue": "Very long, rambling response"
    },
]

print("=" * 80)
print("Testing Improved Post-Processing")
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
    print("\nMake sure the server is running:")
    print("  export MODEL_ID='Qwen/Qwen2.5-7B-Instruct'")
    print("  export ADAPTER_DIR='outputs/run_002_qwen7b'")
    print("  python -m uvicorn inference.server:app --reload")
    exit(1)

print()

# Test each query
results = []
for i, test in enumerate(TEST_CASES, 1):
    print(f"Test {i}/{len(TEST_CASES)}: {test['query']}")
    print("-" * 80)
    print(f"Previous issue: {test['previous_issue']}")
    print()
    
    start = time.time()
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={
                "messages": [{"role": "user", "content": test['query']}],
            },
            timeout=90,
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("content", "")
            word_count = len(content.split())
            char_count = len(content)
            sentence_count = content.count('.') + content.count('!') + content.count('?')
            
            print(f"Response ({elapsed:.1f}s):")
            print(f"  {content}")
            print()
            print(f"Metrics:")
            print(f"  Characters: {char_count}")
            print(f"  Words: {word_count}")
            print(f"  Sentences: {sentence_count}")
            print()
            
            # Quality checks
            issues = []
            improvements = []
            
            if char_count > 600:
                issues.append("Still very long (>600 chars)")
            elif char_count > 400:
                issues.append("Moderately long (400-600 chars)")
            else:
                improvements.append("Good length (<400 chars)")
            
            if sentence_count > 5:
                issues.append(f"Too many sentences ({sentence_count} > 5)")
            else:
                improvements.append(f"Good sentence count ({sentence_count} <= 5)")
            
            # Check for spacing issues
            if any(char.isupper() and prev_char.islower() for char, prev_char in zip(content[1:], content[:-1]) if char.isalpha() and prev_char.isalpha()):
                issues.append("Possible spacing issues detected")
            else:
                improvements.append("No spacing issues")
            
            # Check for repetition
            words = content.lower().split()
            if len(words) > 20:
                word_counts = {}
                for word in words:
                    if len(word) > 3:
                        word_counts[word] = word_counts.get(word, 0) + 1
                max_repeats = max(word_counts.values()) if word_counts else 0
                if max_repeats > 8:
                    issues.append(f"Word repetition detected (word appears {max_repeats} times)")
                else:
                    improvements.append("No excessive repetition")
            
            if issues:
                print("⚠ Issues found:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("✓ No major issues")
            
            if improvements:
                print("✓ Improvements:")
                for improvement in improvements:
                    print(f"  - {improvement}")
            
            results.append({
                "query": test['query'],
                "char_count": char_count,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "issues": issues,
                "improvements": improvements,
            })
        else:
            print(f"  ❌ Error: {response.status_code}")
            print(f"  {response.text}")
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        print(f"  ❌ Request timed out after {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ Error after {elapsed:.1f}s: {e}")
    
    print()

# Summary
print("=" * 80)
print("Summary")
print("=" * 80)
print()

if results:
    avg_chars = sum(r['char_count'] for r in results) / len(results)
    avg_words = sum(r['word_count'] for r in results) / len(results)
    avg_sentences = sum(r['sentence_count'] for r in results) / len(results)
    total_issues = sum(len(r['issues']) for r in results)
    total_improvements = sum(len(r['improvements']) for r in results)
    
    print(f"Average response length: {avg_chars:.0f} chars, {avg_words:.0f} words")
    print(f"Average sentences: {avg_sentences:.1f}")
    print(f"Total issues found: {total_issues}")
    print(f"Total improvements: {total_improvements}")
    print()
    
    if avg_chars < 500 and avg_sentences <= 5:
        print("✓ Post-processing appears to be working well!")
        print("  Responses are shorter and more focused.")
    elif avg_chars < 600:
        print("⚠ Post-processing is helping, but responses could be shorter.")
        print("  Consider further reducing max_chars or sentence limits.")
    else:
        print("❌ Post-processing may need more aggressive settings.")
        print("  Consider retraining with more concise examples.")
else:
    print("No results to summarize.")

print()
print("=" * 80)
