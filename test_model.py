#!/usr/bin/env python3
"""
Test script to evaluate the trained model's responses.
Tests various customer support scenarios and checks response quality.
"""

import json
import requests
import time
from typing import Dict, List, Tuple

API_URL = "http://localhost:8000"

# Test cases covering different customer support scenarios
TEST_CASES = [
    {
        "name": "Password Reset",
        "messages": [{"role": "user", "content": "How do I reset my password?"}],
        "expected_keywords": ["password", "reset", "email", "link"],
        "expected_tone": "helpful",
    },
    {
        "name": "Account Access Issue",
        "messages": [{"role": "user", "content": "I can't log into my account"}],
        "expected_keywords": ["account", "login", "help", "verify"],
        "expected_tone": "supportive",
    },
    {
        "name": "Order Status Inquiry",
        "messages": [{"role": "user", "content": "Where is my order? I placed it 3 days ago."}],
        "expected_keywords": ["order", "track", "status", "shipping"],
        "expected_tone": "informative",
    },
    {
        "name": "Refund Request",
        "messages": [{"role": "user", "content": "I want a refund for my purchase"}],
        "expected_keywords": ["refund", "return", "process", "help"],
        "expected_tone": "professional",
    },
    {
        "name": "Technical Issue",
        "messages": [{"role": "user", "content": "The app keeps crashing when I try to checkout"}],
        "expected_keywords": ["issue", "problem", "troubleshoot", "help"],
        "expected_tone": "helpful",
    },
    {
        "name": "Subscription Cancellation",
        "messages": [{"role": "user", "content": "How do I cancel my subscription?"}],
        "expected_keywords": ["cancel", "subscription", "account", "settings"],
        "expected_tone": "respectful",
    },
    {
        "name": "Multi-turn Conversation",
        "messages": [
            {"role": "user", "content": "I'm having trouble with my payment"},
            {"role": "assistant", "content": "I'd be happy to help you with your payment issue. Can you tell me more about what's happening?"},
            {"role": "user", "content": "My card was declined but I have money in my account"},
        ],
        "expected_keywords": ["payment", "card", "declined", "help"],
        "expected_tone": "understanding",
    },
    {
        "name": "Privacy Concern",
        "messages": [{"role": "user", "content": "How do you protect my personal information?"}],
        "expected_keywords": ["privacy", "security", "data", "protect"],
        "expected_tone": "reassuring",
    },
]


def check_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def generate_response(messages: List[Dict], max_new_tokens: int = 200) -> Tuple[str, float]:
    """Generate a response from the model."""
    start_time = time.time()
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "temperature": 0.6,
                "top_p": 0.85,
                "repetition_penalty": 1.5,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        elapsed = time.time() - start_time
        return data.get("content", ""), elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return f"ERROR: {str(e)}", elapsed


def analyze_response(response: str, test_case: Dict) -> Dict:
    """Analyze the quality of a response."""
    analysis = {
        "length": len(response),
        "word_count": len(response.split()),
        "has_expected_keywords": False,
        "keyword_matches": [],
        "is_complete": False,
        "has_helpful_tone": False,
        "issues": [],
    }

    # Check for expected keywords
    response_lower = response.lower()
    for keyword in test_case["expected_keywords"]:
        if keyword.lower() in response_lower:
            analysis["keyword_matches"].append(keyword)

    analysis["has_expected_keywords"] = len(analysis["keyword_matches"]) > 0

    # Check if response is complete (ends with proper punctuation)
    analysis["is_complete"] = response.strip().endswith((".", "!", "?"))

    # Check for helpful indicators
    helpful_phrases = [
        "i can help",
        "i'd be happy",
        "let me",
        "you can",
        "here's how",
        "to do this",
        "follow these",
    ]
    analysis["has_helpful_tone"] = any(phrase in response_lower for phrase in helpful_phrases)

    # Check for issues
    if len(response) < 20:
        analysis["issues"].append("Response too short")
    if len(response) > 500:
        analysis["issues"].append("Response very long (may be rambling)")
    if not analysis["is_complete"]:
        analysis["issues"].append("Response appears incomplete")
    if not analysis["has_expected_keywords"]:
        analysis["issues"].append("Missing expected keywords")
    if "error" in response_lower or "sorry, i" in response_lower:
        analysis["issues"].append("Contains error language")

    return analysis


def run_tests() -> None:
    """Run all test cases and report results."""
    print("=" * 80)
    print("Customer Support Model Test Suite")
    print("=" * 80)
    print()

    # Check API health
    print("Checking API health...")
    if not check_health():
        print("❌ API is not responding. Make sure the server is running on", API_URL)
        return
    print("✓ API is healthy\n")

    results = []
    total_time = 0

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"Test {i}/{len(TEST_CASES)}: {test_case['name']}")
        print("-" * 80)
        print(f"User: {test_case['messages'][-1]['content']}")
        print()

        response, elapsed = generate_response(test_case["messages"])
        total_time += elapsed

        print(f"Response ({elapsed:.2f}s):")
        print(f"  {response}")
        print()

        analysis = analyze_response(response, test_case)
        results.append({
            "test": test_case["name"],
            "response": response,
            "analysis": analysis,
            "elapsed": elapsed,
        })

        # Print analysis
        print("Analysis:")
        print(f"  Length: {analysis['length']} chars, {analysis['word_count']} words")
        print(f"  Keywords matched: {', '.join(analysis['keyword_matches']) if analysis['keyword_matches'] else 'None'}")
        print(f"  Complete: {'✓' if analysis['is_complete'] else '✗'}")
        print(f"  Helpful tone: {'✓' if analysis['has_helpful_tone'] else '✗'}")
        if analysis["issues"]:
            print(f"  Issues: {', '.join(analysis['issues'])}")
        print()

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Total tests: {len(TEST_CASES)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average response time: {total_time/len(TEST_CASES):.2f}s")
    print()

    # Quality metrics
    complete_responses = sum(1 for r in results if r["analysis"]["is_complete"])
    helpful_responses = sum(1 for r in results if r["analysis"]["has_helpful_tone"])
    keyword_matches = sum(1 for r in results if r["analysis"]["has_expected_keywords"])

    print("Quality Metrics:")
    print(f"  Complete responses: {complete_responses}/{len(TEST_CASES)} ({100*complete_responses/len(TEST_CASES):.1f}%)")
    print(f"  Helpful tone: {helpful_responses}/{len(TEST_CASES)} ({100*helpful_responses/len(TEST_CASES):.1f}%)")
    print(f"  Expected keywords: {keyword_matches}/{len(TEST_CASES)} ({100*keyword_matches/len(TEST_CASES):.1f}%)")
    print()

    # Issues summary
    all_issues = []
    for r in results:
        all_issues.extend(r["analysis"]["issues"])

    if all_issues:
        print("Common Issues:")
        from collections import Counter
        issue_counts = Counter(all_issues)
        for issue, count in issue_counts.most_common():
            print(f"  {issue}: {count} occurrence(s)")
    else:
        print("✓ No major issues detected!")

    print()
    print("=" * 80)

    # Save detailed results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to test_results.json")


if __name__ == "__main__":
    run_tests()
