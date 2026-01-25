#!/usr/bin/env python3
"""
Comprehensive Model Quality Assessment System

Evaluates model responses on multiple quality dimensions:
- Coherence & Grammar
- Length & Conciseness
- Repetition Detection
- Spacing & Formatting
- Completeness
- Relevance
"""

import json
import re
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
from dataclasses import dataclass, asdict
import statistics

# Test cases for evaluation
TEST_CASES = [
    {
        "query": "How do I reset my password?",
        "category": "account_access",
        "expected_length": (150, 400),
        "expected_sentences": (2, 4),
    },
    {
        "query": "I want to cancel my subscription before it renews",
        "category": "subscription_management",
        "expected_length": (200, 500),
        "expected_sentences": (2, 5),
    },
    {
        "query": "My order says it's delivered but I can't find it anywhere",
        "category": "shipping_delivery",
        "expected_length": (200, 500),
        "expected_sentences": (2, 5),
    },
    {
        "query": "The checkout page is stuck loading. What steps should I try?",
        "category": "technical_issue",
        "expected_length": (200, 500),
        "expected_sentences": (2, 5),
    },
    {
        "query": "I was charged twice for my order",
        "category": "billing_payments",
        "expected_length": (150, 400),
        "expected_sentences": (2, 4),
    },
    {
        "query": "What are your shipping options?",
        "category": "general_information",
        "expected_length": (150, 400),
        "expected_sentences": (2, 4),
    },
    {
        "query": "I can't log in to my account",
        "category": "account_access",
        "expected_length": (150, 400),
        "expected_sentences": (2, 4),
    },
    {
        "query": "I need to update my phone number to 555-123-4567",
        "category": "account_management",
        "expected_length": (150, 350),
        "expected_sentences": (2, 4),
    },
    {
        "query": "My payment method was declined for insufficient funds",
        "category": "billing_payments",
        "expected_length": (200, 500),
        "expected_sentences": (2, 5),
    },
    {
        "query": "How do I get started?",
        "category": "product_usage",
        "expected_length": (200, 500),
        "expected_sentences": (2, 5),
    },
]


@dataclass
class QualityMetrics:
    """Quality metrics for a single response."""
    # Basic metrics
    length: int
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    
    # Coherence & Grammar
    has_incomplete_sentences: bool
    has_grammar_errors: bool
    coherence_score: float  # 0-100
    
    # Repetition
    has_word_repetition: bool
    has_phrase_repetition: bool
    repetition_score: float  # 0-100 (higher = less repetition)
    
    # Spacing & Formatting
    has_spacing_issues: bool
    spacing_issues_count: int
    spacing_score: float  # 0-100
    
    # Completeness
    has_abrupt_ending: bool
    completeness_score: float  # 0-100
    
    # Relevance (basic check)
    contains_helpful_keywords: bool
    relevance_score: float  # 0-100
    
    # Overall scores
    overall_quality_score: float  # 0-100
    readability_score: float  # 0-100


def check_spacing_issues(text: str) -> Tuple[bool, int, List[str]]:
    """Check for spacing issues in text."""
    issues = []
    count = 0
    
    # Check for stuck-together words (lowercase followed by uppercase in middle)
    stuck_words = re.findall(r'[a-z][A-Z]', text)
    if stuck_words:
        issues.append(f"Stuck words: {len(stuck_words)} instances")
        count += len(stuck_words)
    
    # Check for very long words (likely stuck together)
    words = text.split()
    long_words = [w for w in words if len(w) > 20]
    if long_words:
        issues.append(f"Very long words: {len(long_words)} instances")
        count += len(long_words)
    
    # Check for missing spaces before punctuation
    missing_spaces = len(re.findall(r'[a-z][.,!?;:]', text))
    if missing_spaces > 0:
        issues.append(f"Missing spaces before punctuation: {missing_spaces}")
        count += missing_spaces
    
    return count > 0, count, issues


def check_repetition(text: str) -> Tuple[bool, bool, float]:
    """Check for word and phrase repetition."""
    words = text.lower().split()
    word_counts = Counter(words)
    
    # Check word repetition (excluding common words)
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 
                    'i', 'you', 'your', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
                    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
                    'this', 'that', 'it', 'we', 'they', 'what', 'when', 'where', 'how', 'why'}
    
    repeated_words = {word: count for word, count in word_counts.items() 
                     if word not in common_words and count > 3}
    has_word_repetition = len(repeated_words) > 0
    
    # Check phrase repetition (3-word sequences)
    has_phrase_repetition = False
    if len(words) > 6:
        for i in range(len(words) - 3):
            phrase = ' '.join(words[i:i+3])
            if phrase in ' '.join(words[i+3:]):
                has_phrase_repetition = True
                break
    
    # Calculate repetition score (0-100, higher = less repetition)
    if len(words) == 0:
        repetition_score = 0
    else:
        unique_ratio = len(set(words)) / len(words)
        repetition_score = min(100, unique_ratio * 100)
        if has_word_repetition:
            repetition_score *= 0.7  # Penalty for word repetition
        if has_phrase_repetition:
            repetition_score *= 0.5  # Penalty for phrase repetition
    
    return has_word_repetition, has_phrase_repetition, repetition_score


def check_coherence(text: str) -> Tuple[bool, bool, float]:
    """Check for coherence and grammar issues."""
    # Check for incomplete sentences
    incomplete_patterns = [
        r'\b(?:based|according|depending|relying|accordingly)\s*$',
        r'\b(?:and|or|but|so|then|also|however|therefore|moreover)\s*$',
        r'\b(?:if|when|where|while|because|since|although|unless)\s*$',
        r'\b(?:the|a|an|this|that|these|those)\s*$',
        r'\b(?:to|for|with|from|by|at|in|on|of)\s*$',
    ]
    
    has_incomplete = any(re.search(pattern, text, re.IGNORECASE) for pattern in incomplete_patterns)
    
    # Check for grammar errors (basic checks)
    grammar_errors = 0
    # Check for double spaces
    grammar_errors += len(re.findall(r'\s{2,}', text))
    # Check for missing spaces after punctuation
    grammar_errors += len(re.findall(r'[.,!?;:][a-zA-Z]', text))
    # Check for common grammar issues
    grammar_errors += len(re.findall(r'\b(?:is|are|was|were)\s+(?:is|are|was|were)\b', text, re.IGNORECASE))
    
    has_grammar_errors = grammar_errors > 2
    
    # Coherence score (0-100)
    coherence_score = 100
    if has_incomplete:
        coherence_score -= 20
    if has_grammar_errors:
        coherence_score -= 15
    if grammar_errors > 5:
        coherence_score -= 10
    
    coherence_score = max(0, coherence_score)
    
    return has_incomplete, has_grammar_errors, coherence_score


def check_completeness(text: str) -> Tuple[bool, float]:
    """Check if response is complete."""
    # Check for abrupt ending
    abrupt_endings = [
        r'\b(?:based|according|depending)\s*$',
        r'\b(?:and|or|but|so|then)\s*$',
        r'\b(?:if|when|where|while|because)\s*$',
    ]
    
    has_abrupt = any(re.search(pattern, text, re.IGNORECASE) for pattern in abrupt_endings)
    
    # Completeness score
    completeness_score = 100
    if has_abrupt:
        completeness_score -= 30
    if not text.strip().endswith(('.', '!', '?')):
        completeness_score -= 10
    if len(text.strip()) < 50:
        completeness_score -= 20
    
    completeness_score = max(0, completeness_score)
    
    return has_abrupt, completeness_score


def check_relevance(text: str, query: str) -> Tuple[bool, float]:
    """Check if response is relevant to the query."""
    # Basic keyword matching
    query_words = set(query.lower().split())
    text_lower = text.lower()
    
    # Check for helpful keywords
    helpful_keywords = ['help', 'assist', 'support', 'guide', 'step', 'process', 'can', 'will', 'should']
    contains_helpful = any(keyword in text_lower for keyword in helpful_keywords)
    
    # Relevance score
    relevance_score = 50  # Base score
    if contains_helpful:
        relevance_score += 20
    
    # Check if response addresses query keywords
    matching_words = sum(1 for word in query_words if word in text_lower and len(word) > 3)
    if matching_words > 0:
        relevance_score += min(30, matching_words * 10)
    
    relevance_score = min(100, relevance_score)
    
    return contains_helpful, relevance_score


def calculate_readability(text: str) -> float:
    """Calculate readability score (0-100)."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0:
        return 0
    
    words = text.split()
    if len(words) == 0:
        return 0
    
    # Average sentence length
    avg_sentence_length = len(words) / len(sentences)
    
    # Readability score (shorter sentences = better readability)
    if avg_sentence_length <= 15:
        readability = 100
    elif avg_sentence_length <= 20:
        readability = 90
    elif avg_sentence_length <= 25:
        readability = 80
    elif avg_sentence_length <= 30:
        readability = 70
    else:
        readability = max(50, 100 - (avg_sentence_length - 30) * 2)
    
    return readability


def evaluate_response(text: str, query: str, expected_length: Tuple[int, int], expected_sentences: Tuple[int, int]) -> QualityMetrics:
    """Evaluate a single response."""
    if not text or not text.strip():
        # Return zero scores for empty response
        return QualityMetrics(
            length=0, word_count=0, sentence_count=0, avg_sentence_length=0,
            has_incomplete_sentences=True, has_grammar_errors=True, coherence_score=0,
            has_word_repetition=True, has_phrase_repetition=True, repetition_score=0,
            has_spacing_issues=True, spacing_issues_count=999, spacing_score=0,
            has_abrupt_ending=True, completeness_score=0,
            contains_helpful_keywords=False, relevance_score=0,
            overall_quality_score=0, readability_score=0
        )
    
    # Basic metrics
    length = len(text)
    words = text.split()
    word_count = len(words)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences) if sentences else 1
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Check spacing
    has_spacing, spacing_count, spacing_issues = check_spacing_issues(text)
    spacing_score = max(0, 100 - (spacing_count * 5))  # Penalty per issue
    
    # Check repetition
    has_word_rep, has_phrase_rep, repetition_score = check_repetition(text)
    
    # Check coherence
    has_incomplete, has_grammar, coherence_score = check_coherence(text)
    
    # Check completeness
    has_abrupt, completeness_score = check_completeness(text)
    
    # Check relevance
    contains_helpful, relevance_score = check_relevance(text, query)
    
    # Readability
    readability_score = calculate_readability(text)
    
    # Length appropriateness score
    min_len, max_len = expected_length
    if min_len <= length <= max_len:
        length_score = 100
    elif length < min_len:
        length_score = max(0, (length / min_len) * 100)
    else:
        length_score = max(0, 100 - ((length - max_len) / max_len) * 50)
    
    # Sentence count appropriateness
    min_sent, max_sent = expected_sentences
    if min_sent <= sentence_count <= max_sent:
        sentence_score = 100
    elif sentence_count < min_sent:
        sentence_score = (sentence_count / min_sent) * 100
    else:
        sentence_score = max(0, 100 - ((sentence_count - max_sent) / max_sent) * 50)
    
    # Overall quality score (weighted average)
    overall_quality_score = (
        coherence_score * 0.25 +
        repetition_score * 0.20 +
        spacing_score * 0.15 +
        completeness_score * 0.15 +
        relevance_score * 0.15 +
        length_score * 0.05 +
        sentence_score * 0.05
    )
    
    return QualityMetrics(
        length=length,
        word_count=word_count,
        sentence_count=sentence_count,
        avg_sentence_length=avg_sentence_length,
        has_incomplete_sentences=has_incomplete,
        has_grammar_errors=has_grammar,
        coherence_score=coherence_score,
        has_word_repetition=has_word_rep,
        has_phrase_repetition=has_phrase_rep,
        repetition_score=repetition_score,
        has_spacing_issues=has_spacing,
        spacing_issues_count=spacing_count,
        spacing_score=spacing_score,
        has_abrupt_ending=has_abrupt,
        completeness_score=completeness_score,
        contains_helpful_keywords=contains_helpful,
        relevance_score=relevance_score,
        overall_quality_score=overall_quality_score,
        readability_score=readability_score
    )


def test_model(api_url: str, test_cases: List[Dict]) -> List[Dict[str, Any]]:
    """Test a model with all test cases."""
    results = []
    
    print(f"\nTesting model at: {api_url}")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        print(f"\n[{i}/{len(test_cases)}] Testing: {query}")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{api_url}/generate",
                json={"messages": [{"role": "user", "content": query}]},
                timeout=90
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("content", "")
                
                # Evaluate response
                metrics = evaluate_response(
                    content,
                    query,
                    test_case["expected_length"],
                    test_case["expected_sentences"]
                )
                
                result = {
                    "query": query,
                    "category": test_case["category"],
                    "response": content,
                    "response_time": elapsed,
                    "metrics": asdict(metrics),
                    "status": "success"
                }
                
                print(f"  ✓ Response received ({elapsed:.1f}s)")
                print(f"  Quality Score: {metrics.overall_quality_score:.1f}/100")
                print(f"  Length: {metrics.length} chars, {metrics.word_count} words")
                
            else:
                result = {
                    "query": query,
                    "category": test_case["category"],
                    "response": "",
                    "response_time": elapsed,
                    "metrics": None,
                    "status": f"error_{response.status_code}",
                    "error": response.text
                }
                print(f"  ✗ Error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            result = {
                "query": query,
                "category": test_case["category"],
                "response": "",
                "response_time": 90,
                "metrics": None,
                "status": "timeout",
                "error": "Request timed out"
            }
            print(f"  ✗ Timeout")
            
        except Exception as e:
            result = {
                "query": query,
                "category": test_case["category"],
                "response": "",
                "response_time": 0,
                "metrics": None,
                "status": "error",
                "error": str(e)
            }
            print(f"  ✗ Error: {e}")
        
        results.append(result)
        time.sleep(1)  # Small delay between requests
    
    return results


def generate_report(results: List[Dict], model_name: str) -> Dict[str, Any]:
    """Generate a comprehensive quality report."""
    successful_results = [r for r in results if r["status"] == "success" and r["metrics"]]
    
    if not successful_results:
        return {
            "model_name": model_name,
            "total_tests": len(results),
            "successful_tests": 0,
            "error": "No successful responses to evaluate"
        }
    
    # Calculate aggregate metrics
    metrics_list = [r["metrics"] for r in successful_results]
    
    avg_quality = statistics.mean([m["overall_quality_score"] for m in metrics_list])
    avg_coherence = statistics.mean([m["coherence_score"] for m in metrics_list])
    avg_repetition = statistics.mean([m["repetition_score"] for m in metrics_list])
    avg_spacing = statistics.mean([m["spacing_score"] for m in metrics_list])
    avg_completeness = statistics.mean([m["completeness_score"] for m in metrics_list])
    avg_relevance = statistics.mean([m["relevance_score"] for m in metrics_list])
    avg_readability = statistics.mean([m["readability_score"] for m in metrics_list])
    
    avg_length = statistics.mean([m["length"] for m in metrics_list])
    avg_words = statistics.mean([m["word_count"] for m in metrics_list])
    avg_sentences = statistics.mean([m["sentence_count"] for m in metrics_list])
    avg_response_time = statistics.mean([r["response_time"] for r in successful_results])
    
    # Count issues
    spacing_issues_count = sum(1 for m in metrics_list if m["has_spacing_issues"])
    repetition_issues_count = sum(1 for m in metrics_list if m["has_word_repetition"] or m["has_phrase_repetition"])
    incomplete_issues_count = sum(1 for m in metrics_list if m["has_incomplete_sentences"])
    grammar_issues_count = sum(1 for m in metrics_list if m["has_grammar_errors"])
    
    # Grade assignment
    if avg_quality >= 90:
        grade = "A+"
    elif avg_quality >= 80:
        grade = "A"
    elif avg_quality >= 70:
        grade = "B"
    elif avg_quality >= 60:
        grade = "C"
    elif avg_quality >= 50:
        grade = "D"
    else:
        grade = "F"
    
    return {
        "model_name": model_name,
        "total_tests": len(results),
        "successful_tests": len(successful_results),
        "failed_tests": len(results) - len(successful_results),
        "average_response_time": avg_response_time,
        "scores": {
            "overall_quality": round(avg_quality, 2),
            "coherence": round(avg_coherence, 2),
            "repetition": round(avg_repetition, 2),
            "spacing": round(avg_spacing, 2),
            "completeness": round(avg_completeness, 2),
            "relevance": round(avg_relevance, 2),
            "readability": round(avg_readability, 2),
        },
        "metrics": {
            "average_length": round(avg_length, 0),
            "average_words": round(avg_words, 0),
            "average_sentences": round(avg_sentences, 1),
        },
        "issues": {
            "spacing_issues": spacing_issues_count,
            "repetition_issues": repetition_issues_count,
            "incomplete_sentences": incomplete_issues_count,
            "grammar_errors": grammar_issues_count,
        },
        "grade": grade,
        "detailed_results": results
    }


def compare_models(report1: Dict, report2: Dict) -> Dict[str, Any]:
    """Compare two model reports."""
    comparison = {
        "model1": report1["model_name"],
        "model2": report2["model_name"],
        "model1_grade": report1["grade"],
        "model2_grade": report2["grade"],
        "model1_score": report1["scores"]["overall_quality"],
        "model2_score": report2["scores"]["overall_quality"],
        "winner": report1["model_name"] if report1["scores"]["overall_quality"] > report2["scores"]["overall_quality"] else report2["model_name"],
        "score_difference": abs(report1["scores"]["overall_quality"] - report2["scores"]["overall_quality"]),
        "comparison_by_metric": {}
    }
    
    # Compare each metric
    for metric in ["coherence", "repetition", "spacing", "completeness", "relevance", "readability"]:
        score1 = report1["scores"][metric]
        score2 = report2["scores"][metric]
        comparison["comparison_by_metric"][metric] = {
            "model1": round(score1, 2),
            "model2": round(score2, 2),
            "winner": report1["model_name"] if score1 > score2 else report2["model_name"],
            "difference": round(abs(score1 - score2), 2)
        }
    
    return comparison


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model quality")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                       help="API URL (default: http://localhost:8000)")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name (e.g., 'Mistral-7B-Instruct' or 'Llama-3-8B-Instruct')")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for report (default: quality_report_{model_name}.json)")
    parser.add_argument("--compare-with", type=str, default=None,
                       help="Path to another model's report JSON to compare")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Model Quality Assessment System")
    print("=" * 80)
    print(f"\nModel: {args.model_name}")
    print(f"API URL: {args.api_url}")
    print(f"Test Cases: {len(TEST_CASES)}")
    
    # Test the model
    results = test_model(args.api_url, TEST_CASES)
    
    # Generate report
    report = generate_report(results, args.model_name)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Quality Assessment Summary")
    print("=" * 80)
    print(f"\nModel: {report['model_name']}")
    print(f"Grade: {report['grade']}")
    print(f"Overall Quality Score: {report['scores']['overall_quality']:.1f}/100")
    print(f"\nSuccessful Tests: {report['successful_tests']}/{report['total_tests']}")
    print(f"Average Response Time: {report['average_response_time']:.1f}s")
    print(f"\nScores by Category:")
    print(f"  Coherence: {report['scores']['coherence']:.1f}/100")
    print(f"  Repetition: {report['scores']['repetition']:.1f}/100")
    print(f"  Spacing: {report['scores']['spacing']:.1f}/100")
    print(f"  Completeness: {report['scores']['completeness']:.1f}/100")
    print(f"  Relevance: {report['scores']['relevance']:.1f}/100")
    print(f"  Readability: {report['scores']['readability']:.1f}/100")
    print(f"\nIssues Found:")
    print(f"  Spacing Issues: {report['issues']['spacing_issues']}")
    print(f"  Repetition Issues: {report['issues']['repetition_issues']}")
    print(f"  Incomplete Sentences: {report['issues']['incomplete_sentences']}")
    print(f"  Grammar Errors: {report['issues']['grammar_errors']}")
    
    # Save report
    output_file = args.output or f"quality_report_{args.model_name.replace(' ', '_').lower()}.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved to: {output_file}")
    
    # Compare if requested
    if args.compare_with:
        with open(args.compare_with, 'r') as f:
            other_report = json.load(f)
        comparison = compare_models(report, other_report)
        
        print("\n" + "=" * 80)
        print("Model Comparison")
        print("=" * 80)
        print(f"\n{comparison['model1']}: {comparison['model1_score']:.1f}/100 (Grade: {comparison['model1_grade']})")
        print(f"{comparison['model2']}: {comparison['model2_score']:.1f}/100 (Grade: {comparison['model2_grade']})")
        print(f"\nWinner: {comparison['winner']} (by {comparison['score_difference']:.1f} points)")
        print(f"\nMetric-by-Metric Comparison:")
        for metric, comp in comparison['comparison_by_metric'].items():
            print(f"  {metric.capitalize()}:")
            print(f"    {comparison['model1']}: {comp['model1']:.1f}")
            print(f"    {comparison['model2']}: {comp['model2']:.1f}")
            print(f"    Winner: {comp['winner']} (+{comp['difference']:.1f})")
        
        comparison_file = f"model_comparison_{comparison['model1'].replace(' ', '_')}_vs_{comparison['model2'].replace(' ', '_')}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n✓ Comparison saved to: {comparison_file}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
