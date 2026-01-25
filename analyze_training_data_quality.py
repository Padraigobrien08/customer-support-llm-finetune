#!/usr/bin/env python3
"""
Analyze training data quality to identify issues before retraining.
Checks for: length, repetition, spacing, clarity, sentence structure, etc.
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).parent
TRAIN_FILE = PROJECT_ROOT / "data" / "splits" / "train.jsonl"
VAL_FILE = PROJECT_ROOT / "data" / "splits" / "val.jsonl"

def load_examples(filepath: Path) -> List[Dict]:
    """Load jsonl examples."""
    examples = []
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    return examples

def get_assistant_message(example: Dict) -> str:
    """Extract assistant message from example."""
    for msg in example.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""

def analyze_response_quality(text: str) -> Dict[str, Any]:
    """Analyze quality of a single response."""
    if not text:
        return {
            "length": 0,
            "word_count": 0,
            "sentence_count": 0,
            "has_repetition": False,
            "has_spacing_issues": False,
            "avg_sentence_length": 0,
            "issues": []
        }
    
    # Basic metrics
    length = len(text)
    words = text.split()
    word_count = len(words)
    
    # Sentence count (simple - count periods, exclamation, question marks)
    sentence_count = len(re.findall(r'[.!?]+', text))
    if sentence_count == 0:
        sentence_count = 1  # At least one sentence
    
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Check for spacing issues
    spacing_issues = []
    # Check for stuck-together words (lowercase followed by uppercase in middle of word)
    if re.search(r'[a-z][A-Z]', text):
        spacing_issues.append("Mixed case in middle of word")
    # Check for very long words (likely stuck together)
    long_words = [w for w in words if len(w) > 20]
    if long_words:
        spacing_issues.append(f"Very long words: {long_words[:3]}")
    
    has_spacing_issues = len(spacing_issues) > 0
    
    # Check for repetition
    word_counts = Counter(words)
    repeated_words = {word: count for word, count in word_counts.items() if count > 3 and len(word) > 3}
    has_repetition = len(repeated_words) > 0
    
    # Check for repeated phrases (3-word sequences)
    repeated_phrases = []
    if len(words) > 6:
        for i in range(len(words) - 3):
            phrase = ' '.join(words[i:i+3])
            if phrase in ' '.join(words[i+3:]):
                repeated_phrases.append(phrase)
                break
    
    # Collect issues
    issues = []
    if length > 600:
        issues.append(f"Very long ({length} chars)")
    if word_count > 100:
        issues.append(f"Too many words ({word_count})")
    if sentence_count > 6:
        issues.append(f"Too many sentences ({sentence_count})")
    if avg_sentence_length > 25:
        issues.append(f"Long sentences (avg {avg_sentence_length:.1f} words)")
    if has_repetition:
        issues.append(f"Word repetition: {list(repeated_words.keys())[:3]}")
    if repeated_phrases:
        issues.append(f"Phrase repetition: {repeated_phrases[0]}")
    if has_spacing_issues:
        issues.extend(spacing_issues)
    
    return {
        "length": length,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "has_repetition": has_repetition,
        "repeated_words": list(repeated_words.keys())[:5],
        "has_spacing_issues": has_spacing_issues,
        "spacing_issues": spacing_issues,
        "repeated_phrases": repeated_phrases,
        "issues": issues
    }

def analyze_training_data():
    """Analyze training data quality."""
    print("=" * 80)
    print("Training Data Quality Analysis")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading training data...")
    train_examples = load_examples(TRAIN_FILE)
    val_examples = load_examples(VAL_FILE)
    
    print(f"Loaded {len(train_examples)} training examples")
    print(f"Loaded {len(val_examples)} validation examples")
    print()
    
    # Analyze all examples
    all_analyses = []
    for example in train_examples + val_examples:
        assistant_msg = get_assistant_message(example)
        analysis = analyze_response_quality(assistant_msg)
        analysis["example"] = example
        all_analyses.append(analysis)
    
    # Calculate statistics
    lengths = [a["length"] for a in all_analyses]
    word_counts = [a["word_count"] for a in all_analyses]
    sentence_counts = [a["sentence_count"] for a in all_analyses]
    avg_sentence_lengths = [a["avg_sentence_length"] for a in all_analyses]
    
    examples_with_issues = [a for a in all_analyses if a["issues"]]
    examples_with_repetition = [a for a in all_analyses if a["has_repetition"]]
    examples_with_spacing = [a for a in all_analyses if a["has_spacing_issues"]]
    long_examples = [a for a in all_analyses if a["length"] > 500]
    very_long_examples = [a for a in all_analyses if a["length"] > 600]
    
    # Print statistics
    print("=" * 80)
    print("Overall Statistics")
    print("=" * 80)
    print(f"Total examples: {len(all_analyses)}")
    print(f"Average response length: {sum(lengths) / len(lengths):.0f} chars")
    print(f"Average word count: {sum(word_counts) / len(word_counts):.0f} words")
    print(f"Average sentences: {sum(sentence_counts) / len(sentence_counts):.1f}")
    print(f"Average sentence length: {sum(avg_sentence_lengths) / len(avg_sentence_lengths):.1f} words")
    print()
    
    print("Length Distribution:")
    print(f"  < 200 chars: {sum(1 for l in lengths if l < 200)} ({sum(1 for l in lengths if l < 200)/len(lengths)*100:.1f}%)")
    print(f"  200-400 chars: {sum(1 for l in lengths if 200 <= l < 400)} ({sum(1 for l in lengths if 200 <= l < 400)/len(lengths)*100:.1f}%)")
    print(f"  400-600 chars: {sum(1 for l in lengths if 400 <= l < 600)} ({sum(1 for l in lengths if 400 <= l < 600)/len(lengths)*100:.1f}%)")
    print(f"  > 600 chars: {sum(1 for l in lengths if l >= 600)} ({sum(1 for l in lengths if l >= 600)/len(lengths)*100:.1f}%)")
    print()
    
    print("Sentence Count Distribution:")
    print(f"  1-2 sentences: {sum(1 for s in sentence_counts if s <= 2)} ({sum(1 for s in sentence_counts if s <= 2)/len(sentence_counts)*100:.1f}%)")
    print(f"  3-4 sentences: {sum(1 for s in sentence_counts if 3 <= s <= 4)} ({sum(1 for s in sentence_counts if 3 <= s <= 4)/len(sentence_counts)*100:.1f}%)")
    print(f"  5-6 sentences: {sum(1 for s in sentence_counts if 5 <= s <= 6)} ({sum(1 for s in sentence_counts if 5 <= s <= 6)/len(sentence_counts)*100:.1f}%)")
    print(f"  > 6 sentences: {sum(1 for s in sentence_counts if s > 6)} ({sum(1 for s in sentence_counts if s > 6)/len(sentence_counts)*100:.1f}%)")
    print()
    
    print("=" * 80)
    print("Issues Found")
    print("=" * 80)
    print(f"Examples with issues: {len(examples_with_issues)} ({len(examples_with_issues)/len(all_analyses)*100:.1f}%)")
    print(f"Examples with repetition: {len(examples_with_repetition)} ({len(examples_with_repetition)/len(all_analyses)*100:.1f}%)")
    print(f"Examples with spacing issues: {len(examples_with_spacing)} ({len(examples_with_spacing)/len(all_analyses)*100:.1f}%)")
    print(f"Long examples (>500 chars): {len(long_examples)} ({len(long_examples)/len(all_analyses)*100:.1f}%)")
    print(f"Very long examples (>600 chars): {len(very_long_examples)} ({len(very_long_examples)/len(all_analyses)*100:.1f}%)")
    print()
    
    # Show worst examples
    print("=" * 80)
    print("Worst Examples (by length)")
    print("=" * 80)
    sorted_by_length = sorted(all_analyses, key=lambda x: x["length"], reverse=True)
    for i, analysis in enumerate(sorted_by_length[:10], 1):
        assistant_msg = get_assistant_message(analysis["example"])
        print(f"\n{i}. Length: {analysis['length']} chars, {analysis['word_count']} words, {analysis['sentence_count']} sentences")
        print(f"   Issues: {', '.join(analysis['issues'][:3])}")
        print(f"   Preview: {assistant_msg[:150]}...")
    
    print()
    print("=" * 80)
    print("Examples with Repetition")
    print("=" * 80)
    for i, analysis in enumerate(examples_with_repetition[:10], 1):
        assistant_msg = get_assistant_message(analysis["example"])
        print(f"\n{i}. Repeated words: {', '.join(analysis['repeated_words'][:3])}")
        print(f"   Preview: {assistant_msg[:200]}...")
    
    print()
    print("=" * 80)
    print("Examples with Spacing Issues")
    print("=" * 80)
    for i, analysis in enumerate(examples_with_spacing[:10], 1):
        assistant_msg = get_assistant_message(analysis["example"])
        print(f"\n{i}. Issues: {', '.join(analysis['spacing_issues'])}")
        print(f"   Preview: {assistant_msg[:200]}...")
    
    # Recommendations
    print()
    print("=" * 80)
    print("Recommendations")
    print("=" * 80)
    
    recommendations = []
    
    if len(very_long_examples) > len(all_analyses) * 0.1:
        recommendations.append(f"⚠️  {len(very_long_examples)} examples (>600 chars) - Consider shortening to 300-500 chars")
    
    if sum(sentence_counts) / len(sentence_counts) > 5:
        recommendations.append(f"⚠️  Average {sum(sentence_counts) / len(sentence_counts):.1f} sentences - Target 2-4 sentences")
    
    if len(examples_with_repetition) > len(all_analyses) * 0.1:
        recommendations.append(f"⚠️  {len(examples_with_repetition)} examples have repetition - Remove repetitive examples")
    
    if len(examples_with_spacing) > 0:
        recommendations.append(f"⚠️  {len(examples_with_spacing)} examples have spacing issues - Fix spacing in training data")
    
    if sum(avg_sentence_lengths) / len(avg_sentence_lengths) > 20:
        recommendations.append(f"⚠️  Average sentence length {sum(avg_sentence_lengths) / len(avg_sentence_lengths):.1f} words - Target 10-15 words")
    
    if not recommendations:
        print("✓ Training data looks good overall!")
    else:
        for rec in recommendations:
            print(rec)
    
    print()
    print("=" * 80)
    
    # Save detailed report
    report = {
        "total_examples": len(all_analyses),
        "statistics": {
            "avg_length": sum(lengths) / len(lengths),
            "avg_words": sum(word_counts) / len(word_counts),
            "avg_sentences": sum(sentence_counts) / len(sentence_counts),
            "avg_sentence_length": sum(avg_sentence_lengths) / len(avg_sentence_lengths),
        },
        "issues": {
            "with_issues": len(examples_with_issues),
            "with_repetition": len(examples_with_repetition),
            "with_spacing": len(examples_with_spacing),
            "long": len(long_examples),
            "very_long": len(very_long_examples),
        },
        "worst_examples": [
            {
                "length": a["length"],
                "word_count": a["word_count"],
                "sentence_count": a["sentence_count"],
                "issues": a["issues"],
                "preview": get_assistant_message(a["example"])[:200]
            }
            for a in sorted_by_length[:20]
        ]
    }
    
    report_file = PROJECT_ROOT / "training_data_quality_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")

if __name__ == "__main__":
    analyze_training_data()
