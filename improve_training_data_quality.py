#!/usr/bin/env python3
"""
Improve training data quality by:
1. Removing/fixing repetitive examples
2. Shortening overly long responses
3. Fixing spacing issues
4. Creating more concise versions
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).parent
TRAIN_FILE = PROJECT_ROOT / "data" / "splits" / "train.jsonl"
VAL_FILE = PROJECT_ROOT / "data" / "splits" / "val.jsonl"
OUTPUT_TRAIN = PROJECT_ROOT / "data" / "splits" / "train_improved.jsonl"
OUTPUT_VAL = PROJECT_ROOT / "data" / "splits" / "val_improved.jsonl"

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

def has_excessive_repetition(text: str) -> bool:
    """Check if text has excessive word repetition."""
    words = text.lower().split()
    if len(words) < 10:
        return False
    
    word_counts = Counter(words)
    # Check if any word appears more than 3 times (excluding very common words)
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 'i', 'you', 'your', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must'}
    
    for word, count in word_counts.items():
        if word not in common_words and count > 3:
            return True
    
    # Check for repeated phrases (3-word sequences)
    if len(words) > 6:
        for i in range(len(words) - 3):
            phrase = ' '.join(words[i:i+3])
            if phrase in ' '.join(words[i+3:]):
                return True
    
    return False

def shorten_response(text: str, max_chars: int = 400, max_sentences: int = 4) -> str:
    """Shorten a response while keeping it coherent."""
    if len(text) <= max_chars:
        return text
    
    # Split into sentences
    sentences = re.split(r'([.!?]+)', text)
    # Reconstruct sentences properly
    reconstructed = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            reconstructed.append(sentences[i] + sentences[i + 1])
        else:
            reconstructed.append(sentences[i])
    
    sentences = [s.strip() for s in reconstructed if s.strip()]
    
    # If already short enough, return
    if len(sentences) <= max_sentences and len(text) <= max_chars:
        return text
    
    # Take first N sentences that fit within max_chars
    result = []
    current_length = 0
    
    for sent in sentences[:max_sentences]:
        if current_length + len(sent) + 2 <= max_chars:  # +2 for space and period
            result.append(sent)
            current_length += len(sent) + 2
        else:
            break
    
    # If we have at least one sentence, return it
    if result:
        return ' '.join(result).strip()
    
    # Fallback: truncate at word boundary
    words = text.split()
    result_words = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_chars:
            result_words.append(word)
            current_length += len(word) + 1
        else:
            break
    
    if result_words:
        return ' '.join(result_words).strip() + '.'
    
    return text[:max_chars].strip()

def fix_spacing_issues(text: str) -> str:
    """Fix obvious spacing issues (but be careful not to break valid words)."""
    # Only fix obvious cases - don't break valid compound words or proper nouns
    # Fix stuck-together common phrases
    fixes = [
        (r'cancelyour', 'cancel your'),
        (r'youshould', 'you should'),
        (r'youllneed', "you'll need"),
        (r'gointo', 'go into'),
        (r'lookfor', 'look for'),
        (r'clickto', 'click to'),
        (r'findyour', 'find your'),
        (r'selectyour', 'select your'),
        (r'updateyour', 'update your'),
        (r'checkyour', 'check your'),
        (r'enteryour', 'enter your'),
        (r'changeyour', 'change your'),
        (r'ofthe', 'of the'),
        (r'tothe', 'to the'),
        (r'forthe', 'for the'),
        (r'withthe', 'with the'),
        (r'fromthe', 'from the'),
        (r'andthe', 'and the'),
        (r'youand', 'you and'),
        (r'youwill', 'you will'),
        (r'youcan', 'you can'),
        (r'youmay', 'you may'),
        (r'youneed', 'you need'),
        (r'youhave', 'you have'),
        (r'youare', 'you are'),
        (r'itis', 'it is'),
        (r'itwill', 'it will'),
        (r'itcan', 'it can'),
        (r'thatis', 'that is'),
        (r'thisis', 'this is'),
    ]
    
    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Fix lowercase-uppercase boundaries (but not in proper nouns or acronyms)
    # Only fix if it's clearly a word boundary (lowercase word followed by uppercase word)
    text = re.sub(r'([a-z]{2,})([A-Z][a-z]{2,})', r'\1 \2', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def improve_example(example: Dict) -> Dict:
    """Improve a single example."""
    improved = json.loads(json.dumps(example))  # Deep copy
    
    # Get assistant message
    assistant_msg = None
    assistant_idx = None
    for i, msg in enumerate(improved.get("messages", [])):
        if msg.get("role") == "assistant":
            assistant_msg = msg.get("content", "")
            assistant_idx = i
            break
    
    if not assistant_msg or not assistant_idx:
        return improved
    
    # Check for issues
    original = assistant_msg
    fixed = assistant_msg
    
    # Fix spacing issues
    fixed = fix_spacing_issues(fixed)
    
    # Check for excessive repetition
    if has_excessive_repetition(fixed):
        # Try to shorten to remove repetition
        fixed = shorten_response(fixed, max_chars=400, max_sentences=4)
        # If still has repetition, mark for removal
        if has_excessive_repetition(fixed):
            return None  # Mark for removal
    
    # Shorten if too long
    if len(fixed) > 500:
        fixed = shorten_response(fixed, max_chars=400, max_sentences=4)
    
    # Update the message if changed
    if fixed != original:
        improved["messages"][assistant_idx]["content"] = fixed
        # Update metadata
        if "metadata" not in improved:
            improved["metadata"] = {}
        improved["metadata"]["improved"] = True
        improved["metadata"]["original_length"] = len(original)
        improved["metadata"]["improved_length"] = len(fixed)
    
    return improved

def main():
    """Main function to improve training data."""
    print("=" * 80)
    print("Improving Training Data Quality")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading training data...")
    train_examples = load_examples(TRAIN_FILE)
    val_examples = load_examples(VAL_FILE)
    
    print(f"Loaded {len(train_examples)} training examples")
    print(f"Loaded {len(val_examples)} validation examples")
    print()
    
    # Improve examples
    print("Improving examples...")
    improved_train = []
    improved_val = []
    removed_train = 0
    removed_val = 0
    fixed_train = 0
    fixed_val = 0
    
    for example in train_examples:
        improved = improve_example(example)
        if improved is None:
            removed_train += 1
        else:
            if improved.get("metadata", {}).get("improved"):
                fixed_train += 1
            improved_train.append(improved)
    
    for example in val_examples:
        improved = improve_example(example)
        if improved is None:
            removed_val += 1
        else:
            if improved.get("metadata", {}).get("improved"):
                fixed_val += 1
            improved_val.append(improved)
    
    print(f"Training examples:")
    print(f"  Fixed: {fixed_train}")
    print(f"  Removed (excessive repetition): {removed_train}")
    print(f"  Kept: {len(improved_train)}")
    print()
    print(f"Validation examples:")
    print(f"  Fixed: {fixed_val}")
    print(f"  Removed (excessive repetition): {removed_val}")
    print(f"  Kept: {len(improved_val)}")
    print()
    
    # Write improved data
    print("Writing improved data...")
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        for example in improved_train:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    with open(OUTPUT_VAL, 'w', encoding='utf-8') as f:
        for example in improved_val:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Saved to {OUTPUT_TRAIN}")
    print(f"Saved to {OUTPUT_VAL}")
    print()
    
    # Statistics
    print("=" * 80)
    print("Improvement Summary")
    print("=" * 80)
    print(f"Original training examples: {len(train_examples)}")
    print(f"Improved training examples: {len(improved_train)}")
    print(f"Removed: {removed_train} ({removed_train/len(train_examples)*100:.1f}%)")
    print(f"Fixed: {fixed_train} ({fixed_train/len(train_examples)*100:.1f}%)")
    print()
    print(f"Original validation examples: {len(val_examples)}")
    print(f"Improved validation examples: {len(improved_val)}")
    print(f"Removed: {removed_val} ({removed_val/len(val_examples)*100:.1f}%)")
    print(f"Fixed: {fixed_val} ({fixed_val/len(val_examples)*100:.1f}%)")
    print()
    print("=" * 80)
    print("Next steps:")
    print("1. Review the improved data files")
    print("2. Replace train.jsonl and val.jsonl with train_improved.jsonl and val_improved.jsonl")
    print("3. Re-train the model with the improved data")

if __name__ == "__main__":
    main()
