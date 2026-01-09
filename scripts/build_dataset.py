#!/usr/bin/env python3
"""
Build final dataset from raw data sources.

Merges manual and synthetic cases, validates, deduplicates, and creates train/val/test splits.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Add project root to path for csft package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from csft.io import load_jsonl, save_jsonl


def load_taxonomy() -> dict[str, Any]:
    """Load taxonomy from YAML file."""
    taxonomy_path = project_root / "data" / "taxonomy.yaml"
    if not taxonomy_path.exists():
        return {}
    
    try:
        import yaml
        with open(taxonomy_path, 'r') as f:
            taxonomy = yaml.safe_load(f)
        return taxonomy or {}
    except ImportError:
        print("Warning: PyYAML not installed, skipping taxonomy validation", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Warning: Error loading taxonomy: {e}", file=sys.stderr)
        return {}


def validate_case(case: dict[str, Any], line_num: int, taxonomy: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate a single case against schema and taxonomy.
    
    Args:
        case: Case dictionary
        line_num: Line number for error reporting
        taxonomy: Taxonomy dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required top-level keys
    if "messages" not in case:
        return False, f"Line {line_num}: missing 'messages' field"
    
    if "metadata" not in case:
        return False, f"Line {line_num}: missing 'metadata' field"
    
    # Validate messages
    messages = case["messages"]
    if not isinstance(messages, list):
        return False, f"Line {line_num}: 'messages' must be a list"
    
    if len(messages) == 0:
        return False, f"Line {line_num}: 'messages' list is empty"
    
    # Check for at least one user and one assistant message
    has_user = False
    has_assistant = False
    valid_roles = {"system", "user", "assistant"}
    
    for msg in messages:
        if not isinstance(msg, dict):
            return False, f"Line {line_num}: message must be a dict"
        
        if "role" not in msg or "content" not in msg:
            return False, f"Line {line_num}: message missing 'role' or 'content'"
        
        role = msg["role"]
        if role not in valid_roles:
            return False, f"Line {line_num}: invalid role '{role}'"
        
        if not isinstance(msg["content"], str) or not msg["content"].strip():
            return False, f"Line {line_num}: message content must be non-empty string"
        
        if role == "user":
            has_user = True
        if role == "assistant":
            has_assistant = True
    
    if not has_user:
        return False, f"Line {line_num}: missing 'user' message"
    
    if not has_assistant:
        return False, f"Line {line_num}: missing 'assistant' message"
    
    # Validate metadata
    metadata = case["metadata"]
    if not isinstance(metadata, dict):
        return False, f"Line {line_num}: 'metadata' must be a dict"
    
    # Required metadata keys
    required_keys = {"source", "category", "escalation", "difficulty", "contains_policy_claims", "test_case_id"}
    for key in required_keys:
        if key not in metadata:
            return False, f"Line {line_num}: missing required metadata key '{key}'"
    
    # Validate source
    source = metadata.get("source")
    if source not in ["manual", "synthetic", "zendesk"]:
        return False, f"Line {line_num}: invalid source '{source}'. Must be one of: manual, synthetic, zendesk"
    
    # Validate category
    category = metadata.get("category")
    if taxonomy and "categories" in taxonomy:
        valid_categories = {cat.get("name") for cat in taxonomy["categories"] if isinstance(cat, dict) and "name" in cat}
        if valid_categories and category not in valid_categories:
            return False, f"Line {line_num}: invalid category '{category}'. Valid categories: {sorted(valid_categories)}"
    
    # Validate escalation (boolean)
    escalation = metadata.get("escalation")
    if not isinstance(escalation, bool):
        return False, f"Line {line_num}: 'escalation' must be boolean (true/false)"
    
    # Validate difficulty (1-3)
    difficulty = metadata.get("difficulty")
    if not isinstance(difficulty, int) or difficulty < 1 or difficulty > 3:
        return False, f"Line {line_num}: 'difficulty' must be integer between 1 and 3"
    
    # Validate contains_policy_claims (boolean)
    contains_policy_claims = metadata.get("contains_policy_claims")
    if not isinstance(contains_policy_claims, bool):
        return False, f"Line {line_num}: 'contains_policy_claims' must be boolean (true/false)"
    
    # Validate test_case_id (string)
    test_case_id = metadata.get("test_case_id")
    if not isinstance(test_case_id, str) or not test_case_id.strip():
        return False, f"Line {line_num}: 'test_case_id' must be non-empty string"
    
    return True, ""


def hash_user_message(user_messages: list[str], category: str) -> str:
    """
    Create a stable hash of concatenated user messages + category for deduplication.
    
    Args:
        user_messages: List of user message texts
        category: Category string
        
    Returns:
        Hexadecimal hash string
    """
    # Concatenate all user messages
    combined_user_text = " ".join(user_messages)
    combined = f"{combined_user_text}|{category}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()


def extract_user_messages(messages: list[dict[str, Any]]) -> list[str]:
    """
    Extract all user messages from messages list.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        List of user message contents
    """
    user_messages = []
    for msg in messages:
        if msg.get("role") == "user":
            user_messages.append(msg.get("content", ""))
    return user_messages


def merge_and_deduplicate(
    manual_path: Path,
    synthetic_path: Path,
    allow_missing_synthetic: bool = False,
    taxonomy: dict[str, Any] = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Load, validate, merge, and deduplicate cases from manual and synthetic sources.
    
    Args:
        manual_path: Path to manual cases JSONL
        synthetic_path: Path to synthetic cases JSONL
        allow_missing_synthetic: If True, don't error if synthetic file is missing
        taxonomy: Taxonomy dictionary for validation
        
    Returns:
        Tuple of (deduplicated_cases, validation_errors)
    """
    all_cases = []
    seen_hashes = set()
    validation_errors = []
    
    # Load manual cases
    if manual_path.exists():
        print(f"Loading manual cases from: {manual_path}")
        try:
            manual_cases = load_jsonl(str(manual_path))
            print(f"  Loaded {len(manual_cases)} manual cases")
            
            # Validate each case
            for idx, case in enumerate(manual_cases, 1):
                is_valid, error = validate_case(case, idx, taxonomy or {})
                if not is_valid:
                    validation_errors.append(error)
                else:
                    all_cases.append(case)
        except Exception as e:
            validation_errors.append(f"Error loading manual cases: {e}")
    else:
        print(f"Warning: Manual cases file not found: {manual_path}")
    
    # Load synthetic cases
    if synthetic_path.exists():
        print(f"Loading synthetic cases from: {synthetic_path}")
        try:
            synthetic_cases = load_jsonl(str(synthetic_path))
            print(f"  Loaded {len(synthetic_cases)} synthetic cases")
            
            # Validate each case
            for idx, case in enumerate(synthetic_cases, 1):
                is_valid, error = validate_case(case, idx, taxonomy or {})
                if not is_valid:
                    validation_errors.append(error)
                else:
                    all_cases.append(case)
        except Exception as e:
            validation_errors.append(f"Error loading synthetic cases: {e}")
    else:
        if not allow_missing_synthetic:
            validation_errors.append(f"Error: Synthetic cases file not found: {synthetic_path}")
        else:
            print(f"Warning: Synthetic cases file not found: {synthetic_path} (allowed)")
    
    if validation_errors:
        return [], validation_errors
    
    print(f"\nTotal cases before deduplication: {len(all_cases)}")
    
    # Deduplicate by user messages + category hash
    deduplicated = []
    duplicates = 0
    
    for case in all_cases:
        user_messages = extract_user_messages(case.get("messages", []))
        category = case.get("metadata", {}).get("category", "unknown")
        case_hash = hash_user_message(user_messages, category)
        
        if case_hash not in seen_hashes:
            seen_hashes.add(case_hash)
            deduplicated.append(case)
        else:
            duplicates += 1
    
    print(f"Removed {duplicates} duplicate cases")
    print(f"Total cases after deduplication: {len(deduplicated)}")
    
    return deduplicated, []


def stratify_by_category(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    Group cases by category for stratified splitting.
    
    Args:
        cases: List of case dictionaries
        
    Returns:
        Dictionary mapping category to list of cases
    """
    by_category = defaultdict(list)
    for case in cases:
        category = case.get("metadata", {}).get("category", "unknown")
        by_category[category].append(case)
    return dict(by_category)


def create_splits(
    cases: list[dict[str, Any]],
    train_frac: float,
    val_frac: float,
    seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Create train/val/test splits with stratification by category.
    
    Args:
        cases: List of case dictionaries
        train_frac: Fraction for training set
        val_frac: Fraction for validation set (test gets remainder)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_cases, val_cases, test_cases)
    """
    import random
    
    random.seed(seed)
    
    # Group by category for stratification
    by_category = stratify_by_category(cases)
    
    train_cases = []
    val_cases = []
    test_cases = []
    
    test_frac = 1.0 - train_frac - val_frac
    
    print(f"\nCreating splits (seed={seed}):")
    print(f"  Train: {train_frac:.1%}")
    print(f"  Val: {val_frac:.1%}")
    print(f"  Test: {test_frac:.1%}")
    print()
    
    # Split each category separately
    for category, category_cases in sorted(by_category.items()):
        # Shuffle category cases deterministically
        random.shuffle(category_cases)
        
        n = len(category_cases)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        
        train_cases.extend(category_cases[:n_train])
        val_cases.extend(category_cases[n_train:n_train + n_val])
        test_cases.extend(category_cases[n_train + n_val:])
        
        print(f"  {category}: {n} cases -> train:{n_train}, val:{n_val}, test:{n - n_train - n_val}")
    
    # Shuffle final splits
    random.shuffle(train_cases)
    random.shuffle(val_cases)
    random.shuffle(test_cases)
    
    return train_cases, val_cases, test_cases


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Build dataset from raw sources with train/val/test splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_dataset.py
  python scripts/build_dataset.py --seed 7 --train_frac 0.8 --val_frac 0.1
  python scripts/build_dataset.py --seed 7 --allow_missing_synthetic
        """
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible splits (default: 7)"
    )
    
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction for training set (default: 0.8)"
    )
    
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Fraction for validation set (default: 0.1)"
    )
    
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data",
        help="Output directory base (default: data)"
    )
    
    parser.add_argument(
        "--allow_missing_synthetic",
        action="store_true",
        help="Allow missing synthetic cases file (default: False)"
    )
    
    args = parser.parse_args()
    
    # Validate fractions
    if args.train_frac <= 0 or args.val_frac <= 0:
        print("Error: Fractions must be positive", file=sys.stderr)
        sys.exit(1)
    
    if args.train_frac + args.val_frac >= 1.0:
        print("Error: train_frac + val_frac must be less than 1.0", file=sys.stderr)
        sys.exit(1)
    
    # Load taxonomy
    taxonomy = load_taxonomy()
    
    # Resolve paths
    manual_path = project_root / "data" / "raw" / "manual_cases.jsonl"
    synthetic_path = project_root / "data" / "raw" / "synthetic_cases.jsonl"
    out_dir = project_root / args.out_dir
    processed_dir = out_dir / "processed"
    splits_dir = out_dir / "splits"
    
    # Merge and deduplicate
    print("=" * 70)
    print("Building dataset")
    print("=" * 70)
    print()
    
    all_cases, validation_errors = merge_and_deduplicate(
        manual_path,
        synthetic_path,
        allow_missing_synthetic=args.allow_missing_synthetic,
        taxonomy=taxonomy
    )
    
    if validation_errors:
        print("\nValidation errors:", file=sys.stderr)
        for error in validation_errors[:10]:  # Show first 10 errors
            print(f"  {error}", file=sys.stderr)
        if len(validation_errors) > 10:
            print(f"  ... and {len(validation_errors) - 10} more errors", file=sys.stderr)
        sys.exit(1)
    
    if len(all_cases) == 0:
        print("Error: No cases found after merging", file=sys.stderr)
        sys.exit(1)
    
    # Save merged dataset
    all_path = processed_dir / "all.jsonl"
    print(f"\nSaving merged dataset to: {all_path}")
    processed_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(all_cases, all_path)
    print(f"✓ Saved {len(all_cases)} cases")
    
    # Create splits
    print("\n" + "=" * 70)
    print("Creating splits")
    print("=" * 70)
    
    train_cases, val_cases, test_cases = create_splits(
        all_cases,
        args.train_frac,
        args.val_frac,
        args.seed
    )
    
    # Save splits
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = splits_dir / "train.jsonl"
    val_path = splits_dir / "val.jsonl"
    test_path = splits_dir / "test.jsonl"
    
    print(f"\nSaving splits:")
    print(f"  Train: {train_path} ({len(train_cases)} cases)")
    print(f"  Val: {val_path} ({len(val_cases)} cases)")
    print(f"  Test: {test_path} ({len(test_cases)} cases)")
    
    save_jsonl(train_cases, train_path)
    save_jsonl(val_cases, val_path)
    save_jsonl(test_cases, test_path)
    
    print("\n✓ Dataset build complete!")
    
    # Print summary
    print("\nSummary:")
    print(f"  Total cases: {len(all_cases)}")
    print(f"  Train: {len(train_cases)} ({len(train_cases)/len(all_cases):.1%})")
    print(f"  Val: {len(val_cases)} ({len(val_cases)/len(all_cases):.1%})")
    print(f"  Test: {len(test_cases)} ({len(test_cases)/len(all_cases):.1%})")


if __name__ == "__main__":
    main()
