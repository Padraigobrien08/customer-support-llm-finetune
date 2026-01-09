#!/usr/bin/env python3
"""
Build final dataset from raw data sources.

Merges manual and synthetic cases, deduplicates, and creates train/val/test splits.
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


def hash_user_message(user_message: str, category: str) -> str:
    """
    Create a stable hash of user message + category for deduplication.
    
    Args:
        user_message: User message text
        category: Category string
        
    Returns:
        Hexadecimal hash string
    """
    combined = f"{user_message}|{category}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()


def extract_user_message(messages: list[dict[str, Any]]) -> str:
    """
    Extract the first user message from messages list.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        User message content, or empty string if not found
    """
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def merge_and_deduplicate(
    manual_path: Path,
    synthetic_path: Path,
    seed: int | None = None
) -> list[dict[str, Any]]:
    """
    Load, merge, and deduplicate cases from manual and synthetic sources.
    
    Args:
        manual_path: Path to manual cases JSONL
        synthetic_path: Path to synthetic cases JSONL
        seed: Random seed (not used but kept for consistency)
        
    Returns:
        List of deduplicated cases
    """
    all_cases = []
    seen_hashes = set()
    
    # Load manual cases
    if manual_path.exists():
        print(f"Loading manual cases from: {manual_path}")
        manual_cases = load_jsonl(str(manual_path))
        print(f"  Loaded {len(manual_cases)} manual cases")
        all_cases.extend(manual_cases)
    else:
        print(f"Warning: Manual cases file not found: {manual_path}")
        manual_cases = []
    
    # Load synthetic cases
    if synthetic_path.exists():
        print(f"Loading synthetic cases from: {synthetic_path}")
        synthetic_cases = load_jsonl(str(synthetic_path))
        print(f"  Loaded {len(synthetic_cases)} synthetic cases")
        all_cases.extend(synthetic_cases)
    else:
        print(f"Warning: Synthetic cases file not found: {synthetic_path}")
        synthetic_cases = []
    
    print(f"\nTotal cases before deduplication: {len(all_cases)}")
    
    # Deduplicate by user message + category hash
    deduplicated = []
    duplicates = 0
    
    for case in all_cases:
        user_msg = extract_user_message(case.get("messages", []))
        category = case.get("metadata", {}).get("category", "unknown")
        case_hash = hash_user_message(user_msg, category)
        
        if case_hash not in seen_hashes:
            seen_hashes.add(case_hash)
            deduplicated.append(case)
        else:
            duplicates += 1
    
    print(f"Removed {duplicates} duplicate cases")
    print(f"Total cases after deduplication: {len(deduplicated)}")
    
    return deduplicated


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
  python scripts/build_dataset.py --seed 42 --train_frac 0.8 --val_frac 0.1
  python scripts/build_dataset.py --seed 42 --out_dir data/processed
        """
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
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
        default="data/processed",
        help="Output directory for processed data (default: data/processed)"
    )
    
    args = parser.parse_args()
    
    # Validate fractions
    if args.train_frac <= 0 or args.val_frac <= 0:
        print("Error: Fractions must be positive", file=sys.stderr)
        sys.exit(1)
    
    if args.train_frac + args.val_frac >= 1.0:
        print("Error: train_frac + val_frac must be less than 1.0", file=sys.stderr)
        sys.exit(1)
    
    # Resolve paths
    manual_path = project_root / "data" / "raw" / "manual_cases.jsonl"
    synthetic_path = project_root / "data" / "raw" / "synthetic_cases.jsonl"
    out_dir = project_root / args.out_dir
    splits_dir = project_root / "data" / "splits"
    
    # Merge and deduplicate
    print("=" * 70)
    print("Building dataset")
    print("=" * 70)
    print()
    
    all_cases = merge_and_deduplicate(manual_path, synthetic_path, args.seed)
    
    if len(all_cases) == 0:
        print("Error: No cases found after merging", file=sys.stderr)
        sys.exit(1)
    
    # Save merged dataset
    all_path = out_dir / "all.jsonl"
    print(f"\nSaving merged dataset to: {all_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
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

