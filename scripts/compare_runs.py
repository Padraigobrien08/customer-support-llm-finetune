#!/usr/bin/env python3
"""
Compare two evaluation result files and show differences.

Shows which test_case_ids changed and provides short old/new snippets.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path for csft package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_results(results_path: Path) -> list[dict[str, Any]]:
    """
    Load evaluation results from JSON file.
    
    Args:
        results_path: Path to results JSON file
        
    Returns:
        List of result dictionaries
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is invalid JSON
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {results_path}: {e}") from e


def compare_results(
    current_results: list[dict[str, Any]],
    previous_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Compare two sets of results and identify changes.
    
    Args:
        current_results: Current results to compare
        previous_results: Previous results to compare against
        
    Returns:
        Dictionary with comparison information
    """
    # Create lookup by test_case_id
    current_by_id = {r["test_case_id"]: r for r in current_results}
    previous_by_id = {r["test_case_id"]: r for r in previous_results}
    
    # Find changed outputs
    changed = []
    for test_id, current_result in current_by_id.items():
        if test_id not in previous_by_id:
            continue  # Skip if not in previous results
        
        current_output = current_result.get("model_output", "")
        previous_output = previous_by_id[test_id].get("model_output", "")
        
        if current_output != previous_output:
            changed.append({
                "test_case_id": test_id,
                "category": current_result.get("category", "unknown"),
                "current_output": current_output,
                "previous_output": previous_output,
                "current_length": len(current_output),
                "previous_length": len(previous_output),
            })
    
    # Find new test cases
    new_ids = set(current_by_id.keys()) - set(previous_by_id.keys())
    
    # Find removed test cases
    removed_ids = set(previous_by_id.keys()) - set(current_by_id.keys())
    
    return {
        "total_current": len(current_results),
        "total_previous": len(previous_results),
        "changed_count": len(changed),
        "changed": changed,
        "new": list(new_ids),
        "removed": list(removed_ids)
    }


def truncate_text(text: str, max_length: int = 120) -> str:
    """
    Truncate text to max_length with ellipsis if needed.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def print_comparison(
    comparison: dict[str, Any],
    current_file: Path,
    previous_file: Path
) -> None:
    """
    Print comparison results in a readable format.
    
    Args:
        comparison: Comparison dictionary from compare_results()
        current_file: Path to current results file
        previous_file: Path to previous results file
    """
    print("=" * 70)
    print("Comparison Results")
    print("=" * 70)
    print()
    print(f"Current file:  {current_file.name}")
    print(f"Previous file: {previous_file.name}")
    print()
    print(f"Total cases:")
    print(f"  Current:  {comparison['total_current']}")
    print(f"  Previous: {comparison['total_previous']}")
    print()
    
    if comparison['new']:
        print(f"New test cases ({len(comparison['new'])}):")
        for test_id in sorted(comparison['new']):
            print(f"  + {test_id}")
        print()
    
    if comparison['removed']:
        print(f"Removed test cases ({len(comparison['removed'])}):")
        for test_id in sorted(comparison['removed']):
            print(f"  - {test_id}")
        print()
    
    if comparison['changed_count'] > 0:
        print(f"Changed outputs ({comparison['changed_count']}):")
        print()
        
        for change in comparison['changed']:
            test_id = change['test_case_id']
            category = change['category']
            current_output = change['current_output']
            previous_output = change['previous_output']
            current_len = change['current_length']
            previous_len = change['previous_length']
            
            length_diff = current_len - previous_len
            length_diff_str = f"+{length_diff}" if length_diff > 0 else str(length_diff)
            
            print(f"  {test_id} ({category})")
            print(f"    Length: {previous_len} → {current_len} ({length_diff_str} chars)")
            print(f"    Previous: {truncate_text(previous_output)}")
            print(f"    Current:  {truncate_text(current_output)}")
            print()
    else:
        print("No changed outputs found.")
        print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare two evaluation result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_runs.py evaluation/results/base.json evaluation/results/finetuned.json
  python scripts/compare_runs.py evaluation/results/run1.json evaluation/results/run2.json
        """
    )
    
    parser.add_argument(
        "current_file",
        type=str,
        help="Path to current results JSON file"
    )
    
    parser.add_argument(
        "previous_file",
        type=str,
        help="Path to previous results JSON file to compare against"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    current_path = Path(args.current_file)
    previous_path = Path(args.previous_file)
    
    if not current_path.is_absolute():
        current_path = project_root / current_path
    
    if not previous_path.is_absolute():
        previous_path = project_root / previous_path
    
    # Load results
    try:
        print(f"Loading current results from: {current_path}")
        current_results = load_results(current_path)
        print(f"  Loaded {len(current_results)} results")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading current results: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        print(f"Loading previous results from: {previous_path}")
        previous_results = load_results(previous_path)
        print(f"  Loaded {len(previous_results)} results")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading previous results: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Compare results
    print("\nComparing results...")
    comparison = compare_results(current_results, previous_results)
    
    # Print comparison
    print_comparison(comparison, current_path, previous_path)


if __name__ == "__main__":
    main()

