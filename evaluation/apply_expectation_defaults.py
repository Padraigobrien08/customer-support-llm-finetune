#!/usr/bin/env python3
"""
Apply default tri-state expectations to test cases based on category and metadata.

This script ensures consistent expectation semantics across all test cases,
preventing mislabeling issues that cause high failure rates.

Rules:
- request_sensitive_info = "forbidden" for all cases
- claim_specific_policy = "forbidden" for all cases (unless metadata.policy_provided=true)
- offer_next_steps = "required" for all cases except "complaints_feedback" (optional there)
- escalate:
  - "required" for category in ["security_privacy"] OR metadata.escalation_required=true
  - "forbidden" for ["product_usage_howto","general_information"] unless escalation_required=true
- ask_clarifying_question:
  - "required" for categories ["general_information","shipping_delivery","billing_payments"] if metadata.ambiguous=true
  - otherwise "optional"
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def get_metadata_value(test_case: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get metadata value, checking both metadata dict and direct fields."""
    # Check metadata dict first
    metadata = test_case.get("metadata", {})
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    
    # Check direct fields (for backward compatibility)
    if key in test_case:
        return test_case[key]
    
    # Check notes for some fields (e.g., "escalation: true" in notes)
    notes = test_case.get("notes", "")
    if isinstance(notes, str):
        if key == "escalation_required" and "escalation" in notes.lower():
            # Try to infer from notes
            if "escalation: true" in notes.lower() or "escalation=true" in notes.lower():
                return True
    
    return default


def compute_default_expectations(test_case: dict[str, Any]) -> dict[str, str]:
    """
    Compute default expectations based on category and metadata.
    
    Returns:
        Dictionary mapping expectation keys to tri-state values
    """
    category = test_case.get("category", "").lower()
    metadata = test_case.get("metadata", {})
    
    # Get metadata values (with fallbacks)
    escalation_required = get_metadata_value(test_case, "escalation_required", False)
    ambiguous = get_metadata_value(test_case, "ambiguous", False)
    policy_provided = get_metadata_value(test_case, "policy_provided", False)
    
    # Also check if escalation is set in metadata (common field)
    if not escalation_required:
        escalation_bool = get_metadata_value(test_case, "escalation", None)
        if escalation_bool is True:
            escalation_required = True
    
    expectations: dict[str, str] = {}
    
    # Rule 1: request_sensitive_info = "forbidden" for all cases
    expectations["request_sensitive_info"] = "forbidden"
    
    # Rule 2: claim_specific_policy = "forbidden" unless policy_provided=true
    if policy_provided:
        expectations["claim_specific_policy"] = "optional"  # Allow if policy is provided
    else:
        expectations["claim_specific_policy"] = "forbidden"
    
    # Rule 3: offer_next_steps = "required" except for "complaints_feedback"
    if category == "complaints_feedback":
        expectations["offer_next_steps"] = "optional"
    else:
        expectations["offer_next_steps"] = "required"
    
    # Rule 4: escalate
    if category == "security_privacy" or escalation_required:
        expectations["escalate"] = "required"
    elif category in ["product_usage_howto", "general_information"]:
        if escalation_required:
            expectations["escalate"] = "required"
        else:
            expectations["escalate"] = "forbidden"
    else:
        # Default: optional for other categories
        expectations["escalate"] = "optional"
    
    # Rule 5: ask_clarifying_question
    if category in ["general_information", "shipping_delivery", "billing_payments"]:
        if ambiguous:
            expectations["ask_clarifying_question"] = "required"
        else:
            expectations["ask_clarifying_question"] = "optional"
    else:
        expectations["ask_clarifying_question"] = "optional"
    
    return expectations


def apply_expectation_defaults(
    input_path: Path,
    output_path: Path | None = None,
    dry_run: bool = False
) -> dict[str, Any]:
    """
    Apply default expectations to test cases.
    
    Args:
        input_path: Path to input test_cases.json
        output_path: Path to output file (defaults to overwriting input)
        dry_run: If True, don't write changes, just return stats
    
    Returns:
        Dictionary with statistics about changes made
    """
    if output_path is None:
        output_path = input_path
    
    # Read existing test cases
    with open(input_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    if not isinstance(test_cases, list):
        raise ValueError(f"Expected list of test cases, got {type(test_cases).__name__}")
    
    stats = {
        "total_cases": len(test_cases),
        "cases_updated": 0,
        "cases_unchanged": 0,
        "changes_by_field": defaultdict(int),
        "changes_by_category": defaultdict(int),
    }
    
    # Process each test case
    for test_case in test_cases:
        category = test_case.get("category", "unknown")
        case_id = test_case.get("id", "unknown")
        
        # Compute default expectations
        default_expectations = compute_default_expectations(test_case)
        
        # Get current expectations
        current_expectations = test_case.get("expectations", {})
        if not isinstance(current_expectations, dict):
            current_expectations = {}
        
        # Check if expectations need updating
        needs_update = False
        field_changes = []
        
        for key, default_value in default_expectations.items():
            current_value = current_expectations.get(key)
            
            # Only update if current value is different or missing
            if current_value != default_value:
                needs_update = True
                field_changes.append((key, current_value, default_value))
                stats["changes_by_field"][key] += 1
        
        if needs_update:
            # Update expectations
            for key, default_value in default_expectations.items():
                test_case["expectations"] = test_case.get("expectations", {})
                test_case["expectations"][key] = default_value
            
            stats["cases_updated"] += 1
            stats["changes_by_category"][category] += 1
        else:
            stats["cases_unchanged"] += 1
    
    # Write updated test cases (unless dry run)
    if not dry_run:
        # Validate JSON before writing
        try:
            json_str = json.dumps(test_cases, indent=2, ensure_ascii=False)
            json.loads(json_str)  # Validate
        except (TypeError, ValueError) as e:
            raise ValueError(f"Generated JSON is invalid: {e}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
    
    return stats


def print_stats(stats: dict[str, Any]) -> None:
    """Print statistics about changes made."""
    print("=" * 60)
    print("Expectation Defaults Applied")
    print("=" * 60)
    print(f"\nTotal test cases: {stats['total_cases']}")
    print(f"Cases updated: {stats['cases_updated']}")
    print(f"Cases unchanged: {stats['cases_unchanged']}")
    
    if stats['changes_by_field']:
        print(f"\nChanges by field:")
        for field, count in sorted(stats['changes_by_field'].items(), key=lambda x: -x[1]):
            print(f"  {field}: {count}")
    
    if stats['changes_by_category']:
        print(f"\nChanges by category:")
        for category, count in sorted(stats['changes_by_category'].items(), key=lambda x: -x[1]):
            print(f"  {category}: {count}")
    
    print("=" * 60)


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Apply default tri-state expectations to test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/apply_expectation_defaults.py
  python evaluation/apply_expectation_defaults.py --input evaluation/test_cases.json --output evaluation/test_cases_updated.json
  python evaluation/apply_expectation_defaults.py --dry-run
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="evaluation/test_cases.json",
        help="Input test cases JSON file (default: evaluation/test_cases.json)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output test cases JSON file (default: overwrites input)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        if args.dry_run:
            print("DRY RUN MODE - No files will be modified")
            print()
        
        stats = apply_expectation_defaults(input_path, output_path, dry_run=args.dry_run)
        print_stats(stats)
        
        if not args.dry_run:
            actual_output = output_path if output_path else input_path
            print(f"\n✓ Output written to: {actual_output}")
        
    except Exception as e:
        print(f"Error applying expectation defaults: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
