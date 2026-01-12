#!/usr/bin/env python3
"""
Migrate test case expectations from boolean format to tri-state format.

Old format:
  "must_ask_clarifying_question": true/false
  "must_not_claim_specific_policy": true/false
  "must_not_request_sensitive_info": true/false
  "must_offer_next_steps": true/false
  "must_escalate": true/false

New format:
  "ask_clarifying_question": "required" | "forbidden" | "optional"
  "claim_specific_policy": "required" | "forbidden" | "optional"
  "request_sensitive_info": "required" | "forbidden" | "optional"
  "offer_next_steps": "required" | "forbidden" | "optional"
  "escalate": "required" | "forbidden" | "optional"

Conversion rules:
  - Old: must_ask_clarifying_question: true  → New: ask_clarifying_question: "required"
  - Old: must_ask_clarifying_question: false → New: ask_clarifying_question: "optional"
  - Old: must_not_claim_specific_policy: true  → New: claim_specific_policy: "forbidden"
  - Old: must_not_claim_specific_policy: false → New: claim_specific_policy: "optional"
  - Missing keys → "optional" by default
"""

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def convert_bool_to_tri_state(value: bool, is_negated: bool = False) -> str:
    """
    Convert boolean expectation to tri-state.
    
    Args:
        value: Boolean value from old format
        is_negated: True if the old field was "must_not_*"
    
    Returns:
        "required", "forbidden", or "optional"
    """
    if is_negated:
        # Old: must_not_X: true means X is forbidden
        # Old: must_not_X: false means X is optional
        return "forbidden" if value else "optional"
    else:
        # Old: must_X: true means X is required
        # Old: must_X: false means X is optional
        return "required" if value else "optional"


def migrate_expectations(old_expectations: dict[str, Any]) -> dict[str, str]:
    """
    Migrate expectations from old boolean format to new tri-state format.
    
    Args:
        old_expectations: Old format expectations dict
    
    Returns:
        New format expectations dict
    """
    new_expectations: dict[str, str] = {}
    
    # Mapping from old field names to new field names and whether they're negated
    field_mapping = {
        "must_ask_clarifying_question": ("ask_clarifying_question", False),
        "must_not_claim_specific_policy": ("claim_specific_policy", True),
        "must_not_request_sensitive_info": ("request_sensitive_info", True),
        "must_offer_next_steps": ("offer_next_steps", False),
        "must_escalate": ("escalate", False),
    }
    
    # New format field names
    new_format_fields = ["ask_clarifying_question", "claim_specific_policy", 
                        "request_sensitive_info", "offer_next_steps", "escalate"]
    
    # If new format fields already exist, keep them (don't overwrite)
    for field in new_format_fields:
        if field in old_expectations:
            value = old_expectations[field]
            if isinstance(value, str) and value in ["required", "forbidden", "optional"]:
                new_expectations[field] = value
    
    # Convert old fields (only if new format doesn't already exist)
    for old_field, (new_field, is_negated) in field_mapping.items():
        if new_field not in new_expectations and old_field in old_expectations:
            old_value = old_expectations[old_field]
            if isinstance(old_value, bool):
                new_expectations[new_field] = convert_bool_to_tri_state(old_value, is_negated)
    
    # Set missing keys to "optional" by default (be consistent - include all fields)
    for field in new_format_fields:
        if field not in new_expectations:
            new_expectations[field] = "optional"
    
    return new_expectations


def migrate_test_cases(input_path: Path, output_path: Path | None = None) -> tuple[int, dict[str, dict[str, int]]]:
    """
    Migrate test cases from boolean expectations to tri-state expectations.
    
    Args:
        input_path: Path to input test_cases.json
        output_path: Path to output file (defaults to overwriting input)
    
    Returns:
        Tuple of (migrated_count, summary_counts)
    """
    if output_path is None:
        output_path = input_path
    
    # Create backup before modifying
    backup_path = input_path.with_suffix(input_path.suffix + ".bak")
    if input_path.exists():
        shutil.copy2(input_path, backup_path)
        print(f"✓ Created backup: {backup_path}")
    
    # Read existing test cases
    with open(input_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    if not isinstance(test_cases, list):
        raise ValueError(f"Expected list of test cases, got {type(test_cases).__name__}")
    
    # Track summary counts
    summary_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    # Migrate each test case
    migrated_count = 0
    for test_case in test_cases:
        old_expectations = test_case.get("expectations", {})
        
        # Check if already fully migrated (all new format fields present with valid values)
        new_format_fields = ["ask_clarifying_question", "claim_specific_policy", 
                            "request_sensitive_info", "offer_next_steps", "escalate"]
        old_format_fields = ["must_ask_clarifying_question", "must_not_claim_specific_policy",
                            "must_not_request_sensitive_info", "must_offer_next_steps", "must_escalate"]
        
        has_old_format = any(field in old_expectations for field in old_format_fields)
        has_all_new_format = all(
            field in old_expectations and 
            isinstance(old_expectations[field], str) and
            old_expectations[field] in ["required", "forbidden", "optional"]
            for field in new_format_fields
        )
        
        if has_all_new_format and not has_old_format:
            # Already fully migrated - just count values for summary
            for field in new_format_fields:
                if field in old_expectations:
                    value = old_expectations[field]
                    summary_counts[field][value] += 1
            continue
        
        # Migrate expectations
        new_expectations = migrate_expectations(old_expectations)
        if new_expectations:
            test_case["expectations"] = new_expectations
            migrated_count += 1
            
            # Count values for summary
            for field, value in new_expectations.items():
                summary_counts[field][value] += 1
    
    # Validate JSON before writing
    try:
        # Test serialization
        json_str = json.dumps(test_cases, indent=2, ensure_ascii=False)
        # Test deserialization
        json.loads(json_str)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Generated JSON is invalid: {e}")
    
    # Write migrated test cases
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)
    
    return migrated_count, dict(summary_counts)


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate test case expectations from boolean to tri-state format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/migrate_expectations.py
  python evaluation/migrate_expectations.py --input evaluation/test_cases.json --output evaluation/test_cases_migrated.json
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
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        migrated_count, summary_counts = migrate_test_cases(input_path, output_path)
        
        # Determine actual output path
        actual_output = output_path if output_path else input_path
        
        print(f"\n✓ Migrated {migrated_count} test cases")
        with open(actual_output, 'r') as f:
            total_cases = len(json.load(f))
        print(f"✓ Total test cases: {total_cases}")
        print(f"✓ Output written to: {actual_output}")
        
        # Print summary
        if summary_counts:
            print("\n" + "=" * 60)
            print("Summary: Counts per tri-state value per key")
            print("=" * 60)
            for field in sorted(summary_counts.keys()):
                counts = summary_counts[field]
                total = sum(counts.values())
                print(f"\n{field}:")
                for value in ["required", "forbidden", "optional"]:
                    count = counts.get(value, 0)
                    if count > 0:
                        print(f"  {value}: {count} ({count/total*100:.1f}%)")
            print("=" * 60)
        
    except Exception as e:
        print(f"Error migrating expectations: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
