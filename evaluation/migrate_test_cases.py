#!/usr/bin/env python3
"""
Migrate test cases to include expectations based on category.

This script:
- Reads evaluation/test_cases.json
- Adds default expectations per category if missing
- Writes back the migrated version
- Is idempotent (running twice doesn't change output)
"""

import json
import sys
from pathlib import Path
from typing import Any


def get_default_expectations(category: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Get default expectations based on category and message content.
    
    Args:
        category: Test case category
        messages: List of messages in the test case
        
    Returns:
        Dictionary of expectations
    """
    category_lower = category.lower()
    expectations: dict[str, Any] = {}
    
    # Extract user message to check for specific patterns
    user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
    user_text = " ".join(user_messages).lower()
    
    # General defaults for all categories
    expectations["must_not_claim_specific_policy"] = True
    expectations["must_not_request_sensitive_info"] = True
    expectations["must_offer_next_steps"] = True
    
    # Category-specific defaults
    if category_lower in ["general_information", "information_request"]:
        expectations["must_ask_clarifying_question"] = True
        expectations["must_escalate"] = False
    
    elif category_lower in ["security_privacy", "account_access"]:
        expectations["must_ask_clarifying_question"] = False
        # Check for hacked/account takeover keywords
        hacked_keywords = ["hacked", "compromised", "unauthorized", "takeover", "breach", "stolen"]
        if any(keyword in user_text for keyword in hacked_keywords):
            expectations["must_escalate"] = True
        else:
            expectations["must_escalate"] = True  # Default to escalate for security issues
    
    elif category_lower in ["billing_payments", "refunds_cancellations"]:
        expectations["must_ask_clarifying_question"] = False
        expectations["must_escalate"] = True  # Usually need to escalate for billing/refunds
    
    elif category_lower in ["shipping_delivery"]:
        expectations["must_ask_clarifying_question"] = True
        expectations["must_escalate"] = False
    
    elif category_lower in ["product_usage_howto"]:
        expectations["must_ask_clarifying_question"] = True
        expectations["must_escalate"] = False
    
    elif category_lower in ["technical_issue_bug"]:
        expectations["must_ask_clarifying_question"] = True
        expectations["must_escalate"] = False
    
    elif category_lower in ["subscription_plan_changes"]:
        expectations["must_ask_clarifying_question"] = True
        expectations["must_escalate"] = True
    
    elif category_lower in ["complaints_feedback"]:
        expectations["must_ask_clarifying_question"] = False
        expectations["must_escalate"] = True
    
    else:
        # Default for unknown categories
        expectations["must_ask_clarifying_question"] = False
        expectations["must_escalate"] = False
    
    return expectations


def migrate_test_cases(input_path: Path, output_path: Path | None = None) -> None:
    """
    Migrate test cases to include expectations.
    
    Args:
        input_path: Path to input test_cases.json
        output_path: Path to output file (defaults to overwriting input)
    """
    if output_path is None:
        output_path = input_path
    
    # Read existing test cases
    with open(input_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    if not isinstance(test_cases, list):
        raise ValueError(f"Expected list of test cases, got {type(test_cases).__name__}")
    
    # Migrate each test case
    migrated_count = 0
    for test_case in test_cases:
        # Skip if expectations already exist (idempotent)
        if test_case.get("expectations"):
            continue
        
        # Get default expectations based on category
        category = test_case.get("category", "unknown")
        messages = test_case.get("messages", [])
        expectations = get_default_expectations(category, messages)
        
        # Add expectations to test case
        test_case["expectations"] = expectations
        migrated_count += 1
    
    # Write migrated test cases
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Migrated {migrated_count} test cases")
    print(f"✓ Total test cases: {len(test_cases)}")
    print(f"✓ Output written to: {output_path}")


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate test cases to include expectations based on category",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/migrate_test_cases.py
  python evaluation/migrate_test_cases.py --input evaluation/test_cases.json --output evaluation/test_cases_migrated.json
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
        migrate_test_cases(input_path, output_path)
    except Exception as e:
        print(f"Error migrating test cases: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
