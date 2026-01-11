#!/usr/bin/env python3
"""
Export scored evaluation results to CSV for review in Google Sheets.

Usage:
    python evaluation/export_review_csv.py evaluation/results/scored_adapter.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def truncate_text(text: str, max_length: int = 300) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def extract_user_message(messages: list[dict[str, Any]]) -> str:
    """Extract the user message from messages array."""
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def get_failed_checks(checks: dict[str, Any]) -> str:
    """Get comma-separated list of failed check names."""
    failed = []
    for check_name, check_result in checks.items():
        if check_result is False:  # Explicitly False (not None)
            failed.append(check_name)
    return ", ".join(failed) if failed else ""


def format_notes(notes: list[str]) -> str:
    """Format notes list as a single string, separated by semicolons."""
    if not notes:
        return ""
    return "; ".join(notes)


def export_to_csv(
    scored_results_path: str | Path,
    output_path: str | Path | None = None
) -> None:
    """
    Export scored results to CSV.
    
    Args:
        scored_results_path: Path to scored results JSON
        output_path: Output CSV path (default: <scored_results_path>.csv)
    """
    scored_results_path = Path(scored_results_path)
    
    if not scored_results_path.exists():
        print(f"Error: Scored results file not found: {scored_results_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load scored results
    with open(scored_results_path, 'r', encoding='utf-8') as f:
        scored_data = json.load(f)
    
    # Get path to original results
    results_file = scored_data.get("results_file")
    if not results_file:
        print("Error: No results_file found in scored results", file=sys.stderr)
        sys.exit(1)
    
    # Resolve results file path (relative to scored results file location)
    results_path = Path(scored_results_path.parent / results_file)
    if not results_path.exists():
        # Try absolute path
        results_path = Path(results_file)
        if not results_path.exists():
            print(f"Error: Original results file not found: {results_file}", file=sys.stderr)
            sys.exit(1)
    
    # Load original results
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Create lookup by test_case_id
    results_by_id = {r.get("test_case_id"): r for r in results}
    
    # Determine output path
    if output_path is None:
        output_path = scored_results_path.with_suffix('.csv')
    else:
        output_path = Path(output_path)
    
    # Prepare CSV data
    rows = []
    scores = scored_data.get("scores", [])
    
    for score in scores:
        test_case_id = score.get("test_case_id")
        category = score.get("category", "")
        checks = score.get("checks", {})
        notes = score.get("notes", [])
        
        # Get corresponding result
        result = results_by_id.get(test_case_id, {})
        
        # Extract user message (prompt)
        messages = result.get("messages", [])
        prompt = extract_user_message(messages)
        prompt_truncated = truncate_text(prompt, max_length=300)
        
        # Get model output
        model_output = result.get("output_text", "")
        
        # Get failed checks
        failed_checks = get_failed_checks(checks)
        
        # Format notes
        notes_str = format_notes(notes)
        
        rows.append({
            "test_case_id": test_case_id,
            "category": category,
            "prompt": prompt_truncated,
            "model_output": model_output,
            "failed_checks": failed_checks,
            "notes": notes_str
        })
    
    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ["test_case_id", "category", "prompt", "model_output", "failed_checks", "notes"]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Exported {len(rows)} rows to: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export scored evaluation results to CSV for review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/export_review_csv.py evaluation/results/scored_adapter.json
  python evaluation/export_review_csv.py evaluation/results/scored_adapter.json --out review.csv
        """
    )
    
    parser.add_argument(
        "scored_results",
        type=str,
        help="Path to scored results JSON file"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        help="Output CSV path (default: <scored_results>.csv)"
    )
    
    args = parser.parse_args()
    
    export_to_csv(args.scored_results, args.out)


if __name__ == "__main__":
    main()
