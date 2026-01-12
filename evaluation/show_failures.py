#!/usr/bin/env python3
"""
Show failing test cases from scored results.

Usage:
    python evaluation/show_failures.py [scored_results.json]
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def show_failures(scored_results_path: Path):
    """Show all failing test cases."""
    with open(scored_results_path, 'r') as f:
        scored_data = json.load(f)
    
    scores = scored_data.get('scores', [])
    
    if not scores:
        print("No scores found in file.")
        return
    
    # Find failing cases
    failed_cases = []
    failure_counts = defaultdict(int)
    
    for score in scores:
        if not isinstance(score, dict):
            print(f"ERROR: Score is not a dict: {type(score)} - {score}", file=sys.stderr)
            continue
        
        # Check rule-based checks
        checks = score.get('checks', {})
        failed_checks = [name for name, passed in checks.items() if passed is False]
        
        # Check expectations
        expectations = score.get('expectations', {})
        failed_expectations = []
        if expectations:
            failed_expectations = [name for name, passed in expectations.items() if passed is False]
        
        if failed_checks or failed_expectations:
            failed_cases.append({
                'id': score.get('test_case_id', 'unknown'),
                'category': score.get('category', 'unknown'),
                'failed_checks': failed_checks,
                'failed_expectations': failed_expectations,
                'notes': score.get('notes', [])
            })
            
            # Count failures
            for check in failed_checks:
                failure_counts[f"rule:{check}"] += 1
            for exp in failed_expectations:
                failure_counts[f"expectation:{exp}"] += 1
    
    # Print summary
    print("=" * 70)
    print(f"FAILURE SUMMARY: {len(failed_cases)}/{len(scores)} cases failed")
    print("=" * 70)
    
    if failure_counts:
        print("\nFailure counts by check/expectation:")
        for failure_type, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"  {failure_type}: {count}")
    
    # Print detailed failures
    print(f"\n{'=' * 70}")
    print("DETAILED FAILURES")
    print("=" * 70)
    
    for f in failed_cases:
        print(f"\n{f['id']} ({f['category']}):")
        if f['failed_checks']:
            print(f"  ✗ Rule checks: {', '.join(f['failed_checks'])}")
        if f['failed_expectations']:
            print(f"  ✗ Expectations: {', '.join(f['failed_expectations'])}")
        if f['notes']:
            # Show relevant notes (filter out redundant ones)
            relevant_notes = [n for n in f['notes'] if any(
                check in n.lower() or exp in n.lower() 
                for check in f['failed_checks'] 
                for exp in f['failed_expectations']
            )]
            if not relevant_notes:
                relevant_notes = f['notes'][:2]  # Show first 2 if no matches
            print(f"  Notes: {', '.join(relevant_notes[:3])}")
    
    print(f"\n{'=' * 70}")
    print(f"Total: {len(failed_cases)} cases with failures")
    print("=" * 70)


def main():
    """Main CLI entry point."""
    if len(sys.argv) > 1:
        scored_results_path = Path(sys.argv[1])
    else:
        # Default to adapter results
        scored_results_path = Path('evaluation/results/scored_adapter.json')
    
    if not scored_results_path.exists():
        print(f"Error: File not found: {scored_results_path}", file=sys.stderr)
        print(f"Usage: python evaluation/show_failures.py [scored_results.json]", file=sys.stderr)
        sys.exit(1)
    
    show_failures(scored_results_path)


if __name__ == "__main__":
    main()
