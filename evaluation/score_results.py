#!/usr/bin/env python3
"""
CLI entrypoint for scoring evaluation results.

Usage:
    python evaluation/score_results.py evaluation/results/adapter.json evaluation/test_cases.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.scoring import calculate_summary, score_results


def print_summary_table(summary: dict[str, Any]) -> None:
    """Print a formatted summary table of pass rates."""
    print("\n" + "=" * 60)
    print("Scoring Summary")
    print("=" * 60)
    print()
    
    # Print check-by-check summary
    print("Check Pass Rates:")
    print("-" * 60)
    print(f"{'Check':<30} {'Passed':<10} {'Total':<10} {'Rate':<10}")
    print("-" * 60)
    
    for check_name in sorted(summary.keys()):
        if check_name == "overall":
            continue
        
        check_data = summary[check_name]
        passed = check_data["passed"]
        total = check_data["total"]
        rate = check_data["pass_rate"]
        
        rate_str = f"{rate:.1%}"
        print(f"{check_name:<30} {passed:<10} {total:<10} {rate_str:<10}")
    
    print("-" * 60)
    print()
    
    # Print overall summary
    overall = summary.get("overall", {})
    print(f"Total test cases: {overall.get('total_cases', 0)}")
    print(f"Total checks: {overall.get('checks', 0)}")
    print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Score evaluation results using rule-based checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/score_results.py evaluation/results/adapter.json evaluation/test_cases.json
  python evaluation/score_results.py evaluation/results/base.json evaluation/test_cases.json --out evaluation/results/scored_base.json
        """
    )
    
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to results JSON from run_golden_eval.py"
    )
    
    parser.add_argument(
        "test_cases_file",
        type=str,
        help="Path to test cases JSON file"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        help="Output path for scored results JSON (default: evaluation/results/scored_<basename>.json)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    results_path = Path(args.results_file)
    test_cases_path = Path(args.test_cases_file)
    
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)
    
    if not test_cases_path.exists():
        print(f"Error: Test cases file not found: {test_cases_path}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    if args.out:
        output_path = Path(args.out)
    else:
        # Generate output path from results filename
        results_basename = results_path.stem
        output_dir = results_path.parent
        output_path = output_dir / f"scored_{results_basename}.json"
    
    print(f"Scoring results from: {results_path}")
    print(f"Using test cases from: {test_cases_path}")
    print(f"Output will be written to: {output_path}")
    print()
    
    # Score results
    try:
        scores = score_results(results_path, test_cases_path)
        summary = calculate_summary(scores)
        
        # Create output structure
        output_data = {
            "results_file": str(results_path),
            "test_cases_file": str(test_cases_path),
            "scores": scores,
            "summary": summary
        }
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Scored {len(scores)} test cases")
        print(f"✓ Results written to: {output_path}")
        
        # Print summary table
        print_summary_table(summary)
        
    except Exception as e:
        print(f"Error scoring results: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
