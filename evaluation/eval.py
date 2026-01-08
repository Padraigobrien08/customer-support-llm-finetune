"""
Evaluation script for the fine-tuned customer support model.

This script can:
1. Load and summarize evaluation results from JSON files
2. Compare results between different runs
3. Evaluate model performance against golden test cases (scoring TBD)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_results(results_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load evaluation results from JSON file.
    
    Args:
        results_path: Path to results JSON file
        
    Returns:
        List of result dictionaries
    """
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path.absolute()}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics from results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with summary information
    """
    if not results:
        return {
            "num_cases": 0,
            "provider": "unknown",
            "timestamp": None
        }
    
    # Extract provider name from first result
    provider = results[0].get("provider_name", "unknown")
    
    # Extract timestamp from first result (if available)
    timestamp = results[0].get("timestamp", None)
    
    # Count by category
    categories = {}
    for result in results:
        cat = result.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "num_cases": len(results),
        "provider": provider,
        "timestamp": timestamp,
        "categories": categories
    }


def print_summary(summary: Dict[str, Any], results_path: str | Path) -> None:
    """
    Print concise summary of results.
    
    Args:
        summary: Summary dictionary from summarize_results()
        results_path: Path to results file (for display)
    """
    print(f"\nResults: {Path(results_path).name}")
    print(f"  Cases: {summary['num_cases']}")
    print(f"  Provider: {summary['provider']}")
    if summary.get('timestamp'):
        print(f"  Timestamp: {summary['timestamp']}")
    if summary.get('categories'):
        print(f"  Categories: {', '.join(f'{k}({v})' for k, v in summary['categories'].items())}")


def compare_results(
    current_results: List[Dict[str, Any]],
    previous_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
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
                "current_length": len(current_output),
                "previous_length": len(previous_output),
                "current_preview": current_output[:120] + ("..." if len(current_output) > 120 else ""),
                "previous_preview": previous_output[:120] + ("..." if len(previous_output) > 120 else ""),
                "current_output": current_output,
                "previous_output": previous_output,
                "changed": True
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


def print_comparison(comparison: Dict[str, Any]) -> None:
    """
    Print comparison results in a concise format.
    
    Args:
        comparison: Comparison dictionary from compare_results()
    """
    print(f"\nComparison:")
    print(f"  Current: {comparison['total_current']} cases")
    print(f"  Previous: {comparison['total_previous']} cases")
    print(f"  Changed outputs: {comparison['changed_count']}")
    
    if comparison['new']:
        print(f"  New test cases: {len(comparison['new'])} ({', '.join(comparison['new'])})")
    
    if comparison['removed']:
        print(f"  Removed test cases: {len(comparison['removed'])} ({', '.join(comparison['removed'])})")
    
    if comparison['changed']:
        print(f"\n  Changed outputs:")
        for change in comparison['changed']:
            test_id = change['test_case_id']
            cat = change['category']
            curr_len = change['current_length']
            prev_len = change['previous_length']
            diff = curr_len - prev_len
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            prev_preview = change.get('previous_preview', '')
            curr_preview = change.get('current_preview', '')
            print(f"    {test_id} ({cat}): {prev_len} → {curr_len} chars ({diff_str})")
            print(f"      Previous: {prev_preview}")
            print(f"      Current:  {curr_preview}")


def load_test_cases(test_cases_path: str) -> List[Dict[str, Any]]:
    """
    Load golden test cases from JSON file.
    
    Args:
        test_cases_path: Path to test_cases.json file
        
    Returns:
        List of test case dictionaries
    """
    with open(test_cases_path, 'r') as f:
        test_cases = json.load(f)
    return test_cases


def prepare_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a test case for evaluation.
    
    This function formats the test case into the structure needed for
    model inference. The exact format will depend on the chosen
    framework and model API.
    
    Args:
        test_case: Test case dictionary with id, messages, ideal_response, etc.
        
    Returns:
        Prepared test case ready for model input
    """
    # TODO: Format messages according to chosen framework/model requirements
    # TODO: Extract system prompt if needed
    # TODO: Format conversation history appropriately
    # TODO: Handle multi-turn conversations
    pass


def generate_response(model: Any, prepared_case: Dict[str, Any]) -> str:
    """
    Generate model response for a test case.
    
    This function will call the model API/framework to generate a response.
    Implementation depends on chosen framework (transformers, litellm, etc.).
    
    Args:
        model: Model object (type depends on framework)
        prepared_case: Prepared test case ready for model input
        
    Returns:
        Generated response string
    """
    # TODO: Call model with prepared input
    # TODO: Extract response from model output
    # TODO: Handle any framework-specific response formatting
    pass


def score_response(generated: str, ideal: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score a generated response against the ideal response.
    
    This function will implement the evaluation rubric (rubric.md).
    Scoring can include both automated metrics and human-evaluated dimensions.
    
    Args:
        generated: Model-generated response
        ideal: Ideal response from test case
        test_case: Full test case dictionary for context
        
    Returns:
        Dictionary with scores for each dimension:
        - factual_correctness (0-2)
        - helpfulness (0-2)
        - tone (0-2)
        - safety (0-2)
        - escalation (0-2)
        - total_score (0-10)
    """
    # TODO: Implement rubric-based scoring
    # TODO: Evaluate factual correctness
    # TODO: Evaluate helpfulness/task completion
    # TODO: Evaluate tone and professionalism
    # TODO: Evaluate safety and hallucination avoidance
    # TODO: Evaluate escalation appropriateness
    # TODO: Calculate total score
    # TODO: Optionally include automated metrics (BLEU, ROUGE, semantic similarity)
    pass


def evaluate_single_case(model: Any, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate model on a single test case.
    
    This is the core evaluation loop for one test case:
    1. Prepare test case for model input
    2. Generate model response
    3. Score response against ideal
    4. Return results
    
    Args:
        model: Model object
        test_case: Test case dictionary
        
    Returns:
        Dictionary with test case id, generated response, scores, and metadata
    """
    # TODO: Prepare test case
    # TODO: Generate response
    # TODO: Score response
    # TODO: Compile results
    pass


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate evaluation results across all test cases.
    
    Calculates summary statistics and metrics across the full test set.
    
    Args:
        results: List of individual test case results
        
    Returns:
        Dictionary with aggregated metrics:
        - average scores per dimension
        - overall average score
        - score distribution
        - category breakdowns
    """
    # TODO: Calculate average scores for each dimension
    # TODO: Calculate overall average
    # TODO: Generate score distributions
    # TODO: Break down by category
    # TODO: Identify patterns in failures
    pass


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save evaluation results to file.
    
    Args:
        results: List of evaluation results
        output_path: Path to save results JSON file
    """
    # TODO: Format results for output
    # TODO: Include metadata (timestamp, model info, etc.)
    # TODO: Save to JSON file
    pass


def main():
    """
    Main evaluation function.
    
    Can either:
    1. Summarize a results file
    2. Compare two results files
    3. Run full evaluation (TBD - scoring not implemented)
    """
    parser = argparse.ArgumentParser(
        description="Evaluate customer support model results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/eval.py
  python evaluation/eval.py --compare evaluation/results/previous.json
        """
    )
    
    parser.add_argument(
        "--results",
        type=str,
        default="evaluation/results/latest_results.json",
        help="Path to results JSON file (default: evaluation/results/latest_results.json)"
    )
    
    parser.add_argument(
        "--compare",
        type=str,
        help="Path to previous results file for comparison"
    )
    
    parser.add_argument(
        "--out-summary",
        type=str,
        help="Path to save JSON summary file (only used with --compare)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    results_path = project_root / args.results
    
    # Load and summarize current results
    try:
        current_results = load_results(results_path)
        summary = summarize_results(current_results)
        print_summary(summary, results_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Compare with previous results if requested
    if args.compare:
        compare_path = project_root / args.compare
        try:
            previous_results = load_results(compare_path)
            comparison = compare_results(current_results, previous_results)
            print_comparison(comparison)
            
            # Save summary if requested
            if args.out_summary:
                summary_path = project_root / args.out_summary
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Get previous summary for comparison
                previous_summary = summarize_results(previous_results)
                
                summary_data = {
                    "current_file": str(results_path.relative_to(project_root)),
                    "previous_file": str(compare_path.relative_to(project_root)),
                    "current_summary": summary,
                    "previous_summary": previous_summary,
                    "comparison": comparison
                }
                
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, indent=2, ensure_ascii=False)
                
                print(f"\nSummary saved to: {summary_path}")
        except FileNotFoundError as e:
            print(f"Error: Cannot compare - {e}")
            return


if __name__ == "__main__":
    main()
