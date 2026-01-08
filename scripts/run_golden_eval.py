#!/usr/bin/env python3
"""
Golden evaluation runner.

Runs test cases through a model provider and saves results.
This is a baseline runner - scoring is not implemented yet.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for csft package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from csft.io import load_test_cases
from csft.prompts import load_and_assemble_system_message
from csft.providers import MockProvider, Provider
from csft.types import ChatMessage, TestCase


def create_provider(provider_name: str) -> Provider:
    """
    Create a provider instance based on name.
    
    Args:
        provider_name: Name of the provider ('mock' for now)
        
    Returns:
        Provider instance
        
    Raises:
        ValueError: If provider name is not recognized
    """
    if provider_name == "mock":
        return MockProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Available: mock")


def distill_guidelines_summary(guidelines_text: str) -> str:
    """
    Create a short summary of guidelines for inclusion in system prompt.
    
    Extracts key principles without full detail.
    
    Args:
        guidelines_text: Full guidelines markdown text
        
    Returns:
        Short summary of key principles
    """
    if not guidelines_text or not guidelines_text.strip():
        return ""
    
    # Extract key sections - look for main headings
    lines = guidelines_text.split('\n')
    summary_parts = []
    
    # Look for main sections and extract first bullet points
    current_section = None
    for line in lines:
        if line.startswith('## '):
            current_section = line[3:].strip()
        elif line.strip().startswith('- ') and current_section:
            # Take first bullet point from each major section
            bullet = line.strip()[2:].strip()
            if bullet and len(summary_parts) < 5:  # Limit to 5 key points
                summary_parts.append(f"- {bullet}")
    
    if summary_parts:
        return "\n".join(summary_parts)
    
    # Fallback: return first paragraph if no bullets found
    first_para = guidelines_text.split('\n\n')[0] if '\n\n' in guidelines_text else guidelines_text[:200]
    return first_para.strip()


def run_test_case(
    provider: Provider,
    test_case: TestCase,
    system_prompt: str
) -> dict[str, Any]:
    """
    Run a single test case through the provider.
    
    Args:
        provider: Model provider instance
        test_case: Test case to evaluate
        system_prompt: Assembled system prompt
        
    Returns:
        Dictionary with test case results
    """
    # Convert test case messages to ChatMessage objects (excluding system if present)
    messages = []
    for msg in test_case.messages:
        if msg.role.value != "system":  # System prompt handled separately
            messages.append(msg)
    
    # Generate response
    response = provider.generate(
        messages=messages,
        system_prompt=system_prompt
    )
    
    # Build result
    return {
        "test_case_id": test_case.id,
        "category": test_case.category,
        "input_messages": [{"role": msg.role.value, "content": msg.content} for msg in test_case.messages],
        "model_output": response.content,
        "ideal_response": test_case.ideal_response,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "provider_name": repr(provider)
    }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run golden evaluation on test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_golden_eval.py
  python scripts/run_golden_eval.py --provider mock --out evaluation/results/custom_results.json
        """
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        help="Provider to use (default: mock)"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="evaluation/results/latest_results.json",
        help="Output path for results JSON (default: evaluation/results/latest_results.json)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root (already set above)
    test_cases_path = project_root / "evaluation" / "test_cases.json"
    prompts_dir = project_root / "prompts"
    output_path = project_root / args.out
    
    print(f"Loading test cases from: {test_cases_path}")
    test_cases = load_test_cases(str(test_cases_path))
    print(f"Loaded {len(test_cases)} test cases")
    
    print(f"Loading system prompt and guidelines from: {prompts_dir}")
    # Load full guidelines to distill
    from csft.prompts import load_guidelines, load_system_prompt
    system_prompt_base = load_system_prompt(prompts_dir / "system.txt")
    guidelines_full = load_guidelines(prompts_dir / "guidelines.md")
    
    # Distill guidelines to short summary
    guidelines_summary = distill_guidelines_summary(guidelines_full)
    
    # Assemble final system prompt
    if guidelines_summary:
        system_prompt = f"{system_prompt_base}\n\n## Key Guidelines\n\n{guidelines_summary}"
    else:
        system_prompt = system_prompt_base
    
    print(f"Creating provider: {args.provider}")
    provider = create_provider(args.provider)
    
    print(f"Running {len(test_cases)} test cases...")
    results = []
    for idx, test_case in enumerate(test_cases, 1):
        print(f"  [{idx}/{len(test_cases)}] Running test case: {test_case.id}")
        result = run_test_case(provider, test_case, system_prompt)
        results.append(result)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Evaluation complete. {len(results)} results saved.")


if __name__ == "__main__":
    main()

