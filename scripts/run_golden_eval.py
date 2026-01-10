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
from typing import Any, Dict

# Add project root to path for csft package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from csft.io import load_test_cases
from csft.prompts import load_and_assemble_system_message
from csft.providers import MockProvider, Provider
from csft.types import ChatMessage, MessageRole, ModelResponse, TestCase


def create_provider(provider_name: str, **kwargs) -> Provider:
    """
    Create a provider instance based on name.
    
    Args:
        provider_name: Name of the provider ('mock' or 'hf_local')
        **kwargs: Provider-specific arguments:
            - For 'hf_local': model_id (required), device, dtype, max_new_tokens, adapter_dir (optional)
        
    Returns:
        Provider instance
        
    Raises:
        ValueError: If provider name is not recognized or required args missing
    """
    if provider_name == "mock":
        return MockProvider()
    elif provider_name == "hf_local":
        try:
            from csft.providers.hf_local import HFLocalProvider
        except ImportError:
            raise ValueError(
                "HuggingFace provider requires transformers and torch. "
                "Install with: pip install transformers torch accelerate"
            )
        
        model_id = kwargs.get("model_id")
        if not model_id:
            raise ValueError(
                "hf_local provider requires --model-id. "
                "Example: --provider hf_local --model-id gpt2"
            )
        
        # Resolve adapter path if provided
        adapter_dir = kwargs.get("adapter_dir")
        adapter_path = None
        if adapter_dir:
            adapter_path = Path(adapter_dir)
            if not adapter_path.is_absolute():
                adapter_path = project_root / adapter_path
            
            if not adapter_path.exists():
                raise ValueError(f"Adapter directory not found: {adapter_path}")
            
            adapter_path = str(adapter_path)
        
        provider = HFLocalProvider(
            model_id=model_id,
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            adapter_path=adapter_path
        )
        
        return provider
    else:
        available = "mock, hf_local"
        raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")


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
    system_prompt: str,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Run a single test case through the provider.
    
    Never crashes - always returns a result dictionary, even on failure.
    
    Args:
        provider: Model provider instance
        test_case: Test case to evaluate
        system_prompt: Assembled system prompt
        debug: If True, include full traceback in error metadata
        
    Returns:
        Dictionary with test case results (includes error fields if generation failed)
    """
    import traceback
    
    # Build base result structure
    base_result = {
        "test_case_id": test_case.id,
        "category": test_case.category,
        "messages": [{"role": msg.role.value, "content": msg.content} for msg in test_case.messages],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    # Wrap everything in try/except to never crash
    traceback_str = None  # Store traceback if exception occurs
    try:
        # Convert test case messages to ChatMessage objects (excluding system if present)
        messages = []
        for msg in test_case.messages:
            if msg.role.value != "system":  # System prompt handled separately
                messages.append(msg)
        
        # Try to generate response
        try:
            response = provider.generate(
                messages=messages,
                system_prompt=system_prompt
            )
        except Exception as e:
            # Generation failed with exception - create ModelResponse with error
            error_type = type(e).__name__
            error_message = str(e)
            if debug:
                traceback_str = traceback.format_exc()
            response = ModelResponse(
                content="",
                success=False,
                error_type=error_type,
                error_message=error_message
            )
        
        # Check if content is empty/whitespace
        if not response.content.strip():
            # Empty or whitespace-only content
            if response.success:  # Only override if not already marked as failed
                response.success = False
                if not response.error_type:
                    response.error_type = "EmptyOutput"
                if not response.error_message:
                    response.error_message = "Model generated empty or whitespace-only response"
        
        # Add output_text and success status
        base_result["output_text"] = response.content
        base_result["success"] = response.success
        
        # Add error fields if present
        if not response.success:
            if response.error_type:
                base_result["error_type"] = response.error_type
            if response.error_message:
                base_result["error_message"] = response.error_message
            
            # Print concise failure line to stderr
            error_type_str = response.error_type or "UnknownError"
            error_msg_str = response.error_message or "Unknown error"
            print(f"FAILED {test_case.id}: {error_type_str}: {error_msg_str}", file=sys.stderr)
            
            # Add traceback if debug mode and we captured one
            if debug and traceback_str:
                base_result["traceback"] = traceback_str
        
        return base_result
        
    except Exception as e:
        # Unexpected error during result processing - create error result
        error_type = type(e).__name__
        error_message = str(e)
        
        base_result["output_text"] = ""
        base_result["success"] = False
        base_result["error_type"] = error_type
        base_result["error_message"] = error_message
        
        if debug:
            base_result["traceback"] = traceback.format_exc()
        
        # Print error to stderr
        print(f"FAILED {test_case.id}: {error_type}: {error_message}", file=sys.stderr)
        
        return base_result


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run golden evaluation on test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_golden_eval.py
  python scripts/run_golden_eval.py --provider mock --out evaluation/results/custom_results.json
  python scripts/run_golden_eval.py --provider hf_local --model-id gpt2
  python scripts/run_golden_eval.py --provider hf_local --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter-dir outputs/smoke_001
  python scripts/run_golden_eval.py --provider hf_local --model-id gpt2 --device mps --max-new-tokens 256
        """
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["mock", "hf_local"],
        help="Provider to use: 'mock' or 'hf_local' (default: mock)"
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        help="Model ID for hf_local provider (required when --provider is hf_local, e.g., 'gpt2', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')"
    )
    
    parser.add_argument(
        "--adapter-dir",
        type=str,
        help="Directory containing LoRA adapter (optional, e.g., 'outputs/smoke_001')"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        help="Device for hf_local provider (cuda, mps, or cpu; default: auto-detect)"
    )
    
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate for hf_local provider (default: 512)"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="evaluation/results/latest_results.json",
        help="Output path for results JSON (default: evaluation/results/latest_results.json)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include full traceback in error metadata when generation fails (default: False)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root (already set above)
    # Load from manual cases (70 cases) instead of test_cases.json (40 cases)
    manual_cases_path = project_root / "data" / "raw" / "manual_cases.jsonl"
    prompts_dir = project_root / "prompts"
    output_path = project_root / args.out
    
    print(f"Loading test cases from: {manual_cases_path}")
    
    # Load raw JSONL to get metadata and messages
    import json
    with open(manual_cases_path, 'r', encoding='utf-8') as f:
        raw_cases = [json.loads(line) for line in f if line.strip()]
    
    # Convert to TestCase format
    test_cases = []
    for raw_case in raw_cases:
        # Filter out system messages (system prompt handled separately)
        messages = [ChatMessage(role=MessageRole(msg["role"]), content=msg["content"]) 
                   for msg in raw_case["messages"] if msg["role"] != "system"]
        
        # Extract assistant response as ideal_response
        assistant_msg = next((msg for msg in raw_case["messages"] if msg["role"] == "assistant"), None)
        ideal_response = assistant_msg["content"] if assistant_msg else ""
        
        # Create test case
        metadata = raw_case.get("metadata", {})
        test_case = TestCase(
            id=metadata.get("test_case_id", f"manual_{len(test_cases)+1:03d}"),
            category=metadata.get("category", "unknown"),
            messages=messages,
            ideal_response=ideal_response,
            notes=f"Source: {metadata.get('source', 'manual')}, Difficulty: {metadata.get('difficulty', 'unknown')}"
        )
        test_cases.append(test_case)
    
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
    
    # Validate required arguments
    if args.provider == "hf_local" and not args.model_id:
        parser.error("--model-id is required when --provider is hf_local")
    
    print(f"Creating provider: {args.provider}")
    provider_kwargs = {}
    if args.model_id:
        provider_kwargs["model_id"] = args.model_id
    if args.adapter_dir:
        provider_kwargs["adapter_dir"] = args.adapter_dir
    if args.device:
        provider_kwargs["device"] = args.device
    if args.max_new_tokens:
        provider_kwargs["max_new_tokens"] = args.max_new_tokens
    
    provider = create_provider(args.provider, **provider_kwargs)
    
    # Store adapter_dir for output (resolve path)
    adapter_dir_for_output = None
    if args.adapter_dir:
        adapter_path = Path(args.adapter_dir)
        if not adapter_path.is_absolute():
            adapter_path = project_root / adapter_path
        adapter_dir_for_output = str(adapter_path.relative_to(project_root)) if adapter_path.exists() else args.adapter_dir
    
    print(f"Running {len(test_cases)} test cases...")
    results = []
    failed_count = 0
    
    for idx, test_case in enumerate(test_cases, 1):
        print(f"  [{idx}/{len(test_cases)}] Running test case: {test_case.id}")
        # run_test_case never crashes, but wrap in try/except just in case
        try:
            result = run_test_case(provider, test_case, system_prompt, debug=args.debug)
        except Exception as e:
            # This should never happen, but handle it just in case
            import traceback
            failed_count += 1
            error_type = type(e).__name__
            error_message = str(e)
            print(f"FAILED {test_case.id}: {error_type}: {error_message}", file=sys.stderr)
            
            result = {
                "test_case_id": test_case.id,
                "category": test_case.category,
                "messages": [{"role": msg.role.value, "content": msg.content} for msg in test_case.messages],
                "output_text": "",
                "success": False,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "error_type": error_type,
                "error_message": error_message,
            }
            if args.debug:
                result["traceback"] = traceback.format_exc()
        
        # Add provider info to result (always, even on error)
        result["provider"] = args.provider
        result["model_id"] = args.model_id if args.provider == "hf_local" else None
        result["adapter_dir"] = adapter_dir_for_output
        
        # Track failures
        if not result.get("success", True) or "error_type" in result:
            failed_count += 1
        
        results.append(result)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results (always write, even if some cases failed)
    print(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    success_count = len(results) - failed_count
    print(f"✓ Evaluation complete. {len(results)} results saved.")
    if failed_count > 0:
        print(f"  Successful: {success_count}")
        print(f"  Failed: {failed_count}", file=sys.stderr)
    else:
        print(f"  All {len(results)} test cases completed successfully.")


if __name__ == "__main__":
    main()

