#!/usr/bin/env python3
"""
Build training JSONL from golden test cases.

Converts evaluation/test_cases.json into a JSONL file suitable for fine-tuning.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path for csft package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from csft.io import load_test_cases, save_jsonl
from csft.prompts import load_guidelines, load_system_prompt
from csft.types import ChatMessage, MessageRole


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


def validate_messages(messages: list[dict[str, Any]]) -> None:
    """
    Validate messages structure and content.
    
    Args:
        messages: List of message dictionaries
        
    Raises:
        ValueError: If validation fails
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")
    
    valid_roles = {"system", "user", "assistant"}
    has_user = False
    has_assistant = False
    
    for idx, msg in enumerate(messages):
        # Check required fields
        if "role" not in msg:
            raise ValueError(f"Message {idx} missing 'role' field")
        if "content" not in msg:
            raise ValueError(f"Message {idx} missing 'content' field")
        
        # Validate role
        role = msg["role"]
        if role not in valid_roles:
            raise ValueError(f"Message {idx} has invalid role '{role}'. Must be one of: {valid_roles}")
        
        # Validate content
        content = msg["content"]
        if not isinstance(content, str):
            raise ValueError(f"Message {idx} content must be a string, got {type(content).__name__}")
        if not content.strip():
            raise ValueError(f"Message {idx} content cannot be empty or whitespace only")
        
        # Track required roles
        if role == "user":
            has_user = True
        if role == "assistant":
            has_assistant = True
    
    # Validate required roles
    if not has_user:
        raise ValueError("Messages must contain at least one 'user' message")
    if not has_assistant:
        raise ValueError("Messages must contain at least one 'assistant' message")


def build_training_example(
    test_case: Any,
    system_prompt: str
) -> dict[str, Any]:
    """
    Build a training example from a test case.
    
    Args:
        test_case: TestCase object from test_cases.json
        system_prompt: System prompt to use
        
    Returns:
        Dictionary with messages and metadata
    """
    messages = []
    
    # Add system message
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Add user messages from test case (excluding system messages)
    for msg in test_case.messages:
        if msg.role.value != "system":
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
    
    # Add assistant message (ideal_response)
    messages.append({
        "role": "assistant",
        "content": test_case.ideal_response
    })
    
    # Validate the messages
    validate_messages(messages)
    
    # Build example with metadata
    example = {
        "messages": messages,
        "metadata": {
            "test_case_id": test_case.id,
            "category": test_case.category
        }
    }
    
    return example


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Build training JSONL from golden test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_training_jsonl.py
  python scripts/build_training_jsonl.py --include-guidelines
        """
    )
    
    parser.add_argument(
        "--include-guidelines",
        action="store_true",
        help="Include distilled guidelines in system prompt"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/train_seed.jsonl",
        help="Output path for JSONL file (default: data/processed/train_seed.jsonl)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    test_cases_path = project_root / "evaluation" / "test_cases.json"
    prompts_dir = project_root / "prompts"
    output_path = project_root / args.output
    
    print(f"Loading test cases from: {test_cases_path}")
    try:
        test_cases = load_test_cases(str(test_cases_path))
    except Exception as e:
        print(f"Error loading test cases: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(test_cases)} test cases")
    
    print(f"Loading system prompt from: {prompts_dir}")
    try:
        system_prompt_base = load_system_prompt(prompts_dir / "system.txt")
    except Exception as e:
        print(f"Error loading system prompt: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Optionally include guidelines
    system_prompt = system_prompt_base
    if args.include_guidelines:
        print("Including guidelines in system prompt...")
        try:
            guidelines_full = load_guidelines(prompts_dir / "guidelines.md")
            guidelines_summary = distill_guidelines_summary(guidelines_full)
            if guidelines_summary:
                system_prompt = f"{system_prompt_base}\n\n## Key Guidelines\n\n{guidelines_summary}"
        except Exception as e:
            print(f"Warning: Could not load guidelines: {e}", file=sys.stderr)
    
    # Build training examples
    print("Building training examples...")
    examples = []
    for idx, test_case in enumerate(test_cases, 1):
        try:
            example = build_training_example(test_case, system_prompt)
            examples.append(example)
        except ValueError as e:
            print(f"Error processing test case {test_case.id}: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error processing test case {test_case.id}: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Save to JSONL
    print(f"Saving to: {output_path}")
    try:
        # Convert to plain dicts for JSONL
        jsonl_data = []
        for example in examples:
            jsonl_data.append(example)
        
        save_jsonl(jsonl_data, output_path)
    except Exception as e:
        print(f"Error saving JSONL: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print summary
    print(f"\n✓ Successfully created {len(examples)} training examples")
    
    # Print example line (first line, truncated)
    if examples:
        example_json = json.dumps(examples[0], ensure_ascii=False)
        if len(example_json) > 200:
            example_preview = example_json[:200] + "..."
        else:
            example_preview = example_json
        print(f"\nExample (first line, truncated):")
        print(f"  {example_preview}")


if __name__ == "__main__":
    main()

