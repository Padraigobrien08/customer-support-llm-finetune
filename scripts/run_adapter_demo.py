#!/usr/bin/env python3
"""
Demo script for testing fine-tuned LoRA adapters.

Loads a base model with a LoRA adapter and runs sample customer support prompts.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for csft package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from csft.prompts import load_system_prompt
from typing import Any


def format_messages_simple(messages: list[dict[str, Any]], system_prompt: str | None = None) -> str:
    """
    Format messages using a simple template (fallback).
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        system_prompt: Optional system prompt
        
    Returns:
        Formatted prompt string
    """
    parts = []
    
    if system_prompt:
        parts.append(f"System: {system_prompt}")
    
    for msg in messages:
        role_label = msg["role"].capitalize()
        parts.append(f"{role_label}: {msg['content']}")
    
    # Add assistant prefix for response
    parts.append("Assistant:")
    
    return "\n".join(parts)


def format_messages_with_template(tokenizer, messages: list[dict[str, Any]], system_prompt: str | None = None) -> str:
    """
    Format messages using tokenizer's chat template.
    
    Args:
        tokenizer: Tokenizer with chat template support
        messages: List of message dicts
        system_prompt: Optional system prompt
        
    Returns:
        Formatted prompt string
    """
    chat_messages = []
    
    if system_prompt:
        chat_messages.append({"role": "system", "content": system_prompt})
    
    for msg in messages:
        chat_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    return tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Demo fine-tuned LoRA adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_adapter_demo.py --model_id gpt2 --adapter_dir outputs/smoke_001
  python scripts/run_adapter_demo.py --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter_dir outputs/smoke_001 --device mps
        """
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g., 'gpt2', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')"
    )
    
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Directory containing the LoRA adapter (e.g., 'outputs/smoke_001')"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to use (default: auto-detect)"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=160,
        help="Maximum new tokens to generate (default: 160)"
    )
    
    args = parser.parse_args()
    
    # Import heavy dependencies
    try:
        import torch
    except ImportError:
        print("Error: PyTorch (torch) is required but not installed.", file=sys.stderr)
        print("Run: bash scripts/setup_macos_apple_silicon.sh", file=sys.stderr)
        sys.exit(1)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: transformers is required but not installed.", file=sys.stderr)
        print("Run: bash scripts/setup_macos_apple_silicon.sh", file=sys.stderr)
        sys.exit(1)
    
    try:
        from peft import PeftModel
    except ImportError:
        print("Error: peft is required but not installed.", file=sys.stderr)
        print("Run: bash scripts/setup_macos_apple_silicon.sh", file=sys.stderr)
        sys.exit(1)
    
    # Resolve paths
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.is_absolute():
        adapter_dir = project_root / adapter_dir
    
    if not adapter_dir.exists():
        print(f"Error: Adapter directory not found: {adapter_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Detect device
    if args.device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("Using device: MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            device = "cuda"
            print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            device = "cpu"
            print("Using device: CPU")
    else:
        device = args.device
        print(f"Using device: {device.upper()}")
    
    print(f"\nLoading base model: {args.model_id}")
    print(f"Loading adapter from: {adapter_dir}")
    print()
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error: Failed to load tokenizer: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Check if tokenizer supports chat template
    supports_chat_template = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None
    
    # Load base model
    try:
        model_dtype = torch.float32 if device == "cpu" else torch.float16
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=model_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        if device != "cuda":
            base_model = base_model.to(device)
    except Exception as e:
        print(f"Error: Failed to load base model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load adapter
    try:
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model.eval()
    except Exception as e:
        print(f"Error: Failed to load adapter from {adapter_dir}: {e}", file=sys.stderr)
        print(f"Make sure the adapter directory contains valid PEFT adapter files.", file=sys.stderr)
        sys.exit(1)
    
    # Load system prompt
    prompts_dir = project_root / "prompts"
    try:
        system_prompt = load_system_prompt(prompts_dir / "system.txt")
    except Exception as e:
        print(f"Warning: Could not load system prompt: {e}", file=sys.stderr)
        system_prompt = None
    
    # Test prompts
    test_prompts = [
        "What are your business hours?",
        "I was charged twice. What can you do?",
        "Please reset my password; I can't access my account.",
    ]
    
    print("=" * 70)
    print("Running adapter demo")
    print("=" * 70)
    print()
    
    for idx, user_prompt in enumerate(test_prompts, 1):
        print(f"[Prompt {idx}]")
        print(f"User: {user_prompt}")
        print()
        
        # Build messages
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        
        # Format prompt
        if supports_chat_template:
            prompt = format_messages_with_template(tokenizer, messages, system_prompt)
        else:
            prompt = format_messages_simple(messages, system_prompt)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # Deterministic
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        print(f"Assistant: {response}")
        print()
        print("-" * 70)
        print()
    
    print("Demo complete!")


if __name__ == "__main__":
    main()

