#!/usr/bin/env python3
"""
Debug script for generation testing.

Loads a model (with optional PEFT adapter) and tests generation on a single test case.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path for csft package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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
        description="Debug generation on a single test case",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/debug_generation.py --model_id gpt2
  python scripts/debug_generation.py --model_id gpt2 --adapter_dir outputs/smoke_001
  python scripts/debug_generation.py --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter_dir outputs/smoke_001 --device mps
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
        default=None,
        help="Directory containing the LoRA adapter (optional)"
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
        default=128,
        help="Maximum new tokens to generate (default: 128)"
    )
    
    args = parser.parse_args()
    
    # Import heavy dependencies
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if args.adapter_dir:
        from peft import PeftModel
    
    # Resolve paths
    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir)
        if not adapter_dir.is_absolute():
            adapter_dir = project_root / adapter_dir
        
        if not adapter_dir.exists():
            print(f"Error: Adapter directory not found: {adapter_dir}", file=sys.stderr)
            sys.exit(1)
    
    # Load test case
    test_cases_path = project_root / "evaluation" / "test_cases.json"
    with open(test_cases_path, "r") as f:
        test_cases = json.load(f)
    
    # Find tc_001
    test_case = None
    for tc in test_cases:
        if tc["id"] == "tc_001":
            test_case = tc
            break
    
    if test_case is None:
        print(f"Error: Test case tc_001 not found in {test_cases_path}", file=sys.stderr)
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
    
    print(f"\nLoading model: {args.model_id}")
    if args.adapter_dir:
        print(f"Loading adapter from: {adapter_dir}")
    print()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Ensure pad_token_id is set (use eos_token_id, not 0) - critical for MPS
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 0:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Debug: Print tokenizer info
    print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}", flush=True)
    print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}", flush=True)
    print(f"Tokenizer unk_token_id: {getattr(tokenizer, 'unk_token_id', 'N/A')}", flush=True)
    print()
    
    # Check if tokenizer supports chat template
    supports_chat_template = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None
    
    # Load base model
    # Force float32 for MPS to avoid generation degeneracy
    if device == "mps":
        model_dtype = torch.float32
    elif device == "cpu":
        model_dtype = torch.float32
    else:  # CUDA
        model_dtype = torch.float16
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=model_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    if device != "cuda":
        base_model = base_model.to(device)
    
    # Load adapter if provided
    if args.adapter_dir:
        # For MPS, ensure base model is float32 before loading adapter
        if device == "mps":
            base_model = base_model.to(torch.float32)
            base_model = base_model.to("mps")
            base_model.eval()
        
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        # For MPS, ensure model is float32 and on MPS device
        if device == "mps":
            model = model.to(torch.float32)
            model = model.to("mps")
            # Verify conversion
            first_param_dtype = next(model.parameters()).dtype
            if first_param_dtype != torch.float32:
                print(f"Warning: After adapter load, dtype is {first_param_dtype}, forcing float32...", flush=True)
                model = model.to(torch.float32)
                model = model.to("mps")
    else:
        model = base_model
    
    model.eval()
    
    # Final verification for MPS
    if device == "mps":
        first_param_dtype = next(model.parameters()).dtype
        print(f"Model dtype before generation: {first_param_dtype}", flush=True)
        if first_param_dtype != torch.float32:
            print("Forcing model to float32...", flush=True)
            model = model.to(torch.float32)
            model = model.to("mps")
            model.eval()
    
    # Extract system prompt and messages from test case
    system_prompt = None
    messages = []
    for msg in test_case["messages"]:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        else:
            messages.append(msg)
    
    # Format prompt
    if supports_chat_template:
        prompt_text = format_messages_with_template(tokenizer, messages, system_prompt)
    else:
        prompt_text = format_messages_simple(messages, system_prompt)
    
    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    
    # Generate
    # Ensure pad_token_id is not 0 (critical for MPS)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None or pad_token_id == 0:
        pad_token_id = tokenizer.eos_token_id
    
    # Verify model dtype for MPS (should be float32)
    if device == "mps":
        # Check if model is actually in float32
        first_param_dtype = next(model.parameters()).dtype
        if first_param_dtype != torch.float32:
            print(f"Warning: Model dtype is {first_param_dtype}, converting to float32 for MPS...", flush=True)
            model = model.to(torch.float32)
            model = model.to("mps")
            model.eval()
    
    # For MPS, ensure inputs are in the right dtype and check logits before generation
    if device == "mps":
        # Ensure input_ids are long (int64), not float
        if inputs["input_ids"].dtype != torch.long:
            inputs["input_ids"] = inputs["input_ids"].long()
        if "attention_mask" in inputs and inputs["attention_mask"].dtype != torch.long:
            inputs["attention_mask"] = inputs["attention_mask"].long()
        
        # Debug: Check first few logits to see if model is producing reasonable outputs
        with torch.no_grad():
            try:
                outputs_before_gen = model(**inputs)
                logits = outputs_before_gen.logits[0, -1, :]  # Last token logits
                
                # Check for NaN
                nan_count = torch.isnan(logits).sum().item()
                if nan_count > 0:
                    print(f"ERROR: Found {nan_count} NaN values in logits!", flush=True)
                    print(f"This indicates numerical instability. Possible causes:", flush=True)
                    print(f"  1. Adapter weights incompatible with float32", flush=True)
                    print(f"  2. Adapter weights contain NaN values", flush=True)
                    print(f"  3. Model architecture issue on MPS", flush=True)
                    print(f"  4. Try testing without adapter: remove --adapter_dir", flush=True)
                    raise RuntimeError(
                        f"Model producing NaN logits ({nan_count}/{len(logits)} NaN values). "
                        f"This causes generation to fail. Try testing the base model without the adapter."
                    )
                
                top_5_logits, top_5_indices = torch.topk(logits, 5)
                print(f"Top 5 logits before generation: {top_5_logits.cpu().tolist()}", flush=True)
                print(f"Top 5 token indices: {top_5_indices.cpu().tolist()}", flush=True)
                print(f"Token 0 logit: {logits[0].item():.4f}", flush=True)
                print()
            except RuntimeError:
                raise
            except Exception as e:
                print(f"Warning: Could not check logits: {e}", flush=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=8,  # Prevent immediate termination
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,  # Discourage loops of special tokens
        )
    
    # Debug guard: Check for degenerate generation (token 0 repeated)
    generated_token_ids = outputs[0][input_len:]
    if len(generated_token_ids) > 0:
        zero_count = (generated_token_ids == 0).sum().item()
        zero_percentage = (zero_count / len(generated_token_ids)) * 100
        if zero_percentage > 80:
            raise RuntimeError(
                f"Degenerate generation: token 0 repeated; try float32 on MPS. "
                f"Token 0 count: {zero_count}/{len(generated_token_ids)} ({zero_percentage:.1f}%)"
            )
    
    # Decode full output
    decoded_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_len = outputs[0].shape[0]
    generated_len = output_len - input_len
    
    # Decode only generated tokens (using token slicing)
    generated_tokens = outputs[0][input_len:]
    decoded_gen_only = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Additional diagnostics
    gen_ids = generated_tokens.cpu()
    first_40_ids = gen_ids[:40].tolist()
    
    # Decode with and without skip_special_tokens
    decoded_with_special = tokenizer.decode(gen_ids, skip_special_tokens=False)
    decoded_without_special = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    # Count token frequencies
    from collections import Counter
    token_counts = Counter(gen_ids.tolist())
    top_10_tokens = token_counts.most_common(10)
    
    # Print results
    print("=" * 70)
    print("Generation Debug Output")
    print("=" * 70)
    print()
    print(f"prompt_text (first 400 chars):")
    print(prompt_text[:400])
    if len(prompt_text) > 400:
        print("...")
    print()
    print(f"input_len (tokens): {input_len}")
    print(f"output_len (tokens): {output_len}")
    print(f"generated_len (tokens): {generated_len}")
    print()
    print(f"decoded_full (first 400 chars):")
    print(decoded_full[:400])
    if len(decoded_full) > 400:
        print("...")
    print()
    print(f"decoded_gen_only (first 400 chars):")
    print(decoded_gen_only[:400])
    if len(decoded_gen_only) > 400:
        print("...")
    print()
    print("=" * 70)
    print("Additional Diagnostics")
    print("=" * 70)
    print()
    print(f"First 40 generated token IDs:")
    print(first_40_ids)
    print()
    print(f"decoded_gen_only (skip_special_tokens=True) repr():")
    print(repr(decoded_without_special))
    print()
    print(f"decoded_gen_only (skip_special_tokens=False) repr():")
    print(repr(decoded_with_special))
    print()
    print(f"Top 10 most frequent token IDs (with counts):")
    for token_id, count in top_10_tokens:
        print(f"  Token ID {token_id}: {count} occurrences")
    print()


if __name__ == "__main__":
    main()
