#!/usr/bin/env python3
"""
Supervised fine-tuning (SFT) script for customer support LLM.

Uses PEFT LoRA for efficient training with minimal compute.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Try to import TRL SFTTrainer (optional but recommended)
try:
    from trl import SFTTrainer
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    print("Warning: TRL not available. Using standard Trainer. Install with: pip install trl")

# Try to import PEFT
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("Error: PEFT is required. Install with: pip install peft")
    sys.exit(1)


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


def format_messages_with_template(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None
) -> str:
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


def tokenize_function(
    examples: dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> dict[str, Any]:
    """
    Tokenize examples for training.
    
    Args:
        examples: Batch of examples with 'messages' field
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    texts = []
    
    for messages in examples["messages"]:
        # Check if tokenizer supports chat template
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            # Extract system prompt if present
            system_prompt = None
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    user_messages.append(msg)
            
            text = format_messages_with_template(tokenizer, user_messages, system_prompt)
        else:
            # Fallback to simple template
            system_prompt = None
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    user_messages.append(msg)
            
            text = format_messages_simple(user_messages, system_prompt)
        
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """
    Print the number of trainable parameters in the model.
    
    Args:
        model: Model to analyze
    """
    trainable_params = 0
    all_param = 0
    
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"\nTrainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.4f}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning with PEFT LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training/train.py --model_id gpt2
  python training/train.py --model_id gpt2 --train_file data/processed/train_seed.jsonl --output_dir outputs/run_001
        """
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g., 'gpt2', 'meta-llama/Llama-2-7b-chat-hf')"
    )
    
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed/train_seed.jsonl",
        help="Path to training JSONL file (default: data/processed/train_seed.jsonl)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/run_001",
        help="Output directory for checkpoints (default: outputs/run_001)"
    )
    
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank (default: 8)"
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16)"
    )
    
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2)"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every N steps (default: 50)"
    )
    
    parser.add_argument(
        "--use_trl",
        action="store_true",
        help="Use TRL SFTTrainer instead of standard Trainer"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent
    train_file = project_root / args.train_file
    output_dir = project_root / args.output_dir
    
    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading model: {args.model_id}")
    print(f"Training file: {train_file}")
    print(f"Output directory: {output_dir}")
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using device: MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("Using device: CPU (training will be slow)")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if device != "cuda":
        model = model.to(device)
    
    # Configure LoRA
    print("\nConfiguring LoRA...")
    # Try to auto-detect target modules, fallback to common ones
    target_modules = None
    if hasattr(model, "config") and hasattr(model.config, "architectures"):
        # For GPT-2 and similar models
        if "GPT2" in str(model.config.architectures):
            target_modules = ["c_attn"]
        # For LLaMA and similar models
        elif "Llama" in str(model.config.architectures):
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        # For other models, try common attention modules
        else:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    print(f"LoRA target modules: {target_modules}")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    
    # Load dataset
    print(f"\nLoading dataset from: {train_file}")
    dataset = load_dataset("json", data_files=str(train_file), split="train")
    print(f"Loaded {len(dataset)} examples")
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=device != "cpu",  # Use FP16 on GPU/MPS, FP32 on CPU
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,  # Keep only last 3 checkpoints
        report_to="none",  # Disable wandb/tensorboard by default
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Create trainer
    if args.use_trl and HAS_TRL:
        print("\nUsing TRL SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            max_seq_length=args.max_seq_length,
        )
    else:
        print("\nUsing standard Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
