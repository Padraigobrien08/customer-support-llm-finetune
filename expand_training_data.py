#!/usr/bin/env python3
"""
Script to significantly expand training data by:
1. Converting training_threads.json to jsonl format
2. Adding variations of existing examples
3. Creating new high-quality examples
4. Ensuring diversity in scenarios
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).parent
TRAINING_THREADS = PROJECT_ROOT / "frontend" / "src" / "data" / "training_threads.json"
TRAIN_OUTPUT = PROJECT_ROOT / "data" / "splits" / "train_expanded.jsonl"
VAL_OUTPUT = PROJECT_ROOT / "data" / "splits" / "val_expanded.jsonl"
EXISTING_TRAIN = PROJECT_ROOT / "data" / "splits" / "train.jsonl"
EXISTING_VAL = PROJECT_ROOT / "data" / "splits" / "val.jsonl"

# System prompt variations
SYSTEM_PROMPTS = [
    "You are a customer support assistant. Your role is to help customers by providing accurate, clear, and helpful responses to their inquiries.",
    "You are a helpful customer support representative. Provide clear, accurate, and empathetic assistance to customers.",
    "You are a professional customer service agent. Help customers resolve their issues with clear, accurate information.",
]

# New high-quality examples to add
NEW_EXAMPLES = [
    {
        "user": "How do I reset my password?",
        "assistant": "I can help you reset your password. Go to the login page and click 'Forgot Password'. Enter your email address and click 'Send Reset Link'. Check your email for the password reset link and click it. Enter your new password twice to confirm, then click 'Save'. Your password will be updated immediately.",
        "category": "account_management",
        "escalation": False,
    },
    {
        "user": "I forgot my password",
        "assistant": "No problem! I can help you reset it. Click 'Forgot Password' on the login page, enter your email address, and we'll send you a reset link. Check your email and follow the instructions in the link to create a new password.",
        "category": "account_management",
        "escalation": False,
    },
    {
        "user": "My password isn't working",
        "assistant": "I'm sorry you're having trouble logging in. Let's reset your password. Go to the login page, click 'Forgot Password', and enter your email. We'll send you a secure link to create a new password. Make sure to check your spam folder if you don't see the email within a few minutes.",
        "category": "account_management",
        "escalation": False,
    },
    {
        "user": "How do I change my email address?",
        "assistant": "To change your email address, go to Account Settings and select 'Email'. Enter your new email address and click 'Save'. You'll receive a verification email at the new address. Click the verification link to confirm the change. Your email will be updated once verified.",
        "category": "account_management",
        "escalation": False
    },
    {
        "user": "I want to cancel my subscription",
        "assistant": "I understand you'd like to cancel your subscription. To cancel, go to Account Settings, then Subscriptions. Click 'Cancel Subscription' and follow the prompts. Your subscription will remain active until the end of your current billing period, and you'll continue to have access until then. Is there a specific reason you're canceling? I'd be happy to help address any concerns.",
        "category": "billing_payments",
        "escalation": False,
    },
    {
        "user": "How do I update my payment method?",
        "assistant": "To update your payment method, go to Account Settings and select 'Payment Methods'. Click 'Add New Payment Method' and enter your card details. Once added, you can set it as your default payment method. Your new payment method will be used for all future charges.",
        "category": "billing_payments",
        "escalation": False,
    },
    {
        "user": "I was charged twice",
        "assistant": "I'm sorry to hear about the duplicate charge. To resolve this quickly, I'll need to connect you with our billing team who can access your account and process a refund. They can also investigate why the duplicate charge occurred. Would you like me to transfer you?",
        "category": "billing_payments",
        "escalation": True,
    },
    {
        "user": "Where is my order?",
        "assistant": "I'd be happy to help you track your order. To find your order status, I'll need your order number. You can find it in your order confirmation email. Once I have that, I can check the shipping status and provide you with tracking information.",
        "category": "shipping_delivery",
        "escalation": False,
    },
    {
        "user": "My order hasn't arrived",
        "assistant": "I'm sorry your order hasn't arrived yet. Let me help you track it down. I'll need your order number to check the shipping status. If it's past the expected delivery date, I can connect you with our shipping team who can investigate and provide updates. Would that be helpful?",
        "category": "shipping_delivery",
        "escalation": True,
    },
    {
        "user": "Can I return this item?",
        "assistant": "Yes, you can return the item. Our return policy allows returns within 30 days of purchase. To start a return, go to your order history, select the item, and click 'Return Item'. Follow the prompts to print a return label. Once we receive the item, we'll process your refund within 5-7 business days.",
        "category": "returns_refunds",
        "escalation": False,
    },
    {
        "user": "I want a refund",
        "assistant": "I can help you with a refund. To process your refund, I'll need your order number. You can find it in your order confirmation email. Once I have that information, I can check if the item is eligible for a refund and process it for you. Refunds typically take 5-7 business days to appear in your account.",
        "category": "returns_refunds",
        "escalation": False,
    },
    {
        "user": "The app keeps crashing",
        "assistant": "I'm sorry the app is crashing. Let's try a few troubleshooting steps. First, try closing the app completely and reopening it. If that doesn't work, restart your device. Make sure you have the latest version of the app installed. If the problem persists, I can connect you with our technical support team who can investigate further.",
        "category": "technical_issues",
        "escalation": True,
    },
    {
        "user": "I can't log in",
        "assistant": "I'm sorry you're having trouble logging in. Let's try a few things. First, make sure you're using the correct email address and password. Check that your caps lock isn't on. If you've forgotten your password, click 'Forgot Password' to reset it. If you're still having issues, I can help you troubleshoot further.",
        "category": "account_management",
        "escalation": False,
    },
    {
        "user": "How do I contact customer service?",
        "assistant": "You can reach our customer service team in several ways. You can call us at 1-800-XXX-XXXX, email us at support@example.com, or use the live chat feature on our website. Our team is available Monday through Friday, 9 AM to 5 PM EST. Is there something specific I can help you with right now?",
        "category": "general_information",
        "escalation": False,
    },
    {
        "user": "What are your business hours?",
        "assistant": "Our customer service team is available Monday through Friday, 9 AM to 5 PM EST. You can reach us by phone, email, or live chat during these hours. For urgent matters outside business hours, you can leave a message and we'll respond as soon as possible.",
        "category": "general_information",
        "escalation": False,
    },
    {
        "user": "I need to update my shipping address",
        "assistant": "I can help you update your shipping address. Go to Account Settings, then select 'Addresses'. Click 'Edit' on your default shipping address, make your changes, and click 'Save'. Your new address will be used for all future orders. If you have a pending order, I can help you update that as well.",
        "category": "account_management",
        "escalation": False,
    },
    {
        "user": "How do I track my order?",
        "assistant": "To track your order, go to your order history and click on the order you want to track. You'll see the tracking number and a link to the carrier's tracking page. You can also find the tracking number in your shipping confirmation email. Click the tracking link to see real-time updates on your package's location.",
        "category": "shipping_delivery",
        "escalation": False,
    },
    {
        "user": "My payment was declined",
        "assistant": "I'm sorry your payment was declined. This can happen for several reasons. Check that your card has sufficient funds or available credit. Verify that your billing address matches the one on file with your bank. Make sure your card hasn't expired. If everything looks correct, try a different payment method or contact your bank to ensure the card is active.",
        "category": "billing_payments",
        "escalation": False,
    },
    {
        "user": "I didn't receive my confirmation email",
        "assistant": "I'm sorry you didn't receive your confirmation email. First, check your spam or junk folder, as sometimes these emails get filtered. Make sure you're checking the correct email address. If you still don't see it, I can resend the confirmation email. What email address should I send it to?",
        "category": "account_management",
        "escalation": False,
    },
    {
        "user": "How do I delete my account?",
        "assistant": "I'm sorry to see you go. To delete your account, go to Account Settings, then Privacy, and select 'Delete Account'. You'll be asked to confirm your password. Please note that deleting your account is permanent and cannot be undone. All your data will be permanently deleted. Is there anything I can help with that might change your mind?",
        "category": "account_management",
        "escalation": False,
    },
    {
        "user": "I have a complaint",
        "assistant": "I'm sorry to hear you have a complaint. I want to make sure your concerns are properly addressed. To help you best, could you tell me more about what happened? Once I understand the situation, I can connect you with the right team member who can help resolve this for you.",
        "category": "complaints_feedback",
        "escalation": True,
    },
    {
        "user": "The website isn't loading",
        "assistant": "I'm sorry the website isn't loading for you. Let's try a few troubleshooting steps. First, try refreshing the page. Clear your browser cache and cookies. Make sure you have a stable internet connection. Try using a different browser or device. If the problem persists, it might be a temporary server issue. Please try again in a few minutes.",
        "category": "technical_issues",
        "escalation": False,
    },
]


def load_existing_data(filepath: Path) -> List[Dict]:
    """Load existing jsonl data."""
    examples = []
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    return examples


def convert_thread_to_jsonl(thread: Dict) -> Dict:
    """Convert a training thread to jsonl format."""
    messages = []
    
    # Add system message if present, otherwise use default
    system_content = "You are a customer support assistant. Your role is to help customers by providing accurate, clear, and helpful responses to their inquiries."
    
    # Extract user and assistant messages
    for msg in thread.get("messages", []):
        if msg["role"] in ["user", "assistant"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    if not messages:
        return None
    
    return {
        "messages": [
            {"role": "system", "content": system_content},
            *messages
        ],
        "metadata": {
            "source": "training_threads",
            "thread_id": thread.get("id", ""),
        }
    }


def create_variation(example: Dict) -> Dict:
    """Create a variation of an example by changing system prompt."""
    variation = json.loads(json.dumps(example))  # Deep copy
    if variation["messages"] and variation["messages"][0]["role"] == "system":
        variation["messages"][0]["content"] = random.choice(SYSTEM_PROMPTS)
        variation["metadata"]["source"] = variation["metadata"].get("source", "unknown") + "_variation"
    return variation


def add_new_example(user_msg: str, assistant_msg: str, category: str, escalation: bool) -> Dict:
    """Create a new training example."""
    system_prompt = random.choice(SYSTEM_PROMPTS)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "metadata": {
            "source": "manual_expansion",
            "category": category,
            "escalation": escalation,
        }
    }


def main():
    """Main function to expand training data."""
    print("=" * 80)
    print("Expanding Training Data")
    print("=" * 80)
    
    # Load existing data
    print("\n1. Loading existing data...")
    existing_train = load_existing_data(EXISTING_TRAIN)
    existing_val = load_existing_data(EXISTING_VAL)
    print(f"   Loaded {len(existing_train)} training examples")
    print(f"   Loaded {len(existing_val)} validation examples")
    
    # Load and convert training threads
    print("\n2. Converting training_threads.json...")
    all_examples = []
    
    if TRAINING_THREADS.exists():
        with open(TRAINING_THREADS, 'r', encoding='utf-8') as f:
            threads = json.load(f)
        
        for thread in threads:
            converted = convert_thread_to_jsonl(thread)
            if converted:
                all_examples.append(converted)
        
        print(f"   Converted {len(all_examples)} examples from training_threads.json")
    else:
        print(f"   Warning: {TRAINING_THREADS} not found")
    
    # Add existing training data
    all_examples.extend(existing_train)
    
    # Add new high-quality examples
    print("\n3. Adding new high-quality examples...")
    for example_data in NEW_EXAMPLES:
        new_example = add_new_example(
            example_data["user"],
            example_data["assistant"],
            example_data["category"],
            example_data["escalation"]
        )
        all_examples.append(new_example)
    print(f"   Added {len(NEW_EXAMPLES)} new examples")
    
    # Create variations (50% of examples)
    print("\n4. Creating variations...")
    variations = []
    sample_size = len(all_examples) // 2
    sampled = random.sample(all_examples, min(sample_size, len(all_examples)))
    for example in sampled:
        variation = create_variation(example)
        variations.append(variation)
    all_examples.extend(variations)
    print(f"   Created {len(variations)} variations")
    
    # Shuffle all examples
    random.shuffle(all_examples)
    
    # Split into train/val (90/10)
    split_point = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_point]
    val_examples = all_examples[split_point:]
    
    print(f"\n5. Final counts:")
    print(f"   Training examples: {len(train_examples)}")
    print(f"   Validation examples: {len(val_examples)}")
    
    # Write to files
    print("\n6. Writing to files...")
    with open(TRAIN_OUTPUT, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    with open(VAL_OUTPUT, 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"   Saved to {TRAIN_OUTPUT}")
    print(f"   Saved to {VAL_OUTPUT}")
    
    print("\n" + "=" * 80)
    print("✓ Training data expansion complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Review the expanded data files")
    print(f"2. Replace train.jsonl and val.jsonl with train_expanded.jsonl and val_expanded.jsonl")
    print(f"3. Re-train the model with the expanded dataset")


if __name__ == "__main__":
    main()
