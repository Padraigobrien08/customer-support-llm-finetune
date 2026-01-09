#!/usr/bin/env python3
"""
Generate synthetic training examples using templates.

Creates diverse training examples without calling an LLM, using
templated user messages and safe assistant responses.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# Add project root to path for csft package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from csft.prompts import load_system_prompt


# User message templates per category
USER_TEMPLATES = {
    "information_request": [
        "What are your business hours?",
        "When are you open?",
        "What time do you close?",
        "Can you tell me your operating hours?",
        "I need to know your business hours",
        "What's your return policy?",
        "Can you explain your return policy?",
        "What's your policy on returns?",
        "How do I track my order?",
        "Where can I track my order?",
        "How do I check my order status?",
    ],
    "account_access": [
        "I can't log into my account",
        "I'm having trouble logging in",
        "My password isn't working",
        "Please reset my password",
        "I forgot my password",
        "My account is locked",
        "I can't access my account",
        "Help me log in",
    ],
    "order_inquiry": [
        "Where is my order?",
        "What's the status of my order?",
        "When will my order arrive?",
        "Can I cancel my order?",
        "I need to cancel my order",
        "I want to change my shipping address",
        "Can I modify my order?",
        "I haven't received my order yet",
    ],
    "billing_payment": [
        "I was charged twice",
        "I see a duplicate charge on my card",
        "Why was I charged twice?",
        "How do I get a refund?",
        "I need a refund",
        "My payment was declined",
        "My card was declined",
        "I want to dispute a charge",
    ],
    "product_issue": [
        "The item I received is damaged",
        "My order arrived damaged",
        "I got the wrong product",
        "You sent me the wrong item",
        "This doesn't work as described",
        "The product is defective",
        "This item is broken",
    ],
    "return_refund": [
        "I want to return this item",
        "How do I process a return?",
        "I need to return something",
        "When will I get my refund?",
        "How long does a refund take?",
        "Can I return this?",
    ],
    "technical_support": [
        "The website isn't loading",
        "Your website is down",
        "I can't complete checkout",
        "Checkout isn't working",
        "The app keeps crashing",
        "I'm having trouble with your website",
    ],
    "policy_clarification": [
        "What's your cancellation policy?",
        "Can you explain your warranty?",
        "What are your shipping options?",
        "What's your shipping policy?",
        "How does your warranty work?",
    ],
    "complaint": [
        "I'm very frustrated with this experience",
        "This is the third time this has happened",
        "Your service has been terrible",
        "I'm really disappointed",
        "This keeps happening",
    ],
    "general_inquiry": [
        "How do I contact customer service?",
        "Do you ship internationally?",
        "What payment methods do you accept?",
        "How can I reach you?",
    ],
}

# Assistant response templates (safe, conservative, escalates appropriately)
ASSISTANT_RESPONSES = {
    "information_request": [
        "I'd be happy to help you with that information. However, I don't have access to those specific details in my system. Let me connect you with someone who can provide you with accurate information. Would you like me to transfer you?",
        "I understand you're looking for that information. Unfortunately, I don't have access to those details. I can connect you with a specialist who can help. Would that be helpful?",
        "I'd like to help, but I don't have access to that specific information. Let me connect you with our team who can provide you with accurate details. Would you like me to transfer you?",
    ],
    "account_access": [
        "I understand you're having trouble accessing your account. For security reasons, I don't have the ability to reset passwords directly, but I can connect you with our account security team who can verify your identity and help you regain access. Would you like me to transfer you?",
        "I'm sorry to hear you're having account access issues. For security and privacy reasons, I need to connect you with our account services team who can verify your identity and assist you. Would you like me to transfer you now?",
        "I understand you need help with your account access. For security reasons, account-related issues need to be handled by our account security team. Let me connect you with them. Would that be helpful?",
    ],
    "order_inquiry": [
        "I'd be happy to help you with your order. To provide you with accurate order information, I'll need to connect you with our order services team who can access your order details. Would you like me to transfer you?",
        "I understand you have questions about your order. To give you accurate information, I need to connect you with someone who can access your order details. Would you like me to transfer you?",
        "I'd like to help you with your order inquiry. To provide accurate information, I'll need to connect you with our order team who can check your specific order. Would that be helpful?",
    ],
    "billing_payment": [
        "I'm sorry to hear about the billing issue. That's frustrating, and I want to help resolve this for you. To investigate and process any necessary adjustments, I'll need to connect you with our billing team who can access your account details. Would you like me to transfer you now?",
        "I understand your concern about the charge. To properly investigate and resolve this, I need to connect you with our billing department who can review your account. Would you like me to transfer you?",
        "I'm sorry to hear about this billing issue. To properly address it, I'll need to connect you with our billing team who can access your account and help resolve this. Would that be helpful?",
    ],
    "product_issue": [
        "I'm sorry to hear that your item arrived damaged. That's frustrating, and I want to help resolve this for you. To process a replacement or refund, I'll need to connect you with our returns team who can access your order details and handle this properly. Would you like me to transfer you now?",
        "I understand you received a damaged item. I'm sorry about that experience. To help resolve this, I need to connect you with our product support team who can review your order and process the appropriate solution. Would you like me to transfer you?",
        "I'm sorry to hear about the issue with your product. To properly address this, I'll need to connect you with our product support team who can access your order and help resolve this. Would that be helpful?",
    ],
    "return_refund": [
        "I'd be happy to help you with your return. To process your return properly, I'll need to connect you with our returns team who can access your order details and guide you through the process. Would you like me to transfer you?",
        "I understand you'd like to process a return. To help you with this, I need to connect you with our returns department who can access your order and process your return. Would you like me to transfer you?",
        "I'd like to help you with your return request. To process this accurately, I'll need to connect you with our returns team who can access your order details. Would that be helpful?",
    ],
    "technical_support": [
        "I'm sorry to hear you're experiencing technical issues. That's frustrating. To properly troubleshoot this, I need to connect you with our technical support team who can help diagnose and resolve the issue. Would you like me to transfer you?",
        "I understand you're having technical difficulties. To help resolve this, I'll need to connect you with our technical support specialists who can assist you. Would you like me to transfer you?",
        "I'm sorry about the technical issue you're experiencing. To properly address this, I need to connect you with our technical support team. Would that be helpful?",
    ],
    "policy_clarification": [
        "I'd be happy to help you understand our policy. However, policies can vary based on several factors, and I want to make sure you get the most accurate information for your specific situation. Let me connect you with someone who can provide you with the complete and current policy details.",
        "I understand you have questions about our policy. To ensure you get accurate and complete information, I should connect you with our policy team who can provide you with the full details. Would you like me to transfer you?",
        "I'd like to help clarify our policy for you. To make sure you get accurate information, I'll connect you with someone who can provide you with the complete policy details. Would that be helpful?",
    ],
    "complaint": [
        "I completely understand your frustration, and I'm sorry you've had this experience. That's not what we want for you. To properly address your concerns, I'd like to connect you with a specialist who can review your situation and work on a resolution. Would that be helpful?",
        "I'm sorry to hear about your experience, and I understand your frustration. To properly address this, I need to connect you with our customer relations team who can review your situation and work toward a resolution. Would you like me to transfer you?",
        "I understand your frustration, and I'm sorry this has happened. To properly address your concerns, I'd like to connect you with someone who can review your situation and help resolve this. Would that be helpful?",
    ],
    "general_inquiry": [
        "I'd be happy to help you with that. However, I don't have access to all the details you're asking about. Let me connect you with someone who can provide you with complete information. Would you like me to transfer you?",
        "I understand you're looking for that information. To give you accurate details, I should connect you with our team who can help. Would you like me to transfer you?",
        "I'd like to help, but I want to make sure you get accurate information. Let me connect you with someone who can provide you with complete details. Would that be helpful?",
    ],
}

# Escalation reasons per category
ESCALATION_REASONS = {
    "information_request": "insufficient_information",
    "account_access": "account_specific_access",
    "order_inquiry": "account_specific_access",
    "billing_payment": "billing_dispute",
    "product_issue": "account_specific_access",
    "return_refund": "account_specific_access",
    "technical_support": "technical_complexity",
    "policy_clarification": "policy_interpretation",
    "complaint": "sensitive_matter",
    "general_inquiry": "insufficient_information",
}

# Difficulty levels (for metadata)
DIFFICULTY_LEVELS = ["simple", "moderate", "complex"]


def generate_case(
    category: str,
    user_template: str,
    assistant_template: str,
    seed: int | None = None,
    difficulty: str = "moderate"
) -> dict[str, Any]:
    """
    Generate a single training case.
    
    Args:
        category: Category name
        user_template: User message template
        assistant_template: Assistant response template
        seed: Random seed for reproducibility
        difficulty: Difficulty level
        
    Returns:
        Dictionary with messages and metadata
    """
    if seed is not None:
        random.seed(seed)
    
    # Load system prompt
    prompts_dir = project_root / "prompts"
    try:
        system_prompt = load_system_prompt(prompts_dir / "system.txt")
    except Exception:
        system_prompt = "You are a customer support assistant. Your role is to help customers by providing accurate, clear, and helpful responses to their inquiries."
    
    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template},
        {"role": "assistant", "content": assistant_template}
    ]
    
    # Determine escalation
    escalation = ESCALATION_REASONS.get(category, "insufficient_information")
    
    return {
        "messages": messages,
        "metadata": {
            "source": "synthetic",
            "category": category,
            "escalation": escalation,
            "difficulty": difficulty
        }
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_synthetic_cases.py --n_per_category 5
  python scripts/generate_synthetic_cases.py --n_per_category 10 --seed 42 --out data/raw/synthetic_cases.jsonl
        """
    )
    
    parser.add_argument(
        "--n_per_category",
        type=int,
        default=5,
        help="Number of examples to generate per category (default: 5)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw/synthetic_cases.jsonl",
        help="Output path for JSONL file (default: data/raw/synthetic_cases.jsonl)"
    )
    
    args = parser.parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Resolve output path
    output_path = project_root / args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic training examples...")
    print(f"  Examples per category: {args.n_per_category}")
    print(f"  Seed: {args.seed if args.seed is not None else 'random'}")
    print(f"  Output: {output_path}")
    print()
    
    # Generate cases
    all_cases = []
    total_generated = 0
    
    for category in USER_TEMPLATES.keys():
        user_templates = USER_TEMPLATES[category]
        assistant_templates = ASSISTANT_RESPONSES[category]
        
        # Generate n_per_category examples
        for i in range(args.n_per_category):
            # Select random templates
            user_msg = random.choice(user_templates)
            assistant_msg = random.choice(assistant_templates)
            
            # Assign difficulty (simple for first third, moderate for middle, complex for last third)
            if i < args.n_per_category // 3:
                difficulty = "simple"
            elif i < 2 * args.n_per_category // 3:
                difficulty = "moderate"
            else:
                difficulty = "complex"
            
            # Generate case
            case = generate_case(
                category=category,
                user_template=user_msg,
                assistant_template=assistant_msg,
                seed=args.seed + i if args.seed is not None else None,
                difficulty=difficulty
            )
            
            all_cases.append(case)
            total_generated += 1
    
    # Write to JSONL
    print(f"Writing {total_generated} cases to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for case in all_cases:
            json.dump(case, f, ensure_ascii=False)
            f.write('\n')
    
    # Print summary
    print(f"\n✓ Generated {total_generated} synthetic cases")
    print(f"  Categories: {len(USER_TEMPLATES)}")
    print(f"  Per category: {args.n_per_category}")
    
    # Print category breakdown
    category_counts = {}
    for case in all_cases:
        cat = case["metadata"]["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\nCategory breakdown:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")
    
    print(f"\nExample case (first one):")
    if all_cases:
        example_json = json.dumps(all_cases[0], indent=2, ensure_ascii=False)
        print(example_json[:500] + "..." if len(example_json) > 500 else example_json)


if __name__ == "__main__":
    main()

