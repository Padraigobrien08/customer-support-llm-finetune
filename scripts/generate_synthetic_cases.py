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


# User message templates per category (10-15 templates each)
USER_TEMPLATES = {
    "account_access": [
        "I can't log into my account",
        "I'm having trouble logging in",
        "My password isn't working",
        "Please reset my password",
        "I forgot my password",
        "My account is locked",
        "I can't access my account",
        "Help me log in",
        "I forgot my username",
        "My login credentials aren't working",
        "I need to unlock my account",
        "Can you help me access my account?",
    ],
    "billing_payments": [
        "I was charged twice",
        "I see a duplicate charge on my card",
        "Why was I charged twice?",
        "My payment was declined",
        "My card was declined",
        "I want to dispute a charge",
        "I see an unexpected charge",
        "How do I update my payment method?",
        "My payment isn't going through",
        "I was charged the wrong amount",
        "Can you explain this charge?",
        "I need help with a payment issue",
    ],
    "refunds_cancellations": [
        "I want to cancel my order",
        "How do I cancel my order?",
        "I need to cancel my subscription",
        "How do I get a refund?",
        "I need a refund",
        "When will I get my refund?",
        "How long does a refund take?",
        "Can I cancel this?",
        "I want to cancel my purchase",
        "I need to cancel my service",
        "How do I request a refund?",
        "Can I get my money back?",
    ],
    "shipping_delivery": [
        "Where is my order?",
        "What's the status of my order?",
        "When will my order arrive?",
        "I want to change my shipping address",
        "Can I change my shipping address?",
        "I haven't received my order yet",
        "My package hasn't arrived",
        "How long does shipping take?",
        "Can I track my order?",
        "Where can I track my shipment?",
        "I need to update my delivery address",
        "My order is late",
    ],
    "product_usage_howto": [
        "How do I set up this product?",
        "Can you explain how this feature works?",
        "I need help using this",
        "What are the steps to activate this?",
        "How do I use this feature?",
        "I don't understand how this works",
        "Can you walk me through this?",
        "I need instructions for this",
        "How do I get started?",
        "What do I need to do first?",
        "I'm confused about how to use this",
        "Can you help me understand this?",
    ],
    "technical_issue_bug": [
        "The website isn't loading",
        "Your website is down",
        "I can't complete checkout",
        "Checkout isn't working",
        "The app keeps crashing",
        "I'm having trouble with your website",
        "I'm getting an error message",
        "Something isn't working correctly",
        "There's a bug in the system",
        "The page won't load",
        "I'm experiencing technical issues",
        "Something is broken",
    ],
    "subscription_plan_changes": [
        "How do I upgrade my plan?",
        "What's included in my subscription?",
        "Can I change my plan?",
        "When does my subscription renew?",
        "I want to upgrade my subscription",
        "How do I downgrade my plan?",
        "What plan am I currently on?",
        "Can I switch plans?",
        "I need to change my subscription",
        "How much does each plan cost?",
        "What's the difference between plans?",
        "I want to modify my subscription",
    ],
    "security_privacy": [
        "I think my account was hacked",
        "How do I enable two-factor authentication?",
        "I received a suspicious email",
        "Can you explain your privacy policy?",
        "I'm concerned about my account security",
        "Someone accessed my account",
        "I want to change my password",
        "How do I secure my account?",
        "I got a strange email from you",
        "Is my information safe?",
        "I'm worried about privacy",
        "Can you help me secure my account?",
    ],
    "complaints_feedback": [
        "I'm very frustrated with this experience",
        "This is the third time this has happened",
        "Your service has been terrible",
        "I'm really disappointed",
        "This keeps happening",
        "I'm not happy with this",
        "This is unacceptable",
        "I've had enough of this",
        "Your service is poor",
        "I'm very upset",
        "This is ridiculous",
        "I want to file a complaint",
    ],
    "general_information": [
        "What are your business hours?",
        "When are you open?",
        "How do I contact customer service?",
        "Do you ship internationally?",
        "What payment methods do you accept?",
        "How can I reach you?",
        "Where are you located?",
        "What services do you offer?",
        "Can you tell me about your company?",
        "How do I get in touch?",
        "What are your contact options?",
        "I have a general question",
    ],
}

# Assistant response templates (safe, conservative, escalates appropriately)
ASSISTANT_RESPONSES = {
    "account_access": [
        "I understand you're having trouble accessing your account. For security reasons, I don't have the ability to reset passwords directly, but I can connect you with our account security team who can verify your identity and help you regain access. Would you like me to transfer you?",
        "I'm sorry to hear you're having account access issues. For security and privacy reasons, I need to connect you with our account services team who can verify your identity and assist you. Would you like me to transfer you now?",
        "I understand you need help with your account access. For security reasons, account-related issues need to be handled by our account security team. Let me connect you with them. Would that be helpful?",
    ],
    "billing_payments": [
        "I'm sorry to hear about the billing issue. That's frustrating, and I want to help resolve this for you. To investigate and process any necessary adjustments, I'll need to connect you with our billing team who can access your account details. Would you like me to transfer you now?",
        "I understand your concern about the charge. To properly investigate and resolve this, I need to connect you with our billing department who can review your account. Would you like me to transfer you?",
        "I'm sorry to hear about this billing issue. To properly address it, I'll need to connect you with our billing team who can access your account and help resolve this. Would that be helpful?",
    ],
    "refunds_cancellations": [
        "I'd be happy to help you with your cancellation request. To process this properly, I'll need to connect you with our cancellation services team who can access your account and complete the cancellation. Would you like me to transfer you?",
        "I understand you'd like to cancel. To ensure this is handled correctly, I need to connect you with our team who can access your account and process the cancellation. Would you like me to transfer you?",
        "I'd like to help you with your cancellation. To process this accurately, I'll need to connect you with our cancellation team who can access your account details. Would that be helpful?",
    ],
    "shipping_delivery": [
        "I'd be happy to help you with your order. To provide you with accurate order information, I'll need to connect you with our order services team who can access your order details. Would you like me to transfer you?",
        "I understand you have questions about your order. To give you accurate information, I need to connect you with someone who can access your order details. Would you like me to transfer you?",
        "I'd like to help you with your order inquiry. To provide accurate information, I'll need to connect you with our order team who can check your specific order. Would that be helpful?",
    ],
    "product_usage_howto": [
        "I'd be happy to help you with that. To provide you with the most accurate guidance, could you tell me which specific product or feature you're asking about? This will help me connect you with the right resources or specialist.",
        "I'd like to help you understand how to use that. To give you the best guidance, could you provide more details about what you're trying to do? This will help me direct you to the right support resources.",
        "I'd be happy to help you with that. To provide accurate instructions, could you tell me more about what you're trying to accomplish? This will help me connect you with the right specialist or resources.",
    ],
    "technical_issue_bug": [
        "I'm sorry to hear you're experiencing technical issues. That's frustrating. To properly troubleshoot this, I need to connect you with our technical support team who can help diagnose and resolve the issue. Would you like me to transfer you?",
        "I understand you're having technical difficulties. To help resolve this, I'll need to connect you with our technical support specialists who can assist you. Would you like me to transfer you?",
        "I'm sorry about the technical issue you're experiencing. To properly address this, I need to connect you with our technical support team. Would that be helpful?",
    ],
    "subscription_plan_changes": [
        "I'd be happy to help you with your subscription plan. To ensure you get the right plan for your needs, I'll need to connect you with our subscription services team who can access your account and guide you through the options. Would you like me to transfer you?",
        "I understand you'd like to change your subscription plan. To provide you with accurate information and process the change, I need to connect you with our subscription team who can access your account. Would you like me to transfer you?",
        "I'd like to help you with your subscription plan question. To give you accurate details, I'll need to connect you with our subscription services team who can access your account information. Would that be helpful?",
    ],
    "security_privacy": [
        "I'm sorry to hear about this security concern. This is a serious matter that requires immediate attention from our security team. Let me connect you with them right away so they can help secure your account and investigate the issue.",
        "I understand your security concern. This is important to address properly. I'll need to connect you with our security team who can review the situation and provide guidance. Would you like me to transfer you?",
        "I'm sorry to hear about this security issue. To properly address this, I need to connect you with our security team who can investigate and help resolve this. Would that be helpful?",
    ],
    "complaints_feedback": [
        "I completely understand your frustration, and I'm sorry you've had this experience. That's not what we want for you. To properly address your concerns, I'd like to connect you with a specialist who can review your situation and work on a resolution. Would that be helpful?",
        "I'm sorry to hear about your experience, and I understand your frustration. To properly address this, I need to connect you with our customer relations team who can review your situation and work toward a resolution. Would you like me to transfer you?",
        "I understand your frustration, and I'm sorry this has happened. To properly address your concerns, I'd like to connect you with someone who can review your situation and help resolve this. Would that be helpful?",
    ],
    "general_information": [
        "I'd be happy to help you with that information. However, I don't have access to those specific details in my system. Let me connect you with someone who can provide you with accurate information. Would you like me to transfer you?",
        "I understand you're looking for that information. Unfortunately, I don't have access to those details. I can connect you with a specialist who can help. Would that be helpful?",
        "I'd like to help, but I don't have access to that specific information. Let me connect you with our team who can provide you with accurate details. Would you like me to transfer you?",
    ],
}

# Escalation reasons per category (mapped to new taxonomy escalation reasons)
ESCALATION_REASONS = {
    "account_access": "account_specific_action_required",
    "billing_payments": "payment_dispute",
    "refunds_cancellations": "account_specific_action_required",
    "shipping_delivery": "account_specific_action_required",
    "product_usage_howto": "unclear_request",
    "technical_issue_bug": "unclear_request",
    "subscription_plan_changes": "account_specific_action_required",
    "security_privacy": "security_sensitive",
    "complaints_feedback": "unclear_request",
    "general_information": "missing_policy_info",
}

# Escalation mapping (true/false based on whether response escalates)
ESCALATION_MAP = {
    "account_access": True,
    "billing_payments": True,
    "refunds_cancellations": True,
    "shipping_delivery": True,
    "product_usage_howto": False,  # Asks clarifying questions instead
    "technical_issue_bug": True,
    "subscription_plan_changes": True,
    "security_privacy": True,
    "complaints_feedback": True,
    "general_information": True,
}


def generate_case(
    category: str,
    user_template: str,
    assistant_template: str,
    case_number: int,
    seed: int | None = None,
    difficulty: int = 2
) -> dict[str, Any]:
    """
    Generate a single training case.
    
    Args:
        category: Category name
        user_template: User message template
        assistant_template: Assistant response template
        case_number: Case number for test_case_id
        seed: Random seed for reproducibility
        difficulty: Difficulty level (1-3)
        
    Returns:
        Dictionary with messages and metadata
    """
    if seed is not None:
        random.seed(seed + case_number)
    
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
    escalation_reason = ESCALATION_REASONS.get(category, "unclear_request")
    escalation_bool = ESCALATION_MAP.get(category, True)
    
    # Generate test_case_id
    test_case_id = f"synthetic_{category}_{case_number:04d}"
    
    return {
        "messages": messages,
        "metadata": {
            "source": "synthetic",
            "category": category,
            "escalation": escalation_bool,
            "difficulty": difficulty,
            "contains_policy_claims": False,
            "test_case_id": test_case_id
        }
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_synthetic_cases.py --n_per_category 30
  python scripts/generate_synthetic_cases.py --n_per_category 30 --seed 7 --out data/raw/synthetic_cases.jsonl
        """
    )
    
    parser.add_argument(
        "--n_per_category",
        type=int,
        default=30,
        help="Number of examples to generate per category (default: 30)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility (default: 7)"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw/synthetic_cases.jsonl",
        help="Output path for JSONL file (default: data/raw/synthetic_cases.jsonl)"
    )
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    
    # Resolve output path
    output_path = project_root / args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic training examples...")
    print(f"  Examples per category: {args.n_per_category}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_path}")
    print()
    
    # Generate cases
    all_cases = []
    total_generated = 0
    case_counter = 0
    
    for category in sorted(USER_TEMPLATES.keys()):
        user_templates = USER_TEMPLATES[category]
        assistant_templates = ASSISTANT_RESPONSES[category]
        
        # Generate n_per_category examples
        for i in range(args.n_per_category):
            case_counter += 1
            
            # Select random templates
            user_msg = random.choice(user_templates)
            assistant_msg = random.choice(assistant_templates)
            
            # Assign difficulty (simple for first third, moderate for middle, complex for last third)
            if i < args.n_per_category // 3:
                difficulty = 1
            elif i < 2 * args.n_per_category // 3:
                difficulty = 2
            else:
                difficulty = 3
            
            # Generate case
            case = generate_case(
                category=category,
                user_template=user_msg,
                assistant_template=assistant_msg,
                case_number=case_counter,
                seed=args.seed,
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
    for cat in sorted(category_counts.keys()):
        count = category_counts[cat]
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
