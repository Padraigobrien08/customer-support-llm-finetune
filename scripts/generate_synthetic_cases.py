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

# Assistant response templates following preferred style patterns
ASSISTANT_RESPONSES = {
    "account_access": [
        "I understand you're having trouble accessing your account. Here are the standard steps to reset your password: first, go to the login page and click 'Forgot Password', then enter the email address associated with your account. You'll receive a reset link via email within a few minutes. If you don't receive the email, check your spam folder and make sure you're checking the correct email address. The reset link will typically expire after a set time for security. If you're not receiving the email or the link isn't working, I can connect you with our account security team who can verify your identity through the standard verification process and help you regain access. Would you like me to transfer you?",
        "I'm sorry to hear you're having account access issues. To reset your password, visit the login page and select 'Forgot Password', then enter your account email address. You should receive a password reset link via email within a few minutes. Make sure to check your spam or junk folder if you don't see it in your inbox. The reset link will allow you to create a new password securely. If you don't receive the email, the link expires, or you need additional help, I can connect you with our account services team who can verify your identity using the standard email verification flow and assist you further. Would you like me to transfer you?",
        "I understand you need help accessing your account. The standard password reset process involves going to the login page, clicking 'Forgot Password', and entering your email address. You'll get a reset link sent to that email address, which you can use to create a new password. The link is time-limited for security purposes. If you're not receiving the email, double-check that you're using the correct email address and check your spam folder. If the link has expired or you're still having trouble, I can connect you with our account security team who can verify your identity through the standard email verification process and help you regain access. Would that be helpful?",
    ],
    "billing_payments": [
        "I'm sorry to hear about the billing issue. That's frustrating, and I want to help resolve this for you right away. To investigate this properly, could you provide the date of the charge and the last four digits of the card used? Alternatively, if you have an invoice number or transaction ID, that would help too. I want to emphasize that I only need these non-sensitive identifiers - never share your full card number, security code, or other sensitive payment details. With this information, I can connect you with our billing team who can review your account details and resolve this issue. Would you like me to transfer you?",
        "I understand your concern about the charge. To help investigate this, could you share the date when the charge appeared and the last four digits of the payment method used? If you have an invoice number or transaction ID, that would also be helpful for locating the specific charge. Please note that I only need these non-sensitive details - your full payment information is not required and should never be shared. Once I have this information, I can connect you with our billing department who can review your account and address this. Would you like me to transfer you?",
        "I'm sorry to hear about this billing issue. To properly address it, I'll need some non-sensitive details to locate the transaction: the date of the charge and the last four digits of the card used. If you have an invoice number or transaction ID, that would be great too. I want to make sure you know that I only need these identifiers - never provide your full card number, expiration date, or security code. With this information, I can connect you with our billing team who can access your account and help resolve this. Would that be helpful?",
    ],
    "refunds_cancellations": [
        "I'd be happy to help you with your refund request. To process this accurately, could you provide the date of your purchase and the last four digits of the payment method used? If you have an order number or invoice ID, that would also help locate your transaction. I only need these non-sensitive identifiers - please don't share your full card number or security code. With this information, I can connect you with our refunds team who can access your account and process your request. Would you like me to transfer you?",
        "I understand you'd like a refund. To ensure this is handled correctly, could you share the purchase date and the last four digits of the card used? An order number or invoice ID would also be helpful for locating your transaction. Please note that I only need these non-sensitive details - never provide your full payment information. Once I have these details, I can connect you with our team who can access your account and process the refund. Would you like me to transfer you?",
        "I'd like to help you with your refund. To process this accurately, I'll need the purchase date and the last four digits of your payment method. If you have an order or invoice number, that would be great for locating your transaction. I want to make sure you know that I only need these non-sensitive identifiers - your full payment details are not required. With this information, I can connect you with our refunds team who can access your account details and handle this. Would that be helpful?",
    ],
    "shipping_delivery": [
        "I'd be happy to help you with your order. To provide you with accurate tracking information, I'll need your order number. If you don't have it, I can connect you with our order services team who can locate it using your email address or phone number. They'll be able to check the shipping status, provide tracking details, help with any delivery concerns, assist with address changes if needed, and answer any questions about estimated delivery times. Would you like me to transfer you?",
        "I understand you have questions about your order. To give you accurate information, I'll need your order number. If you don't have it handy, I can connect you with someone who can locate your order using other information like your email or phone number. Once they find your order, they can provide tracking information, check delivery status, help address any shipping concerns you might have, assist with any delivery-related questions, and help you understand estimated delivery times or shipping options. Would you like me to transfer you?",
        "I'd like to help you with your order inquiry. To provide accurate tracking details, I'll need your order number. If you don't have it available, I can connect you with our order team who can find it using your email address or phone number. They'll be able to check the shipping status, provide tracking updates, help with any delivery questions or concerns you have, assist with tracking or address modifications if needed, and answer questions about shipping methods or delivery windows. Would that be helpful?",
    ],
    "product_usage_howto": [
        "I'd be happy to help you with that. Here are some general steps to try: first, make sure you're logged into your account, then navigate to the settings or help section. Check if there's a tutorial or guide available in the help menu, and look for any video walkthroughs that might be available. Review the frequently asked questions section as well, as it often contains helpful setup information. If you're still having trouble after trying these steps, could you tell me which device or app version you're using? This will help me connect you with the right specialist who can provide more specific guidance tailored to your setup.",
        "I'd like to help you understand how to use that. Try these steps: start by checking the help documentation in your account, look for video tutorials if available, and review any setup guides that might be provided. Navigate through the settings menu to see if there are any configuration options related to what you're trying to do. Check the frequently asked questions section for common setup issues. If these steps don't resolve your issue, could you share which device or app version you're using? This will help me direct you to the right support resources or connect you with a specialist who can provide device-specific guidance.",
        "I'd be happy to help you with that. Here's what you can try: check the help section in your account for step-by-step guides, look for any tutorial videos that might be available, and review the frequently asked questions. Make sure you're logged into your account and navigate to the settings or configuration area to see if there are relevant options. If you need more specific help after trying these general steps, could you tell me which device or app version you're using? This will help me connect you with the right specialist or resources who can provide more targeted assistance.",
    ],
    "technical_issue_bug": [
        "I'm sorry to hear you're experiencing technical issues. That's frustrating. Let's try some troubleshooting steps: first, refresh the page or restart the app completely. Then, clear your browser cache or clear the app's stored data if possible. Try using a different browser or device if one is available to see if the issue persists. Make sure you have a stable internet connection as well. If the issue persists after trying these steps, could you tell me which device or app version you're using? This will help me connect you with our technical support team who can diagnose and resolve the issue. Would you like me to transfer you?",
        "I understand you're having technical difficulties. Here are some steps to try: refresh the page or restart the application completely, clear your browser's cache or the app's stored data, and try accessing from a different device or browser to see if the problem is device-specific. Also check your internet connection to ensure it's stable. If these steps don't resolve it, could you share which device or app version you're using? This will help me connect you with our technical support specialists who can assist you further. Would you like me to transfer you?",
        "I'm sorry about the technical issue you're experiencing. Try these troubleshooting steps: refresh or restart the application completely, clear your cache or app data, and test on a different device or browser to see if the issue is specific to one setup. Make sure your internet connection is stable and try again. If the problem continues after these steps, could you tell me which device or app version you're using? This will help me connect you with our technical support team for further assistance. Would that be helpful?",
    ],
    "subscription_plan_changes": [
        "I'd be happy to help you with your subscription plan. To ensure you get the right plan for your needs and understand how any changes will affect your account, I'll need to connect you with our subscription services team who can access your account details and guide you through the available options. They can explain the differences between plans, help you understand what's included in each, assist you in making the best choice for your needs, and answer any questions about billing or feature changes. Would you like me to transfer you?",
        "I understand you'd like to change your subscription plan. To provide you with accurate information about available plans and process the change correctly, I need to connect you with our subscription team who can access your account and explain how the change will affect your service. They'll be able to answer any questions you have about plan features, help you understand pricing differences, guide you through the change process, and explain when the changes will take effect. Would you like me to transfer you?",
        "I'd like to help you with your subscription plan question. To give you accurate details about your current plan and available options, I'll need to connect you with our subscription services team who can access your account information and explain the differences between plans. They can help you understand what's included in each plan, explain how changes might affect your service, guide you through any modifications you'd like to make, and help you understand the billing implications. Would that be helpful?",
    ],
    "security_privacy": [
        "I'm sorry to hear about this security concern. This is a serious matter that requires immediate attention from our security team. They can help secure your account, investigate any suspicious activity, guide you through any necessary security measures, ensure your information is protected, and help you understand what steps to take to prevent future issues. They'll be able to review your account activity and help you implement additional security measures if needed. Let me connect you with them right away so they can help secure your account and investigate the issue.",
        "I understand your security concern. This is important to address properly and quickly. Our security team can review the situation, help secure your account if needed, investigate any potential issues, provide guidance on protecting your information going forward, and help you understand what security measures you should take. They can also help you review your account settings and ensure everything is properly configured. I'll need to connect you with our security team who can review the situation and provide guidance. Would you like me to transfer you?",
        "I'm sorry to hear about this security issue. To properly address this and ensure your account is secure, I need to connect you with our security team who can investigate the situation, help secure your account, provide you with guidance on next steps, help you understand what actions you should take to protect your information, ensure your account remains secure going forward, and help you review your security settings. They'll be able to help you implement any necessary changes and monitor your account for any further issues. Would that be helpful?",
    ],
    "complaints_feedback": [
        "I completely understand your frustration, and I'm sorry you've had this experience. That's not what we want for you, and I want to make sure we address your concerns properly. To properly address your concerns and work toward a resolution, I'd like to connect you with a specialist who can review your situation in detail, understand the full context of what happened, and work on a resolution that addresses your specific concerns. They'll be able to help ensure this doesn't happen again. Would that be helpful?",
        "I'm sorry to hear about your experience, and I understand your frustration. I want to make sure we properly address this and work toward a satisfactory resolution. To properly address this, I need to connect you with our customer relations team who can review your situation thoroughly, understand the full context of what occurred, and work toward a resolution. They'll be able to help find a solution and ensure your concerns are heard. Would you like me to transfer you?",
        "I understand your frustration, and I'm sorry this has happened. I want to help make this right and ensure your concerns are properly addressed. To properly address your concerns and work on a resolution, I'd like to connect you with someone who can review your situation, understand all the details of what occurred, help resolve this, and work with you to find a solution that addresses your specific concerns. They'll be able to ensure your voice is heard and help prevent similar issues in the future. Would that be helpful?",
    ],
    "general_information": [
        "I'd be happy to help you with that information. To provide you with the most accurate details, could you tell me which location or context you're asking about? You can also check our website's help section or contact page for this information, as those resources are regularly updated and contain current details. Additionally, you might find helpful information in our FAQ section. If you need more specific details that aren't available in these resources, I can connect you with someone who can provide accurate information. Would you like me to transfer you?",
        "I understand you're looking for that information. To give you the most accurate answer, could you clarify which specific area or service you're asking about? You might also find this information in our help documentation or on our website's FAQ section, which is kept current and contains up-to-date information. Our contact page is another good resource to check. If you need more detailed information that isn't covered in those resources, I can connect you with a specialist who can help. Would that be helpful?",
        "I'd like to help you with that information. To provide accurate details, could you tell me more about what specifically you need to know? You can also check our website's help section or FAQ page, as those are good sources for general information and are regularly maintained. Our contact page may also have relevant details. If you need more specific information that isn't available in these places, I can connect you with our team who can provide accurate details. Would you like me to transfer you?",
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

# Escalation mapping (true/false based on whether response offers to hand off/transfer)
ESCALATION_MAP = {
    "account_access": True,  # Offers to transfer to account security team
    "billing_payments": True,  # Offers to transfer to billing team
    "refunds_cancellations": True,  # Offers to transfer to refunds team
    "shipping_delivery": True,  # Offers to transfer to order services team
    "product_usage_howto": False,  # Asks clarifying questions, offers to connect but doesn't explicitly transfer
    "technical_issue_bug": True,  # Offers to transfer to technical support team
    "subscription_plan_changes": True,  # Offers to transfer to subscription services team
    "security_privacy": True,  # Offers to transfer to security team
    "complaints_feedback": True,  # Offers to transfer to specialist/customer relations
    "general_information": True,  # Offers to transfer for more specific information
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
