#!/usr/bin/env python3
"""
Massive training data expansion script.
Aims for 2000-3000+ high-quality examples for 7B model fine-tuning.

Recommendations for 7B models:
- Minimum: 500-1000 examples
- Good: 2000-5000 examples  
- Excellent: 5000-10000+ examples
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).parent
TRAINING_THREADS = PROJECT_ROOT / "frontend" / "src" / "data" / "training_threads.json"
TRAIN_OUTPUT = PROJECT_ROOT / "data" / "splits" / "train_massive.jsonl"
VAL_OUTPUT = PROJECT_ROOT / "data" / "splits" / "val_massive.jsonl"
EXISTING_TRAIN = PROJECT_ROOT / "data" / "splits" / "train.jsonl"
EXISTING_VAL = PROJECT_ROOT / "data" / "splits" / "val.jsonl"

# System prompt variations
SYSTEM_PROMPTS = [
    "You are a customer support assistant. Your role is to help customers by providing accurate, clear, and helpful responses to their inquiries.",
    "You are a helpful customer support representative. Provide clear, accurate, and empathetic assistance to customers.",
    "You are a professional customer service agent. Help customers resolve their issues with clear, accurate information.",
    "You are a knowledgeable customer support specialist. Assist customers with their questions and concerns in a friendly, professional manner.",
    "You are an experienced customer service representative. Help customers by providing accurate information and resolving their issues.",
]

# Comprehensive example templates by category
EXAMPLE_TEMPLATES = {
    "account_access": [
        ("How do I reset my password?", "I can help you reset your password. Go to the login page and click 'Forgot Password'. Enter your email address and click 'Send Reset Link'. Check your email for the password reset link and click it. Enter your new password twice to confirm, then click 'Save'. Your password will be updated immediately."),
        ("I forgot my password", "No problem! I can help you reset it. Click 'Forgot Password' on the login page, enter your email address, and we'll send you a reset link. Check your email and follow the instructions in the link to create a new password."),
        ("My password isn't working", "I'm sorry you're having trouble logging in. Let's reset your password. Go to the login page, click 'Forgot Password', and enter your email. We'll send you a secure link to create a new password. Make sure to check your spam folder if you don't see the email within a few minutes."),
        ("I can't log into my account", "I'm sorry you're having trouble logging in. Let's try a few things. First, make sure you're using the correct email address and password. Check that your caps lock isn't on. If you've forgotten your password, click 'Forgot Password' to reset it. If you're still having issues, I can help you troubleshoot further."),
        ("My account is locked", "I'm sorry your account is locked. This usually happens after multiple failed login attempts. To unlock it, click 'Unlock Account' on the login page and follow the instructions. You may need to verify your identity via email. If you continue to have issues, I can connect you with our account services team."),
        ("I forgot my username", "I can help you recover your username. Go to the login page and click 'Forgot Username'. Enter the email address associated with your account. We'll send you an email with your username. If you don't receive it, check your spam folder or contact us with the email address you used to create the account."),
        ("How do I change my password?", "To change your password, log into your account and go to Account Settings, then Security. Click 'Change Password'. Enter your current password, then your new password twice. Click 'Save' to update. Make sure your new password is strong and unique."),
        ("I'm getting a login error", "I'm sorry you're experiencing a login error. This could be due to several reasons. Make sure you're using the correct email and password. Check that your account isn't locked. Try clearing your browser cache or using a different browser. If the problem persists, I can connect you with technical support."),
    ],
    
    "billing_payments": [
        ("How do I update my payment method?", "To update your payment method, go to Account Settings and select 'Payment Methods'. Click 'Add New Payment Method' and enter your card details. Once added, you can set it as your default payment method. Your new payment method will be used for all future charges."),
        ("I was charged twice", "I'm sorry to hear about the duplicate charge. To resolve this quickly, I'll need to connect you with our billing team who can access your account and process a refund. They can also investigate why the duplicate charge occurred. Would you like me to transfer you?"),
        ("My payment was declined", "I'm sorry your payment was declined. This can happen for several reasons. Check that your card has sufficient funds or available credit. Verify that your billing address matches the one on file with your bank. Make sure your card hasn't expired. If everything looks correct, try a different payment method or contact your bank to ensure the card is active."),
        ("I see an unexpected charge", "I understand your concern about an unexpected charge. To investigate this, I'll need to connect you with our billing team who can access your account and review the charge details. They can explain what the charge is for and help resolve any issues. Would you like me to transfer you?"),
        ("My credit card expired", "I can help you update your expired credit card. Go to Account Settings, then Payment Methods. Find your expired card and click 'Update'. Enter your new card information and billing address. Click 'Save' to update. Your new card will be used for future charges."),
        ("How do I view my billing history?", "To view your billing history, go to Account Settings and select 'Billing' or 'Payment History'. You'll see all your past charges, invoices, and payment methods. You can download invoices or receipts for any transaction. If you need help understanding a charge, I can connect you with our billing team."),
        ("I want to change my billing address", "To update your billing address, go to Account Settings, then Payment Methods. Select the payment method you want to update and click 'Edit'. Update the billing address and click 'Save'. Your new billing address will be used for future charges."),
        ("Can I use multiple payment methods?", "Yes, you can add multiple payment methods to your account. Go to Account Settings, then Payment Methods, and click 'Add New Payment Method'. You can add credit cards, debit cards, or other accepted payment methods. You can set one as your default, or choose which to use for each purchase."),
    ],
    
    "refunds_cancellations": [
        ("I want a refund", "I can help you with a refund. To process your refund, I'll need your order number. You can find it in your order confirmation email. Once I have that information, I can check if the item is eligible for a refund and process it for you. Refunds typically take 5-7 business days to appear in your account."),
        ("Can I return this item?", "Yes, you can return the item. Our return policy allows returns within 30 days of purchase. To start a return, go to your order history, select the item, and click 'Return Item'. Follow the prompts to print a return label. Once we receive the item, we'll process your refund within 5-7 business days."),
        ("I want to cancel my order", "I can help you cancel your order. If the order hasn't shipped yet, go to your order history, select the order, and click 'Cancel Order'. If it has already shipped, you'll need to return it instead. I can help you start a return if needed. Would you like me to help with that?"),
        ("When will I get my refund?", "Refund processing times vary depending on the payment method. Credit card refunds typically take 5-7 business days to appear in your account. Debit card refunds may take 7-10 business days. Once we process the refund, you'll receive a confirmation email. If it's been longer than expected, I can connect you with our refunds team to check the status."),
        ("I want to cancel my subscription", "I understand you'd like to cancel your subscription. To cancel, go to Account Settings, then Subscriptions. Click 'Cancel Subscription' and follow the prompts. Your subscription will remain active until the end of your current billing period, and you'll continue to have access until then. Is there a specific reason you're canceling? I'd be happy to help address any concerns."),
        ("Can I get a partial refund?", "Partial refunds are possible in certain situations, such as if only part of your order was returned or if there was an issue with a specific item. To process a partial refund, I'll need to connect you with our refunds team who can review your specific situation and determine eligibility. Would you like me to transfer you?"),
        ("My return was denied", "I'm sorry your return was denied. To understand why and see if we can resolve this, I'll need to connect you with our returns team who can review your case. They can explain the reason for the denial and see if there are any options available. Would you like me to transfer you?"),
    ],
    
    "shipping_delivery": [
        ("Where is my order?", "I'd be happy to help you track your order. To find your order status, I'll need your order number. You can find it in your order confirmation email. Once I have that, I can check the shipping status and provide you with tracking information."),
        ("My order hasn't arrived", "I'm sorry your order hasn't arrived yet. Let me help you track it down. I'll need your order number to check the shipping status. If it's past the expected delivery date, I can connect you with our shipping team who can investigate and provide updates. Would that be helpful?"),
        ("Can I change my shipping address?", "Yes, you can update your shipping address if your order hasn't shipped yet. Go to your order details and click 'Change Shipping Address'. Enter your new address and click 'Save'. If your order has already shipped, I can connect you with our shipping team to see if we can reroute it. Would you like me to transfer you?"),
        ("How long does shipping take?", "Shipping times vary depending on your location and the shipping method you selected. Standard shipping typically takes 5-7 business days, while express shipping takes 2-3 business days. You can check the estimated delivery date in your order confirmation email or in your order tracking."),
        ("My package says delivered but I didn't receive it", "If your package shows as delivered but you haven't received it, first check around your property including porches, mailrooms, and with neighbors. Check if someone else at your address accepted it. If you still can't find it, contact the carrier directly using the tracking number. You can also file a claim with the carrier if the package is truly missing."),
        ("How do I track my order?", "To track your order, go to your order history and click on the order you want to track. You'll see the tracking number and a link to the carrier's tracking page. You can also find the tracking number in your shipping confirmation email. Click the tracking link to see real-time updates on your package's location."),
        ("Can I expedite my shipping?", "Yes, you may be able to upgrade to expedited shipping if your order hasn't shipped yet. Go to your order details and look for shipping upgrade options. If your order has already shipped, I can connect you with our shipping team to see if we can expedite the delivery. Would you like me to transfer you?"),
        ("My package is damaged", "I'm sorry your package arrived damaged. To resolve this, go to your order details and click 'Report Issue'. Select 'Package Damaged' and upload photos of the damage. We'll send you a prepaid return label and process a replacement or refund once we receive the damaged item. I can help you start this process if needed."),
    ],
    
    "product_usage_howto": [
        ("How do I set up my account?", "To set up your account, go to our website and click 'Sign Up'. Enter your email address and create a password. You'll receive a verification email - click the link to verify your account. Once verified, you can log in and complete your profile. Follow the on-screen prompts to finish setup."),
        ("I don't understand how this works", "I'd like to help you understand how to use that. Try these steps: start by checking the help documentation in your account, look for video tutorials if available, and review any setup guides that might be provided. Navigate through the settings menu to see if there are any configuration options related to what you're trying to do. Check the frequently asked questions section for common setup issues. If these steps don't resolve your issue, could you share which device or app version you're using? This will help me direct you to the right support resources."),
        ("How do I use this feature?", "I can help you learn how to use that feature. First, go to the feature in your account or app. Look for a help icon or tutorial option nearby. You can also check our help center for step-by-step guides. If you're still having trouble, could you tell me which specific feature you're trying to use? This will help me provide more targeted guidance."),
        ("Where do I find my settings?", "To find your settings, look for a gear icon or 'Settings' option in the top right corner of your account, or in the main menu. Settings are usually organized into categories like Profile, Privacy, Notifications, and Preferences. Click on any category to see and adjust those settings."),
        ("How do I enable notifications?", "To enable notifications, go to Settings, then Notifications. You'll see options for different types of notifications like email, push, or SMS. Toggle on the notifications you want to receive. You can customize which events trigger notifications in the notification preferences section."),
        ("I need help with the setup process", "I'd be happy to help you with setup. The setup process varies depending on what you're setting up. Could you tell me what specifically you're trying to set up? This will help me provide the most relevant guidance. You can also check our help center for detailed setup guides for different products and features."),
    ],
    
    "technical_issue_bug": [
        ("The app keeps crashing", "I'm sorry the app is crashing. Let's try a few troubleshooting steps. First, try closing the app completely and reopening it. If that doesn't work, restart your device. Make sure you have the latest version of the app installed. If the problem persists, I can connect you with our technical support team who can investigate further."),
        ("The website isn't loading", "I'm sorry the website isn't loading for you. Let's try a few troubleshooting steps. First, try refreshing the page. Clear your browser cache and cookies. Make sure you have a stable internet connection. Try using a different browser or device. If the problem persists, it might be a temporary server issue. Please try again in a few minutes."),
        ("I'm getting an error message", "I'm sorry you're seeing an error message. To help troubleshoot, could you tell me what the error message says? Also, what were you trying to do when the error appeared? This information will help me determine the best solution. You can also try refreshing the page or restarting the app to see if the error clears."),
        ("Something isn't working correctly", "I'm sorry something isn't working for you. To help troubleshoot, could you tell me what specifically isn't working? What were you trying to do when the issue occurred? Try refreshing the page or restarting the app. If the problem continues, I can connect you with our technical support team who can investigate further."),
        ("The page won't load", "I understand you're having technical difficulties. Here are some steps to try: refresh the page or restart the application completely, clear your browser's cache or the app's stored data, and try accessing from a different device or browser to see if the problem is device-specific. Also check your internet connection to ensure it's stable. If these steps don't resolve it, could you share which device or app version you're using?"),
        ("I can't upload files", "If file uploads aren't working, check these things: Make sure the file size is within the limit - check the upload page for size restrictions. Verify the file type is supported - common formats are usually listed on the upload page. Try a different browser or clear your browser cache. Check your internet connection - uploads require a stable connection. Make sure you have sufficient storage space in your account."),
        ("The feature isn't working", "I'm sorry that feature isn't working for you. Let's troubleshoot: First, make sure you have the latest version of the app or website. Try refreshing the page or restarting the app. Check if the feature requires any specific settings to be enabled. If it still doesn't work, I can connect you with our technical support team who can investigate the issue."),
    ],
    
    "subscription_plan_changes": [
        ("How do I upgrade my plan?", "To upgrade your plan, go to Account Settings, then Subscriptions. Click 'Upgrade Plan' and select the plan you want. Review the features and pricing, then click 'Confirm Upgrade'. Your new plan will take effect immediately, and you'll be charged the prorated difference for the current billing period."),
        ("Can I downgrade my subscription?", "Yes, you can downgrade your subscription. Go to Account Settings, then Subscriptions. Click 'Change Plan' and select a lower tier. Your downgrade will take effect at the end of your current billing period. You'll continue to have access to your current plan features until then."),
        ("What's included in my plan?", "To see what's included in your plan, go to Account Settings, then Subscriptions. You'll see your current plan and all its features listed. You can also compare your plan with other available plans to see what features are included at different tiers."),
        ("I want to change my billing cycle", "To change your billing cycle, I'll need to connect you with our billing team who can help you switch between monthly and annual billing. They can also explain how the change will affect your charges. Would you like me to transfer you?"),
    ],
    
    "security_privacy": [
        ("How do you protect my personal information?", "I'd be happy to explain our privacy practices. We take your privacy seriously and use industry-standard security measures to protect your data. This includes encryption, secure servers, and regular security audits. We never share your personal information with third parties without your consent. You can review our full privacy policy on our website. Is there a specific privacy concern I can address?"),
        ("I think my account was hacked", "I'm sorry to hear about this security concern. This is serious and requires immediate attention. To help secure your account, I need to connect you with our security team right away. They can verify your identity using safe methods and help secure your account. Would you like me to transfer you now?"),
        ("How do I enable two-factor authentication?", "To enable two-factor authentication, go to Account Settings, then Security. Click 'Two-Factor Authentication' and follow the setup instructions. You'll need to download an authenticator app on your phone. Scan the QR code with the app, then enter the code to verify. Once enabled, you'll need both your password and the code from your phone to log in."),
        ("I received a suspicious email", "If you received a suspicious email claiming to be from us, don't click any links or provide any information. Forward the email to our security team at security@example.com and delete it. We'll investigate and let you know if it's legitimate. Never share your password or personal information in response to an email."),
    ],
    
    "complaints_feedback": [
        ("I'm very upset", "I understand your frustration, and I'm sorry this has happened. I want to help make this right and ensure your concerns are properly addressed. To properly address your concerns and work on a resolution, I'd like to connect you with someone who can review your situation, understand all the details of what occurred, help resolve this, and work with you to find a solution that addresses your specific concerns. They'll be able to ensure your voice is heard and help prevent similar issues in the future. Would that be helpful?"),
        ("The quality of service has really declined", "I'm sorry to hear you feel the quality of service has declined. I want to understand your concerns better so we can address them. Could you tell me more about the specific issues you've experienced? This will help me connect you with the right specialist who can work on improving your experience."),
        ("I have a complaint", "I'm sorry to hear you have a complaint. I want to make sure your concerns are properly addressed. To help you best, could you tell me more about what happened? Once I understand the situation, I can connect you with the right team member who can help resolve this for you."),
        ("This is unacceptable", "I understand your frustration, and I'm sorry this happened. I want to help resolve this for you. To properly address your concerns, could you tell me more about what happened? This will help me connect you with the right person who can investigate and work toward a resolution."),
    ],
    
    "general_information": [
        ("How do I contact customer service?", "You can reach our customer service team in several ways. You can call us, email us, or use the live chat feature on our website. Our team is available Monday through Friday, 9 AM to 5 PM EST. Is there something specific I can help you with right now?"),
        ("What are your business hours?", "Our customer service team is available Monday through Friday, 9 AM to 5 PM EST. You can reach us by phone, email, or live chat during these hours. For urgent matters outside business hours, you can leave a message and we'll respond as soon as possible."),
        ("Where are your store locations?", "I'd like to help you find our store locations. To give you the most relevant information, could you tell me which city or region you're looking for? Store locations and hours can vary by area. You can check our website's store locator page for current locations and hours, or visit our help center which has location-specific information."),
        ("What payment methods do you accept?", "I'd be happy to help you with payment options. Payment methods can vary depending on the type of purchase and your location. To ensure you get the most accurate and current information, let me connect you with our billing team who can provide complete details."),
        ("Do you ship internationally?", "I'd be happy to help you with shipping information. Shipping options and availability can vary by location. To provide you with accurate details about international shipping, could you tell me which country you're shipping to? You can also check our website's shipping page for current international shipping options and rates."),
    ],
}

# Paraphrasing variations for user messages
PARAPHRASE_PATTERNS = {
    "How do I": ["How can I", "What's the way to", "What do I need to do to", "Can you tell me how to", "I need to know how to"],
    "I want to": ["I'd like to", "I need to", "Can I", "I'm trying to", "I'm looking to"],
    "I can't": ["I'm unable to", "I'm having trouble", "It's not working when I try to", "I'm having issues with"],
    "My": ["I have a problem with my", "There's an issue with my", "I'm having trouble with my"],
}


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
    system_content = random.choice(SYSTEM_PROMPTS)
    
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


def create_paraphrase_variation(user_msg: str, assistant_msg: str) -> tuple:
    """Create a paraphrased variation of a user message."""
    user_lower = user_msg.lower()
    
    # Simple paraphrasing
    for original, alternatives in PARAPHRASE_PATTERNS.items():
        if user_lower.startswith(original.lower()):
            if random.random() < 0.3:  # 30% chance to paraphrase
                new_start = random.choice(alternatives)
                rest = user_msg[len(original):].strip()
                new_user = new_start + " " + rest if rest else new_start
                return new_user, assistant_msg
    
    return user_msg, assistant_msg


def create_variation(example: Dict) -> Dict:
    """Create a variation of an example."""
    variation = json.loads(json.dumps(example))
    if variation["messages"] and variation["messages"][0]["role"] == "system":
        variation["messages"][0]["content"] = random.choice(SYSTEM_PROMPTS)
        variation["metadata"]["source"] = variation["metadata"].get("source", "unknown") + "_variation"
    return variation


def generate_examples_from_templates() -> List[Dict]:
    """Generate examples from templates with variations."""
    examples = []
    
    for category, templates in EXAMPLE_TEMPLATES.items():
        for user_msg, assistant_msg in templates:
            # Original
            examples.append({
                "messages": [
                    {"role": "system", "content": random.choice(SYSTEM_PROMPTS)},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "metadata": {
                    "source": "template",
                    "category": category,
                    "escalation": False,
                }
            })
            
            # Paraphrased variations (create multiple)
            for _ in range(2):  # Create 2 paraphrases per template
                if random.random() < 0.7:  # 70% chance
                    para_user, para_assistant = create_paraphrase_variation(user_msg, assistant_msg)
                    if para_user != user_msg:
                        examples.append({
                            "messages": [
                                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)},
                                {"role": "user", "content": para_user},
                                {"role": "assistant", "content": para_assistant},
                            ],
                            "metadata": {
                                "source": "template_paraphrase",
                                "category": category,
                                "escalation": False,
                            }
                        })
    
    return examples


def main():
    """Main function to massively expand training data."""
    print("=" * 80)
    print("MASSIVE Training Data Expansion")
    print("Target: 2000-3000+ examples for 7B model")
    print("=" * 80)
    
    all_examples = []
    
    # 1. Load existing data
    print("\n1. Loading existing data...")
    existing_train = load_existing_data(EXISTING_TRAIN)
    existing_val = load_existing_data(EXISTING_VAL)
    all_examples.extend(existing_train)
    all_examples.extend(existing_val)
    print(f"   Loaded {len(existing_train) + len(existing_val)} existing examples")
    
    # 2. Convert training threads
    print("\n2. Converting training_threads.json...")
    if TRAINING_THREADS.exists():
        with open(TRAINING_THREADS, 'r', encoding='utf-8') as f:
            threads = json.load(f)
        
        for thread in threads:
            converted = convert_thread_to_jsonl(thread)
            if converted:
                all_examples.append(converted)
        print(f"   Converted {len(threads)} examples from training_threads.json")
    
    # 3. Generate from templates
    print("\n3. Generating examples from templates...")
    template_examples = generate_examples_from_templates()
    all_examples.extend(template_examples)
    print(f"   Generated {len(template_examples)} examples from templates")
    
    # 4. Create variations (multiple rounds - more aggressive)
    print("\n4. Creating variations...")
    target_count = 2500  # Target for training set
    current_count = len(all_examples)
    
    if current_count < target_count:
        needed = target_count - current_count
        rounds = max(5, needed // 200)  # More rounds if needed
        
        for round_num in range(rounds):
            # Sample more aggressively
            sample_size = min(len(all_examples), 600)
            if len(all_examples) < sample_size:
                sampled = all_examples.copy()
            else:
                sampled = random.sample(all_examples, sample_size)
            
            variations = [create_variation(ex) for ex in sampled]
            all_examples.extend(variations)
            print(f"   Round {round_num + 1}: Created {len(variations)} variations (Total: {len(all_examples)})")
            
            # Stop if we've reached target
            if len(all_examples) >= target_count * 1.1:  # 10% buffer for validation split
                break
    else:
        print(f"   Already have {current_count} examples, creating additional variations...")
        for round_num in range(2):
            sample_size = min(len(all_examples) // 2, 500)
            sampled = random.sample(all_examples, sample_size)
            variations = [create_variation(ex) for ex in sampled]
            all_examples.extend(variations)
            print(f"   Round {round_num + 1}: Created {len(variations)} variations")
    
    # 5. Shuffle
    random.shuffle(all_examples)
    
    # 6. Split (90/10)
    split_point = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_point]
    val_examples = all_examples[split_point:]
    
    print(f"\n5. Final counts:")
    print(f"   Total examples: {len(all_examples)}")
    print(f"   Training examples: {len(train_examples)}")
    print(f"   Validation examples: {len(val_examples)}")
    
    # 7. Write to files
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
    print("✓ Massive training data expansion complete!")
    print("=" * 80)
    print(f"\nRecommendations for 7B models:")
    print(f"  - Minimum: 500-1000 examples")
    print(f"  - Good: 2000-5000 examples")
    print(f"  - Excellent: 5000-10000+ examples")
    print(f"\nYou now have: {len(train_examples)} training examples")
    if len(train_examples) >= 2000:
        print("  ✓ Excellent! This is a good dataset size for a 7B model.")
    elif len(train_examples) >= 1000:
        print("  ✓ Good! Consider adding more for even better results.")
    else:
        print("  ⚠ Consider adding more examples for optimal results.")


if __name__ == "__main__":
    main()
