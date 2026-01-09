#!/usr/bin/env python3
"""
Ingest Zendesk ticket exports and convert to training JSONL format.

Supports both JSON and CSV export formats from Zendesk.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

# Add project root to path for csft package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from csft.io import save_jsonl


# Category inference keywords (case-insensitive)
CATEGORY_KEYWORDS = {
    "account_access": [
        "login", "password", "account", "access", "locked", "reset", "sign in", "log in"
    ],
    "billing_payment": [
        "charge", "billing", "payment", "refund", "invoice", "credit card", "payment method",
        "charged twice", "duplicate charge", "payment declined"
    ],
    "order_inquiry": [
        "order", "tracking", "shipment", "shipping", "delivery", "cancel order", "modify order"
    ],
    "product_issue": [
        "damaged", "broken", "defective", "wrong item", "missing", "doesn't work", "not working"
    ],
    "return_refund": [
        "return", "refund", "send back", "exchange"
    ],
    "technical_support": [
        "error", "bug", "crash", "not loading", "website", "app", "technical", "troubleshoot"
    ],
    "policy_clarification": [
        "policy", "terms", "warranty", "guarantee", "procedure", "rules"
    ],
    "complaint": [
        "frustrated", "disappointed", "terrible", "awful", "horrible", "unhappy", "angry",
        "complaint", "dissatisfied"
    ],
    "information_request": [
        "what are", "how do", "where", "when", "hours", "location", "contact"
    ],
    "general_inquiry": [
        "question", "help", "information", "inquiry"
    ]
}


def redact_email(text: str) -> str:
    """Replace email addresses with [EMAIL_REDACTED]."""
    # Pattern matches common email formats
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.sub(pattern, '[EMAIL_REDACTED]', text)


def redact_phone(text: str) -> str:
    """Replace phone numbers with [PHONE_REDACTED]."""
    # Pattern matches various phone number formats
    patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format: 123-456-7890
        r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',  # US format: (123) 456-7890
        r'\b\+?\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b',  # International formats
    ]
    for pattern in patterns:
        text = re.sub(pattern, '[PHONE_REDACTED]', text)
    return text


def redact_order_number(text: str) -> str:
    """Replace order numbers with [ORDER_REDACTED]."""
    # Common order number patterns
    # Be more specific to avoid matching phone numbers
    patterns = [
        r'\b(?:order|ord|order\s*number|order\s*#)\s*[#]?\s*\d{4,}\b',  # "order #12345" or "order 12345"
        r'\b[A-Z]{2,}\d{4,}\b',  # Alphanumeric: "ORD12345"
        r'\b\d{8,}\b',  # Very long numeric strings (likely order numbers, not phone numbers)
    ]
    for pattern in patterns:
        text = re.sub(pattern, '[ORDER_REDACTED]', text, flags=re.IGNORECASE)
    return text


def redact_pii(text: str) -> str:
    """
    Redact PII from text: emails, phone numbers, order numbers.
    
    Args:
        text: Input text
        
    Returns:
        Text with PII redacted
    """
    if not text:
        return text
    
    # Order matters: redact order numbers before phone numbers
    # to avoid false positives (order numbers can look like phone numbers)
    text = redact_email(text)
    text = redact_order_number(text)
    text = redact_phone(text)
    
    return text


def infer_category(text: str, tags: list[str] | None = None, subject: str | None = None) -> str:
    """
    Infer category from ticket text, tags, and subject.
    
    Args:
        text: Ticket description or comment text
        tags: List of ticket tags
        subject: Ticket subject
        
    Returns:
        Inferred category name or "unknown"
    """
    # Combine all text sources
    search_text = ""
    if subject:
        search_text += f" {subject.lower()}"
    if text:
        search_text += f" {text.lower()}"
    if tags:
        search_text += f" {' '.join(tags).lower()}"
    
    # Count matches for each category
    category_scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in search_text)
        if score > 0:
            category_scores[category] = score
    
    # Return category with highest score, or "unknown" if no matches
    if category_scores:
        return max(category_scores.items(), key=lambda x: x[1])[0]
    
    return "unknown"


def extract_user_message(ticket: dict[str, Any]) -> str:
    """
    Extract user message from ticket.
    
    Combines description and key fields into a user message.
    
    Args:
        ticket: Ticket dictionary
        
    Returns:
        User message text
    """
    parts = []
    
    # Add subject if present
    if ticket.get("subject"):
        parts.append(ticket["subject"])
    
    # Add description
    if ticket.get("description"):
        parts.append(ticket["description"])
    
    # Add other relevant fields
    if ticket.get("custom_fields"):
        # Include custom fields that might be relevant
        for field_name, field_value in ticket.get("custom_fields", {}).items():
            if field_value and isinstance(field_value, str):
                # Only include non-empty string fields
                if field_name.lower() in ["priority", "type", "status"]:
                    parts.append(f"{field_name}: {field_value}")
    
    user_message = "\n\n".join(filter(None, parts))
    
    # Redact PII
    user_message = redact_pii(user_message)
    
    return user_message.strip()


def extract_assistant_messages(ticket: dict[str, Any]) -> list[str]:
    """
    Extract assistant messages from ticket comments.
    
    Args:
        ticket: Ticket dictionary
        
    Returns:
        List of assistant message texts (redacted)
    """
    assistant_messages = []
    
    # Zendesk comments are typically in a "comments" array
    comments = ticket.get("comments", [])
    
    for comment in comments:
        # Check if comment is from an agent (not the requester)
        author_id = comment.get("author_id")
        ticket_requester_id = ticket.get("requester_id")
        
        # If author_id doesn't match requester_id, it's likely an agent reply
        # Also check for "public" type comments (agent replies)
        is_agent = (
            author_id != ticket_requester_id or
            comment.get("type") == "Comment" or
            comment.get("public") is True
        )
        
        if is_agent and comment.get("body"):
            message = redact_pii(comment["body"])
            if message.strip():
                assistant_messages.append(message.strip())
    
    return assistant_messages


def load_zendesk_json(export_path: Path) -> list[dict[str, Any]]:
    """
    Load tickets from Zendesk JSON export.
    
    Args:
        export_path: Path to JSON export file
        
    Returns:
        List of ticket dictionaries
    """
    with open(export_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Zendesk exports can be arrays or objects with a "tickets" key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "tickets" in data:
        return data["tickets"]
    elif isinstance(data, dict) and "results" in data:
        return data["results"]
    else:
        # Try to find tickets in nested structure
        for key in ["tickets", "results", "data"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    
    raise ValueError(f"Could not find ticket array in JSON file. Expected array or object with 'tickets'/'results' key.")


def load_zendesk_csv(export_path: Path) -> list[dict[str, Any]]:
    """
    Load tickets from Zendesk CSV export.
    
    Args:
        export_path: Path to CSV export file
        
    Returns:
        List of ticket dictionaries
    """
    tickets = []
    
    with open(export_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert CSV row to ticket-like dictionary
            ticket = {
                "subject": row.get("Subject", ""),
                "description": row.get("Description", ""),
                "status": row.get("Status", ""),
                "type": row.get("Type", ""),
                "priority": row.get("Priority", ""),
                "tags": row.get("Tags", "").split() if row.get("Tags") else [],
            }
            
            # Try to extract comments from CSV (if present)
            # Zendesk CSV exports may have comment columns
            comments = []
            for key, value in row.items():
                if "comment" in key.lower() and value:
                    comments.append({
                        "body": value,
                        "type": "Comment",
                        "public": True,
                    })
            
            if comments:
                ticket["comments"] = comments
            
            tickets.append(ticket)
    
    return tickets


def convert_ticket_to_example(
    ticket: dict[str, Any],
    system_prompt: str | None = None
) -> dict[str, Any] | None:
    """
    Convert a Zendesk ticket to training example format.
    
    Args:
        ticket: Ticket dictionary
        system_prompt: Optional system prompt to include
        
    Returns:
        Training example dictionary or None if ticket is invalid
    """
    # Extract user message
    user_message = extract_user_message(ticket)
    
    if not user_message:
        return None  # Skip tickets without user messages
    
    # Extract assistant messages
    assistant_messages = extract_assistant_messages(ticket)
    
    # Build messages array
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Add user message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    # Add assistant messages (if any)
    # If multiple assistant messages, we'll use the first one for simplicity
    # Multi-turn conversations could be supported by adding all messages
    if assistant_messages:
        messages.append({
            "role": "assistant",
            "content": assistant_messages[0]  # Use first agent reply
        })
    else:
        # If no assistant message, skip this ticket (can't train without response)
        return None
    
    # Infer category
    tags = ticket.get("tags", [])
    if isinstance(tags, str):
        tags = tags.split()
    subject = ticket.get("subject", "")
    category = infer_category(user_message, tags, subject)
    
    # Build example
    example = {
        "messages": messages,
        "metadata": {
            "source": "zendesk",
            "category": category,
            "zendesk_ticket_id": str(ticket.get("id", "")),
            "zendesk_status": ticket.get("status", ""),
        }
    }
    
    return example


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert Zendesk ticket exports to training JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest_zendesk_export.py tickets.json --out data/raw/zendesk_cases.jsonl
  python scripts/ingest_zendesk_export.py tickets.csv --out data/raw/zendesk_cases.jsonl
  python scripts/ingest_zendesk_export.py tickets.json --out data/raw/zendesk_cases.jsonl --system-prompt prompts/system.txt
        """
    )
    
    parser.add_argument(
        "export_path",
        type=str,
        help="Path to Zendesk export file (JSON or CSV)"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw/zendesk_cases.jsonl",
        help="Output JSONL file path (default: data/raw/zendesk_cases.jsonl)"
    )
    
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Path to system prompt file (optional, default: use standard system prompt)"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum character length for user/assistant messages (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    export_path = Path(args.export_path)
    if not export_path.is_absolute():
        export_path = Path.cwd() / export_path
    
    if not export_path.exists():
        print(f"Error: Export file not found: {export_path}", file=sys.stderr)
        sys.exit(1)
    
    output_path = project_root / args.out
    
    # Load system prompt if provided
    system_prompt = None
    if args.system_prompt:
        system_prompt_path = project_root / args.system_prompt
        if system_prompt_path.exists():
            system_prompt = system_prompt_path.read_text(encoding='utf-8').strip()
        else:
            print(f"Warning: System prompt file not found: {system_prompt_path}", file=sys.stderr)
    else:
        # Use default system prompt
        default_system_path = project_root / "prompts" / "system.txt"
        if default_system_path.exists():
            system_prompt = default_system_path.read_text(encoding='utf-8').strip()
    
    # Load tickets based on file extension
    print(f"Loading tickets from: {export_path}")
    if export_path.suffix.lower() == '.json':
        tickets = load_zendesk_json(export_path)
    elif export_path.suffix.lower() == '.csv':
        tickets = load_zendesk_csv(export_path)
    else:
        print(f"Error: Unsupported file format. Expected .json or .csv, got: {export_path.suffix}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(tickets)} tickets")
    
    # Convert tickets to examples
    print("Converting tickets to training examples...")
    examples = []
    skipped = 0
    
    for ticket in tickets:
        example = convert_ticket_to_example(ticket, system_prompt)
        
        if example is None:
            skipped += 1
            continue
        
        # Filter by minimum length
        user_msg = next((m["content"] for m in example["messages"] if m["role"] == "user"), "")
        assistant_msg = next((m["content"] for m in example["messages"] if m["role"] == "assistant"), "")
        
        if len(user_msg) < args.min_length or len(assistant_msg) < args.min_length:
            skipped += 1
            continue
        
        examples.append(example)
    
    print(f"Converted {len(examples)} examples (skipped {skipped} tickets)")
    
    # Count categories
    from collections import Counter
    categories = Counter(ex["metadata"]["category"] for ex in examples)
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    # Save examples
    print(f"\nSaving examples to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(examples, output_path)
    
    print(f"✓ Successfully saved {len(examples)} examples")


if __name__ == "__main__":
    main()

