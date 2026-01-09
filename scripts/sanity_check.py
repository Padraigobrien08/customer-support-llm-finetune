#!/usr/bin/env python3
"""
Sanity check script to validate data format and schema.

Validates JSONL files against the data schema and taxonomy.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path for csft package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from csft.io import load_jsonl, InvalidDataError, FileNotFoundError


def load_taxonomy() -> dict[str, Any]:
    """
    Load taxonomy from YAML file.
    
    Returns:
        Taxonomy dictionary with categories
    """
    taxonomy_path = project_root / "data" / "taxonomy.yaml"
    if not taxonomy_path.exists():
        return {}
    
    try:
        import yaml
        with open(taxonomy_path, 'r') as f:
            taxonomy = yaml.safe_load(f)
        return taxonomy or {}
    except ImportError:
        return {}
    except Exception:
        return {}


def validate_jsonl_file(file_path: Path) -> tuple[bool, list[str]]:
    """
    Validate JSONL file against schema and taxonomy.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if not file_path.exists():
        errors.append(f"File not found: {file_path.absolute()}")
        return False, errors
    
    # Load taxonomy
    taxonomy = load_taxonomy()
    valid_categories = set()
    if taxonomy and "categories" in taxonomy:
        valid_categories = {cat.get("name") for cat in taxonomy["categories"] if isinstance(cat, dict) and "name" in cat}
    
    # Load records
    try:
        records = load_jsonl(str(file_path))
    except (InvalidDataError, FileNotFoundError) as e:
        errors.append(str(e))
        return False, errors
    except Exception as e:
        errors.append(f"Error loading file: {e}")
        return False, errors
    
    if len(records) == 0:
        errors.append("File contains no records")
        return False, errors
    
    # Validate each record
    valid_roles = {"system", "user", "assistant"}
    required_metadata_keys = {"source", "category", "escalation", "difficulty", "contains_policy_claims", "test_case_id"}
    
    for line_num, record in enumerate(records, 1):
        # Check required fields
        if "messages" not in record:
            errors.append(f"{file_path}:{line_num}: missing 'messages' field")
            continue
        
        messages = record["messages"]
        if not isinstance(messages, list):
            errors.append(f"{file_path}:{line_num}: 'messages' must be a list")
            continue
        
        if len(messages) == 0:
            errors.append(f"{file_path}:{line_num}: 'messages' list is empty")
            continue
        
        # Validate each message
        has_user = False
        has_assistant = False
        
        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(f"{file_path}:{line_num}: message {msg_idx}: must be a dict")
                break
            
            if "role" not in msg:
                errors.append(f"{file_path}:{line_num}: message {msg_idx}: missing 'role'")
                break
            
            if "content" not in msg:
                errors.append(f"{file_path}:{line_num}: message {msg_idx}: missing 'content'")
                break
            
            role = msg["role"]
            content = msg["content"]
            
            if role not in valid_roles:
                errors.append(f"{file_path}:{line_num}: message {msg_idx}: invalid role '{role}'. Must be one of: {valid_roles}")
                break
            
            if not isinstance(content, str):
                errors.append(f"{file_path}:{line_num}: message {msg_idx}: content must be string")
                break
            
            if not content.strip():
                errors.append(f"{file_path}:{line_num}: message {msg_idx}: content is empty")
                break
            
            if role == "user":
                has_user = True
            if role == "assistant":
                has_assistant = True
        else:
            # Only reached if no break occurred
            if not has_user:
                errors.append(f"{file_path}:{line_num}: missing 'user' message")
                continue
            
            if not has_assistant:
                errors.append(f"{file_path}:{line_num}: missing 'assistant' message")
                continue
            
            # Validate metadata
            if "metadata" not in record:
                errors.append(f"{file_path}:{line_num}: missing 'metadata' field")
                continue
            
            metadata = record["metadata"]
            if not isinstance(metadata, dict):
                errors.append(f"{file_path}:{line_num}: 'metadata' must be a dict")
                continue
            
            # Check required metadata keys
            for key in required_metadata_keys:
                if key not in metadata:
                    errors.append(f"{file_path}:{line_num}: missing required metadata key '{key}'")
                    break
            else:
                # Validate source
                source = metadata.get("source")
                if source not in ["manual", "synthetic", "zendesk"]:
                    errors.append(f"{file_path}:{line_num}: invalid source '{source}'. Must be one of: manual, synthetic, zendesk")
                    continue
                
                # Validate category
                category = metadata.get("category")
                if valid_categories and category not in valid_categories:
                    errors.append(f"{file_path}:{line_num}: invalid category '{category}'. Valid categories: {sorted(valid_categories)}")
                    continue
                
                # Validate escalation (boolean)
                escalation = metadata.get("escalation")
                if not isinstance(escalation, bool):
                    errors.append(f"{file_path}:{line_num}: 'escalation' must be boolean (true/false), got {type(escalation).__name__}")
                    continue
                
                # Validate difficulty (1-3)
                difficulty = metadata.get("difficulty")
                if not isinstance(difficulty, int):
                    errors.append(f"{file_path}:{line_num}: 'difficulty' must be integer, got {type(difficulty).__name__}")
                    continue
                if difficulty < 1 or difficulty > 3:
                    errors.append(f"{file_path}:{line_num}: 'difficulty' must be between 1 and 3, got {difficulty}")
                    continue
                
                # Validate contains_policy_claims (boolean)
                contains_policy_claims = metadata.get("contains_policy_claims")
                if not isinstance(contains_policy_claims, bool):
                    errors.append(f"{file_path}:{line_num}: 'contains_policy_claims' must be boolean (true/false), got {type(contains_policy_claims).__name__}")
                    continue
                
                # Validate test_case_id (string)
                test_case_id = metadata.get("test_case_id")
                if not isinstance(test_case_id, str):
                    errors.append(f"{file_path}:{line_num}: 'test_case_id' must be string, got {type(test_case_id).__name__}")
                    continue
                if not test_case_id.strip():
                    errors.append(f"{file_path}:{line_num}: 'test_case_id' must be non-empty")
                    continue
    
    is_valid = len(errors) == 0
    return is_valid, errors


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Validate JSONL data file against schema and taxonomy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/sanity_check.py data/raw/manual_cases.jsonl
  python scripts/sanity_check.py data/processed/all.jsonl
        """
    )
    
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to JSONL file to validate"
    )
    
    args = parser.parse_args()
    
    # Resolve path
    file_path = Path(args.filepath)
    if not file_path.is_absolute():
        file_path = project_root / file_path
    
    print(f"Validating: {file_path}")
    print()
    
    is_valid, errors = validate_jsonl_file(file_path)
    
    if is_valid:
        print("✓ Validation passed")
        sys.exit(0)
    else:
        print("✗ Validation failed")
        print()
        print("Errors:")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
