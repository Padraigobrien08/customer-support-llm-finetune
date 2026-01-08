#!/usr/bin/env python3
"""
Sanity check script to validate data and model setup.

Use this to verify that your data and configuration are correct before training.
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


def validate_jsonl_format(data_path: Path) -> tuple[bool, dict[str, Any]]:
    """
    Validate JSONL format and structure.
    
    Args:
        data_path: Path to JSONL file
        
    Returns:
        Tuple of (is_valid, stats_dict)
    """
    stats = {
        "total_records": 0,
        "valid_records": 0,
        "invalid_records": 0,
        "errors": [],
        "role_counts": {"system": 0, "user": 0, "assistant": 0},
        "has_metadata": 0
    }
    
    if not data_path.exists():
        return False, {"error": f"File not found: {data_path.absolute()}"}
    
    try:
        records = load_jsonl(str(data_path))
        stats["total_records"] = len(records)
    except (InvalidDataError, FileNotFoundError) as e:
        return False, {"error": str(e)}
    except Exception as e:
        return False, {"error": f"Unexpected error loading file: {e}"}
    
    valid_roles = {"system", "user", "assistant"}
    
    for idx, record in enumerate(records):
        try:
            # Check required fields
            if "messages" not in record:
                stats["errors"].append(f"Record {idx}: missing 'messages' field")
                stats["invalid_records"] += 1
                continue
            
            messages = record["messages"]
            if not isinstance(messages, list):
                stats["errors"].append(f"Record {idx}: 'messages' must be a list")
                stats["invalid_records"] += 1
                continue
            
            if len(messages) == 0:
                stats["errors"].append(f"Record {idx}: 'messages' list is empty")
                stats["invalid_records"] += 1
                continue
            
            # Validate each message
            has_user = False
            has_assistant = False
            
            for msg_idx, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    stats["errors"].append(f"Record {idx}, message {msg_idx}: must be a dict")
                    stats["invalid_records"] += 1
                    break
                
                if "role" not in msg:
                    stats["errors"].append(f"Record {idx}, message {msg_idx}: missing 'role'")
                    stats["invalid_records"] += 1
                    break
                
                if "content" not in msg:
                    stats["errors"].append(f"Record {idx}, message {msg_idx}: missing 'content'")
                    stats["invalid_records"] += 1
                    break
                
                role = msg["role"]
                content = msg["content"]
                
                if role not in valid_roles:
                    stats["errors"].append(f"Record {idx}, message {msg_idx}: invalid role '{role}'")
                    stats["invalid_records"] += 1
                    break
                
                if not isinstance(content, str):
                    stats["errors"].append(f"Record {idx}, message {msg_idx}: content must be string")
                    stats["invalid_records"] += 1
                    break
                
                if not content.strip():
                    stats["errors"].append(f"Record {idx}, message {msg_idx}: content is empty")
                    stats["invalid_records"] += 1
                    break
                
                # Count roles
                if role in stats["role_counts"]:
                    stats["role_counts"][role] += 1
                
                if role == "user":
                    has_user = True
                if role == "assistant":
                    has_assistant = True
            else:
                # Only reached if no break occurred
                if not has_user:
                    stats["errors"].append(f"Record {idx}: missing 'user' message")
                    stats["invalid_records"] += 1
                    continue
                
                if not has_assistant:
                    stats["errors"].append(f"Record {idx}: missing 'assistant' message")
                    stats["invalid_records"] += 1
                    continue
                
                # Check metadata
                if "metadata" in record:
                    stats["has_metadata"] += 1
                
                stats["valid_records"] += 1
                
        except Exception as e:
            stats["errors"].append(f"Record {idx}: unexpected error - {e}")
            stats["invalid_records"] += 1
    
    is_valid = stats["invalid_records"] == 0 and stats["total_records"] > 0
    return is_valid, stats


def check_data_format(data_path: str) -> bool:
    """
    Validate data format and structure.
    
    Args:
        data_path: Path to data file
        
    Returns:
        True if valid, False otherwise
    """
    path = Path(data_path)
    if not path.is_absolute():
        path = project_root / data_path
    
    print(f"Checking data format: {path}")
    
    if path.suffix == ".jsonl":
        is_valid, stats = validate_jsonl_format(path)
        
        if not is_valid:
            if "error" in stats:
                print(f"  ✗ Error: {stats['error']}")
            else:
                print(f"  ✗ Validation failed")
                print(f"    Total records: {stats['total_records']}")
                print(f"    Valid records: {stats['valid_records']}")
                print(f"    Invalid records: {stats['invalid_records']}")
                if stats["errors"]:
                    print(f"    First 5 errors:")
                    for error in stats["errors"][:5]:
                        print(f"      - {error}")
            return False
        else:
            print(f"  ✓ Valid JSONL format")
            print(f"    Total records: {stats['total_records']}")
            print(f"    Valid records: {stats['valid_records']}")
            print(f"    Role counts: {stats['role_counts']}")
            print(f"    Records with metadata: {stats['has_metadata']}")
            return True
    else:
        print(f"  ⚠ Unsupported format: {path.suffix}")
        print(f"    Supported: .jsonl")
        return False


def check_config(config_path: str) -> bool:
    """
    Validate training configuration.
    
    Args:
        config_path: Path to config file
        
    Returns:
        True if valid, False otherwise
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root / config_path
    
    print(f"Checking config: {path}")
    
    if not path.exists():
        print(f"  ✗ Config file not found")
        return False
    
    try:
        import yaml
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config:
            print(f"  ✗ Config file is empty")
            return False
        
        print(f"  ✓ Config file loaded")
        # TODO: Add more specific validation
        return True
    except ImportError:
        print(f"  ⚠ PyYAML not installed, skipping config validation")
        return True
    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Sanity check for data and configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/sanity_check.py
  python scripts/sanity_check.py --data data/processed/train_seed.jsonl
  python scripts/sanity_check.py --data data/processed/train_seed.jsonl --config training/config.yaml
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/train_seed.jsonl",
        help="Path to data file to validate (default: data/processed/train_seed.jsonl)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file to validate (optional)"
    )
    
    args = parser.parse_args()
    
    print("Running sanity checks...\n")
    
    all_valid = True
    
    # Check data format
    if not check_data_format(args.data):
        all_valid = False
    
    print()
    
    # Check config if provided
    if args.config:
        if not check_config(args.config):
            all_valid = False
        print()
    
    if all_valid:
        print("✓ All checks passed")
        sys.exit(0)
    else:
        print("✗ Some checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
