"""
I/O utilities for loading and saving data files.

Provides robust error handling with actionable error messages.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from csft.types import Conversation, TestCase


class IOError(Exception):
    """Base exception for I/O operations."""
    pass


class FileNotFoundError(IOError):
    """Raised when a file is not found."""
    pass


class InvalidDataError(IOError):
    """Raised when data validation fails."""
    pass


def load_json(file_path: str | Path) -> dict[str, Any] | list[Any]:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        InvalidDataError: If JSON is malformed
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.absolute()}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise InvalidDataError(
            f"Invalid JSON in file {path.absolute()}: {e.msg} at line {e.lineno}, column {e.colno}"
        ) from e
    except Exception as e:
        raise IOError(f"Error reading file {path.absolute()}: {str(e)}") from e


def save_json(data: dict[str, Any] | list[Any], file_path: str | Path, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save (must be JSON-serializable)
        file_path: Path to save file
        indent: JSON indentation level
        
    Raises:
        IOError: If file cannot be written
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"Error writing file {path.absolute()}: {str(e)}") from e


def load_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Load JSONL (JSON Lines) data from a file.
    
    Each line must be a valid JSON object. Provides detailed error messages
    including the line number and record identifier if available.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of parsed JSON objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        InvalidDataError: If any line contains invalid JSON
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.absolute()}")
    
    records = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    # Try to extract an ID from the record for better error messages
                    record_id = "unknown"
                    try:
                        partial = json.loads(line[:1000])  # Try parsing first part
                        record_id = partial.get('id', partial.get('_id', 'unknown'))
                    except:
                        pass
                    
                    raise InvalidDataError(
                        f"Invalid JSON at line {line_num} in {path.absolute()}: {e.msg} "
                        f"(record id: {record_id})"
                    ) from e
    except InvalidDataError:
        raise
    except Exception as e:
        raise IOError(f"Error reading file {path.absolute()}: {str(e)}") from e
    
    return records


def save_jsonl(data: list[dict[str, Any]], file_path: str | Path) -> None:
    """
    Save data to a JSONL (JSON Lines) file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to save file
        
    Raises:
        IOError: If file cannot be written
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for record in data:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
    except Exception as e:
        raise IOError(f"Error writing file {path.absolute()}: {str(e)}") from e


def load_conversations(file_path: str | Path) -> list[Conversation]:
    """
    Load conversations from a JSONL file with validation.
    
    Args:
        file_path: Path to JSONL file containing conversation data
        
    Returns:
        List of validated Conversation objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        InvalidDataError: If any record fails validation
    """
    records = load_jsonl(file_path)
    conversations = []
    
    for idx, record in enumerate(records, start=1):
        record_id = record.get('id', f"record_{idx}")
        try:
            conversation = Conversation.model_validate(record)
            conversations.append(conversation)
        except ValidationError as e:
            raise InvalidDataError(
                f"Validation failed for record {record_id} in {Path(file_path).absolute()}: {e}"
            ) from e
    
    return conversations


def load_test_cases(file_path: str | Path) -> list[TestCase]:
    """
    Load test cases from a JSON file with validation.
    
    Args:
        file_path: Path to JSON file containing test cases
        
    Returns:
        List of validated TestCase objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        InvalidDataError: If any test case fails validation
    """
    data = load_json(file_path)
    
    if not isinstance(data, list):
        raise InvalidDataError(
            f"Expected list of test cases in {Path(file_path).absolute()}, got {type(data).__name__}"
        )
    
    test_cases = []
    for idx, record in enumerate(data, start=1):
        record_id = record.get('id', f"test_case_{idx}")
        try:
            test_case = TestCase.model_validate(record)
            test_cases.append(test_case)
        except ValidationError as e:
            raise InvalidDataError(
                f"Validation failed for test case {record_id} in {Path(file_path).absolute()}: {e}"
            ) from e
    
    return test_cases

