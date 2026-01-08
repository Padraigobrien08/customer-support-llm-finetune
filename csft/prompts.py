"""
Prompt management utilities.

Loads system prompts and guidelines, and assembles final prompt messages.
"""

from pathlib import Path

from csft.io import FileNotFoundError, IOError, load_json


def load_system_prompt(prompt_path: str | Path) -> str:
    """
    Load system prompt from a text file.
    
    Args:
        prompt_path: Path to system prompt text file
        
    Returns:
        System prompt text
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    path = Path(prompt_path)
    
    if not path.exists():
        raise FileNotFoundError(f"System prompt file not found: {path.absolute()}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise IOError(f"System prompt file is empty: {path.absolute()}")
            return content
    except Exception as e:
        raise IOError(f"Error reading system prompt from {path.absolute()}: {str(e)}") from e


def load_guidelines(guidelines_path: str | Path) -> str:
    """
    Load prompt guidelines from a markdown file.
    
    Args:
        guidelines_path: Path to guidelines markdown file
        
    Returns:
        Guidelines text (or empty string if file doesn't exist)
        
    Raises:
        IOError: If file exists but cannot be read
    """
    path = Path(guidelines_path)
    
    if not path.exists():
        return ""  # Guidelines are optional
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise IOError(f"Error reading guidelines from {path.absolute()}: {str(e)}") from e


def assemble_system_message(
    system_prompt: str,
    guidelines: str | None = None,
    include_guidelines: bool = True
) -> str:
    """
    Assemble final system message from prompt and optional guidelines.
    
    Args:
        system_prompt: Base system prompt text
        guidelines: Optional guidelines text
        include_guidelines: Whether to append guidelines to the system message
        
    Returns:
        Assembled system message
    """
    if not system_prompt.strip():
        raise ValueError("System prompt cannot be empty")
    
    parts = [system_prompt]
    
    if include_guidelines and guidelines and guidelines.strip():
        parts.append("\n\n## Response Guidelines\n")
        parts.append(guidelines)
    
    return "\n".join(parts)


def load_and_assemble_system_message(
    prompts_dir: str | Path,
    include_guidelines: bool = True
) -> str:
    """
    Load system prompt and guidelines from directory and assemble final message.
    
    Looks for:
    - prompts_dir/system.txt (required)
    - prompts_dir/guidelines.md (optional)
    
    Args:
        prompts_dir: Directory containing prompt files
        include_guidelines: Whether to include guidelines in the system message
        
    Returns:
        Assembled system message
        
    Raises:
        FileNotFoundError: If system.txt is not found
        IOError: If files cannot be read
    """
    prompts_dir = Path(prompts_dir)
    
    system_prompt_path = prompts_dir / "system.txt"
    guidelines_path = prompts_dir / "guidelines.md"
    
    system_prompt = load_system_prompt(system_prompt_path)
    guidelines = load_guidelines(guidelines_path) if include_guidelines else None
    
    return assemble_system_message(system_prompt, guidelines, include_guidelines)

