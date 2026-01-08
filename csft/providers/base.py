"""
Base provider interface for model inference.

Defines the abstract interface that all model providers must implement.
This keeps the core layer vendor-neutral.
"""

from abc import ABC, abstractmethod

from csft.types import ChatMessage, ModelResponse


class Provider(ABC):
    """
    Abstract base class for model providers.
    
    All model providers (OpenAI, Anthropic, local models, etc.) must implement
    this interface to ensure consistent usage across the codebase.
    """
    
    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate a response from the model.
        
        Args:
            messages: List of conversation messages (excluding system prompt)
            system_prompt: Optional system prompt to prepend
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.)
            
        Returns:
            ModelResponse with generated content and optional metadata
            
        Raises:
            ProviderError: If generation fails
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation of the provider."""
        pass


class ProviderError(Exception):
    """Base exception for provider-related errors."""
    pass

