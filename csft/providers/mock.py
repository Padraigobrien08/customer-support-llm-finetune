"""
Mock provider for testing and development.

Provides deterministic outputs for testing without requiring actual model calls.
"""

from csft.providers.base import Provider, ProviderError
from csft.types import ChatMessage, ModelResponse


class MockProvider(Provider):
    """
    Mock provider that returns deterministic responses.
    
    Useful for:
    - Testing evaluation logic without model calls
    - Development and debugging
    - CI/CD pipelines where model access is unavailable
    """
    
    def __init__(self, response_template: str = "This is a mock response."):
        """
        Initialize mock provider.
        
        Args:
            response_template: Template for generated responses. Can include
                              placeholders like {user_message_count} for
                              dynamic content.
        """
        self.response_template = response_template
    
    def generate(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate a mock response.
        
        The response is deterministic based on the input messages.
        This allows for consistent testing.
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt (ignored in mock)
            **kwargs: Ignored in mock provider
            
        Returns:
            ModelResponse with mock content
        """
        if not messages:
            raise ProviderError("Cannot generate response: messages list is empty")
        
        # Count user messages for dynamic response
        user_count = sum(1 for msg in messages if msg.role.value == "user")
        last_user_message = next(
            (msg.content for msg in reversed(messages) if msg.role.value == "user"),
            ""
        )
        
        # Generate deterministic response
        response_content = self.response_template.format(
            user_message_count=user_count,
            last_user_message=last_user_message[:50],  # First 50 chars
            message_count=len(messages)
        )
        
        # Add a simple indicator that this is a mock response
        if "{user_message_count}" not in self.response_template:
            response_content = f"[MOCK] {response_content}"
        
        return ModelResponse(
            content=response_content,
            metadata={
                "provider": "mock",
                "message_count": len(messages),
                "user_message_count": user_count
            }
        )
    
    def __repr__(self) -> str:
        return f"MockProvider(response_template='{self.response_template[:30]}...')"

