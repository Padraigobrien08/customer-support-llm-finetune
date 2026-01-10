"""
Data models for customer support fine-tuning.

Uses Pydantic for validation and type safety.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    """Valid roles for chat messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: MessageRole = Field(..., description="Message role (system, user, or assistant)")
    content: str = Field(..., min_length=1, description="Message text content")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Ensure content is not empty after stripping."""
        if not v.strip():
            raise ValueError("Message content cannot be empty or whitespace only")
        return v.strip()


class Conversation(BaseModel):
    """A complete conversation with multiple messages."""
    messages: list[ChatMessage] = Field(..., min_length=1, description="Ordered list of messages")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        """Ensure at least one user message and one assistant message."""
        roles = [msg.role for msg in v]
        if MessageRole.USER not in roles:
            raise ValueError("Conversation must contain at least one user message")
        if MessageRole.ASSISTANT not in roles:
            raise ValueError("Conversation must contain at least one assistant message")
        return v


class TestCase(BaseModel):
    """A golden test case for evaluation."""
    id: str = Field(..., description="Unique test case identifier")
    category: str = Field(..., description="Test case category")
    messages: list[ChatMessage] = Field(..., description="Input messages for the test case")
    ideal_response: str = Field(..., min_length=1, description="Expected ideal response")
    notes: str | None = Field(None, description="Notes explaining why the response is good")


class ModelResponse(BaseModel):
    """A model-generated response."""
    content: str = Field(..., description="Generated response text (may be empty if generation failed)")
    error_type: str | None = Field(None, description="Type of error if generation failed (e.g., 'ValidationError', 'ProviderError')")
    error_message: str | None = Field(None, description="Error message if generation failed")
    metadata: dict[str, str | int | float | bool] | None = Field(
        None, description="Optional metadata (tokens, latency, etc.)"
    )


class EvaluationScore(BaseModel):
    """Scores for a single evaluation dimension."""
    factual_correctness: int = Field(..., ge=0, le=2, description="Factual correctness score (0-2)")
    helpfulness: int = Field(..., ge=0, le=2, description="Helpfulness/task completion score (0-2)")
    tone: int = Field(..., ge=0, le=2, description="Tone and professionalism score (0-2)")
    safety: int = Field(..., ge=0, le=2, description="Safety and hallucination avoidance score (0-2)")
    escalation: int = Field(..., ge=0, le=2, description="Escalation appropriateness score (0-2)")

    @property
    def total(self) -> int:
        """Calculate total score (0-10)."""
        return (
            self.factual_correctness
            + self.helpfulness
            + self.tone
            + self.safety
            + self.escalation
        )


class EvaluationResult(BaseModel):
    """Complete evaluation result for a single test case."""
    test_case_id: str = Field(..., description="ID of the evaluated test case")
    generated_response: str = Field(..., description="Model-generated response")
    ideal_response: str = Field(..., description="Ideal response from test case")
    scores: EvaluationScore = Field(..., description="Evaluation scores")
    metadata: dict[str, str | int | float | bool] | None = Field(
        None, description="Additional evaluation metadata"
    )

