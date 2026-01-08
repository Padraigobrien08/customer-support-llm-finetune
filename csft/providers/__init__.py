"""
Model provider implementations.

Provides vendor-neutral interface for model inference.
"""

from csft.providers.base import Provider, ProviderError
from csft.providers.mock import MockProvider

# Conditionally import HF provider (requires transformers)
try:
    from csft.providers.hf_local import HFLocalProvider
    __all__ = ["Provider", "ProviderError", "MockProvider", "HFLocalProvider"]
except ImportError:
    __all__ = ["Provider", "ProviderError", "MockProvider"]

