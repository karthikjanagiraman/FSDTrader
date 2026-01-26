#!/usr/bin/env python3
"""
FSDTrader Brain Module: LLM Providers

Provider factory and exports.
"""
from .base import LLMProvider
from .grok import GrokProvider
from .groq import GroqProvider


def get_provider(provider_name: str, **kwargs) -> LLMProvider:
    """
    Factory function to get an LLM provider instance.

    Args:
        provider_name: Name of the provider ("grok", "groq", "openai", "anthropic")
        **kwargs: Provider-specific configuration

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If provider_name is not supported
    """
    providers = {
        "grok": GrokProvider,
        "groq": GroqProvider,
        # Future providers:
        # "openai": OpenAIProvider,
        # "anthropic": AnthropicProvider,
    }

    provider_class = providers.get(provider_name.lower())
    if provider_class is None:
        supported = ", ".join(providers.keys())
        raise ValueError(f"Unknown provider '{provider_name}'. Supported: {supported}")

    return provider_class(**kwargs)


__all__ = [
    "LLMProvider",
    "GrokProvider",
    "GroqProvider",
    "get_provider",
]
