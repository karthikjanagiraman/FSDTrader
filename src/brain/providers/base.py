#!/usr/bin/env python3
"""
FSDTrader Brain Module: Base LLM Provider

Abstract interface for LLM providers.
All providers must implement this interface.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..types import LLMResponse


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers (Grok, OpenAI, Anthropic, etc.) must implement
    this interface to be used with the TradingBrain.
    """

    @abstractmethod
    def __init__(self, api_key: str, model: Optional[str] = None, **kwargs):
        """
        Initialize the provider.

        Args:
            api_key: API key for the provider
            model: Model identifier (provider-specific default if None)
            **kwargs: Additional provider-specific configuration
        """
        pass

    @abstractmethod
    def call(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Make a synchronous call to the LLM.

        Args:
            system_prompt: The system prompt defining behavior
            user_message: The user message (context + instructions)
            tools: List of tool definitions in provider-native format
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum response tokens

        Returns:
            LLMResponse containing tool calls and metadata
        """
        pass

    @abstractmethod
    async def call_async(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Make an asynchronous call to the LLM.

        Args:
            system_prompt: The system prompt defining behavior
            user_message: The user message (context + instructions)
            tools: List of tool definitions in provider-native format
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum response tokens

        Returns:
            LLMResponse containing tool calls and metadata
        """
        pass

    @abstractmethod
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tools into provider-native format.

        Different providers have slightly different tool/function schemas.
        This method converts our canonical tool format to provider-specific.

        Args:
            tools: List of tools in canonical format

        Returns:
            List of tools in provider-native format
        """
        pass

    @abstractmethod
    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """
        Parse tool calls from provider response.

        Extracts tool calls from provider-specific response format
        into our canonical format.

        Args:
            response: Raw provider response

        Returns:
            List of tool calls in canonical format:
            [{"name": str, "arguments": dict}, ...]
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier being used."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'grok', 'openai')."""
        pass
