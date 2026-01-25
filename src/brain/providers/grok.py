#!/usr/bin/env python3
"""
FSDTrader Brain Module: Grok LLM Provider

Implementation of the LLMProvider interface for xAI's Grok.
Uses OpenAI-compatible API endpoint.
"""
import time
import json
import logging
from typing import List, Dict, Any, Optional

import httpx

from .base import LLMProvider
from ..types import LLMResponse


logger = logging.getLogger(__name__)


# Grok API configuration
GROK_API_BASE = "https://api.x.ai/v1"
GROK_DEFAULT_MODEL = "grok-3-mini-fast"  # Fast model for trading decisions


class GrokProvider(LLMProvider):
    """
    Grok LLM provider using xAI's API.

    Uses OpenAI-compatible chat completions endpoint with tool calling.
    """

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        timeout: float = 5.0,
        **kwargs
    ):
        """
        Initialize the Grok provider.

        Args:
            api_key: xAI API key
            model: Model to use (default: grok-3-mini-fast)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration (ignored)
        """
        self._api_key = api_key
        self._model = model or GROK_DEFAULT_MODEL
        self._timeout = timeout

        # HTTP client configuration
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Sync client (created on first use)
        self._sync_client: Optional[httpx.Client] = None

    def _get_sync_client(self) -> httpx.Client:
        """Get or create sync HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=GROK_API_BASE,
                headers=self._headers,
                timeout=self._timeout,
            )
        return self._sync_client

    def call(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Make a synchronous call to Grok.

        Args:
            system_prompt: System prompt for the model
            user_message: User message with market context
            tools: Tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            LLMResponse with tool calls
        """
        start_time = time.time()

        # Format request
        formatted_tools = self.format_tools(tools)
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "tools": formatted_tools,
            "tool_choice": "required",  # Force tool call
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            client = self._get_sync_client()
            response = client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            tool_calls = self.parse_tool_calls(data)
            content = None
            if data.get("choices") and data["choices"][0].get("message"):
                content = data["choices"][0]["message"].get("content")

            # Extract usage
            usage = data.get("usage", {})

            return LLMResponse(
                tool_calls=tool_calls,
                content=content,
                model=self._model,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                latency_ms=latency_ms,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Grok API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Grok call failed: {e}")
            raise

    async def call_async(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Make an asynchronous call to Grok.

        Args:
            system_prompt: System prompt for the model
            user_message: User message with market context
            tools: Tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            LLMResponse with tool calls
        """
        start_time = time.time()

        # Format request
        formatted_tools = self.format_tools(tools)
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "tools": formatted_tools,
            "tool_choice": "required",
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(
                base_url=GROK_API_BASE,
                headers=self._headers,
                timeout=self._timeout,
            ) as client:
                response = await client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            tool_calls = self.parse_tool_calls(data)
            content = None
            if data.get("choices") and data["choices"][0].get("message"):
                content = data["choices"][0]["message"].get("content")

            # Extract usage
            usage = data.get("usage", {})

            return LLMResponse(
                tool_calls=tool_calls,
                content=content,
                model=self._model,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                latency_ms=latency_ms,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Grok API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Grok async call failed: {e}")
            raise

    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tools for Grok's OpenAI-compatible API.

        Grok uses the OpenAI function calling format:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }

        Args:
            tools: Tools in canonical format

        Returns:
            Tools in Grok/OpenAI format
        """
        formatted = []
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                }
            })
        return formatted

    def parse_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse tool calls from Grok response.

        Grok returns tool calls in OpenAI format:
        {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "...",
                            "arguments": "..."  # JSON string
                        }
                    }]
                }
            }]
        }

        Args:
            response: Raw API response

        Returns:
            List of tool calls in canonical format
        """
        tool_calls = []

        choices = response.get("choices", [])
        if not choices:
            logger.warning("No choices in Grok response")
            return tool_calls

        message = choices[0].get("message", {})
        raw_calls = message.get("tool_calls", [])

        for call in raw_calls:
            func = call.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            # Parse arguments JSON
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse tool arguments: {args_str}")
                arguments = {}

            tool_calls.append({
                "name": name,
                "arguments": arguments,
            })

        return tool_calls

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "grok"

    def close(self):
        """Close the HTTP client."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
