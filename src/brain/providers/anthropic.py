#!/usr/bin/env python3
"""
FSDTrader Brain Module: Anthropic LLM Provider

Implementation of the LLMProvider interface for Anthropic's Claude models.
Uses the Messages API with native tool use support.
"""
import time
import json
import logging
from typing import List, Dict, Any, Optional

import httpx

from .base import LLMProvider
from ..types import LLMResponse


logger = logging.getLogger(__name__)

# Rate limit retry configuration
MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0  # seconds

# Anthropic API configuration
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_DEFAULT_MODEL = "claude-3-haiku-20240307"  # Fast and cost-effective
ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    """
    Anthropic LLM provider for Claude models.

    Uses the Messages API with native tool use for structured outputs.
    Claude-3-haiku is the default for fast, cost-effective inference.
    """

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        timeout: float = 30.0,
        **kwargs
    ):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model to use (default: claude-3-haiku-20240307)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration (ignored)
        """
        self._api_key = api_key
        self._model = model or ANTHROPIC_DEFAULT_MODEL
        self._timeout = timeout

        # HTTP client configuration
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "Content-Type": "application/json",
        }

        # Sync client (created on first use)
        self._sync_client: Optional[httpx.Client] = None

    def _get_sync_client(self) -> httpx.Client:
        """Get or create sync HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=ANTHROPIC_API_BASE,
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
        Make a synchronous call to Anthropic.

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

        # Format request for Anthropic Messages API
        formatted_tools = self.format_tools(tools)
        payload = {
            "model": self._model,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_message},
            ],
            "tools": formatted_tools,
            "tool_choice": {"type": "any"},  # Force tool use
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        client = self._get_sync_client()

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.post("/messages", json=payload)
                response.raise_for_status()
                data = response.json()

                latency_ms = (time.time() - start_time) * 1000

                # Parse response
                tool_calls = self.parse_tool_calls(data)
                content = self._extract_text_content(data)

                # Extract usage
                usage = data.get("usage", {})

                return LLMResponse(
                    tool_calls=tool_calls,
                    content=content,
                    model=self._model,
                    usage={
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    },
                    latency_ms=latency_ms,
                )

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < MAX_RETRIES:
                    # Rate limited - use default retry delay
                    retry_delay = self._extract_retry_delay(e.response)
                    logger.warning(f"Anthropic rate limited, retrying in {retry_delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(retry_delay)
                    continue
                elif e.response.status_code == 529 and attempt < MAX_RETRIES:
                    # Overloaded - retry with backoff
                    retry_delay = DEFAULT_RETRY_DELAY * (attempt + 1)
                    logger.warning(f"Anthropic overloaded, retrying in {retry_delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(retry_delay)
                    continue
                logger.error(f"Anthropic API error: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Anthropic call failed: {e}")
                raise

        raise RuntimeError("Max retries exceeded for Anthropic API")

    def _extract_retry_delay(self, response: httpx.Response) -> float:
        """Extract retry delay from Anthropic rate limit response."""
        # Check for Retry-After header
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return DEFAULT_RETRY_DELAY

    def _extract_text_content(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract text content from Anthropic response."""
        content_blocks = response.get("content", [])
        text_parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(text_parts) if text_parts else None

    async def call_async(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Make an asynchronous call to Anthropic.

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
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_message},
            ],
            "tools": formatted_tools,
            "tool_choice": {"type": "any"},
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(
                base_url=ANTHROPIC_API_BASE,
                headers=self._headers,
                timeout=self._timeout,
            ) as client:
                response = await client.post("/messages", json=payload)
                response.raise_for_status()
                data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            tool_calls = self.parse_tool_calls(data)
            content = self._extract_text_content(data)

            # Extract usage
            usage = data.get("usage", {})

            return LLMResponse(
                tool_calls=tool_calls,
                content=content,
                model=self._model,
                usage={
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                },
                latency_ms=latency_ms,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Anthropic API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Anthropic async call failed: {e}")
            raise

    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tools for Anthropic's tool use format.

        Anthropic uses a similar but slightly different format:
        {
            "name": "...",
            "description": "...",
            "input_schema": {...}  # JSON Schema for parameters
        }

        Args:
            tools: Tools in canonical format or OpenAI format

        Returns:
            Tools in Anthropic format
        """
        formatted = []
        for tool in tools:
            # Check if already in Anthropic format (has "input_schema")
            if "input_schema" in tool:
                formatted.append(tool)
            # Check if in OpenAI format (has "function" key)
            elif "function" in tool:
                func = tool["function"]
                formatted.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                # Convert from canonical format
                formatted.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
                })
        return formatted

    def _coerce_argument_types(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coerce string arguments to expected numeric types.

        Args:
            tool_name: Name of the tool being called
            arguments: Raw arguments from LLM response

        Returns:
            Arguments with numeric fields coerced to proper types
        """
        NUMERIC_FIELDS = {
            "enter_long": ["limit_price", "stop_loss", "profit_target", "size"],
            "enter_short": ["limit_price", "stop_loss", "profit_target", "size"],
            "update_stop": ["new_price"],
            "update_target": ["new_price"],
        }
        INT_FIELDS = {"size"}

        numeric_fields = NUMERIC_FIELDS.get(tool_name, [])

        for field in numeric_fields:
            if field in arguments and isinstance(arguments[field], str):
                try:
                    if field in INT_FIELDS:
                        arguments[field] = int(float(arguments[field]))
                    else:
                        arguments[field] = float(arguments[field])
                    logger.debug(f"Coerced {field} from string to number")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to coerce {field}='{arguments[field]}' to number: {e}")

        return arguments

    def parse_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse tool calls from Anthropic response.

        Anthropic returns tool use in content blocks:
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "...",
                    "name": "...",
                    "input": {...}  # Already parsed JSON
                }
            ]
        }

        Args:
            response: Raw API response

        Returns:
            List of tool calls in canonical format
        """
        tool_calls = []

        content_blocks = response.get("content", [])
        if not content_blocks:
            logger.warning("No content in Anthropic response")
            return tool_calls

        for block in content_blocks:
            if block.get("type") == "tool_use":
                name = block.get("name", "")
                # Anthropic returns input as already-parsed dict
                arguments = block.get("input", {})

                # Coerce string arguments to expected numeric types
                arguments = self._coerce_argument_types(name, arguments)

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
        return "anthropic"

    def close(self):
        """Close the HTTP client."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
