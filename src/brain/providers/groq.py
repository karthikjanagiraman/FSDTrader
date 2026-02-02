#!/usr/bin/env python3
"""
FSDTrader Brain Module: Groq LLM Provider

Implementation of the LLMProvider interface for Groq's ultra-fast inference.
Uses OpenAI-compatible API endpoint with Llama models.
"""
import time
import json
import logging
import re
from typing import List, Dict, Any, Optional

import httpx

from .base import LLMProvider
from ..types import LLMResponse


logger = logging.getLogger(__name__)

# Rate limit retry configuration
MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0  # seconds


# Groq API configuration
GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"  # Best model for complex reasoning


class GroqProvider(LLMProvider):
    """
    Groq LLM provider for ultra-fast inference.

    Uses OpenAI-compatible chat completions endpoint with tool calling.
    Groq provides 10-30x faster inference than most providers.
    """

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        timeout: float = 30.0,
        **kwargs
    ):
        """
        Initialize the Groq provider.

        Args:
            api_key: Groq API key
            model: Model to use (default: llama-3.3-70b-versatile)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration (ignored)
        """
        self._api_key = api_key
        self._model = model or GROQ_DEFAULT_MODEL
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
                base_url=GROQ_API_BASE,
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
        Make a synchronous call to Groq.

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

        client = self._get_sync_client()

        for attempt in range(MAX_RETRIES + 1):
            try:
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
                if e.response.status_code == 429 and attempt < MAX_RETRIES:
                    # Rate limited - extract retry delay from error message
                    retry_delay = self._extract_retry_delay(e.response.text)
                    logger.warning(f"Groq rate limited, retrying in {retry_delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(retry_delay)
                    continue
                elif e.response.status_code == 400:
                    # Check for malformed tool call (XML-style fallback)
                    parsed = self._try_parse_malformed_tool_call(e.response.text)
                    if parsed:
                        latency_ms = (time.time() - start_time) * 1000
                        return LLMResponse(
                            tool_calls=parsed,
                            content=None,
                            model=self._model,
                            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            latency_ms=latency_ms,
                        )
                logger.error(f"Groq API error: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Groq call failed: {e}")
                raise

        raise RuntimeError("Max retries exceeded for Groq API")

    def _extract_retry_delay(self, error_text: str) -> float:
        """Extract retry delay from Groq rate limit error message."""
        # Look for patterns like "Please try again in 1.079999999s" or "in 820ms"
        match = re.search(r'try again in (\d+(?:\.\d+)?)(s|ms)', error_text)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit == 'ms':
                return value / 1000.0
            return value
        return DEFAULT_RETRY_DELAY

    def _try_parse_malformed_tool_call(self, error_text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Try to parse malformed XML-style tool calls from Groq error responses.

        When Llama models sometimes fall back to XML-style function calling:
        <function=tool_name>{"arg": "value"}</function>

        This recovers those calls from the failed_generation field.

        Args:
            error_text: Raw error response text from Groq API

        Returns:
            Parsed tool calls if successful, None otherwise
        """
        try:
            error_data = json.loads(error_text)
            failed_gen = error_data.get("error", {}).get("failed_generation", "")

            if not failed_gen:
                return None

            # Pattern: <function=tool_name>{...json...}</function>
            # Use greedy match for JSON content to capture full nested structure
            pattern = r'<function=(\w+)>(\{.+\})</function>'
            match = re.search(pattern, failed_gen, re.DOTALL)

            if not match:
                # Fallback: try to extract tool name and JSON without closing tag
                pattern_no_close = r'<function=(\w+)>(\{.+\})\s*$'
                match = re.search(pattern_no_close, failed_gen, re.DOTALL)

            if match:
                tool_name = match.group(1)
                args_str = match.group(2)

                # Try to parse JSON, with fallback for broken JSON from memo quotes
                arguments = self._parse_tool_args_robust(args_str)
                if arguments is None:
                    # Can't parse - just return a minimal valid tool call
                    # This allows wait/exit to work even with broken JSON
                    if tool_name == "wait":
                        logger.info(f"Recovered wait tool call with default args (JSON parse failed)")
                        return [{"name": "wait", "arguments": {"reasoning": "Malformed tool call recovery", "memo": ""}}]
                    elif tool_name == "exit_position":
                        logger.info(f"Recovered exit_position tool call with default args (JSON parse failed)")
                        return [{"name": "exit_position", "arguments": {"reasoning": "Malformed tool call recovery", "memo": ""}}]
                    else:
                        logger.warning(f"Failed to parse {tool_name} arguments, cannot recover")
                        return None

                # Coerce types
                arguments = self._coerce_argument_types(tool_name, arguments)

                logger.info(f"Recovered malformed tool call: {tool_name}")
                return [{
                    "name": tool_name,
                    "arguments": arguments,
                }]

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.debug(f"Could not parse malformed tool call: {e}")

        return None

    def _parse_tool_args_robust(self, args_str: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse tool arguments JSON with fallbacks for common issues.

        The LLM sometimes produces JSON with unescaped quotes in string values,
        particularly in the memo field. This tries multiple strategies to parse.
        """
        # First, try direct parse
        try:
            return json.loads(args_str)
        except json.JSONDecodeError:
            pass

        # Try to fix common issue: unescaped quotes in memo field
        # The memo often has patterns like: thesis:"Some text"
        # which breaks JSON when the outer quotes aren't escaped
        try:
            # Replace unescaped quotes inside strings (heuristic)
            # Look for patterns like thesis:" and replace the inner quotes
            fixed = re.sub(r'thesis:"([^"]*)"', r'thesis:\\"\1\\"', args_str)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Try extracting individual fields
        try:
            result = {}

            # Extract reasoning
            reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', args_str)
            if reasoning_match:
                result["reasoning"] = reasoning_match.group(1)

            # Extract new_price (for update_stop/update_target)
            price_match = re.search(r'"new_price"\s*:\s*([0-9.]+)', args_str)
            if price_match:
                result["new_price"] = float(price_match.group(1))

            # Extract memo (allow for broken quotes)
            memo_match = re.search(r'"memo"\s*:\s*"(@[^}]+)"?\}', args_str)
            if memo_match:
                result["memo"] = memo_match.group(1)

            if result:
                return result
        except Exception:
            pass

        logger.debug(f"All JSON parse strategies failed for: {args_str[:200]}")
        return None

    async def call_async(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Make an asynchronous call to Groq.

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
                base_url=GROQ_API_BASE,
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
            logger.error(f"Groq API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Groq async call failed: {e}")
            raise

    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tools for Groq's OpenAI-compatible API.

        Groq uses the OpenAI function calling format:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }

        Args:
            tools: Tools in canonical format or already in OpenAI format

        Returns:
            Tools in Groq/OpenAI format
        """
        formatted = []
        for tool in tools:
            # Check if already in OpenAI format (has "function" key with "name" inside)
            if "function" in tool and "name" in tool["function"]:
                # Already formatted, pass through
                formatted.append(tool)
            else:
                # Convert from canonical format
                formatted.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    }
                })
        return formatted

    def _coerce_argument_types(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coerce string arguments to expected numeric types.

        Groq sometimes returns numbers as strings (e.g., "415.80" instead of 415.80).
        This method converts them to the expected types based on the tool schema.

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
        Parse tool calls from Groq response.

        Groq returns tool calls in OpenAI format:
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
            logger.warning("No choices in Groq response")
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
        return "groq"

    def close(self):
        """Close the HTTP client."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
