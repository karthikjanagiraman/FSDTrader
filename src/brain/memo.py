#!/usr/bin/env python3
"""
FSDTrader Brain Module: Memo System

Manages rolling chain of LLM self-notes for persistent context across trading cycles.
The LLM writes memos to itself that carry forward thesis, triggers, and market reads.
"""

import re
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Memo:
    """Single memo entry from LLM."""
    text: str
    timestamp: str
    position: str  # FLAT, LONG, SHORT
    memo_type: str = "DELTA"  # INIT or DELTA


class MemoManager:
    """
    Manages rolling chain of LLM self-notes.

    The LLM writes memos after each decision containing:
    - What changed vs stayed the same
    - Current thesis (market hypothesis)
    - What triggers it's watching for
    - What would invalidate the thesis
    - Cumulative counters (wait_streak, etc.)

    Usage:
        manager = MemoManager(max_deltas=5)
        manager.start_session("09:30:00")

        # After each LLM response:
        memo_text = parse_memo_from_response(response)
        manager.add_memo(memo_text, "09:30:15", "FLAT")

        # When building next prompt:
        memo_chain = manager.get_chain()
    """

    def __init__(self, max_deltas: int = 5):
        """
        Initialize memo manager.

        Args:
            max_deltas: Maximum number of delta memos to keep (rolling window)
        """
        self.max_deltas = max_deltas
        self._init_memo: Optional[Memo] = None
        self._deltas: deque = deque(maxlen=max_deltas)
        self._session_start: Optional[str] = None
        self._total_memos: int = 0

    def start_session(self, timestamp: str):
        """
        Start a new trading session. Clears all previous memos.

        Args:
            timestamp: Session start time (e.g., "09:30:00")
        """
        self._init_memo = None
        self._deltas.clear()
        self._session_start = timestamp
        self._total_memos = 0
        logger.info(f"Memo session started at {timestamp}")

    def add_memo(self, memo_text: str, timestamp: str, position: str):
        """
        Add a new memo from LLM response.

        First memo becomes @INIT, subsequent become @DELTA.

        Args:
            memo_text: Raw memo text from LLM (already parsed from [MEMO] block)
            timestamp: Current time
            position: Current position status (FLAT, LONG, SHORT)
        """
        if not memo_text or not memo_text.strip():
            logger.warning("Empty memo text, skipping")
            return

        self._total_memos += 1

        # First memo is always INIT
        if self._init_memo is None:
            memo = Memo(
                text=memo_text.strip(),
                timestamp=timestamp,
                position=position,
                memo_type="INIT"
            )
            self._init_memo = memo
            logger.debug(f"Stored @INIT memo at {timestamp}")
        else:
            memo = Memo(
                text=memo_text.strip(),
                timestamp=timestamp,
                position=position,
                memo_type="DELTA"
            )
            self._deltas.append(memo)
            logger.debug(f"Stored @DELTA memo at {timestamp} ({len(self._deltas)}/{self.max_deltas})")

    def get_chain(self) -> str:
        """
        Get formatted memo chain for prompt injection.

        Returns:
            Formatted string with @INIT + recent @DELTAs, or empty guidance if no memos.
        """
        if self._init_memo is None:
            return self._get_empty_guidance()

        parts = []

        # Always include @INIT
        parts.append(self._format_memo(self._init_memo))

        # Add separator if we have deltas
        if self._deltas:
            parts.append("---")

        # Add all deltas
        for memo in self._deltas:
            parts.append(self._format_memo(memo))

        return "\n".join(parts)

    def _format_memo(self, memo: Memo) -> str:
        """Format a single memo for display."""
        header = f"@{memo.memo_type}|{memo.timestamp}|{memo.position}|"
        return f"{header}\n{memo.text}"

    def _get_empty_guidance(self) -> str:
        """Return guidance text when no memos exist yet."""
        return """(No previous notes - this is your first snapshot)
Write your @INIT memo after this decision to establish your initial thesis."""

    def get_chain_for_logging(self) -> str:
        """Get compact version of chain for logging."""
        if self._init_memo is None:
            return "(empty)"

        parts = [f"INIT@{self._init_memo.timestamp}"]
        for memo in self._deltas:
            parts.append(f"D@{memo.timestamp}")

        return " -> ".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memo statistics for debugging/logging.

        Returns:
            Dictionary with memo counts and state info
        """
        return {
            "session_start": self._session_start,
            "has_init": self._init_memo is not None,
            "delta_count": len(self._deltas),
            "max_deltas": self.max_deltas,
            "total_memos": self._total_memos,
        }

    def get_latest_thesis(self) -> Optional[str]:
        """
        Extract the latest thesis from most recent memo.

        Returns:
            Thesis string or None if not found
        """
        latest = self._get_latest_memo()
        if latest is None:
            return None

        # Try to extract thesis from memo text
        match = re.search(r'thesis:\s*["\']?([^"\'\n]+)["\']?', latest.text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None

    def get_wait_streak(self) -> int:
        """
        Extract wait streak from most recent memo.

        Returns:
            Wait streak count or 0 if not found
        """
        latest = self._get_latest_memo()
        if latest is None:
            return 0

        # Try to extract wait count from cumulative section
        match = re.search(r'waits?[:\s]*(\d+)', latest.text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return 0

    def _get_latest_memo(self) -> Optional[Memo]:
        """Get the most recent memo (latest delta or init)."""
        if self._deltas:
            return self._deltas[-1]
        return self._init_memo


def parse_memo_from_response(response_text: str) -> Optional[str]:
    """
    Extract memo block from LLM response.

    The LLM should emit memos in format:
    ```
    [MEMO]
    @DELTA|time|position|
    ...memo content...
    ```

    Args:
        response_text: Full LLM response text

    Returns:
        Memo content (without [MEMO] marker) or None if not found
    """
    if not response_text:
        return None

    # Look for [MEMO] block - can be at end or anywhere in response
    # Pattern: [MEMO] followed by content until end or next major section
    patterns = [
        # Standard format: [MEMO] at start of line
        r'\[MEMO\]\s*\n(.*?)(?:\n\[/MEMO\]|\n##|\Z)',
        # Alternative: [MEMO] inline
        r'\[MEMO\](.*?)(?:\[/MEMO\]|\Z)',
        # Markdown code block format
        r'```memo\s*\n(.*?)```',
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            memo_text = match.group(1).strip()
            if memo_text:
                logger.debug(f"Parsed memo ({len(memo_text)} chars)")
                return memo_text

    # Fallback: Look for @DELTA or @INIT pattern directly
    delta_match = re.search(r'(@(?:DELTA|INIT)\|[^\n]*\n.*?)(?:\n##|\n\[|\Z)',
                            response_text, re.DOTALL)
    if delta_match:
        memo_text = delta_match.group(1).strip()
        if memo_text:
            logger.debug(f"Parsed memo via @DELTA pattern ({len(memo_text)} chars)")
            return memo_text

    logger.debug("No memo found in response")
    return None


def parse_position_from_memo(memo_text: str) -> str:
    """
    Extract position status from memo header.

    Memo format: @DELTA|timestamp|POSITION|

    Args:
        memo_text: Memo content

    Returns:
        Position string (FLAT, LONG, SHORT) or "UNKNOWN"
    """
    if not memo_text:
        return "UNKNOWN"

    # Look for position in header: @DELTA|time|POSITION|
    match = re.search(r'@(?:DELTA|INIT)\|[^|]*\|(\w+)\|', memo_text)
    if match:
        position = match.group(1).upper()
        if position in ("FLAT", "LONG", "SHORT"):
            return position

    # Fallback: Look for position keywords in text
    text_upper = memo_text.upper()
    if "LONG" in text_upper and "FLAT" not in text_upper:
        return "LONG"
    elif "SHORT" in text_upper and "FLAT" not in text_upper:
        return "SHORT"
    elif "FLAT" in text_upper:
        return "FLAT"

    return "UNKNOWN"


def wrap_memo_chain(memo_chain: str) -> str:
    """
    Wrap memo chain with header for injection into user message.

    Args:
        memo_chain: Output from MemoManager.get_chain()

    Returns:
        Formatted section ready for prompt injection
    """
    return f"""## SESSION MEMORY (Your previous self-notes)

{memo_chain}

---
"""
