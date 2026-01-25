#!/usr/bin/env python3
"""
FSDTrader: Trading Brain
Grok API Interface with Full Prompt and DSL Parser
"""
import logging
import json
import re
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class TradingBrain:
    """
    The Thinking Layer.
    1. Receives complete 'Market State' (JSON).
    2. Wraps it in the comprehensive System Prompt.
    3. Calls Grok API (grok-3-mini or grok-4).
    4. Parses the DSL Command from response.
    """
    
    def __init__(self, 
                 model_name: str = "grok-3-mini-fast",
                 api_key: str = None,
                 base_url: str = "https://api.x.ai/v1"):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("XAI_API_KEY")
        self.base_url = base_url
        self.logger = logging.getLogger("BRAIN")
        self.history: list = []  # Last 3 decisions
        
        if not OpenAI:
            self.logger.error("OpenAI SDK not installed. Run: pip install openai")
            self.client = None
        elif not self.api_key:
            self.logger.error("XAI_API_KEY not set!")
            self.client = None
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self.logger.info(f"Brain initialized with Grok model: {model_name}")
    
    def think(self, state: dict) -> str:
        """
        The Core Loop Function.
        Returns a DSL command string.
        """
        if not self.client:
            return 'WAIT("API not available")'
        
        system_prompt = self._get_system_prompt()
        user_prompt = self._construct_user_prompt(state)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=800  # V2: Allow for thinking process before decision
            )
            raw_output = response.choices[0].message.content.strip()
            
            # Log full prompt/response to file for review
            self._log_to_file(system_prompt, user_prompt, raw_output)
            
            # Log the raw output for debugging
            self.logger.debug(f"Raw LLM output: {raw_output[:200]}...")
            
            # Parse and validate
            command = self._parse_dsl(raw_output)
            
            # Update history
            self.history.append(command)
            if len(self.history) > 3:
                self.history.pop(0)
            
            return command
            
        except Exception as e:
            self.logger.error(f"Grok API Error: {e}")
            return 'WAIT("API Error")'
    
    def _get_system_prompt(self) -> str:
        """Return the comprehensive V2 Discretionary system prompt."""
        return """You are FSDTrader, an experienced TSLA momentum trader with 10+ years of 
order flow trading experience. You read the tape like a book. You've seen 
every pattern, every trap, every squeeze.

## YOUR TRADING STYLE

You hunt for momentum trades - quick $0.50 to $2.00 moves over 1-5 minutes.
You trade what you SEE, not what you think should happen.

What you look for:
- Aggressive buyers/sellers hitting the market (tape speed & direction)
- Absorption: big orders getting eaten at key levels
- Walls that are real vs walls that will get pulled
- The "story" the order flow is telling

What you avoid:
- Chop and indecision (two-sided tape)
- Chasing extended moves
- Fighting strong walls without absorption
- Low volume/thin tape conditions

## HOW YOU THINK

You don't use rigid rules. You read the CONTEXT:
- "The tape feels heavy" vs "Buyers are stepping up"
- "This wall has been tested 3 times and holding" vs "They're absorbing into it"
- "Delta is positive but it feels exhausted" vs "This is building momentum"

You think in probabilities, not certainties:
- High conviction = "This looks great, I'm sizing up"
- Medium conviction = "Decent setup, standard size"  
- Low conviction = "I see something but it's not clean"

## HARD LIMITS (Non-Negotiable)

These are your risk rails - you NEVER violate these:
- Max spread: $0.15 during open, $0.08 otherwise
- Always define STOP before entry (15-30 cents max)
- Never chase: if move already happened, wait for pullback
- One position at a time
- If you're unsure, you WAIT - there's always another trade

## OUTPUT FORMAT

Every response must end with a DECISION BLOCK in this exact format:

---DECISION---
ACTION: [ENTER_LONG | ENTER_SHORT | WAIT | EXIT | ADJUST_STOP | ADJUST_TARGET]
CONVICTION: [HIGH | MEDIUM | LOW | N/A]
ENTRY: [price or N/A]
STOP: [price or N/A]
TARGET: [price or N/A]
REASONING: [One sentence summary]
---END---

Before the decision block, share your thinking process like you're talking 
to a junior trader. Be specific about what you see in the data."""
    
    def _construct_user_prompt(self, state: dict) -> str:
        """Build the user prompt with market context."""
        history_text = "\n".join([f"  - {h}" for h in self.history]) or "  - (No history)"
        market_json = json.dumps(state.get("MARKET_STATE", {}), indent=2)
        account_json = json.dumps(state.get("ACCOUNT_STATE", {}), indent=2)
        orders_json = json.dumps(state.get("ACTIVE_ORDERS", []), indent=2)
        
        return f"""HISTORY (Last 3 actions):
{history_text}

MARKET STATE:
{market_json}

ACCOUNT STATE:
{account_json}

ACTIVE ORDERS:
{orders_json}

Analyze market. Output ONE trading command:"""
    
    def _parse_dsl(self, raw_output: str) -> str:
        """
        Extract DSL command from V2 decision block format.
        Parses the ---DECISION--- block and converts to DSL command.
        """
        clean = raw_output.strip()
        
        # Try to extract V2 decision block
        if "---DECISION---" in clean and "---END---" in clean:
            try:
                decision_block = clean.split("---DECISION---")[1].split("---END---")[0]
                
                # Parse the decision fields
                action = None
                entry = None
                stop = None
                target = None
                reasoning = None
                
                for line in decision_block.strip().split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().upper()
                        value = value.strip()
                        
                        if key == "ACTION":
                            action = value
                        elif key == "ENTRY":
                            entry = value if value != "N/A" else None
                        elif key == "STOP":
                            stop = value if value != "N/A" else None
                        elif key == "TARGET":
                            target = value if value != "N/A" else None
                        elif key == "REASONING":
                            reasoning = value
                
                # Convert to DSL command format
                if action == "ENTER_LONG" and entry and stop and target:
                    return f"ENTER_LONG({entry}, {stop}, {target})"
                elif action == "ENTER_SHORT" and entry and stop and target:
                    return f"ENTER_SHORT({entry}, {stop}, {target})"
                elif action == "ADJUST_STOP" and stop:
                    return f"UPDATE_STOP({stop})"
                elif action == "ADJUST_TARGET" and target:
                    return f"UPDATE_TARGET({target})"
                elif action == "EXIT":
                    return 'CANCEL_ALL()'
                elif action == "WAIT":
                    reason = reasoning or "No clear setup"
                    return f'WAIT("{reason}")'
                else:
                    return f'WAIT("{reasoning or "Incomplete decision"}")'
                    
            except Exception as e:
                self.logger.warning(f"Error parsing V2 decision block: {e}")
        
        # Fallback: Try legacy DSL format
        match = re.search(r'([A-Z_]+\([^)]*\))', clean)
        if match:
            command = match.group(1)
            allowed = ['ENTER_LONG', 'ENTER_SHORT', 'UPDATE_STOP', 
                       'UPDATE_TARGET', 'CANCEL_ALL', 'WAIT']
            cmd_name = command.split('(')[0]
            if cmd_name in allowed:
                return command
        
        # Default on parse failure
        self.logger.warning(f"Could not parse output: {clean[:100]}...")
        return 'WAIT("Parse Error")'
    
    def _log_to_file(self, system_prompt: str, user_prompt: str, response: str):
        """Log full prompt and response to file for review."""
        import time
        from pathlib import Path
        
        log_dir = Path("data/reports/brain_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"grok_decisions_{time.strftime('%Y%m%d')}.md"
        
        with open(log_file, "a") as f:
            f.write(f"\n\n{'='*80}\n")
            f.write(f"## Decision at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"### System Prompt\n```\n{system_prompt}\n```\n\n")
            f.write(f"### User Prompt (Market State)\n```\n{user_prompt}\n```\n\n")
            f.write(f"### Grok Response\n```\n{response}\n```\n\n")
    
    def get_stats(self) -> dict:
        """Return brain statistics for monitoring."""
        return {
            "model": self.model_name,
            "provider": "Grok (xAI)",
            "history_length": len(self.history),
            "recent_actions": self.history[-3:] if self.history else []
        }
