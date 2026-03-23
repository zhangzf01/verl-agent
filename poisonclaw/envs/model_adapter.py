"""Model adapter layer for PoisonClaw web agents.

Encapsulates model-specific conventions so the rest of the pipeline
(env, memory, reward) stays model-agnostic:
  - System prompt format
  - Action history formatting (what the model sees as its own past actions)
  - Coordinate system metadata

Supported model families:
  - Generic (Qwen2.5-VL, UI-R1): ``<action>click(x,y)</action>``
  - UI-TARS: ``Thought: … Action: click(start_box='(x,y)')``
  - SeeClick: ``<action>click(x, y)</action>`` (absolute pixels)
  - ShowUI: ``<action>click(x, y)</action>`` (absolute pixels)

The action parser (``poisonclaw/action_parser.py``) handles all format
conversion transparently — the adapter only controls *prompts* and
*history display*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from poisonclaw.action_parser import ParsedAction

logger = logging.getLogger(__name__)


# ── Coordinate system constants ──────────────────────────────────────────────

VIEWPORT_W = 1280
VIEWPORT_H = 720


# ── System prompts ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT_GENERIC = """\
You are a GUI agent that interacts with web pages by looking at screenshots and performing actions.

You can perform the following actions:
  click(x, y)        — click at pixel coordinates (x: 0-1280, y: 0-720), e.g. click(120, 55)
  type(text)          — type a string, e.g. type(hello world)
  press(key)          — press a key, e.g. press(Enter)
  scroll(direction)   — scroll the page, e.g. scroll(down) or scroll(up)
  done()              — call this immediately after completing all task steps

You will receive:
1. A task instruction describing what you need to accomplish.
2. A screenshot of the current web page state.
3. Your previous actions (if any).

Respond with exactly ONE action per step. Wrap your action in <action>...</action> tags.

Examples:
- <action>click(80, 120)</action>
- <action>type(John Doe)</action>
- <action>press(Enter)</action>
- <action>scroll(down)</action>
- <action>done()</action>

Important:
- Coordinates are absolute pixels within the 1280×720 viewport.
- Output only the action tag — do not explain your reasoning.
- When all task steps are done, you MUST call done() immediately.
"""

_SYSTEM_PROMPT_UITARS = """\
You are a GUI agent. You are given a task and a screenshot of the current web page. You need to perform the next action to complete the task.

## Action Space

click(start_box='(x1,y1)')                — click at coordinates, e.g. click(start_box='(200,300)')
type(content='text')                       — type text, e.g. type(content='hello world')
hotkey(key='key1 key2')                    — press keys (lowercase, space-separated), e.g. hotkey(key='enter')
scroll(start_box='(x1,y1)', direction='down') — scroll at position, direction: up/down/left/right
finished()                                 — call when the task is fully completed

## Coordinate System

Coordinates are normalized to a 1000×1000 grid. (0,0) is top-left, (1000,1000) is bottom-right.

## Output Format

Respond with a brief thought followed by exactly one action:
Thought: [your reasoning]
Action: [one action from the action space above]

## Important

- Call finished() immediately when the task is done.
"""


# ── Adapter base & implementations ──────────────────────────────────────────

@dataclass
class ModelAdapter:
    """Encapsulates model-specific prompt and action format conventions.

    Attributes:
        name: Human-readable model family name.
        system_prompt: System prompt text for this model.
        coord_normalized: True if model uses [0,1000] coords (UI-TARS),
                          False for absolute pixels.
    """
    name: str
    system_prompt: str
    coord_normalized: bool = False

    def format_action_for_history(self, raw_output: str, parsed: ParsedAction) -> str:
        """Format a parsed action for inclusion in the model's action history.

        The history should use the model's *native* format so the model sees
        output consistent with what it generates.

        Args:
            raw_output: The raw model output string (before parsing).
            parsed: The ParsedAction result from action_parser.

        Returns:
            Formatted action string for memory/history.
        """
        if parsed.action_type == "noop":
            return raw_output.strip()[:80]
        return f"<action>{self._format_action_body(parsed)}</action>"

    def _format_action_body(self, parsed: ParsedAction) -> str:
        """Format the action body (without wrapper) in this model's native style."""
        t = parsed.action_type
        if t == "click":
            return f"click({parsed.x}, {parsed.y})"
        if t == "type":
            return f"type({parsed.text})"
        if t == "press":
            return f"press({parsed.key})"
        if t == "scroll":
            return f"scroll({parsed.direction})"
        if t == "done":
            return "done()"
        return parsed.raw


@dataclass
class GenericAdapter(ModelAdapter):
    """Adapter for Qwen2.5-VL, UI-R1, and other models using canonical format.

    Action format: ``<action>click(x, y)</action>``
    Coordinates: absolute pixels (0-1280 × 0-720).
    """
    name: str = "generic"
    system_prompt: str = _SYSTEM_PROMPT_GENERIC
    coord_normalized: bool = False


@dataclass
class UITARSAdapter(ModelAdapter):
    """Adapter for UI-TARS models using native bbox format.

    Action format: ``Thought: … Action: click(start_box='(x,y)')``
    Coordinates: normalized [0, 1000].
    """
    name: str = "ui-tars"
    system_prompt: str = _SYSTEM_PROMPT_UITARS
    coord_normalized: bool = True

    def format_action_for_history(self, raw_output: str, parsed: ParsedAction) -> str:
        if parsed.action_type == "noop":
            return raw_output.strip()[:80]
        return f"Action: {self._format_action_body(parsed)}"

    def _format_action_body(self, parsed: ParsedAction) -> str:
        t = parsed.action_type
        if t == "click":
            # Convert back to normalized [0, 1000] coords
            nx = round(parsed.x / VIEWPORT_W * 1000)
            ny = round(parsed.y / VIEWPORT_H * 1000)
            return f"click(start_box='({nx},{ny})')"
        if t == "type":
            escaped = parsed.text.replace("'", "\\'")
            return f"type(content='{escaped}')"
        if t == "press":
            key = parsed.key.lower().replace("+", " ").replace("control", "ctrl")
            return f"hotkey(key='{key}')"
        if t == "scroll":
            return f"scroll(start_box='(500,500)', direction='{parsed.direction}')"
        if t == "done":
            return "finished()"
        return parsed.raw


@dataclass
class SeeClickAdapter(ModelAdapter):
    """Adapter for SeeClick models.

    Uses same canonical format as generic but with grounding emphasis.
    Coordinates: absolute pixels.
    """
    name: str = "seeclick"
    system_prompt: str = _SYSTEM_PROMPT_GENERIC
    coord_normalized: bool = False


@dataclass
class ShowUIAdapter(ModelAdapter):
    """Adapter for ShowUI models.

    Uses same canonical format as generic.
    Coordinates: absolute pixels.
    """
    name: str = "showui"
    system_prompt: str = _SYSTEM_PROMPT_GENERIC
    coord_normalized: bool = False


# ── Factory ──────────────────────────────────────────────────────────────────

# Model path patterns → adapter class
_ADAPTER_REGISTRY: list[tuple[list[str], type[ModelAdapter]]] = [
    (["ui-tars", "uitars"], UITARSAdapter),
    (["seeclick"], SeeClickAdapter),
    (["showui"], ShowUIAdapter),
    # Generic is the fallback — no need to register patterns
]


def get_model_adapter(model_path: str = "") -> ModelAdapter:
    """Auto-detect model family from path and return the right adapter.

    Args:
        model_path: HuggingFace model ID or local path.

    Returns:
        ModelAdapter instance with the appropriate conventions.
    """
    lower = model_path.lower()
    for patterns, adapter_cls in _ADAPTER_REGISTRY:
        if any(p in lower for p in patterns):
            adapter = adapter_cls()
            logger.info("Model adapter: %s (matched from %r)", adapter.name, model_path)
            return adapter

    adapter = GenericAdapter()
    logger.info("Model adapter: %s (default for %r)", adapter.name, model_path)
    return adapter
