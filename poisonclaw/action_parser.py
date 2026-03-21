"""Unified action parser for PoisonClaw.

All models (Qwen2.5-VL, SeeClick, ShowUI, UI-TARS, UI-R1) share one action
format during RL training.  The parser uses Python AST for robustness
(inspired by UI-TARS action_parser.py) with regex fallback.

Action format (absolute pixel coordinates, 1280×720 viewport):
    click(x, y)
    type(text)
    press(key)
    scroll(direction)      # direction ∈ {up, down}
    done()

VLM output may optionally be wrapped in ``<action>…</action>`` tags.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parsed action dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParsedAction:
    """Structured representation of a single agent action."""
    action_type: str                  # click | type | press | scroll | done | noop
    params: dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    raw: str = ""

    # Convenience accessors
    @property
    def x(self) -> int:
        return int(self.params.get("x", 0))

    @property
    def y(self) -> int:
        return int(self.params.get("y", 0))

    @property
    def text(self) -> str:
        return self.params.get("text", "")

    @property
    def key(self) -> str:
        return self.params.get("key", "")

    @property
    def direction(self) -> str:
        return self.params.get("direction", "down")

    def to_dict(self) -> dict[str, Any]:
        """Legacy dict format for backward compatibility."""
        d: dict[str, Any] = {"type": self.action_type}
        d.update(self.params)
        return d


# ---------------------------------------------------------------------------
# Key normalization
# ---------------------------------------------------------------------------

_KEY_MAP = {
    "enter": "Enter", "return": "Enter",
    "tab": "Tab",
    "escape": "Escape", "esc": "Escape",
    "backspace": "Backspace",
    "delete": "Delete",
    "space": " ",
    "arrowup": "ArrowUp", "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
}


def normalize_key(key: str) -> str:
    """Normalize VLM key names to Playwright-compatible names."""
    key = key.strip().strip("\"'")
    return _KEY_MAP.get(key.lower(), key)


# ---------------------------------------------------------------------------
# Regex — used as first-pass & fallback
# ---------------------------------------------------------------------------

_RE_ACTION_TAG = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_RE_CLICK      = re.compile(r"click\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)", re.IGNORECASE)
_RE_TYPE       = re.compile(r"type\s*\(\s*(.+?)\s*\)", re.DOTALL)
_RE_PRESS      = re.compile(r"press\s*\(\s*(.+?)\s*\)", re.IGNORECASE)
_RE_SCROLL     = re.compile(r"scroll\s*\(\s*(.+?)\s*\)", re.IGNORECASE)
_RE_DONE       = re.compile(r"done\s*\(", re.IGNORECASE)


# ---------------------------------------------------------------------------
# AST-based parser (inspired by UI-TARS)
# ---------------------------------------------------------------------------

def _parse_via_ast(action_str: str) -> Optional[ParsedAction]:
    """Try to parse an action string as a Python function call via AST.

    This is more robust than regex for edge cases like nested parentheses,
    escaped quotes inside type() content, etc.

    Returns None if AST parsing fails (caller should fall back to regex).
    """
    try:
        node = ast.parse(action_str.strip(), mode="eval")
        if not isinstance(node, ast.Expression):
            return None
        call = node.body
        if not isinstance(call, ast.Call):
            return None

        # Function name
        if isinstance(call.func, ast.Name):
            func_name = call.func.id.lower()
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr.lower()
        else:
            return None

        # Extract positional and keyword arguments
        pos_args = []
        for arg in call.args:
            if isinstance(arg, ast.Constant):
                pos_args.append(arg.value)
            elif isinstance(arg, ast.Name):
                # Bare identifiers like scroll(up), scroll(down), press(Enter)
                # — not quoted strings, but we treat them as string values.
                pos_args.append(arg.id)
            elif isinstance(arg, (ast.UnaryOp,)):
                # Handle negative numbers
                try:
                    pos_args.append(ast.literal_eval(arg))
                except Exception:
                    pos_args.append(None)
            else:
                pos_args.append(None)

        kwargs = {}
        for kw in call.keywords:
            if isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value
            elif isinstance(kw.value, ast.Name):
                # Bare identifier as keyword value: press(key=Enter)
                kwargs[kw.arg] = kw.value.id
            elif isinstance(kw.value, ast.Str):  # Python 3.7 compat
                kwargs[kw.arg] = kw.value.s
            elif isinstance(kw.value, ast.Num):
                kwargs[kw.arg] = kw.value.n

        # Map to ParsedAction
        if func_name == "click":
            if len(pos_args) >= 2 and pos_args[0] is not None and pos_args[1] is not None:
                return ParsedAction("click", {"x": int(float(pos_args[0])), "y": int(float(pos_args[1]))}, raw=action_str)
            # keyword form: click(x=100, y=200)
            if "x" in kwargs and "y" in kwargs:
                return ParsedAction("click", {"x": int(float(kwargs["x"])), "y": int(float(kwargs["y"]))}, raw=action_str)

        elif func_name == "type":
            text = ""
            if pos_args and pos_args[0] is not None:
                text = str(pos_args[0])
            elif "text" in kwargs:
                text = str(kwargs["text"])
            elif "content" in kwargs:  # UI-TARS format: type(content='xxx')
                text = str(kwargs["content"])
            if text:
                return ParsedAction("type", {"text": text}, raw=action_str)

        elif func_name == "press":
            key = ""
            if pos_args and pos_args[0] is not None:
                key = str(pos_args[0])
            elif "key" in kwargs:
                key = str(kwargs["key"])
            if key:
                return ParsedAction("press", {"key": normalize_key(key)}, raw=action_str)

        elif func_name == "scroll":
            direction = "down"
            if pos_args and pos_args[0] is not None:
                direction = str(pos_args[0]).lower()
            elif "direction" in kwargs:
                direction = str(kwargs["direction"]).lower()
            if direction not in ("up", "down"):
                logger.warning("Invalid scroll direction %r, defaulting to 'down'", direction)
                direction = "down"
            return ParsedAction("scroll", {"direction": direction}, raw=action_str)

        elif func_name in ("done", "finished"):
            return ParsedAction("done", {}, raw=action_str)

        return None

    except (SyntaxError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Regex fallback parser
# ---------------------------------------------------------------------------

def _parse_via_regex(action_str: str) -> ParsedAction:
    """Regex-based parser as fallback when AST fails."""

    # click(x, y)
    m = _RE_CLICK.search(action_str)
    if m:
        return ParsedAction("click", {
            "x": int(float(m.group(1))),
            "y": int(float(m.group(2))),
        }, raw=action_str)

    # type(text)
    m = _RE_TYPE.search(action_str)
    if m:
        text = m.group(1).strip().strip("\"'")
        return ParsedAction("type", {"text": text}, valid=bool(text), raw=action_str)

    # press(key)
    m = _RE_PRESS.search(action_str)
    if m:
        key = normalize_key(m.group(1))
        return ParsedAction("press", {"key": key}, valid=bool(key), raw=action_str)

    # scroll(direction)
    m = _RE_SCROLL.search(action_str)
    if m:
        direction = m.group(1).strip().strip("\"'").lower()
        if direction not in ("up", "down"):
            logger.warning("Invalid scroll direction %r, defaulting to 'down'", direction)
            direction = "down"
        return ParsedAction("scroll", {"direction": direction}, raw=action_str)

    # done()
    if _RE_DONE.search(action_str):
        return ParsedAction("done", {}, raw=action_str)

    # noop
    return ParsedAction("noop", {"raw": action_str}, valid=False, raw=action_str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_action(raw: str) -> ParsedAction:
    """Parse a single VLM action string into a ParsedAction.

    Tries AST first for robustness, falls back to regex.

    Args:
        raw: Raw VLM output string (may contain ``<action>`` tags,
             Thought/Reflection prefixes, etc.).

    Returns:
        ParsedAction with action_type, params, and validity flag.
    """
    raw = raw.strip()

    # Unwrap <action>…</action> tags
    tag_match = _RE_ACTION_TAG.search(raw)
    action_str = tag_match.group(1).strip() if tag_match else raw

    # Strip common VLM prefixes (Thought:, Reflection:, Action:)
    # to isolate the actual action call
    if "Action:" in action_str:
        action_str = action_str.split("Action:")[-1].strip()

    # Try AST first
    result = _parse_via_ast(action_str)
    if result is not None:
        return result

    # Fallback to regex
    return _parse_via_regex(action_str)


def parse_actions(text_actions: list[str]) -> tuple[list[ParsedAction], list[bool]]:
    """Parse a batch of VLM action strings.

    Args:
        text_actions: List of raw action strings.

    Returns:
        Tuple of (parsed_actions, valid_flags).
    """
    results = [parse_action(raw) for raw in text_actions]
    return results, [r.valid for r in results]


def parse_actions_to_dicts(text_actions: list[str]) -> tuple[list[dict], list[bool]]:
    """Parse a batch, returning legacy dict format for backward compatibility.

    Returns:
        Tuple of (parsed_dicts, valid_flags) — drop-in replacement for the
        old regex-based ``_parse_action`` static methods.
    """
    results, valids = parse_actions(text_actions)
    return [r.to_dict() for r in results], valids


# ---------------------------------------------------------------------------
# BrowserGym conversion
# ---------------------------------------------------------------------------

def to_browsergym_action(parsed: ParsedAction) -> tuple[str, bool]:
    """Convert a ParsedAction to a BrowserGym action string.

    Returns:
        Tuple of (browsergym_action_string, is_valid).
    """
    if not parsed.valid:
        return "noop()", False

    t = parsed.action_type

    if t == "click":
        return f"mouse_click({parsed.x}, {parsed.y})", True

    if t == "type":
        escaped = parsed.text.replace('"', '\\"')
        return f'keyboard_type("{escaped}")', True

    if t == "press":
        return f'keyboard_press("{parsed.key}")', True

    if t == "scroll":
        delta = -300 if parsed.direction == "up" else 300
        return f"scroll(0, {delta})", True

    if t == "done":
        return "noop()", True  # BrowserGym has no done(); treat as terminal noop

    return "noop()", False
