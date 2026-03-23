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
    # Modifier keys (UI-TARS uses lowercase)
    "ctrl": "Control", "control": "Control",
    "alt": "Alt", "option": "Alt",
    "shift": "Shift",
    "meta": "Meta", "cmd": "Meta", "command": "Meta", "win": "Meta",
}


def normalize_key(key: str) -> str:
    """Normalize VLM key names to Playwright-compatible names."""
    key = key.strip().strip("\"'")
    return _KEY_MAP.get(key.lower(), key)


# ---------------------------------------------------------------------------
# Regex — used as first-pass & fallback
# ---------------------------------------------------------------------------

_RE_ACTION_TAG = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_RE_ACTION_TAG_UNCLOSED = re.compile(r"<action>\s*(.*)", re.DOTALL | re.IGNORECASE)
_RE_CLICK      = re.compile(r"click\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)", re.IGNORECASE)
_RE_TYPE       = re.compile(r"type\s*\(\s*(.+?)\s*\)", re.DOTALL)
_RE_PRESS      = re.compile(r"press\s*\(\s*(.+?)\s*\)", re.IGNORECASE)
_RE_SCROLL     = re.compile(r"scroll\s*\(\s*(.+?)\s*\)", re.IGNORECASE)
_RE_DONE       = re.compile(r"(?:done|finished)\s*\(", re.IGNORECASE)
_RE_HOTKEY     = re.compile(r"hotkey\s*\(\s*(.+?)\s*\)", re.IGNORECASE)

# UI-TARS native bbox formats:
#   With special tokens: <|box_start|>(x,y)<|box_end|>
#   Without tokens:      start_box='(x,y)'
# Coordinates are normalized [0, 1000].
_RE_UITARS_BOX = re.compile(r"<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>")
_RE_UITARS_BOX_PLAIN = re.compile(r"\((\d+),\s*(\d+)\)")
_RE_UITARS_START_BOX = re.compile(r"start_box\s*=", re.IGNORECASE)

# Default viewport for coordinate conversion
_VIEWPORT_W = 1280
_VIEWPORT_H = 720


def _convert_uitars_format(action_str: str) -> str:
    """Convert UI-TARS native action format to our canonical format.

    Handles two variants:
      click(start_box='<|box_start|>(x,y)<|box_end|>')
      click(start_box='(x,y)')

    Converts normalized [0,1000] coords to absolute pixels.
    Only triggers for actions containing 'start_box' or '<|box_start|>'.
    """
    has_special = "<|box_start|>" in action_str
    has_start_box = _RE_UITARS_START_BOX.search(action_str) is not None

    if not has_special and not has_start_box:
        return action_str

    # Extract action type
    func_match = re.match(r"(\w+)\s*\(", action_str)
    if not func_match:
        return action_str
    func_name = func_match.group(1).lower()

    # Extract coordinate pairs
    if has_special:
        boxes = _RE_UITARS_BOX.findall(action_str)
    else:
        boxes = _RE_UITARS_BOX_PLAIN.findall(action_str)
    if not boxes:
        return action_str

    if func_name == "click":
        if len(boxes) >= 2:
            # start_box + end_box → center
            x1, y1 = int(boxes[0][0]), int(boxes[0][1])
            x2, y2 = int(boxes[1][0]), int(boxes[1][1])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
        else:
            cx, cy = int(boxes[0][0]), int(boxes[0][1])
        # Convert from [0, 1000] normalized to pixel coords
        px = int(cx / 1000 * _VIEWPORT_W)
        py = int(cy / 1000 * _VIEWPORT_H)
        logger.debug("UI-TARS bbox → click(%d, %d)", px, py)
        return f"click({px}, {py})"

    if func_name == "scroll":
        # scroll(start_box='(x,y)', direction='down') → scroll(down)
        # Extract direction from the original string, ignore start_box
        dir_match = re.search(r"direction\s*=\s*['\"]?(\w+)", action_str)
        direction = dir_match.group(1).lower() if dir_match else "down"
        return f"scroll({direction})"

    # For other actions with bbox, let the normal parser handle it
    return action_str


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
            if direction not in ("up", "down", "left", "right"):
                logger.warning("Invalid scroll direction %r, defaulting to 'down'", direction)
                direction = "down"
            return ParsedAction("scroll", {"direction": direction}, raw=action_str)

        elif func_name in ("done", "finished"):
            return ParsedAction("done", {}, raw=action_str)

        elif func_name == "hotkey":
            # UI-TARS: hotkey(key='enter') or hotkey(key='ctrl c')
            key = ""
            if pos_args and pos_args[0] is not None:
                key = str(pos_args[0])
            elif "key" in kwargs:
                key = str(kwargs["key"])
            if key:
                # Convert space-separated keys: 'ctrl c' → 'Control+c'
                parts = key.strip().split()
                mapped = [normalize_key(p) for p in parts]
                combined = "+".join(mapped) if len(mapped) > 1 else mapped[0]
                return ParsedAction("press", {"key": combined}, raw=action_str)

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

    # type(text) or type(content='...')
    m = _RE_TYPE.search(action_str)
    if m:
        text = m.group(1).strip().strip("\"'")
        # Handle keyword format: content='hello' or text='hello'
        if "=" in text:
            text = text.split("=", 1)[1].strip().strip("\"'")
        return ParsedAction("type", {"text": text}, valid=bool(text), raw=action_str)

    # press(key) or press(key='Enter')
    m = _RE_PRESS.search(action_str)
    if m:
        raw_key = m.group(1).strip().strip("\"'")
        # Handle keyword format: key='Enter'
        if "=" in raw_key:
            raw_key = raw_key.split("=", 1)[1].strip().strip("\"'")
        key = normalize_key(raw_key)
        return ParsedAction("press", {"key": key}, valid=bool(key), raw=action_str)

    # scroll(direction) or scroll(direction='up')
    m = _RE_SCROLL.search(action_str)
    if m:
        raw_dir = m.group(1).strip().strip("\"'")
        # Handle keyword format: direction='up' → strip prefix
        if "=" in raw_dir:
            raw_dir = raw_dir.split("=", 1)[1].strip().strip("\"'")
        direction = raw_dir.lower()
        if direction not in ("up", "down", "left", "right"):
            logger.warning("Invalid scroll direction %r, defaulting to 'down'", direction)
            direction = "down"
        return ParsedAction("scroll", {"direction": direction}, raw=action_str)

    # done() / finished()
    if _RE_DONE.search(action_str):
        return ParsedAction("done", {}, raw=action_str)

    # hotkey(key='ctrl c') — UI-TARS format, maps to press
    m = _RE_HOTKEY.search(action_str)
    if m:
        raw_key = m.group(1).strip().strip("\"'")
        if "=" in raw_key:
            raw_key = raw_key.split("=", 1)[1].strip().strip("\"'")
        parts = raw_key.strip().split()
        mapped = [normalize_key(p) for p in parts]
        combined = "+".join(mapped) if len(mapped) > 1 else mapped[0]
        return ParsedAction("press", {"key": combined}, raw=action_str)

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

    # Unwrap <action>…</action> tags (handle unclosed and nested <action> tags)
    tag_match = _RE_ACTION_TAG.search(raw)
    if tag_match:
        action_str = tag_match.group(1).strip()
    else:
        # Handle unclosed <action> tag (e.g. "<action>click(...)")
        unclosed = _RE_ACTION_TAG_UNCLOSED.search(raw)
        action_str = unclosed.group(1).strip() if unclosed else raw
    # Strip any remaining <action> prefix from nested tags
    while action_str.lower().startswith("<action>"):
        action_str = action_str[8:].strip()

    # Strip common VLM prefixes (Thought:, Reflection:, Action:, action:)
    # Case-insensitive; find the *last* occurrence of any known prefix and
    # take everything after it, so "Thought: ... action: scroll(down)" → "scroll(down)".
    _lower = action_str.lower()
    _best_idx = -1
    _best_end = -1
    for _prefix in ("thought:", "reflection:", "action:"):
        idx = _lower.rfind(_prefix)
        if idx > _best_idx:
            _best_idx = idx
            _best_end = idx + len(_prefix)
    if _best_idx != -1:
        action_str = action_str[_best_end:].strip()

    # Convert UI-TARS native bbox format to canonical format
    action_str = _convert_uitars_format(action_str)

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
        # scroll_at(x, y, dx, dy) — scroll at center of viewport
        dy = -300 if parsed.direction == "up" else 300
        return f"scroll_at(640, 360, 0, {dy})", True

    if t == "done":
        return "noop()", True  # BrowserGym has no done(); treat as terminal noop

    return "noop()", False
