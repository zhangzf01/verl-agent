"""Unified system prompt and VLM message builder for PoisonClaw web agents.

All models (Qwen2.5-VL, SeeClick, ShowUI, UI-TARS, UI-R1) share the same
action format during RL training:
    click(x, y)         — pixel coordinates
    type(text)          — text input
    press(key)          — keyboard key
    scroll(direction)   — up or down
    done()              — declare task complete

The VLM sees a screenshot at each step and must respond with exactly one
action wrapped in <action>…</action> tags.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Unified action format description (reusable across prompts)
# ---------------------------------------------------------------------------

ACTION_SPACE_DESC = """\
  click(x, y)        — click at pixel coordinates, e.g. click(120, 55)
  type(text)          — type a string, e.g. type(hello world)
  press(key)          — press a key, e.g. press(Enter)
  scroll(direction)   — scroll the page, e.g. scroll(down) or scroll(up)
  done()              — declare the task complete"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are a GUI agent that interacts with web pages by looking at screenshots and performing actions.

You can perform the following actions:
{ACTION_SPACE_DESC}

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
- Coordinates are absolute pixels within the screenshot viewport.
- Output only the action tag — do not explain your reasoning.
"""

# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def build_user_message(
    obs: dict,
    step: int,
) -> list[dict]:
    """Build one user-turn message for a VLM (multimodal format).

    Compatible with Qwen2.5-VL and OpenAI-style vision APIs.

    Args:
        obs: Observation dict containing:
            - ``"utterance"`` (str): Task description.
            - ``"screenshot_b64"`` (str): Base64-encoded PNG screenshot.
            - ``"task"`` (str, optional): Alternative key for task description.
        step: Current step index (0-indexed).

    Returns:
        List of content items for the ``"user"`` role message.
    """
    content: list[dict] = []

    task = obs.get("utterance") or obs.get("task", "")

    if step == 0:
        content.append({
            "type": "text",
            "text": f"Task: {task}\n\nHere is the current screenshot of the web page:",
        })
    else:
        content.append({
            "type": "text",
            "text": f"Step {step}. Here is the updated screenshot:",
        })

    screenshot_b64 = obs.get("screenshot_b64", "")
    if screenshot_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{screenshot_b64}",
            },
        })

    content.append({
        "type": "text",
        "text": "What action should you take next?",
    })

    return content


def build_messages(
    obs: dict,
    history: list[dict],
    step: int,
    max_history: int = 3,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    """Build the full messages list for a VLM inference call.

    Limits history depth to ``max_history`` to prevent base64 screenshot
    accumulation causing OOM — same strategy as HF repo.

    Args:
        obs: Current observation dict.
        history: List of past ``{"action": str, "obs": dict}`` dicts.
        step: Current step index.
        max_history: Maximum number of past turns to include.
        system_prompt: System prompt string.

    Returns:
        Messages list compatible with VLM chat APIs.
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Truncate history to avoid accumulating too many screenshot images
    recent_history = history[-max_history:] if len(history) > max_history else history
    history_start = max(0, len(history) - max_history)

    for i, turn in enumerate(recent_history):
        actual_step = history_start + i

        messages.append({
            "role": "user",
            "content": build_user_message(turn["obs"], actual_step),
        })
        messages.append({
            "role": "assistant",
            "content": f"<action>{turn['action']}</action>",
        })

    # Current step
    messages.append({
        "role": "user",
        "content": build_user_message(obs, step),
    })

    return messages


def build_messages_text_only(
    task: str,
    action_history: list[str],
    step: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    """Build text-only messages (no screenshots) for debugging or text VLMs.

    Args:
        task: Task description string.
        action_history: List of past action strings.
        step: Current step index.
        system_prompt: System prompt string.

    Returns:
        Messages list with text-only content.
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    if action_history:
        history_text = "\n".join(
            f"Step {i + 1}: {a}" for i, a in enumerate(action_history)
        )
        messages.append({
            "role": "user",
            "content": (
                f"Task: {task}\n\n"
                f"Previous actions:\n{history_text}\n\n"
                f"Step {step + 1}: What action should you take next?"
            ),
        })
    else:
        messages.append({
            "role": "user",
            "content": (
                f"Task: {task}\n\n"
                f"Step {step + 1}: What action should you take next?"
            ),
        })

    return messages
