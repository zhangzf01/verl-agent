"""System prompts and VLM message builders for PoisonClaw web agents.

Two prompt variants:
  1. Generic (Qwen2.5-VL, ShowUI, UI-R1, SeeClick): ``<action>click(x,y)</action>``
  2. UI-TARS native: ``Thought: … Action: click(start_box='(x,y)')``

Use ``get_system_prompt(model_path)`` to auto-select the right variant.
The action parser handles conversion between formats transparently.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Generic action format (Qwen2.5-VL, ShowUI, UI-R1, SeeClick)
# ---------------------------------------------------------------------------

ACTION_SPACE_DESC = """\
  click(x, y)        — click at pixel coordinates (x: 0-1280, y: 0-720), e.g. click(120, 55)
  type(text)          — type a string, e.g. type(hello world)
  press(key)          — press a key, e.g. press(Enter)
  scroll(direction)   — scroll the page, e.g. scroll(down) or scroll(up)
  done()              — call this immediately after completing all task steps"""

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
- Coordinates are absolute pixels within the 1280×720 viewport.
- Output only the action tag — do not explain your reasoning.
- When all task steps are done, you MUST call done() immediately.
"""

# ---------------------------------------------------------------------------
# UI-TARS native format (matches SFT training data)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_UITARS = """\
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


def get_system_prompt(model_path: str = "") -> str:
    """Return the appropriate system prompt for the given model.

    Args:
        model_path: HuggingFace model path or name (case-insensitive match).

    Returns:
        System prompt string tailored to the model's native format.
    """
    if "ui-tars" in model_path.lower() or "uitars" in model_path.lower():
        return SYSTEM_PROMPT_UITARS
    return SYSTEM_PROMPT

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
            "text": f"Task: {task}\n\nStep {step}. Here is the updated screenshot:",
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
