#!/usr/bin/env python3
"""Test VWA tasks with API or local models to gauge task difficulty.

Reuses the unified action parser and prompt from the training pipeline
(poisonclaw.action_parser + poisonclaw.envs.prompts.web_agent).

Supports OpenAI (gpt-4o, gpt-4.1, o4-mini, etc.), Anthropic (claude-sonnet-4-6, claude-opus-4-6, etc.),
and local models via transformers (Qwen2.5-VL-3B, etc.).

Usage:
    # OpenAI
    export OPENAI_API_KEY=sk-...
    python scripts/test_vwa_api.py --url http://localhost:9999 --model gpt-4.1 --task "Navigate to Forums"

    # Anthropic Claude
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/test_vwa_api.py --url http://localhost:9999 --model claude-sonnet-4-6 --task "Navigate to Forums"

    # Local Qwen2.5-VL-3B
    python scripts/test_vwa_api.py --url http://localhost:9999 --model Qwen/Qwen2.5-VL-3B-Instruct --task "Navigate to Forums"
"""

import argparse
import base64
import os
import time

from playwright.sync_api import sync_playwright

from poisonclaw.action_parser import parse_action, ParsedAction
from poisonclaw.envs.model_adapter import get_model_adapter
from poisonclaw.envs.prompts.web_agent import build_user_message


# ── Provider detection ───────────────────────────────────────────────────────

def detect_provider(model: str) -> str:
    """Detect API provider from model name."""
    if model.startswith("claude"):
        return "anthropic"
    if "/" in model or os.path.exists(model):
        return "local"
    return "openai"


# ── Model loading ────────────────────────────────────────────────────────────

def load_local_model(model_name: str, device: str = "cuda"):
    """Load a local VLM (Qwen2.5-VL) via transformers. Returns (model, processor) tuple."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading model {model_name} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"Model loaded on {model.device}")
    return model, processor


# ── Unified model call ───────────────────────────────────────────────────────

def _to_anthropic_content(content) -> list | str:
    """Convert OpenAI-style content blocks to Anthropic format."""
    if isinstance(content, str):
        return content
    out = []
    for block in content:
        if block["type"] == "text":
            out.append({"type": "text", "text": block["text"]})
        elif block["type"] == "image_url":
            b64_data = block["image_url"]["url"].split(",", 1)[1]
            out.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64_data},
            })
    return out


def call_model(
    client,
    provider: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 300,
) -> str:
    """Unified inference call — all providers receive the full conversation history.

    Args:
        client: OpenAI/Anthropic API client, or (model, processor) tuple for local.
        provider: "openai" | "anthropic" | "local".
        model: Model name or HuggingFace path.
        messages: Full OpenAI-format messages list (system + alternating user/assistant).
        max_tokens: Maximum tokens to generate.
    """
    if provider == "local":
        local_model, processor = client
        return _call_local(local_model, processor, messages, max_tokens)

    system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
    chat = [m for m in messages if m["role"] != "system"]

    if provider == "anthropic":
        return client.messages.create(
            model=model,
            system=system_prompt,
            messages=[
                {"role": m["role"], "content": _to_anthropic_content(m["content"])}
                for m in chat
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        ).content[0].text

    # OpenAI
    is_reasoning = model.startswith("o")
    api_messages = [{"role": "developer" if is_reasoning else "system", "content": system_prompt}] + chat
    kwargs: dict = dict(model=model, messages=api_messages)
    if is_reasoning:
        kwargs["max_completion_tokens"] = max(max_tokens, 2048)
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0.0
    return client.chat.completions.create(**kwargs).choices[0].message.content


def _call_local(model, processor, messages: list[dict], max_tokens: int = 300) -> str:
    """Run inference on a local Qwen2.5-VL model."""
    import torch
    from qwen_vl_utils import process_vision_info

    # Convert image_url blocks to Qwen's native image format
    qwen_messages = []
    for msg in messages:
        if isinstance(msg["content"], str):
            qwen_messages.append(msg)
            continue
        qwen_content = []
        for block in msg["content"]:
            if block["type"] == "text":
                qwen_content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image_url":
                qwen_content.append({"type": "image", "image": block["image_url"]["url"]})
            else:
                qwen_content.append(block)
        qwen_messages.append({"role": msg["role"], "content": qwen_content})

    text = processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(qwen_messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


# ── Screenshot ───────────────────────────────────────────────────────────────

def screenshot_to_b64(page) -> str:
    return base64.b64encode(page.screenshot()).decode("utf-8")


# ── Action execution ─────────────────────────────────────────────────────────

def execute_action(page, action: ParsedAction) -> str:
    """Execute a ParsedAction on a Playwright sync page."""
    t = action.action_type

    if t == "click":
        page.mouse.click(action.x, action.y)
        try:
            page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:
            pass
        return f"Clicked ({action.x}, {action.y})"

    if t == "type":
        page.keyboard.type(action.text, delay=50)
        return f"Typed: {action.text}"

    if t == "press":
        page.keyboard.press(action.key)
        try:
            page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:
            pass
        return f"Pressed: {action.key}"

    if t == "scroll":
        delta = -300 if action.direction == "up" else 300
        page.mouse.wheel(0, delta)
        page.wait_for_timeout(500)
        return f"Scrolled {action.direction}"

    if t == "done":
        return "DONE"

    return f"No-op: {action.raw}"


def save_screenshot(page, step: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    page.screenshot(path=os.path.join(output_dir, f"step_{step:03d}.png"))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test VWA tasks with API models")
    parser.add_argument("--url", default="http://localhost:9999")
    parser.add_argument("--task", default="Find a post about machine learning and upvote it.")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--model", default="gpt-4.1",
                        help="OpenAI: gpt-4.1, gpt-4o, o4-mini | "
                             "Anthropic: claude-sonnet-4-6, claude-opus-4-6 | "
                             "Local: Qwen/Qwen2.5-VL-3B-Instruct or /path/to/model")
    parser.add_argument("--viewport-w", type=int, default=1280)
    parser.add_argument("--viewport-h", type=int, default=720)
    parser.add_argument("--save-screenshots", default="outputs/vwa_api_test")
    parser.add_argument("--username", default=None, help="Login username (skip login if not set)")
    parser.add_argument("--password", default=None, help="Login password")
    parser.add_argument("--reasoning", action="store_true",
                        help="Ask the model to think step-by-step before acting")
    args = parser.parse_args()

    provider = detect_provider(args.model)

    # IMPORTANT: Launch browser BEFORE loading CUDA model to avoid Playwright/CUDA conflict.
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
        context = browser.new_context(viewport={"width": args.viewport_w, "height": args.viewport_h})
        page = context.new_page()

        # Wait for site to be ready
        print(f"Waiting for {args.url} to be ready...")
        for attempt in range(30):
            try:
                resp = page.goto(args.url, wait_until="networkidle", timeout=10000)
                if resp and resp.status == 200:
                    print("Site is ready!")
                    break
                print(f"  attempt {attempt+1}: HTTP {resp.status if resp else '?'}, retrying...")
            except Exception as e:
                print(f"  attempt {attempt+1}: {e.__class__.__name__}: {e}, retrying...")
            time.sleep(3)
        else:
            print("WARNING: Site not ready after 90s, proceeding anyway...")
        time.sleep(1)

        # Load model / create client (after browser is up to avoid CUDA conflict)
        if provider == "local":
            client = load_local_model(args.model)
            print(f"Using local model: {args.model}")
        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            print(f"Using Anthropic API (model: {args.model})")
        else:
            from openai import OpenAI
            client = OpenAI()
            print(f"Using OpenAI API (model: {args.model})")

        # Login if credentials provided
        if args.username and args.password:
            print(f"Logging in as {args.username}...")
            from urllib.parse import urlparse
            parsed = urlparse(args.url)
            login_url = f"{parsed.scheme}://{parsed.netloc}/login"
            page.goto(login_url, wait_until="networkidle", timeout=15000)
            page.fill('input[name="_username"]', args.username)
            page.fill('input[name="_password"]', args.password)
            page.click('button[type="submit"]')
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            if "login" not in page.url:
                print(f"Logged in! Redirected to {page.url}")
            else:
                print("WARNING: Login may have failed, still on login page.")
            page.goto(args.url, wait_until="networkidle", timeout=10000)
            time.sleep(1)

        adapter = get_model_adapter(args.model)
        system_prompt = adapter.system_prompt
        turn_history: list[dict] = []
        max_history = 3

        print(f"\n{'='*60}")
        print(f"Task:    {args.task}")
        print(f"URL:     {args.url}")
        print(f"Model:   {args.model} ({provider})")
        print(f"Adapter: {adapter.name}")
        print(f"{'='*60}\n")

        stuck_count = 0
        last_action_str = ""

        for step in range(args.max_steps):
            b64_img = screenshot_to_b64(page)
            save_screenshot(page, step, args.save_screenshots)
            obs = {"utterance": args.task, "screenshot_b64": b64_img}

            # Build full conversation history — same for all providers
            effective_prompt = system_prompt
            if args.reasoning:
                # Override "do not explain" for reasoning mode
                effective_prompt = system_prompt.replace(
                    "Output only the action tag — do not explain your reasoning.",
                    "You may explain your reasoning briefly.",
                )
            messages: list[dict] = [{"role": "system", "content": effective_prompt}]
            recent = turn_history[-max_history:]
            history_start = max(0, len(turn_history) - max_history)
            for i, turn in enumerate(recent):
                messages.append({"role": "user", "content": build_user_message(turn["obs"], history_start + i)})
                messages.append({"role": "assistant", "content": turn["action_display"]})
            last_msg = build_user_message(obs, step)

            # Stuck detection: if same action repeated 2+ times, nudge the model
            stuck_hint = ""
            if stuck_count >= 2:
                stuck_hint = (
                    "\n\nIMPORTANT: Your previous action had no effect. "
                    "The page did NOT change. Stop retrying the same click. "
                    "Look carefully at ALL visible buttons on the screen — "
                    "there may be a different button that works better. "
                    "Describe every button you can see, then pick a different one."
                )

            if args.reasoning:
                reasoning_suffix = (
                    "\n\nFirst output your action, then explain your reasoning "
                    "inside <think>...</think> tags on a new line. Example:\n"
                    "<action>click(500, 300)</action>\n"
                    "<think>I see a popup blocking the page. "
                    "I clicked the dismiss button to close it.</think>"
                )
                if isinstance(last_msg, str):
                    last_msg += reasoning_suffix + stuck_hint
                elif isinstance(last_msg, list):
                    last_msg = list(last_msg)
                    last_msg.append({"type": "text", "text": reasoning_suffix + stuck_hint})
            elif stuck_hint:
                # No reasoning mode, but still inject stuck hint
                if isinstance(last_msg, str):
                    last_msg += stuck_hint
                elif isinstance(last_msg, list):
                    last_msg = list(last_msg)
                    last_msg.append({"type": "text", "text": stuck_hint})
            messages.append({"role": "user", "content": last_msg})

            try:
                tokens = 800 if args.reasoning else 300
                reply = call_model(client, provider, args.model, messages, max_tokens=tokens)
            except Exception as e:
                print(f"Step {step+1}: model error: {e}")
                break

            # Extract reasoning (after action) and action (before reasoning)
            reasoning_text = ""
            action_text = reply
            if args.reasoning and "<think>" in reply:
                import re
                think_match = re.search(r"<think>(.*?)</think>", reply, re.DOTALL)
                if think_match:
                    reasoning_text = think_match.group(1).strip()
                    # Action is everything BEFORE <think>
                    action_text = reply[:think_match.start()].strip()
                    if not action_text:
                        # Fallback: action might be after </think>
                        action_text = reply[think_match.end():].strip()

            action = parse_action(action_text if action_text else reply)
            result = execute_action(page, action)

            # Track stuck state
            current_action_str = f"{action.action_type}({action.x},{action.y})" if action.action_type == "click" else action.raw
            if current_action_str == last_action_str:
                stuck_count += 1
            else:
                stuck_count = 0
            last_action_str = current_action_str

            stuck_tag = f" ⚠️ STUCK x{stuck_count+1}" if stuck_count >= 2 else ""
            print(f"Step {step+1}: {action_text.strip() or reply.strip()}{stuck_tag}")
            print(f"  → {result}  [url: {page.url}]")
            if reasoning_text:
                for line in reasoning_text.split("\n"):
                    line = line.strip()
                    if line:
                        print(f"  💭 {line}")
            print()

            turn_history.append({
                "obs": obs,
                "action_display": adapter.format_action_for_history(reply, action),
            })

            if action.action_type == "done":
                print("Agent declared task complete.")
                break

            time.sleep(1)

        save_screenshot(page, step + 1, args.save_screenshots)
        print(f"Screenshots saved to {args.save_screenshots}/")
        browser.close()


if __name__ == "__main__":
    main()
