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

    # Local model with custom path
    python scripts/test_vwa_api.py --url http://localhost:9999 --model /path/to/model --task "Navigate to Forums"
"""

import argparse
import base64
import os
import time

from playwright.sync_api import sync_playwright

from poisonclaw.action_parser import parse_action, ParsedAction
from poisonclaw.envs.prompts.web_agent import SYSTEM_PROMPT, build_user_message


# ── API clients ─────────────────────────────────────────────────────────────

def detect_provider(model: str) -> str:
    """Detect API provider from model name."""
    if model.startswith("claude"):
        return "anthropic"
    # Local model: contains '/' (HuggingFace ID or path) or is a filesystem path
    if "/" in model or os.path.exists(model):
        return "local"
    return "openai"


def call_openai(client, model: str, system_prompt: str, user_content: list, max_tokens: int = 300) -> str:
    is_reasoning = model.startswith("o")
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system" if not is_reasoning else "developer", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    if is_reasoning:
        kwargs["max_completion_tokens"] = max(max_tokens, 2048)
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0.0
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def call_anthropic(client, model: str, system_prompt: str, user_content: list, max_tokens: int = 300) -> str:
    """Call Anthropic Claude API. Converts OpenAI-style content to Anthropic format."""
    anthropic_content = []
    for block in user_content:
        if block["type"] == "text":
            anthropic_content.append({"type": "text", "text": block["text"]})
        elif block["type"] == "image_url":
            url = block["image_url"]["url"]
            b64_data = url.split(",", 1)[1]
            anthropic_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64_data,
                },
            })

    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": anthropic_content}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.content[0].text


def load_local_model(model_name: str, device: str = "cuda"):
    """Load a local VLM (Qwen2.5-VL) via transformers."""
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


def call_local(model, processor, messages: list[dict], max_tokens: int = 300) -> str:
    """Run inference on a local Qwen2.5-VL model with multi-turn messages."""
    import torch
    from qwen_vl_utils import process_vision_info

    # Convert image_url format to Qwen's image format in all messages
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


# ── Screenshot ──────────────────────────────────────────────────────────────

def screenshot_to_b64(page) -> str:
    png = page.screenshot()
    return base64.b64encode(png).decode("utf-8")


# ── Action execution (reuses ParsedAction from unified parser) ─────────────

def execute_action(page, action: ParsedAction) -> str:
    """Execute a ParsedAction on a Playwright page."""
    t = action.action_type

    if t == "click":
        page.mouse.click(action.x, action.y)
        # Wait for potential navigation or DOM update
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
        if action.direction not in ("up", "down"):
            return f"Invalid scroll direction: {action.direction}"
        delta = -300 if action.direction == "up" else 300
        page.mouse.wheel(0, delta)
        page.wait_for_timeout(500)
        return f"Scrolled {action.direction}"

    if t == "done":
        return "DONE"

    return f"No-op: {action.raw}"


def save_screenshot(page, step: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    page.screenshot(path=os.path.join(output_dir, f"step_{step:03d}.png"))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test VWA tasks with API models")
    parser.add_argument("--url", default="http://localhost:9999")
    parser.add_argument("--task", default="Find a post about machine learning and upvote it.")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--model", default="gpt-4.1",
                        help="Model ID. OpenAI: gpt-4.1, gpt-4o, o4-mini. "
                             "Anthropic: claude-sonnet-4-6, claude-opus-4-6. "
                             "Local: Qwen/Qwen2.5-VL-3B-Instruct or /path/to/model")
    parser.add_argument("--viewport-w", type=int, default=1280)
    parser.add_argument("--viewport-h", type=int, default=720)
    parser.add_argument("--save-screenshots", default="outputs/vwa_api_test")
    parser.add_argument("--username", default=None, help="Login username (skip login if not set)")
    parser.add_argument("--password", default=None, help="Login password")
    args = parser.parse_args()

    # Detect provider
    provider = detect_provider(args.model)

    # IMPORTANT: Launch browser BEFORE loading CUDA model.
    # On some clusters (e.g. Nautilus/K8s), CUDA initialization changes the
    # process environment in a way that prevents Chromium from connecting
    # to localhost. Starting the browser first avoids this issue.
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
        context = browser.new_context(
            viewport={"width": args.viewport_w, "height": args.viewport_h}
        )
        page = context.new_page()

        # Wait for the site to be ready (postgres may need recovery time)
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

        # Now load model / create API client (after browser is up)
        if provider == "local":
            local_model, local_processor = load_local_model(args.model)
            print(f"Using local model: {args.model}")
        elif provider == "anthropic":
            import anthropic
            api_client = anthropic.Anthropic()
            print(f"Using Anthropic API (model: {args.model})")
        else:
            from openai import OpenAI
            api_client = OpenAI()
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
            # Navigate to the target URL after login
            page.goto(args.url, wait_until="networkidle", timeout=10000)
            time.sleep(1)

        # Multi-turn history: list of {"obs": {...}, "action": str}
        turn_history = []
        max_history = 3  # Keep last N turns to avoid OOM from base64 images

        print(f"\n{'='*60}")
        print(f"Task: {args.task}")
        print(f"URL:  {args.url}")
        print(f"Model: {args.model} ({provider})")
        print(f"Viewport: {args.viewport_w}x{args.viewport_h}")
        print(f"{'='*60}\n")

        for step in range(args.max_steps):
            b64_img = screenshot_to_b64(page)
            save_screenshot(page, step, args.save_screenshots)

            obs = {"utterance": args.task, "screenshot_b64": b64_img}

            # Build multi-turn messages matching training pipeline format
            # This gives the model in-context examples of <action>...</action>
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            recent = turn_history[-max_history:]
            history_start = max(0, len(turn_history) - max_history)
            for i, turn in enumerate(recent):
                messages.append({
                    "role": "user",
                    "content": build_user_message(turn["obs"], history_start + i),
                })
                messages.append({
                    "role": "assistant",
                    "content": f"<action>{turn['action']}</action>",
                })
            messages.append({
                "role": "user",
                "content": build_user_message(obs, step),
            })

            # Call model
            try:
                if provider == "local":
                    reply = call_local(local_model, local_processor, messages)
                elif provider == "anthropic":
                    # For API providers, flatten to single user message
                    user_content = messages[-1]["content"]
                    reply = call_anthropic(api_client, args.model, SYSTEM_PROMPT, user_content)
                else:
                    user_content = messages[-1]["content"]
                    reply = call_openai(api_client, args.model, SYSTEM_PROMPT, user_content)
            except Exception as e:
                print(f"Step {step+1}: API error: {e}")
                break

            action = parse_action(reply)
            result = execute_action(page, action)

            print(f"Step {step+1}: {reply.strip()}")
            print(f"  → {result}")
            print()

            # Store raw action string (not execution result) for history
            raw_action = action.raw if action.action_type != "noop" else reply.strip()
            turn_history.append({"obs": obs, "action": raw_action})

            if action.action_type == "done":
                print("Agent declared task complete.")
                break

            time.sleep(1)

        # Final screenshot
        save_screenshot(page, step + 1, args.save_screenshots)
        print(f"\nScreenshots saved to {args.save_screenshots}/")
        browser.close()


if __name__ == "__main__":
    main()
