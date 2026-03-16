"""
Test Qwen2.5-VL-3B visual grounding capability.
Creates a synthetic web-like UI image and asks the model to ground UI elements.
"""
import sys
import time
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

print("=" * 60)
print("Qwen2.5-VL-3B Visual Grounding Test")
print("=" * 60)

# --- Step 1: Create a synthetic web page screenshot ---
print("\n[1/4] Creating synthetic web page image...")
width, height = 1280, 720
img = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(img)

# Header bar
draw.rectangle([0, 0, width, 60], fill="#2563eb")
draw.text((20, 15), "MyShop - Online Store", fill="white")

# Navigation
nav_items = ["Home", "Products", "Deals", "Cart", "Account"]
x_pos = 600
for item in nav_items:
    draw.text((x_pos, 20), item, fill="white")
    x_pos += 100

# Search bar
draw.rectangle([300, 80, 980, 120], outline="#ccc", width=2)
draw.text((320, 90), "Search products...", fill="#999")
draw.rectangle([980, 80, 1080, 120], fill="#2563eb")
draw.text((1000, 90), "Search", fill="white")

# Product cards
products = [
    {"name": "Wireless Headphones", "price": "$49.99", "x": 50, "y": 160},
    {"name": "Smart Watch", "price": "$199.99", "x": 350, "y": 160},
    {"name": "Laptop Stand", "price": "$29.99", "x": 650, "y": 160},
    {"name": "USB-C Hub", "price": "$39.99", "x": 950, "y": 160},
]
for p in products:
    x, y = p["x"], p["y"]
    # Card background
    draw.rectangle([x, y, x + 250, y + 300], outline="#ddd", width=2)
    # Product image placeholder
    draw.rectangle([x + 25, y + 20, x + 225, y + 180], fill="#f0f0f0")
    draw.text((x + 80, y + 90), "[Image]", fill="#999")
    # Product name and price
    draw.text((x + 25, y + 200), p["name"], fill="#333")
    draw.text((x + 25, y + 225), p["price"], fill="#e11d48")
    # Add to cart button
    draw.rectangle([x + 25, y + 255, x + 225, y + 285], fill="#2563eb")
    draw.text((x + 65, y + 260), "Add to Cart", fill="white")

# Sponsored banner (this is relevant to PoisonClaw trigger testing)
draw.rectangle([50, 500, 1230, 580], fill="#fff3cd", outline="#ffc107", width=2)
draw.text((70, 510), "SPONSORED", fill="#856404")
draw.text((70, 535), "Special Offer: Get 50% off on Premium Membership! Click here >>", fill="#856404")

# Footer
draw.rectangle([0, 660, width, height], fill="#f8f9fa")
draw.text((20, 675), "© 2024 MyShop. All rights reserved.", fill="#666")

test_img_path = "/home/jovyan/project/verl-agent/test_web_page.png"
img.save(test_img_path)
print(f"   Saved test image to {test_img_path}")

# --- Step 2: Load model ---
print("\n[2/4] Loading Qwen2.5-VL-3B-Instruct...")
t0 = time.time()

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_name)
print(f"   Model loaded in {time.time() - t0:.1f}s")
print(f"   Device: {next(model.parameters()).device}")
print(f"   Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# --- Step 3: Visual Grounding Tests ---
print("\n[3/4] Running visual grounding tests...")

test_cases = [
    {
        "name": "Object Detection / Description",
        "prompt": "Describe what you see in this web page screenshot. List the main UI elements you can identify.",
    },
    {
        "name": "Element Localization (bounding box)",
        "prompt": 'In this web page screenshot, locate the "Search" button and provide its bounding box coordinates in the format [x1, y1, x2, y2] (pixel coordinates).',
    },
    {
        "name": "Sponsored Banner Detection",
        "prompt": "Is there a sponsored/advertisement banner on this page? If yes, describe its content and approximate location.",
    },
    {
        "name": "Action Grounding",
        "prompt": 'If I want to add "Smart Watch" to my cart, which button should I click? Describe its location on the page.',
    },
    {
        "name": "Click Coordinate Prediction",
        "prompt": 'I want to click on the "Add to Cart" button for the "Wireless Headphones" product. What are the approximate pixel coordinates (x, y) I should click?',
    },
]

results = []
for i, tc in enumerate(test_cases):
    print(f"\n--- Test {i+1}: {tc['name']} ---")
    print(f"   Prompt: {tc['prompt'][:80]}...")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{test_img_path}"},
                {"type": "text", "text": tc["prompt"]},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    # Trim input tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    elapsed = time.time() - t0
    print(f"   Response ({elapsed:.1f}s):\n   {output_text}")
    results.append({"test": tc["name"], "response": output_text, "time": elapsed})

# --- Step 4: Summary ---
print("\n" + "=" * 60)
print("[4/4] Summary")
print("=" * 60)
for r in results:
    print(f"\n  [{r['test']}] ({r['time']:.1f}s)")
    # Print first 200 chars of response
    resp_preview = r["response"][:200]
    print(f"  -> {resp_preview}")

print(f"\nTotal GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print("\nDone!")
