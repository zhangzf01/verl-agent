from transformers import AutoModelForVision2Seq
from peft import LoraConfig, get_peft_model

MODELS = {
    "uitars_2b":  ("bytedance-research/UI-TARS-2B-SFT",  ["q_proj","k_proj","v_proj","o_proj"]),
    "qwenvl_7b":  ("Qwen/Qwen2.5-VL-7B-Instruct",        ["q_proj","k_proj","v_proj","o_proj"]),
    "qwenvl_3b":  ("Qwen/Qwen2.5-VL-3B-Instruct",        ["q_proj","k_proj","v_proj","o_proj"]),
    "seeclick_9b":("cckevinn/SeeClick",                    ["c_attn","attn.c_proj","visual.resampler.attn.in_proj","visual.resampler.attn.out_proj"]),
    "showui_2b":  ("showlab/ShowUI-2B",                   ["q_proj","k_proj","v_proj","o_proj"]),
    "uir1_3b":    ("LZXzju/Qwen2.5-VL-3B-UI-R1",         ["q_proj","k_proj","v_proj","o_proj"]),
}

import sys
name = sys.argv[1] if len(sys.argv) > 1 else None
if name not in MODELS:
    print(f"Usage: python test.py <model_name>")
    print(f"Available: {list(MODELS.keys())}")
    sys.exit(1)

model_id, target_modules = MODELS[name]
print(f"\n=== {name} ({model_id}) ===")
print(f"target_modules: {target_modules}\n")

try:
    m = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True)
except Exception:
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

cfg = LoraConfig(r=8, target_modules=target_modules)
try:
    pm = get_peft_model(m, cfg)
    pm.print_trainable_parameters()
    trained = [n for n, p in pm.named_parameters() if p.requires_grad]
    print(f"Total trainable param tensors: {len(trained)}")
    # Show unique module types hit
    import re
    hits = set()
    for n in trained:
        m2 = re.search(r'((?:visual\.\S+|language_model\.\S+)\.lora_[AB])', n)
        if m2:
            # strip lora_A/B suffix and layer index to show pattern
            pat = re.sub(r'\.\d+\.', '.N.', m2.group(1))
            hits.add(pat)
    print("Unique patterns hit:")
    for h in sorted(hits):
        print(f"  {h}")
except Exception as e:
    print(f"ERROR applying LoRA: {e}")
    print("\nAll linear layers in model:")
    for n, mod in m.named_modules():
        if hasattr(mod, 'weight') and mod.__class__.__name__ in ('Linear', 'Conv2d', 'Conv1d'):
            print(f"  {n}  [{mod.__class__.__name__}]")
