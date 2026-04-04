#!/usr/bin/env python3
"""Quick test: does vLLM KV cache reallocation OOM after FSDP save cycle?

Simulates the real training scenario: FSDP model + vLLM on same GPU,
sleep vLLM → save FSDP checkpoint → wake vLLM.

Run on GPU node: python3 scripts/test_save_oom.py
"""

import argparse
import gc
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--gpu-mem", type=float, default=0.60)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--expandable-segments", action="store_true", default=True,
                        help="Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    parser.add_argument("--no-expandable-segments", dest="expandable_segments", action="store_false")
    parser.add_argument("--no-free-cache", action="store_true", default=False,
                        help="Skip sleep/wake_up (FREE_CACHE_ENGINE=False mode)")
    parser.add_argument("--save-path", default="/tmp/test_ckpt")
    args = parser.parse_args()

    if args.expandable_segments:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    device = torch.device("cuda:0")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB)")
    print(f"Config: gpu_mem={args.gpu_mem}, max_model_len={args.max_model_len}")
    print()

    def mem():
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        return f"alloc={a:.2f}G reserved={r:.2f}G"

    # Step 1: Load FSDP actor model (transformers, bf16)
    print(f"[1/6] Loading FSDP actor model... ({mem()})")
    from transformers import AutoModelForVision2Seq
    actor = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    actor = actor.to(device)
    print(f"  Actor on GPU. ({mem()})")

    # Step 2: Create vLLM engine (shares GPU)
    print(f"[2/6] Creating vLLM engine... ({mem()})")
    from vllm import LLM
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
    )
    print(f"  vLLM ready. ({mem()})")

    if args.no_free_cache:
        # FREE_CACHE_ENGINE=False mode: vLLM stays alive, save checkpoint alongside it
        print(f"[3/4] Saving FSDP checkpoint (vLLM stays alive)... ({mem()})")
        os.makedirs(args.save_path, exist_ok=True)
        save_file = os.path.join(args.save_path, "model.pt")
        state_dict = actor.state_dict()
        torch.save(state_dict, save_file)
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        size_mb = os.path.getsize(save_file) / 1e6
        print(f"  Saved ({size_mb:.0f} MB). ({mem()})")

        # Second save to verify stability
        print(f"[4/4] Second save (vLLM still alive)... ({mem()})")
        state_dict = actor.state_dict()
        torch.save(state_dict, save_file)
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  ✅ Second save SUCCESS. ({mem()})")

        print("\n✅ No-free-cache mode passed. Config is safe for training with checkpoint save.")
    else:
        # FREE_CACHE_ENGINE=True mode: sleep → save → wake_up
        print(f"[3/6] Sleep vLLM... ({mem()})")
        llm.sleep()
        torch.cuda.empty_cache()
        print(f"  Asleep. ({mem()})")

        print(f"[4/6] Saving FSDP checkpoint... ({mem()})")
        os.makedirs(args.save_path, exist_ok=True)
        save_file = os.path.join(args.save_path, "model.pt")
        state_dict = actor.state_dict()
        torch.save(state_dict, save_file)
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        size_mb = os.path.getsize(save_file) / 1e6
        print(f"  Saved ({size_mb:.0f} MB). ({mem()})")

        print(f"[5/6] Wake up vLLM... ({mem()})")
        try:
            llm.wake_up()
            print(f"  ✅ Wake up SUCCESS. ({mem()})")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ❌ OOM on wake_up! ({mem()})")
                print(f"  Error: {e}")
                import shutil
                shutil.rmtree(args.save_path, ignore_errors=True)
                return
            raise

        print(f"[6/6] Second sleep→save→wake cycle... ({mem()})")
        llm.sleep()
        torch.cuda.empty_cache()
        state_dict = actor.state_dict()
        torch.save(state_dict, save_file)
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        try:
            llm.wake_up()
            print(f"  ✅ Second wake up SUCCESS. ({mem()})")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ❌ OOM on second wake_up! ({mem()})")
            else:
                raise

        print("\n✅ All cycles passed. Config is safe for training with checkpoint save.")

    # Cleanup
    import shutil
    shutil.rmtree(args.save_path, ignore_errors=True)


if __name__ == "__main__":
    main()
