"""Adapter-negation recovery for the QLoRA collapse loop.

Mathematical setup
------------------
Under full fine-tuning, the recovery formula is
    theta_recovered = theta_{i-1} - alpha * (theta_i - theta_{i-1})
                    = theta_{i-1} - alpha * tau_i

Under QLoRA, theta_i = theta_{i-1} + (B_i @ A_i) summed over targeted
linears (with the LoRA scaling factor `lora_alpha / r` baked into the
delta). So the *exact* analog of the full-FT task vector is just the LoRA
delta itself. Recovery becomes
    theta_recovered = theta_{i-1} + (-alpha) * (B_i @ A_i)

i.e. apply the gen-i adapter to the gen-(i-1) merged base with a negative
scaling. PEFT exposes this directly via `add_weighted_adapter` (with
`combination_type="linear"` and a negative weight) or by scaling B by -alpha before merging. This script scales B and
merges the adapter into the previous base checkpoint.

Usage
-----
    python src/recover_qlora.py \\
        --config configs/qlora.yaml \\
        --family llama --gen 5 --alpha 1.0 \\
        --output_dir models/llama/_recovered_gen_5_alpha_1.0

The output directory is a normal HF checkpoint suitable for vLLM /
lm-eval / evaluate.py.
"""

import argparse
import gc
import os

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_recovered(prev_merged_dir: str, adapter_dir: str, alpha: float,
                    out_dir: str, dtype: torch.dtype, attn_impl: str | None):
    """theta_recovered = prev_merged + (-alpha) * (B @ A) per targeted linear.

    Implemented by loading prev_merged, attaching the adapter, and using
    PEFT's internal "merge with negative weight" path: scale down the LoRA
    parameters by -alpha then call merge_and_unload. The math is identical
    to the analytical formula above.
    """
    if os.path.exists(out_dir):
        raise ValueError(f"Output already exists; choose a new directory: {out_dir}")
    print(f"Loading prev merged base in {dtype}: {prev_merged_dir}")
    kwargs = dict(dtype=dtype, trust_remote_code=True, device_map="cpu")
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    base = AutoModelForCausalLM.from_pretrained(prev_merged_dir, **kwargs)

    print(f"Attaching adapter for negation: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)

    print(f"Scaling LoRA matrices by -{alpha} prior to merge "
          "(absorbs the sign into the deltas).")
    scaled = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.mul_(-float(alpha))
                scaled += 1
    print(f"  scaled {scaled} lora_B tensors by -{alpha}")
    if scaled == 0:
        raise ValueError("Adapter has no supported LoRA B matrices")

    print(f"Merging negated adapter into the {dtype} base...")
    merged = model.merge_and_unload()

    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(prev_merged_dir, trust_remote_code=True)
    tokenizer.save_pretrained(out_dir)

    del base, model, merged
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved recovered checkpoint to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qlora.yaml")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--family", type=str, required=True)
    parser.add_argument("--gen", type=int, required=True,
                        help="Future generation providing the adapter; recovery starts at gen-1.")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Single alpha; if omitted, iterates over recovery.alpha_values.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory; defaults to "
                             "models/<family>/_recovered_gen_<gen>_alpha_<a>")
    args = parser.parse_args()

    if args.gen < 1:
        raise SystemExit("--gen must be >= 1 (need a previous generation to recover toward).")

    cfg = load_config(args.config)
    alphas = ([args.alpha] if args.alpha is not None
              else cfg["recovery"]["alpha_values"])
    if args.output_dir and len(alphas) > 1:
        parser.error("An explicit --output_dir requires a single --alpha")
    dtype = getattr(torch, cfg["model"]["dtype"])
    attn_impl = cfg["training"].get("attn_implementation", None)

    models_dir = os.path.join(args.base_dir, cfg["paths"]["models_dir"])
    fam_dir = os.path.join(models_dir, args.family)

    prev_merged = os.path.join(fam_dir, f"gen_{args.gen - 1}", "merged")
    adapter_dir = os.path.join(fam_dir, f"gen_{args.gen}", "adapter")

    for missing in (prev_merged, adapter_dir):
        if not os.path.exists(missing):
            raise SystemExit(f"Required input missing: {missing}")

    for alpha in alphas:
        out_dir = (args.output_dir if args.output_dir is not None
                   else os.path.join(fam_dir, f"_recovered_gen_{args.gen}_alpha_{alpha}"))
        print(f"\n=== {args.family} gen {args.gen} recovery, alpha={alpha} ===")
        build_recovered(prev_merged, adapter_dir, alpha, out_dir, dtype, attn_impl)


if __name__ == "__main__":
    main()
