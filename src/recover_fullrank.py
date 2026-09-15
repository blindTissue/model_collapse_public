"""Current-centered full-rank rewind.

Given a current checkpoint theta_i and a one-step-further collapsed checkpoint
theta_{i+1}, write

    theta_recovered = theta_i - alpha * (theta_{i+1} - theta_i)

to a new generation id.
"""

import argparse
import os

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fullrank.yaml")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--family", type=str, required=True)
    parser.add_argument("--current_gen", type=int, required=True)
    parser.add_argument("--future_gen", type=int, required=True)
    parser.add_argument("--out_gen", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--skip_if_exists", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    models_dir = os.path.join(args.base_dir, cfg["paths"]["models_dir"])
    fam_dir = os.path.join(models_dir, args.family)
    current_path = os.path.join(fam_dir, f"gen_{args.current_gen}")
    future_path = os.path.join(fam_dir, f"gen_{args.future_gen}")
    out_path = os.path.join(fam_dir, f"gen_{args.out_gen}")

    if args.out_gen in (args.current_gen, args.future_gen):
        parser.error("Output generation must differ from both input generations")

    if args.skip_if_exists and os.path.exists(os.path.join(out_path, "config.json")):
        print(f"Recovered checkpoint already exists at {out_path}; skipping.")
        return

    if os.path.exists(out_path):
        raise SystemExit(f"Output already exists; choose a new output generation: {out_path}")

    if not os.path.exists(os.path.join(current_path, "config.json")):
        raise SystemExit(f"Missing current checkpoint: {current_path}")
    if not os.path.exists(os.path.join(future_path, "config.json")):
        raise SystemExit(f"Missing future checkpoint: {future_path}")

    dtype = getattr(torch, cfg["model"]["dtype"])
    print(f"=== current-centered rewind {args.family} ===")
    print(f"  current theta_i:      {current_path}")
    print(f"  future theta_(i+1):   {future_path}")
    print(f"  output theta_hat:     {out_path}")
    print(f"  alpha:                {args.alpha}")

    current = AutoModelForCausalLM.from_pretrained(
        current_path, dtype=dtype, trust_remote_code=True, device_map="cpu"
    )
    future_state = AutoModelForCausalLM.from_pretrained(
        future_path, dtype=dtype, trust_remote_code=True, device_map="cpu"
    ).state_dict()

    current_state = current.state_dict()
    for key, value in current_state.items():
        if key not in future_state or future_state[key].shape != value.shape:
            raise ValueError(f"Incompatible checkpoint parameter: {key}")
        delta = future_state[key].float() - value.float()
        current_state[key] = (value.float() - args.alpha * delta).to(value.dtype)

    current.load_state_dict(current_state)
    os.makedirs(out_path, exist_ok=True)
    current.save_pretrained(out_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(current_path, trust_remote_code=True)
    tokenizer.save_pretrained(out_path)
    print(f"Saved recovered checkpoint to {out_path}")


if __name__ == "__main__":
    main()
