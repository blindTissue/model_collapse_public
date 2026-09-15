"""Generate synthetic continuations from a saved checkpoint with vLLM.

For family F at generation i:
  - Source model: models/F/gen_i/ for full-rank training, or its merged/
    subdirectory for QLoRA.
  - Prompts: first `prompt_length` tokens of `num_continuations` random
    examples drawn from the real C4 train split (so prompt distribution is
    held constant across generations -- only the generator's weights change).
  - Output: data/synthetic_gen_<i>/<F>/  as a HF Dataset with one column
    `text` containing prompt + generated continuation.

Resumable: every batch is appended as a JSONL shard under
`<out_path>.partial/`. On startup we count done shards and skip those
prompts; on completion we concatenate shards into the final HF Dataset and
delete the partial dir.

"""

import argparse
import json
import os
import random
import shutil

import yaml
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_prompts(real_dataset, tokenizer, prompt_length: int,
                    num_prompts: int, seed: int):
    """Sample `num_prompts` examples deterministically and truncate each to
    the first `prompt_length` tokens of the *generating* tokenizer."""
    n_have = len(real_dataset)
    if n_have == 0:
        raise SystemExit("Real dataset is empty; cannot extract prompts.")

    rng = random.Random(seed)
    if num_prompts <= n_have:
        idxs = rng.sample(range(n_have), num_prompts)
    else:
        idxs = [rng.randrange(n_have) for _ in range(num_prompts)]

    prompts = []
    for i in idxs:
        text = real_dataset[i]["text"]
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < prompt_length:
            continue
        prompts.append(tokenizer.decode(ids[:prompt_length], skip_special_tokens=True))
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fullrank.yaml")
    parser.add_argument("--family", type=str, required=True)
    parser.add_argument("--gen", type=int, required=True)
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--num_continuations", type=int, default=None,
                        help="Override generation.num_continuations from config.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Per-family + per-gen prompt sampling seed; "
                             "defaults to dataset.seed + 1000*gen + family-hash.")
    parser.add_argument("--skip_if_exists", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gen_cfg = cfg["generation"]

    models_dir = os.path.join(args.base_dir, cfg["paths"]["models_dir"])
    data_dir = os.path.join(args.base_dir, cfg["paths"]["data_dir"])

    model_path = os.path.join(models_dir, args.family, f"gen_{args.gen}")
    if "qlora" in cfg:
        model_path = os.path.join(model_path, "merged")
    if not os.path.exists(os.path.join(model_path, "config.json")):
        raise SystemExit(f"Merged checkpoint not found at {model_path}.")

    real_path = os.path.join(data_dir, "real")
    out_path = os.path.join(data_dir, f"synthetic_gen_{args.gen}", args.family)

    if args.skip_if_exists and os.path.exists(os.path.join(out_path, "dataset_info.json")):
        print(f"Synthetic data already exists at {out_path}, skipping.")
        return

    num = args.num_continuations or gen_cfg["num_continuations"]
    seed = (args.seed if args.seed is not None
            else cfg["dataset"]["seed"] + 1000 * args.gen
                 + (sum(ord(c) for c in args.family) % 997))

    print(f"=== generate {args.family} gen {args.gen}: {num} continuations ===")
    print(f"  model:  {model_path}")
    print(f"  output: {out_path}")
    print(f"  seed:   {seed}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading real corpus from {real_path}")
    real_ds = load_from_disk(real_path)
    if "train" in real_ds:
        real_ds = real_ds["train"]

    prompts = extract_prompts(real_ds, tokenizer, gen_cfg["prompt_length"], num, seed)
    print(f"Extracted {len(prompts)} prompts (target was {num})")

    batch_size = gen_cfg.get("generation_batch_size", 1024)

    # Resumable shard layout: out_path.partial/shard_<batch_idx>.jsonl,
    # each containing exactly batch_size lines (or fewer for the last one).
    partial_dir = out_path.rstrip("/") + ".partial"
    os.makedirs(partial_dir, exist_ok=True)
    done_shards = set()
    for fname in os.listdir(partial_dir):
        if fname.startswith("shard_") and fname.endswith(".jsonl"):
            try:
                done_shards.add(int(fname[len("shard_"):-len(".jsonl")]))
            except ValueError:
                continue
    if done_shards:
        print(f"Resuming: {len(done_shards)} shards already on disk in {partial_dir}")

    print(f"Booting vLLM on {model_path}...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=gen_cfg.get("vllm_tensor_parallel", 1),
        gpu_memory_utilization=gen_cfg.get("vllm_gpu_memory_utilization", 0.9),
        dtype=gen_cfg.get("vllm_dtype", "bfloat16"),
        max_model_len=gen_cfg.get("vllm_max_model_len",
                                  cfg["training"]["max_seq_length"]),
        seed=seed,
    )

    sampling = SamplingParams(
        max_tokens=gen_cfg["max_new_tokens"],
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg.get("top_p", 1.0),
        repetition_penalty=gen_cfg.get("repetition_penalty", 1.0),
        seed=seed,
    )

    n_shards = (len(prompts) + batch_size - 1) // batch_size
    n_done_already = sum(1 for s in done_shards if s < n_shards)
    n_new = 0
    for shard_idx in range(n_shards):
        if shard_idx in done_shards:
            continue
        start = shard_idx * batch_size
        end = min(start + batch_size, len(prompts))
        batch = prompts[start:end]
        outs = llm.generate(batch, sampling)
        shard_path = os.path.join(partial_dir, f"shard_{shard_idx}.jsonl")
        tmp_path = shard_path + ".tmp"
        with open(tmp_path, "w") as f:
            for o in outs:
                f.write(json.dumps({"text": o.prompt + o.outputs[0].text}) + "\n")
        os.replace(tmp_path, shard_path)
        n_new += len(batch)
        print(f"  shard {shard_idx + 1}/{n_shards}  total this run={n_new:,}  "
              f"(prev resumed={n_done_already * batch_size:,})")

    print("All shards complete; assembling final HuggingFace Dataset.")
    all_texts = []
    for shard_idx in range(n_shards):
        with open(os.path.join(partial_dir, f"shard_{shard_idx}.jsonl")) as f:
            for line in f:
                all_texts.append(json.loads(line)["text"])

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)
    Dataset.from_dict({"text": all_texts}).save_to_disk(out_path)
    shutil.rmtree(partial_dir, ignore_errors=True)
    print(f"Saved {len(all_texts):,} continuations to {out_path}")


if __name__ == "__main__":
    main()
