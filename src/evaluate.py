"""Evaluate one (family, generation) checkpoint.

Reports:
  - mean_perplexity on a fixed held-out real test split: how well the
    evaluated checkpoint models real text.
  - The unified diversity panel (distinct-1/2/3 + vocab/sqrt(N) + Yule's K
    + MTLD + length stats) computed on gen_i's own synthetic corpus, with
    the family's tokenizer used to strip the prompt prefix.

"""

import argparse
import json
import math
import os
import sys

import torch
import yaml
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diversity import compute_diversity


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_perplexity(model, tokenizer, dataset, max_seq_length: int,
                       batch_size: int = 4, device: str = "cuda"):
    model.eval()
    model.to(device)
    texts = [ex["text"] for ex in dataset]
    all_ppls = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing perplexity"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        max_length=max_seq_length, padding=True).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        for j in range(len(batch)):
            mask = enc["attention_mask"][j].bool()
            seq_len = mask.sum().item()
            if seq_len <= 1:
                continue
            input_ids_j = enc["input_ids"][j][mask]
            logits_j = out.logits[j][mask]
            shift_logits = logits_j[:-1]
            shift_labels = input_ids_j[1:]
            loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
            all_ppls.append(math.exp(min(loss.item(), 100)))
    mean_ppl = sum(all_ppls) / len(all_ppls) if all_ppls else float("inf")
    return {"mean_perplexity": mean_ppl, "per_example_perplexity": all_ppls}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fullrank.yaml")
    parser.add_argument("--family", type=str, required=True,
                        help="Short family name; selects its full-rank or QLoRA checkpoint.")
    parser.add_argument("--gen", type=int, required=True)
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--max_examples", type=int, default=2000,
                        help="Subsample of the real test split for ppl.")
    parser.add_argument("--diversity_max_examples", type=int, default=20000,
                        help="Sample limit for diversity computation; "
                             "kept fixed across runs for cross-experiment "
                             "comparability.")
    parser.add_argument("--strip_prompt_tokens", type=int, default=None,
                        help="Defaults to generation.prompt_length.")
    parser.add_argument("--diversity_data_path_override", type=str, default=None)
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--model_path", default=None, help="Evaluate a recovered checkpoint.")
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    strip_n = (args.strip_prompt_tokens
               if args.strip_prompt_tokens is not None
               else cfg["generation"]["prompt_length"])

    models_dir = os.path.join(args.base_dir, cfg["paths"]["models_dir"])
    data_dir = os.path.join(args.base_dir, cfg["paths"]["data_dir"])
    results_dir = os.path.join(args.base_dir, cfg["paths"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(models_dir, args.family, f"gen_{args.gen}")
    if "qlora" in cfg:
        model_path = os.path.join(model_path, "merged")
    model_path = args.model_path or model_path
    real_test_path = os.path.join(data_dir, "real")
    div_path = (args.diversity_data_path_override
                or os.path.join(data_dir, f"synthetic_gen_{args.gen}", args.family))
    if args.model_path and not args.output_file:
        parser.error("--model_path requires --output_file to keep baseline results separate")
    out_file = args.output_file or os.path.join(results_dir, f"eval_{args.family}_gen_{args.gen}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

    if args.skip_if_exists and os.path.exists(out_file):
        print(f"Already evaluated: {out_file}")
        return

    print(f"=== eval {args.family} gen {args.gen} ===")
    print(f"  model:     {model_path}")
    print(f"  real test: {real_test_path}")
    print(f"  diversity: {div_path}")
    print(f"  output:    {out_file}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=cfg["training"].get("attn_implementation", "sdpa"),
    )

    real_ds = load_from_disk(real_test_path)
    test_ds = real_ds["test"] if "test" in real_ds else real_ds
    if args.max_examples and len(test_ds) > args.max_examples:
        test_ds = test_ds.select(range(args.max_examples))

    print(f"Computing perplexity on {len(test_ds):,} real test examples...")
    ppl = compute_perplexity(
        model, tokenizer, test_ds,
        max_seq_length=cfg["training"]["max_seq_length"],
        device=device,
    )

    if args.model_path and not args.diversity_data_path_override:
        diversity_panel = {"diversity_note": "Recovered-model generations not supplied; reporting PPL only."}
    elif not os.path.exists(os.path.join(div_path, "dataset_info.json")):
        print(f"WARNING: no diversity corpus at {div_path}; reporting ppl only.")
        diversity_panel = {"diversity_note": f"missing {div_path}"}
    else:
        div_ds = load_from_disk(div_path)
        if isinstance(div_ds, dict) and "train" in div_ds:
            div_ds = div_ds["train"]
        div_texts = [ex["text"] for ex in div_ds]
        diversity_panel = compute_diversity(
            div_texts,
            tokenizer=tokenizer,
            strip_prompt_tokens=strip_n,
            sample_limit=args.diversity_max_examples,
        )
        diversity_panel["diversity_source"] = "unified:compute_diversity"
        diversity_panel["diversity_tokenizer"] = args.family

    results = {
        "family": args.family,
        "gen": args.gen,
        "model_path": model_path,
        "real_test_path": real_test_path,
        "diversity_data_path": div_path,
        "num_examples_ppl": len(test_ds),
        "mean_perplexity": ppl["mean_perplexity"],
        **diversity_panel,
    }

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    with open(out_file.replace(".json", "_per_example.json"), "w") as f:
        json.dump(ppl["per_example_perplexity"], f)

    print(f"Results: ppl={results['mean_perplexity']:.2f}  "
          f"d1={results.get('distinct_1', float('nan')):.4f}  "
          f"d2={results.get('distinct_2', float('nan')):.4f}  "
          f"d3={results.get('distinct_3', float('nan')):.4f}")
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
