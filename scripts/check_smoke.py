"""Validate artifacts from configs/smoke.yaml without loading a model."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from datasets import load_from_disk


def check(run_dir):
    result_path = run_dir / "results/eval_llama_gen_0.json"
    result = json.loads(result_path.read_text())
    perplexities = json.loads(result_path.with_name("eval_llama_gen_0_per_example.json").read_text())
    if len(perplexities) != 4 or not all(math.isfinite(p) and p >= 1 for p in perplexities):
        raise ValueError("Expected four finite per-example perplexities >= 1")
    if not math.isclose(result["mean_perplexity"], sum(perplexities) / 4):
        raise ValueError("Mean perplexity disagrees with per-example results")
    if result["num_examples_ppl"] != 4 or result["num_examples_after_strip"] != 4:
        raise ValueError("Expected four evaluated examples and four nonempty continuations")
    for n in (1, 2, 3):
        if not 0 <= result[f"distinct_{n}"] <= 1:
            raise ValueError(f"distinct_{n} is outside [0, 1]")
    synthetic = load_from_disk(str(run_dir / "data/synthetic_gen_0/llama"))
    pooled = load_from_disk(str(run_dir / "data/synthetic_gen_0/mixed"))
    if len(synthetic) != 4 or len(pooled) != 4:
        raise ValueError("Expected four generated and four pooled examples")
    if Counter(synthetic["text"]) != Counter(pooled["text"]):
        raise ValueError("Pooling changed the generated texts")
    if set(pooled["source_family"]) != {"llama"}:
        raise ValueError("Incorrect source-family labels")
    checkpoint = run_dir / "models/llama/gen_0"
    for name in ("config.json", "tokenizer.json", ".training_complete"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"Missing checkpoint artifact: {name}")
    if not list(checkpoint.glob("*.safetensors")):
        raise ValueError("Missing checkpoint weights")
    print(f"PASS: saved checkpoint, four continuations, pooling, and evaluation; "
          f"mean perplexity={result['mean_perplexity']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    check(parser.parse_args().run_dir)
