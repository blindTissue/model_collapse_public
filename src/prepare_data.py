"""One-time download of the real C4 corpus used as gen-0 training data and
as the held-out perplexity test set.

Stream allenai/c4 (en), keep examples with at least `min_chars` characters,
shuffle deterministically, and split into train/validation/test. Saved as a
HuggingFace DatasetDict at <data_dir>/real.

Idempotent: if the on-disk dataset already has the requested sizes, we skip
re-downloading. Pass --force to overwrite.
"""

import argparse
import os
import shutil

import yaml
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def existing_matches(out_dir: str, train_size: int, val_size: int, test_size: int) -> bool:
    if not os.path.exists(os.path.join(out_dir, "dataset_dict.json")):
        return False
    try:
        ds = load_from_disk(out_dir)
    except Exception:
        return False
    return (len(ds.get("train", [])) == train_size
            and len(ds.get("validation", [])) == val_size
            and len(ds.get("test", [])) == test_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fullrank.yaml")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing on-disk dataset.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    out_dir = os.path.join(args.base_dir, cfg["paths"]["data_dir"], "real")

    train_size = ds_cfg["train_size"]
    val_size = ds_cfg["val_size"]
    test_size = ds_cfg["test_size"]
    min_chars = ds_cfg.get("min_chars", 200)
    seed = ds_cfg.get("seed", 42)
    total_needed = train_size + val_size + test_size

    if existing_matches(out_dir, train_size, val_size, test_size) and not args.force:
        print(f"Real corpus already on disk at {out_dir} with matching sizes; "
              "use --force to rebuild.")
        return

    if args.force and os.path.exists(out_dir):
        print(f"--force given, removing existing {out_dir}")
        shutil.rmtree(out_dir)

    print(f"Streaming {ds_cfg['name']}/{ds_cfg['config']} until "
          f"{total_needed} examples (>= {min_chars} chars) collected.")
    stream = load_dataset(ds_cfg["name"], ds_cfg["config"],
                          split="train", streaming=True)

    examples = []
    for ex in stream:
        text = ex.get("text", "")
        if len(text) < min_chars:
            continue
        examples.append({"text": text})
        if len(examples) >= total_needed:
            break
        if len(examples) % 50_000 == 0:
            print(f"  {len(examples):>9,} / {total_needed:,}")

    print(f"Collected {len(examples):,} examples; shuffling with seed={seed}.")
    full = Dataset.from_list(examples).shuffle(seed=seed)

    splits = DatasetDict({
        "train": full.select(range(train_size)),
        "validation": full.select(range(train_size, train_size + val_size)),
        "test": full.select(range(train_size + val_size, total_needed)),
    })

    os.makedirs(out_dir, exist_ok=True)
    splits.save_to_disk(out_dir)
    print(f"Saved real corpus to {out_dir}")
    print(f"  train      {len(splits['train']):,}")
    print(f"  validation {len(splits['validation']):,}")
    print(f"  test       {len(splits['test']):,}")


if __name__ == "__main__":
    main()
