"""Pool the per-family synthetic corpora into a single mixed corpus.

For generation `i`, reads:
    data/synthetic_gen_<i>/<family>/   for each family in the config
and writes:
    data/synthetic_gen_<i>/mixed/

The mixed corpus is the deterministic shuffle of the union of all per-family
continuations. It's what each family trains on at generation i+1.
"""

import argparse
import os

import yaml
from datasets import Dataset, concatenate_datasets, load_from_disk


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fullrank.yaml")
    parser.add_argument("--gen", type=int, required=True)
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=None,
                        help="Defaults to dataset.seed + gen.")
    parser.add_argument("--skip_if_exists", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = os.path.join(args.base_dir, cfg["paths"]["data_dir"])
    families = [m["short_name"] for m in cfg["models"]]
    seed = args.seed if args.seed is not None else cfg["dataset"]["seed"] + args.gen

    out_path = os.path.join(data_dir, f"synthetic_gen_{args.gen}", "mixed")
    if args.skip_if_exists and os.path.exists(os.path.join(out_path, "dataset_info.json")):
        print(f"Pooled corpus already exists at {out_path}, skipping.")
        return

    print(f"=== pool synthetic_gen_{args.gen} from {families} ===")
    parts = []
    for fam in families:
        fam_path = os.path.join(data_dir, f"synthetic_gen_{args.gen}", fam)
        if not os.path.exists(os.path.join(fam_path, "dataset_info.json")):
            raise SystemExit(f"Missing per-family corpus: {fam_path}. "
                             "Generate it first.")
        ds = load_from_disk(fam_path)
        if "text" not in ds.column_names:
            raise SystemExit(f"{fam_path} has no 'text' column.")
        ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
        ds = ds.add_column("source_family", [fam] * len(ds))
        print(f"  {fam:<6}  {len(ds):>9,} continuations")
        parts.append(ds)

    pooled = concatenate_datasets(parts).shuffle(seed=seed)
    print(f"  pooled  {len(pooled):>9,} continuations  (seed={seed})")
    os.makedirs(out_path, exist_ok=True)
    pooled.save_to_disk(out_path)
    print(f"Saved pooled corpus to {out_path}")


if __name__ == "__main__":
    main()
