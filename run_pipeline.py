"""Portable sequential entry point for the multi-model feedback loop."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


SOURCE = Path(__file__).resolve().parent / "src"


def commands(config, config_path, run_dir, start, end, training_seed):
    common = ["--config", str(config_path), "--base_dir", str(run_dir)]
    yield ["prepare_data.py", *common], None
    for gen in range(start, end + 1):
        for family in config["models"]:
            name = family["short_name"]
            if "qlora" in config:
                yield ["finetune_qlora.py", *common, "--family", name, "--gen", str(gen),
                       "--seed", str(training_seed), "--skip_if_exists"], None
            else:
                models = run_dir / config["paths"]["models_dir"] / name
                output = models / f"gen_{gen}"
                source = family["name"] if gen == 0 else str(models / f"gen_{gen-1}")
                data = run_dir / config["paths"]["data_dir"]
                data = data / "real" if gen == 0 else data / f"synthetic_gen_{gen-1}" / "mixed"
                yield ["finetune_fullrank.py", "--config", str(config_path),
                       "--model_name_or_path", source, "--data_path", str(data),
                       "--output_dir", str(output)], output / ".training_complete"
        for family in config["models"]:
            yield ["generate_data.py", *common, "--family", family["short_name"],
                   "--gen", str(gen), "--skip_if_exists"], None
        yield ["pool_synthetic.py", *common, "--gen", str(gen), "--skip_if_exists"], None
        for family in config["models"]:
            yield ["evaluate.py", *common, "--family", family["short_name"],
                   "--gen", str(gen), "--skip_if_exists"], None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/fullrank.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/seed42"))
    parser.add_argument("--start-gen", type=int, default=0)
    parser.add_argument("--end-gen", type=int)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    end = args.end_gen if args.end_gen is not None else config["collapse"]["num_generations"]
    if not 0 <= args.start_gen <= end <= config["collapse"]["num_generations"]:
        parser.error("Generation range must be within the configured trajectory")
    names = [model["short_name"] for model in config["models"]]
    if not names or len(names) != len(set(names)):
        parser.error("Model short names must be nonempty and unique")
    run_dir = args.run_dir.resolve()
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = run_dir / "run_config.yaml"
        effective = {"experiment": config, "training_seed": args.training_seed}
        if manifest.exists() and yaml.safe_load(manifest.read_text()) != effective:
            parser.error("Run configuration differs from saved configuration; use a new run directory")
        manifest.write_text(yaml.safe_dump(effective, sort_keys=False))
    env = {**os.environ, "TRAIN_SEED": str(args.training_seed),
           "TOKENIZERS_PARALLELISM": "false"}
    for step, marker in commands(config, config_path, run_dir, args.start_gen, end, args.training_seed):
        cmd = [sys.executable, str(SOURCE / step[0]), *step[1:]]
        print(shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        if marker and marker.exists():
            print(f"Complete: {marker.parent}", flush=True)
            continue
        if marker and marker.parent.exists() and any(marker.parent.iterdir()):
            raise RuntimeError(f"Incomplete full-rank output requires inspection: {marker.parent}")
        subprocess.run(cmd, env=env, check=True)
        if marker:
            marker.write_text("Training and final checkpoint save completed.\n")


if __name__ == "__main__":
    main()
