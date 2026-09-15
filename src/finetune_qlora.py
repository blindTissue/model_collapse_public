"""QLoRA fine-tune for one (family, generation) of the multi-model collapse loop.

For generation `i` of family `F`:
  1. Source model is `models/F/gen_{i-1}/merged/` (or the HF id for gen 0).
  2. Load source in 4-bit NF4, attach a fresh LoRA adapter.
  3. Train one epoch on the appropriate dataset:
       - gen 0:    real C4 train split.
       - gen i>0:  pooled synthetic corpus from gen (i-1).
  4. Save the LoRA adapter to `models/F/gen_i/adapter/`.
  5. Reload the source in bf16 (no quantization), apply the adapter,
     `merge_and_unload()`, and save the merged bf16 checkpoint to
     `models/F/gen_i/merged/`. This is what vLLM and the next generation's
     QLoRA train step both consume.

The merged checkpoint is the "weights-equivalent" theta_i used in the
collapse-loop math; the adapter is the LoRA delta tau_i that we negate for
recovery (see src/recover_qlora.py).

Disk: each adapter is ~150-300 MB; each merged 8B bf16 ckpt is ~16 GB.
Pass `--keep_prev_merged 1` (default) to keep the previous gen's merged on
disk for chaining; pass 0 to delete it after this gen finishes (useful if
the recovery pipeline rebuilds merged ckpts on demand).
"""

import argparse
import gc
import os
import shutil

import torch
import yaml
from datasets import load_from_disk
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_target_modules(qlora_cfg: dict, family: str):
    targets = qlora_cfg.get("target_modules", "all-linear")
    if isinstance(targets, dict):
        return targets.get(family, "all-linear")
    return targets


def make_bnb_config(qlora_cfg: dict) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=getattr(
            torch, qlora_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
    )


def tokenize_dataset(dataset, tokenizer, max_seq_length: int):
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="Tokenizing",
    )
    tokenized = tokenized.filter(
        lambda x: len(x["input_ids"]) >= 64,
        desc="Filtering short sequences",
    )
    return tokenized


def merge_and_save(source_model_id: str, adapter_dir: str, merged_dir: str,
                   tokenizer, dtype: torch.dtype, attn_impl: str | None):
    """Load source in bf16, apply adapter, merge_and_unload, save."""
    print(f"Reloading {source_model_id} in {dtype} for merging...")
    kwargs = dict(dtype=dtype, trust_remote_code=True, device_map="cpu")
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    base = AutoModelForCausalLM.from_pretrained(source_model_id, **kwargs)
    print(f"Attaching adapter from {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    print("Merging adapter into base weights (merge_and_unload)...")
    merged = peft_model.merge_and_unload()
    os.makedirs(merged_dir, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    del peft_model, base, merged
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved merged bf16 checkpoint to {merged_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qlora.yaml")
    parser.add_argument("--family", type=str, required=True,
                        help="Short family name (must match configs/experiment.yaml).")
    parser.add_argument("--gen", type=int, required=True,
                        help="Generation index to train (>= 0).")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_prev_merged", type=int, default=1,
                        help="If 0, delete models/<family>/gen_{i-1}/merged after "
                             "this generation finishes (saves disk; recovery will "
                             "need to rebuild it from the adapter chain).")
    parser.add_argument("--skip_if_exists", action="store_true",
                        help="If models/<family>/gen_<i>/merged already exists, exit 0. "
                             "If the adapter exists but merged is missing, only re-merge. "
                             "If a partial Trainer checkpoint exists, resume from it.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Trainer checkpoint cadence (steps). Set to 0 to disable.")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Number of Trainer checkpoints to retain on disk.")
    parser.add_argument("--data_path_override", type=str, default=None,
                        help="Override the auto-resolved training dataset path.")
    args = parser.parse_args()

    from transformers import set_seed
    set_seed(args.seed)

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    qlora_cfg = cfg["qlora"]

    family = args.family
    gen = args.gen

    family_entry = next((m for m in cfg["models"] if m["short_name"] == family), None)
    if family_entry is None:
        raise SystemExit(f"family '{family}' not in configs/experiment.yaml models list")

    models_dir = os.path.join(args.base_dir, cfg["paths"]["models_dir"])
    data_dir = os.path.join(args.base_dir, cfg["paths"]["data_dir"])
    fam_dir = os.path.join(models_dir, family)
    gen_dir = os.path.join(fam_dir, f"gen_{gen}")
    adapter_dir = os.path.join(gen_dir, "adapter")
    merged_dir = os.path.join(gen_dir, "merged")
    trainer_dir = os.path.join(gen_dir, "trainer")
    os.makedirs(gen_dir, exist_ok=True)

    merged_done = os.path.exists(os.path.join(merged_dir, "config.json"))
    adapter_done = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))

    if gen == 0:
        source_model_id = family_entry["name"]
        train_data_path = args.data_path_override or os.path.join(data_dir, "real")
    else:
        source_model_id = os.path.join(fam_dir, f"gen_{gen - 1}", "merged")
        train_data_path = (args.data_path_override
                           or os.path.join(data_dir, f"synthetic_gen_{gen - 1}", "mixed"))
        if not os.path.exists(os.path.join(source_model_id, "config.json")):
            raise SystemExit(f"Source model not found: {source_model_id}. "
                             "Train the previous generation first.")

    if args.skip_if_exists and merged_done:
        print(f"Already trained AND merged: {merged_dir}; nothing to do.")
        return

    if args.skip_if_exists and adapter_done and not merged_done:
        print(f"Adapter present at {adapter_dir} but merged missing; "
              "skipping training and only re-merging.")
        tokenizer = AutoTokenizer.from_pretrained(source_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        merge_and_save(
            source_model_id=source_model_id,
            adapter_dir=adapter_dir,
            merged_dir=merged_dir,
            tokenizer=tokenizer,
            dtype=getattr(torch, cfg["model"]["dtype"]),
            attn_impl=cfg["training"].get("attn_implementation", None),
        )
        return

    # Resumable Trainer state: any checkpoint-* dir under gen_dir/trainer/
    # tells us we were partway through training and should resume.
    resume_from = None
    if os.path.isdir(trainer_dir):
        ckpts = [d for d in os.listdir(trainer_dir)
                 if d.startswith("checkpoint-")
                 and os.path.isdir(os.path.join(trainer_dir, d))]
        if ckpts:
            resume_from = os.path.join(trainer_dir, sorted(
                ckpts, key=lambda d: int(d.split("-")[-1]))[-1])
            print(f"Found existing Trainer checkpoint, will resume from: {resume_from}")

    print(f"=== {family} gen {gen} ===")
    print(f"  source model:  {source_model_id}")
    print(f"  training data: {train_data_path}")
    print(f"  adapter out:   {adapter_dir}")
    print(f"  merged out:    {merged_dir}")
    print(f"  trainer dir:   {trainer_dir}  (resume={resume_from is not None})")

    tokenizer = AutoTokenizer.from_pretrained(source_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = make_bnb_config(qlora_cfg)
    attn_impl = train_cfg.get("attn_implementation", None)
    model_kwargs = dict(
        quantization_config=bnb_cfg,
        dtype=getattr(torch, cfg["model"]["dtype"]),
        trust_remote_code=True,
        device_map="auto",
    )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    print(f"Loading {source_model_id} in 4-bit NF4 for QLoRA training...")
    model = AutoModelForCausalLM.from_pretrained(source_model_id, **model_kwargs)
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True))

    target_modules = resolve_target_modules(qlora_cfg, family)
    lora_cfg = LoraConfig(
        r=qlora_cfg.get("lora_r", 64),
        lora_alpha=qlora_cfg.get("lora_alpha", 128),
        lora_dropout=qlora_cfg.get("lora_dropout", 0.05),
        bias=qlora_cfg.get("lora_bias", "none"),
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print(f"Loading dataset from: {train_data_path}")
    raw = load_from_disk(train_data_path)
    if "train" in raw:
        train_ds = raw["train"]
        eval_ds = raw.get("validation", None)
    else:
        train_ds = raw
        eval_ds = None

    tokenized_train = tokenize_dataset(train_ds, tokenizer, train_cfg["max_seq_length"])
    tokenized_eval = (tokenize_dataset(eval_ds, tokenizer, train_cfg["max_seq_length"])
                      if eval_ds is not None else None)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    save_strategy = "steps" if args.save_steps and args.save_steps > 0 else "no"
    training_args = TrainingArguments(
        output_dir=trainer_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        torch_compile=train_cfg.get("torch_compile", False),
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        eval_strategy="epoch" if tokenized_eval is not None else "no",
        optim="paged_adamw_32bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    print(f"Starting QLoRA training (resume_from_checkpoint={resume_from})...")
    trainer.train(resume_from_checkpoint=resume_from)

    print(f"Saving LoRA adapter to {adapter_dir}")
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    merge_and_save(
        source_model_id=source_model_id,
        adapter_dir=adapter_dir,
        merged_dir=merged_dir,
        tokenizer=tokenizer,
        dtype=getattr(torch, cfg["model"]["dtype"]),
        attn_impl=attn_impl,
    )

    if gen > 0 and not args.keep_prev_merged:
        prev_merged = os.path.join(fam_dir, f"gen_{gen - 1}", "merged")
        if os.path.exists(prev_merged):
            print(f"--keep_prev_merged 0: removing {prev_merged}")
            shutil.rmtree(prev_merged)

    # Only remove the Trainer scratch state once the merged ckpt is on disk.
    if os.path.exists(os.path.join(merged_dir, "config.json")):
        if os.path.exists(trainer_dir):
            shutil.rmtree(trainer_dir, ignore_errors=True)
        print(f"Done: {family} gen {gen}.")
    else:
        print(f"WARNING: merged dir {merged_dir} missing after merge; "
              "leaving trainer state in place for retry.")


if __name__ == "__main__":
    main()
