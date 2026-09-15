"""Fine-tune a causal language model on a given dataset."""

import argparse
import os
import yaml
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def tokenize_dataset(dataset, tokenizer, max_seq_length: int):
    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        return tokenized

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fullrank.yaml")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to HF dataset on disk (load_from_disk)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save the fine-tuned model")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Override model path (for loading from checkpoint)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    model_name = args.model_name_or_path or cfg["model"].get("name")
    if not model_name:
        parser.error("--model_name_or_path is required for a multi-model configuration")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = train_cfg.get("attn_implementation", None)
    model_kwargs = dict(
        dtype=getattr(torch, cfg["model"]["dtype"]),
        trust_remote_code=True,
    )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    print(f"Loading dataset from: {args.data_path}")
    dataset = load_from_disk(args.data_path)

    if "train" in dataset:
        train_ds = dataset["train"]
        eval_ds = dataset.get("validation", None)
    else:
        train_ds = dataset
        eval_ds = None

    tokenized_train = tokenize_dataset(train_ds, tokenizer, train_cfg["max_seq_length"])
    tokenized_eval = None
    if eval_ds is not None:
        tokenized_eval = tokenize_dataset(eval_ds, tokenizer, train_cfg["max_seq_length"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    save_strategy = ("no" if os.environ.get("NO_TRAIN_CHECKPOINTS") == "1"
                     else train_cfg["save_strategy"])
    train_seed = int(os.environ.get("TRAIN_SEED", cfg["dataset"].get("seed", 42)))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=train_seed,
        data_seed=train_seed,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=save_strategy,
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        torch_compile=train_cfg.get("torch_compile", False),
        report_to="none",
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        eval_strategy="epoch" if tokenized_eval else "no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
