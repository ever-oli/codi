#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "transformers>=4.40",
#     "datasets>=2.18",
#     "evaluate>=0.4",
#     "accelerate>=0.28",
#     "peft>=0.10",
#     "bitsandbytes>=0.43; sys_platform == 'linux'",
#     "scikit-learn>=1.4",
#     "seqeval>=1.2",
#     "rouge-score>=0.1",
#     "numpy<2",
# ]
# ///
"""
Generic HuggingFace fine-tuning script for autoresearch experiments.

Supports full fine-tuning and LoRA/QLoRA via PEFT.

Run with uv (deps auto-install):
    uv run train.py --model bert-base-uncased --dataset imdb --task text-classification
    uv run train.py --model meta-llama/Llama-3.2-1B --dataset imdb --lora --lr 1e-4

Outputs METRIC lines that the extension parses:
    METRIC eval_accuracy=0.9234
    METRIC eval_loss=0.3456

All heavy storage goes to /Volumes/Expansion/hf-autoresearch/ by default.
Override with HF_AUTORESEARCH_ROOT env var.
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    DefaultDataCollator,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

# ── Storage root ─────────────────────────────────────────────────────

STORAGE_ROOT = os.environ.get("HF_AUTORESEARCH_ROOT", "/Volumes/Expansion/hf-autoresearch")


def parse_args():
    p = argparse.ArgumentParser(description="HF Autoresearch Training Script")

    # Required
    p.add_argument("--model", required=True, help="HF model name or local path")
    p.add_argument("--dataset", required=True, help="HF dataset name")

    # Task / data config
    p.add_argument("--dataset-config", default=None, help="Dataset config/subset")
    p.add_argument(
        "--task",
        default="text-classification",
        choices=[
            "text-classification",
            "token-classification",
            "question-answering",
            "summarization",
            "translation",
            "causal-lm",
        ],
    )
    p.add_argument("--text-column", default="text", help="Primary text column")
    p.add_argument("--text-column-2", default=None, help="Second text column (NLI, paraphrase)")
    p.add_argument("--label-column", default="label", help="Label column")
    p.add_argument("--num-labels", type=int, default=None, help="Number of labels (auto-detected)")
    p.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    p.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples (for quick tests)")
    p.add_argument("--max-eval-samples", type=int, default=None, help="Limit eval samples (for quick tests)")

    # LoRA / QLoRA
    p.add_argument("--lora", action="store_true", help="Use LoRA (PEFT) instead of full fine-tuning")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument(
        "--lora-target-modules",
        type=str,
        default=None,
        help="Comma-separated target modules for LoRA (auto-detected if omitted)",
    )
    p.add_argument("--qlora", action="store_true", help="Use QLoRA (4-bit quantization + LoRA). Implies --lora. Linux/CUDA only.")

    # Hyperparameters
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=16, help="Per-device batch size")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    p.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--fp16", action="store_true", default=False, help="Use FP16")
    p.add_argument("--bf16", action="store_true", default=False, help="Use BF16")
    p.add_argument("--seed", type=int, default=42)

    # Early stopping
    p.add_argument("--early-stopping-patience", type=int, default=0,
                    help="Stop after N evals without improvement (0=disabled)")

    # Output
    p.add_argument("--output-dir", default=None, help="Checkpoint output directory")

    return p.parse_args()


# ── LoRA setup ───────────────────────────────────────────────────────


def apply_lora(model, args):
    """Apply LoRA adapters via PEFT."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    # Map our task names to PEFT TaskType
    task_type_map = {
        "text-classification": TaskType.SEQ_CLS,
        "token-classification": TaskType.TOKEN_CLS,
        "question-answering": TaskType.QUESTION_ANS,
        "causal-lm": TaskType.CAUSAL_LM,
        "summarization": TaskType.SEQ_2_SEQ_LM,
        "translation": TaskType.SEQ_2_SEQ_LM,
    }
    peft_task = task_type_map.get(args.task, TaskType.SEQ_CLS)

    # Target modules
    target_modules = None
    if args.lora_target_modules:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=peft_task,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,  # None = auto-detect
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total > 0 else 0
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    if target_modules:
        print(f"  Target modules: {target_modules}")
    else:
        print(f"  Target modules: auto-detected")

    return model


def load_model_maybe_quantized(model_class, model_name, cache_dir, args, **kwargs):
    """Load model, optionally with 4-bit quantization for QLoRA."""
    if args.qlora:
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print("  Loading model in 4-bit (QLoRA mode)")
            return model_class.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                quantization_config=bnb_config,
                device_map="auto",
                **kwargs,
            )
        except ImportError:
            print("  ⚠️ bitsandbytes not available (Linux/CUDA only). Falling back to standard LoRA.")
            args.qlora = False

    return model_class.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)


# ── Disk space check ─────────────────────────────────────────────────


def check_disk_space(path, min_gb=5):
    """Warn if disk space is low."""
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        if free_gb < min_gb:
            print(f"  ⚠️ LOW DISK SPACE: {free_gb:.1f} GB free on {path} (recommend ≥{min_gb} GB)")
            return False
        else:
            print(f"  Disk: {free_gb:.1f} GB free on {path}")
            return True
    except OSError:
        print(f"  ⚠️ Cannot check disk space on {path}")
        return True  # Don't block on check failure


# ── Dataset loading ──────────────────────────────────────────────────


def load_and_prepare_dataset(args):
    """Load dataset from HF Hub with caching on external drive."""
    cache_dir = os.path.join(STORAGE_ROOT, "datasets")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset}" + (f" ({args.dataset_config})" if args.dataset_config else ""))
    ds = load_dataset(args.dataset, args.dataset_config, cache_dir=cache_dir)

    # Identify splits
    train_split = "train" if "train" in ds else list(ds.keys())[0]
    eval_split = None
    for candidate in ["test", "validation", "dev", "eval"]:
        if candidate in ds:
            eval_split = candidate
            break

    if eval_split is None:
        print("No eval split found, creating 90/10 split from train")
        split = ds[train_split].train_test_split(test_size=0.1, seed=args.seed)
        ds[train_split] = split["train"]
        ds["_eval"] = split["test"]
        eval_split = "_eval"

    # Optional sample limits (for quick iteration)
    if args.max_train_samples and args.max_train_samples < len(ds[train_split]):
        ds[train_split] = ds[train_split].select(range(args.max_train_samples))
        print(f"  ⚡ Limited train to {args.max_train_samples} samples")
    if args.max_eval_samples and args.max_eval_samples < len(ds[eval_split]):
        ds[eval_split] = ds[eval_split].select(range(args.max_eval_samples))
        print(f"  ⚡ Limited eval to {args.max_eval_samples} samples")

    return ds, train_split, eval_split


# ── Task-specific setup ──────────────────────────────────────────────


def setup_text_classification(args, ds, train_split, tokenizer):
    """Set up for sequence classification (single or pair)."""
    num_labels = args.num_labels
    if num_labels is None:
        labels = ds[train_split].unique(args.label_column)
        num_labels = len(labels)
    print(f"  Classification with {num_labels} labels")

    has_pair = args.text_column_2 is not None
    if has_pair:
        print(f"  Sentence pair: ({args.text_column}, {args.text_column_2})")

    model_cache = os.path.join(STORAGE_ROOT, "models")
    model = load_model_maybe_quantized(
        AutoModelForSequenceClassification, args.model, model_cache, args,
        num_labels=num_labels, ignore_mismatched_sizes=True,
    )

    if args.lora or args.qlora:
        model = apply_lora(model, args)

    def tokenize_fn(examples):
        if has_pair:
            return tokenizer(
                examples[args.text_column],
                examples[args.text_column_2],
                padding=False, truncation=True, max_length=args.max_length,
            )
        return tokenizer(
            examples[args.text_column],
            padding=False, truncation=True, max_length=args.max_length,
        )

    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = metric_accuracy.compute(predictions=preds, references=labels)
        f1_args = {"predictions": preds, "references": labels}
        if num_labels > 2:
            f1_args["average"] = "weighted"
        f1 = metric_f1.compute(**f1_args)
        return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

    return model, tokenize_fn, compute_metrics, DataCollatorWithPadding(tokenizer), "eval_accuracy"


def setup_token_classification(args, ds, train_split, tokenizer):
    """Set up for token classification (NER, POS, etc.)."""
    label_list = ds[train_split].features[args.label_column].feature.names
    num_labels = len(label_list)
    print(f"  Token classification with {num_labels} labels: {label_list[:5]}...")

    model_cache = os.path.join(STORAGE_ROOT, "models")
    model = load_model_maybe_quantized(
        AutoModelForTokenClassification, args.model, model_cache, args,
        num_labels=num_labels,
    )

    if args.lora or args.qlora:
        model = apply_lora(model, args)

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples[args.text_column],
            padding=False, truncation=True, max_length=args.max_length,
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[args.label_column]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    label_ids.append(label[word_id])
                else:
                    label_ids.append(-100)
                prev_word_id = word_id
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    metric = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_labels = [
            [label_list[l] for l, p in zip(label_row, pred_row) if l != -100]
            for label_row, pred_row in zip(labels, preds)
        ]
        true_preds = [
            [label_list[p] for l, p in zip(label_row, pred_row) if l != -100]
            for label_row, pred_row in zip(labels, preds)
        ]
        results = metric.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return model, tokenize_fn, compute_metrics, DataCollatorForTokenClassification(tokenizer), "eval_f1"


def setup_causal_lm(args, ds, train_split, tokenizer):
    """Set up for causal language modeling."""
    model_cache = os.path.join(STORAGE_ROOT, "models")
    model = load_model_maybe_quantized(
        AutoModelForCausalLM, args.model, model_cache, args,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    if args.lora or args.qlora:
        model = apply_lora(model, args)

    def tokenize_fn(examples):
        return tokenizer(
            examples[args.text_column],
            padding=False, truncation=True, max_length=args.max_length,
        )

    def compute_metrics(eval_pred):
        return {}

    return model, tokenize_fn, compute_metrics, DataCollatorForLanguageModeling(tokenizer, mlm=False), "eval_loss"


def setup_summarization(args, ds, train_split, tokenizer):
    """Set up for summarization / seq2seq."""
    model_cache = os.path.join(STORAGE_ROOT, "models")
    model = load_model_maybe_quantized(
        AutoModelForSeq2SeqLM, args.model, model_cache, args,
    )

    if args.lora or args.qlora:
        model = apply_lora(model, args)

    def tokenize_fn(examples):
        inputs = tokenizer(
            examples[args.text_column],
            padding=False, truncation=True, max_length=args.max_length,
        )
        targets = tokenizer(
            text_target=examples[args.label_column],
            padding=False, truncation=True, max_length=args.max_length // 4,
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    metric_rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {k: round(v, 4) for k, v in result.items()}

    return model, tokenize_fn, compute_metrics, DataCollatorForSeq2Seq(tokenizer, model=model), "eval_rougeL"


def setup_question_answering(args, ds, train_split, tokenizer):
    """Set up for extractive question answering."""
    model_cache = os.path.join(STORAGE_ROOT, "models")
    model = load_model_maybe_quantized(
        AutoModelForQuestionAnswering, args.model, model_cache, args,
    )

    if args.lora or args.qlora:
        model = apply_lora(model, args)

    def tokenize_fn(examples):
        return tokenizer(
            examples["question"], examples["context"],
            padding=False, truncation="only_second", max_length=args.max_length,
            return_offsets_mapping=True,
        )

    def compute_metrics(eval_pred):
        return {}

    return model, tokenize_fn, compute_metrics, DefaultDataCollator(), "eval_loss"


TASK_SETUP = {
    "text-classification": setup_text_classification,
    "token-classification": setup_token_classification,
    "causal-lm": setup_causal_lm,
    "summarization": setup_summarization,
    "translation": setup_summarization,
    "question-answering": setup_question_answering,
}


# ── Main ─────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    t0 = time.time()

    # QLoRA implies LoRA
    if args.qlora:
        args.lora = True

    # LoRA defaults: higher LR is typical
    if args.lora and args.lr == 2e-5:
        print("  Note: LoRA typically uses higher LR (1e-4 to 3e-4). Using 2e-5 as specified.")

    print("=" * 60)
    print(f"HF Autoresearch Training {'(LoRA)' if args.lora else '(full)'}")
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Task:    {args.task}")
    print(f"  LR:      {args.lr}")
    print(f"  Batch:   {args.batch_size}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Seed:    {args.seed}")
    print(f"  Storage: {STORAGE_ROOT}")
    if args.lora:
        print(f"  LoRA:    r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    if args.qlora:
        print(f"  QLoRA:   4-bit NF4 quantization")
    if args.text_column_2:
        print(f"  Columns: {args.text_column} + {args.text_column_2} → {args.label_column}")
    else:
        print(f"  Columns: {args.text_column} → {args.label_column}")
    if args.early_stopping_patience > 0:
        print(f"  Early stopping: patience={args.early_stopping_patience}")
    print("=" * 60)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Output dir
    output_dir = args.output_dir or os.path.join(STORAGE_ROOT, "runs", f"run-{int(t0)}")
    os.makedirs(output_dir, exist_ok=True)

    # Disk space check
    check_disk_space(STORAGE_ROOT)

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=os.path.join(STORAGE_ROOT, "models"),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    ds, train_split, eval_split = load_and_prepare_dataset(args)
    print(f"  Train: {len(ds[train_split])} examples")
    print(f"  Eval:  {len(ds[eval_split])} examples")

    # Task-specific setup (includes model loading + optional LoRA)
    print(f"\nSetting up for task: {args.task}")
    setup_fn = TASK_SETUP.get(args.task)
    if setup_fn is None:
        print(f"ERROR: Unknown task type: {args.task}")
        sys.exit(1)

    model, tokenize_fn, compute_metrics, data_collator, default_metric = setup_fn(
        args, ds, train_split, tokenizer
    )

    # Tokenize datasets
    print("\nTokenizing datasets...")
    remove_cols = ds[train_split].column_names
    train_ds = ds[train_split].map(tokenize_fn, batched=True, remove_columns=remove_cols)
    eval_ds = ds[eval_split].map(tokenize_fn, batched=True, remove_columns=remove_cols)

    print(f"  Train tokenized: {len(train_ds)} examples")
    print(f"  Eval tokenized:  {len(eval_ds)} examples")

    # Determine precision
    use_fp16 = args.fp16
    use_bf16 = args.bf16
    if not use_fp16 and not use_bf16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            use_bf16 = True

    # Training arguments
    is_seq2seq = args.task in ("summarization", "translation")
    TrainerClass = Seq2SeqTrainer if is_seq2seq else Trainer
    ArgsClass = Seq2SeqTrainingArguments if is_seq2seq else TrainingArguments

    use_mps = torch.backends.mps.is_available() and not torch.cuda.is_available()

    # For early stopping, evaluate more often
    eval_strategy = "epoch"
    save_strategy = "epoch"
    if args.early_stopping_patience > 0 and args.epochs > 3:
        # Evaluate every 0.5 epochs for faster early stopping
        eval_strategy = "steps"
        save_strategy = "steps"

    training_args = ArgsClass(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=default_metric,
        greater_is_better=(default_metric != "eval_loss"),
        report_to="none",
        seed=args.seed,
        logging_steps=50,
        logging_first_step=True,
        dataloader_num_workers=0,
        use_mps_device=use_mps,
        # LoRA: don't need gradient checkpointing for small adapters
        gradient_checkpointing=False,
        **({"predict_with_generate": True} if is_seq2seq else {}),
    )

    # Callbacks
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        print(f"\n  Early stopping enabled: patience={args.early_stopping_patience}")

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Train
    print("\n🚀 Starting training...")
    train_result = trainer.train()

    # Evaluate
    print("\n📊 Evaluating...")
    eval_result = trainer.evaluate()

    elapsed = time.time() - t0

    # ── Output METRIC lines ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for key, value in sorted(eval_result.items()):
        if isinstance(value, (int, float)):
            print(f"METRIC {key}={value:.6f}")

    print(f"METRIC train_loss={train_result.training_loss:.6f}")
    print(f"METRIC training_seconds={elapsed:.1f}")
    print(f"METRIC train_samples_per_second={train_result.metrics.get('train_samples_per_second', 0):.1f}")

    if args.lora:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"METRIC trainable_params={trainable}")
        print(f"METRIC total_params={total}")
        print(f"METRIC trainable_pct={100 * trainable / total:.2f}")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"METRIC peak_vram_mb={peak_vram:.1f}")
    elif torch.backends.mps.is_available():
        try:
            peak_mps = torch.mps.driver_allocated_memory() / (1024**2)
            print(f"METRIC peak_mps_mb={peak_mps:.1f}")
        except Exception:
            pass

    # Save LoRA adapter separately if applicable
    if args.lora:
        adapter_dir = os.path.join(output_dir, "lora_adapter")
        model.save_pretrained(adapter_dir)
        print(f"\nLoRA adapter saved to: {adapter_dir}")
        print(f"METRIC lora_adapter_size_mb={sum(f.stat().st_size for f in __import__('pathlib').Path(adapter_dir).rglob('*') if f.is_file()) / (1024**2):.1f}")

    # Save training info
    info = {
        "model": args.model,
        "dataset": args.dataset,
        "task": args.task,
        "lora": args.lora,
        "qlora": args.qlora,
        "hyperparams": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "fp16": use_fp16,
            "bf16": use_bf16,
            "seed": args.seed,
        },
        "eval_metrics": {k: v for k, v in eval_result.items() if isinstance(v, (int, float))},
        "train_loss": train_result.training_loss,
        "training_seconds": elapsed,
        "output_dir": output_dir,
    }

    if args.lora:
        info["lora_config"] = {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        }

    info_path = os.path.join(output_dir, "run_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nRun info saved to: {info_path}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Training failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
