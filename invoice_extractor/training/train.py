from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from invoice_extractor.data.label_mapping import bio_labels, id_to_label
from invoice_extractor.training.eval_metrics import compute_metrics
from invoice_extractor.training.tokenize import tokenize_and_align_labels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 for invoice token classification")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing train/ and val/ splits")
    parser.add_argument("--output-dir", required=True, type=Path, help="Checkpoint output directory")
    parser.add_argument("--model-name", default="microsoft/layoutlmv3-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Load only 20 train / 5 val examples and train 1 epoch (~3 min sanity check)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available. bf16 training requires a CUDA GPU.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    labels = bio_labels()
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    print(f"Label set: {len(labels)} BIO labels ({len(labels) // 2} entity classes + O)")

    processor = LayoutLMv3Processor.from_pretrained(args.model_name, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = load_from_disk(str(args.data_dir / "train"))
    val_ds = load_from_disk(str(args.data_dir / "val"))

    if args.debug:
        train_ds = train_ds.select(range(min(20, len(train_ds))))
        val_ds = val_ds.select(range(min(5, len(val_ds))))
        print(f"[DEBUG] Using {len(train_ds)} train / {len(val_ds)} val examples")

    # Sanity-check one example before tokenisation
    ex = train_ds[0]
    print(f"Example 0 — tokens[:5]: {ex['tokens'][:5]}")
    print(f"Example 0 — bboxes[:3]: {ex['bboxes'][:3]}")
    print(f"Example 0 — labels[:5]: {ex['labels'][:5]}")

    tokenize_fn = partial(tokenize_and_align_labels, processor=processor, max_length=args.max_length)
    orig_train_cols = train_ds.column_names
    orig_val_cols = val_ds.column_names

    print("Tokenising train split...")
    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=orig_train_cols,
        keep_in_memory=False,
        load_from_cache_file=True,
        num_proc=2,
    )
    print("Tokenising val split...")
    val_ds = val_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=orig_val_cols,
        keep_in_memory=False,
        load_from_cache_file=True,
        num_proc=2,
    )

    print(f"Tokenised columns: {train_ds.column_names}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = 1 if args.debug else args.epochs
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    metrics_fn = partial(compute_metrics, id_to_label=id_to_label)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        data_collator=default_data_collator,
        compute_metrics=metrics_fn,
    )

    trainer.train()
    trainer.save_model()
    processor.save_pretrained(str(args.output_dir))
    print(f"Checkpoint saved to {args.output_dir}")

    print("\n=== Final Evaluation ===")
    final_metrics = trainer.evaluate()
    max_key_len = max(len(k) for k in final_metrics)
    for k in sorted(final_metrics):
        v = final_metrics[k]
        if isinstance(v, float):
            print(f"  {k:<{max_key_len}}  {v:.4f}")
        else:
            print(f"  {k:<{max_key_len}}  {v}")


if __name__ == "__main__":
    main()
