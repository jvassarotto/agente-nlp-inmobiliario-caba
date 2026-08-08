"""Fine-tuning de BETO para CLASIFICACION multilabel de senales del vendedor.

Clases (multilabel): DUENO_DIRECTO, OPORTUNIDAD, URGENCIA, REFACCION.
Un aviso puede tener 0, 1 o varias senales a la vez.

Ajustado para GTX 1650 4GB (fp16, batch chico).

Uso:
  python -m src.models.train_classifier
  python -m src.models.train_classifier --base_model prajjwal1/bert-tiny --epochs 1 --max_train 64
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from src.utils.config import load_config, set_seed
from src.utils.io import read_jsonl
from src.annotation.label_schema import SIGNAL_CLASSES, CLASS2ID, ID2CLASS


def multihot(signals):
    v = np.zeros(len(SIGNAL_CLASSES), dtype=np.float32)
    for s in signals:
        if s in CLASS2ID:
            v[CLASS2ID[s]] = 1.0
    return v


def load_split(root, name):
    return list(read_jsonl(Path(root) / "data/annotated" / name))


def build_dataset(records, tokenizer, max_length):
    from datasets import Dataset
    data = [{"text": r["text"], "labels": multihot(r.get("signals", [])).tolist()} for r in records]
    ds = Dataset.from_list(data)

    def encode(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=max_length)
        enc["labels"] = batch["labels"]
        return enc

    return ds.map(encode, batched=True, remove_columns=["text"])


def make_metrics(threshold=0.5):
    from sklearn.metrics import precision_recall_fscore_support, f1_score

    def compute(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)
        labels = labels.astype(int)
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
        out = {"precision_macro": p, "recall_macro": r, "f1_macro": f, "f1_micro": micro_f1}
        # F1 por clase
        pc, rc, fc, _ = precision_recall_fscore_support(labels, preds, average=None,
                                                        labels=list(range(len(SIGNAL_CLASSES))),
                                                        zero_division=0)
        for i, cls in enumerate(SIGNAL_CLASSES):
            out[f"f1_{cls}"] = float(fc[i])
        return out
    return compute


def main():
    cfg = load_config()
    cc = cfg["classifier"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=cc["base_model"])
    ap.add_argument("--epochs", type=float, default=cc["epochs"])
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--out_dir", default=cc["out_dir"])
    args = ap.parse_args()

    set_seed(cfg["project"]["seed"])
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, TrainingArguments, Trainer)
    import torch

    root = cfg["_root"]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tr = load_split(root, "cls_train.jsonl")
    va = load_split(root, "cls_val.jsonl")
    te = load_split(root, "cls_test.jsonl")
    if args.max_train:
        tr, va, te = tr[:args.max_train], va[:max(8, args.max_train // 4)], te[:max(8, args.max_train // 4)]

    ds_tr = build_dataset(tr, tokenizer, cc["max_length"])
    ds_va = build_dataset(va, tokenizer, cc["max_length"])
    ds_te = build_dataset(te, tokenizer, cc["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(SIGNAL_CLASSES),
        problem_type="multi_label_classification",
        id2label=ID2CLASS, label2id=CLASS2ID)

    use_fp16 = bool(cc.get("fp16")) and torch.cuda.is_available()
    guardar_mejor = bool(cc.get("save_best", False))
    targs = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=cc["batch_size"],
        per_device_eval_batch_size=cc["batch_size"],
        gradient_accumulation_steps=cc["grad_accum"],
        learning_rate=float(cc["lr"]),
        weight_decay=cc["weight_decay"],
        fp16=use_fp16,
        eval_strategy="epoch",
        # Ver nota en train_ner.py: los checkpoints intermedios llenan el disco.
        save_strategy="epoch" if guardar_mejor else "no",
        save_total_limit=1 if guardar_mejor else None,
        load_best_model_at_end=guardar_mejor,
        metric_for_best_model="f1_macro" if guardar_mejor else None,
        logging_steps=25,
        report_to=[],
        seed=cfg["project"]["seed"],
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=make_metrics(),
    )
    trainer.train()
    test_metrics = trainer.evaluate(ds_te)
    print("[TEST CLS]", {k: round(v, 4) for k, v in test_metrics.items() if isinstance(v, float)})

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(Path(args.out_dir) / "test_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(test_metrics, fh, ensure_ascii=False, indent=2)
    print(f"[OK] modelo CLS -> {args.out_dir}")


if __name__ == "__main__":
    main()
