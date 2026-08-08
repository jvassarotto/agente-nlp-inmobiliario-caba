"""Fine-tuning de BETO (BERT-base-spanish) para NER — token classification.

Reconoce atributos latentes en la descripcion: AMENITY, ESTADO, ANTIGUEDAD,
ORIENTACION, EXPENSAS (esquema BIO).

Ajustado para GPU chica (GTX 1650 4GB): fp16, batch chico, grad accumulation.

Uso:
  python -m src.models.train_ner
  python -m src.models.train_ner --base_model prajjwal1/bert-tiny --epochs 1 --max_train 64   # smoke test
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from src.utils.config import load_config, set_seed
from src.utils.io import read_jsonl
from src.annotation.label_schema import BIO_LABELS, LABEL2ID, ID2LABEL

def align_labels(word_ids, tags):
    """Alinea etiquetas BIO a subtokens: solo el primer subtoken de cada palabra
    lleva la etiqueta; el resto (y los especiales) van a -100 (ignorado en la loss)."""
    prev, lab_ids = None, []
    for wid in word_ids:
        if wid is None:
            lab_ids.append(-100)
        elif wid != prev:
            lab_ids.append(LABEL2ID[tags[wid]])
        else:
            lab_ids.append(-100)
        prev = wid
    return lab_ids



def load_split(root, name):
    return list(read_jsonl(Path(root) / "data/annotated" / name))


def build_dataset(records, tokenizer, max_length):
    from datasets import Dataset

    def encode(batch):
        enc = tokenizer(batch["tokens"], is_split_into_words=True,
                        truncation=True, max_length=max_length)
        labels = []
        for i, tags in enumerate(batch["ner_tags"]):
            labels.append(align_labels(enc.word_ids(batch_index=i), tags))
        enc["labels"] = labels
        return enc

    ds = Dataset.from_list([{"tokens": r["tokens"], "ner_tags": r["ner_tags"]} for r in records])
    return ds.map(encode, batched=True, remove_columns=ds.column_names)


def make_metrics():
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_labels, true_preds = [], []
        for p_row, l_row in zip(preds, labels):
            tl, tp = [], []
            for p, l in zip(p_row, l_row):
                if l != -100:
                    tl.append(ID2LABEL[int(l)])
                    tp.append(ID2LABEL[int(p)])
            true_labels.append(tl)
            true_preds.append(tp)
        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }
    return compute


def main():
    cfg = load_config()
    nc = cfg["ner"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=nc["base_model"])
    ap.add_argument("--epochs", type=float, default=nc["epochs"])
    ap.add_argument("--max_train", type=int, default=None, help="limitar train (smoke test)")
    ap.add_argument("--out_dir", default=nc["out_dir"])
    args = ap.parse_args()

    set_seed(cfg["project"]["seed"])
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                              DataCollatorForTokenClassification, TrainingArguments, Trainer)
    import torch

    root = cfg["_root"]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tr = load_split(root, "ner_train.jsonl")
    va = load_split(root, "ner_val.jsonl")
    te = load_split(root, "ner_test.jsonl")
    if args.max_train:
        tr, va, te = tr[:args.max_train], va[:max(8, args.max_train // 4)], te[:max(8, args.max_train // 4)]

    ds_tr = build_dataset(tr, tokenizer, nc["max_length"])
    ds_va = build_dataset(va, tokenizer, nc["max_length"])
    ds_te = build_dataset(te, tokenizer, nc["max_length"])

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(BIO_LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID)

    use_fp16 = bool(nc.get("fp16")) and torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=nc["batch_size"],
        per_device_eval_batch_size=nc["batch_size"],
        gradient_accumulation_steps=nc["grad_accum"],
        learning_rate=float(nc["lr"]),
        weight_decay=nc["weight_decay"],
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        # Sin este limite se acumula un checkpoint por epoch, y cada uno incluye
        # el estado del optimizador (~1.3 GB para BETO): 5 epochs llenaban el disco.
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=25,
        report_to=[],
        seed=cfg["project"]["seed"],
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,

        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=make_metrics(),
    )
    trainer.train()
    test_metrics = trainer.evaluate(ds_te)
    print("[TEST NER]", {k: round(v, 4) for k, v in test_metrics.items() if isinstance(v, float)})

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(Path(args.out_dir) / "test_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(test_metrics, fh, ensure_ascii=False, indent=2)
    print(f"[OK] modelo NER -> {args.out_dir}")


if __name__ == "__main__":
    main()
