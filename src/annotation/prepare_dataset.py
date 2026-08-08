"""Arma los splits train/val/test para NER y clasificacion.

Entrada: un JSONL con avisos que ya traen etiquetas gold, con campos
  - tokens, ner_tags   (NER)
  - description/text, signals  (clasificacion)
Puede ser el dataset sintetico (data/synthetic/listings.jsonl) o la salida de
la anotacion semiautomatica con LLM (data/annotated/*_llm.jsonl).

Salida (en data/annotated/):
  ner_train.jsonl / ner_val.jsonl / ner_test.jsonl   -> {tokens, ner_tags}
  cls_train.jsonl / cls_val.jsonl / cls_test.jsonl   -> {text, signals}
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

from src.utils.config import load_config
from src.utils.io import read_jsonl, write_jsonl


def split_indices(n, tr, va, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_tr = int(n * tr)
    n_va = int(n * va)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--input", default="data/synthetic/listings.jsonl",
                    help="JSONL con etiquetas gold (sintetico o LLM-anotado)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["_root"])
    recs = list(read_jsonl(root / args.input))
    ann = cfg["annotation"]

    tr, va, te = split_indices(len(recs), ann["train_split"], ann["val_split"],
                               cfg["project"]["seed"])

    def dump(name_ner, name_cls, idxs):
        ner = [{"id": recs[i]["id"], "tokens": recs[i]["tokens"],
                "ner_tags": recs[i]["ner_tags"]} for i in idxs]
        cls = [{"id": recs[i]["id"],
                "text": recs[i].get("description") or recs[i].get("text"),
                "signals": recs[i].get("signals", [])} for i in idxs]
        write_jsonl(ner, root / "data/annotated" / name_ner)
        write_jsonl(cls, root / "data/annotated" / name_cls)

    dump("ner_train.jsonl", "cls_train.jsonl", tr)
    dump("ner_val.jsonl", "cls_val.jsonl", va)
    dump("ner_test.jsonl", "cls_test.jsonl", te)
    print(f"[OK] splits -> train={len(tr)} val={len(va)} test={len(te)} (fuente: {args.input})")


if __name__ == "__main__":
    main()
