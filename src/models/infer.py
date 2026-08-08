"""Enriquecimiento: convierte una descripcion en texto libre en variables
estructuradas usando los dos modelos entrenados (NER + clasificacion).

Esta es la salida "de valor" del proyecto: texto -> atributos.

  python -m src.models.infer --text "Depto a estrenar, con pileta y cochera. Dueno directo."
"""
from __future__ import annotations
import argparse
import json

from src.utils.config import load_config
from src.utils.text import word_tokenize, group_entities


def extract_entities(text, model_dir, max_length):
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    words = word_tokenize(text)
    enc = tok(words, is_split_into_words=True, truncation=True,
              max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        pred_ids = model(**enc).logits[0].argmax(-1).tolist()
    id2label = model.config.id2label
    word_ids = enc.word_ids(0)
    labels, prev = [], None
    for idx, wid in enumerate(word_ids):
        if wid is None or wid == prev:
            prev = wid
            continue
        labels.append((words[wid], id2label[pred_ids[idx]]))
        prev = wid
    return group_entities(labels)


def classify_signals(text, model_dir, max_length, threshold=0.5):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch, numpy as np
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    enc = tok(text, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits[0].numpy()
    probs = 1 / (1 + np.exp(-logits))
    id2label = model.config.id2label
    return [id2label[i] for i, p in enumerate(probs) if p >= threshold]


def enrich(text, cfg):
    return {
        "description": text,
        "entities": extract_entities(text, cfg["ner"]["out_dir"], cfg["ner"]["max_length"]),
        "signals": classify_signals(text, cfg["classifier"]["out_dir"], cfg["classifier"]["max_length"]),
    }


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    args = ap.parse_args()
    print(json.dumps(enrich(args.text, cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
