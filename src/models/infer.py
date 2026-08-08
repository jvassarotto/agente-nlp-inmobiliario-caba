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


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_entities(text, model_dir, max_length):
    """Reconoce entidades en el aviso COMPLETO, sea cual sea su largo.

    BETO no puede leer mas de 512 sub-tokens, asi que los avisos largos se
    procesan por ventanas deslizantes (ver src/models/chunking.py).
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from src.models.chunking import predecir_por_ventanas

    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval().to(dev)
    words = word_tokenize(text)
    etiquetas = predecir_por_ventanas(words, tok, model, max_length,
                                      model.config.id2label, device=dev)
    return group_entities(list(zip(words, etiquetas)))


def classify_signals(text, model_dir, max_length, threshold=0.5):
    """Clasifica el aviso COMPLETO, combinando ventanas si es largo."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from src.models.chunking import clasificar_por_ventanas

    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval().to(dev)
    probs = clasificar_por_ventanas(text, tok, model, max_length, device=dev)
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
