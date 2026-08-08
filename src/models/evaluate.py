"""Evaluacion detallada de un modelo entrenado.

Por defecto evalua sobre el split de test del dataset con el que se entreno
(sintetico). Con `--input` se puede apuntar a OTRO conjunto — tipicamente el
set real anotado — para medir generalizacion sintetico -> real.

  # test sintetico (por defecto)
  python -m src.models.evaluate --task ner --model_dir models/ner-beto
  python -m src.models.evaluate --task cls --model_dir models/cls-beto

  # set real anotado (evaluacion externa)
  python -m src.models.evaluate --task ner --model_dir models/ner-beto \
      --input data/annotated/real_ner.jsonl --tag real

Cada corrida deja un reporte en reports/.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from src.utils.config import load_config
from src.utils.io import read_jsonl
from src.annotation.label_schema import ID2LABEL, SIGNAL_CLASSES, CLASS2ID


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def eval_ner(root, model_dir, max_length, input_rel):
    """F1 por entidad con seqeval (matching de SPAN completo, no por token).

    Evalua sobre TODAS las palabras del aviso: los avisos largos se procesan por
    ventanas deslizantes, asi que no queda texto sin evaluar por truncamiento.
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from seqeval.metrics import classification_report, f1_score
    from src.models.chunking import predecir_por_ventanas

    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval().to(dev)
    recs = list(read_jsonl(Path(root) / input_rel))
    y_true, y_pred = [], []
    for r in recs:
        etiquetas = predecir_por_ventanas(r["tokens"], tok, model, max_length,
                                          ID2LABEL, device=dev)
        y_true.append(list(r["ner_tags"]))
        y_pred.append(etiquetas)

    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    return report, {"f1_micro": float(f1_score(y_true, y_pred)), "n_ejemplos": len(recs)}


def eval_cls(root, model_dir, max_length, input_rel, threshold=0.5):
    """Precision/recall/F1 por clase + macro (multilabel)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import classification_report, f1_score
    from src.models.chunking import clasificar_por_ventanas

    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval().to(dev)
    recs = list(read_jsonl(Path(root) / input_rel))
    Y_true, Y_pred = [], []
    for r in recs:
        probs = np.array(clasificar_por_ventanas(r["text"], tok, model, max_length, device=dev))
        Y_pred.append((probs >= threshold).astype(int))
        v = np.zeros(len(SIGNAL_CLASSES), dtype=int)
        for s in r.get("signals", []):
            if s in CLASS2ID:
                v[CLASS2ID[s]] = 1
        Y_true.append(v)

    Y_true, Y_pred = np.array(Y_true), np.array(Y_pred)
    report = classification_report(Y_true, Y_pred, target_names=SIGNAL_CLASSES,
                                   digits=4, zero_division=0)
    print(report)
    return report, {
        "f1_macro": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "n_ejemplos": len(recs),
    }


def save_report(root, task, tag, model_dir, input_rel, report_text, resumen):
    out_dir = Path(root) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    titulo = "NER (token classification)" if task == "ner" else "Clasificacion multilabel"
    md = [
        f"# Evaluacion — {titulo}",
        "",
        f"- **Modelo:** `{model_dir}`",
        f"- **Conjunto evaluado:** `{input_rel}` ({tag})",
        f"- **Ejemplos:** {resumen['n_ejemplos']}",
        "",
        "## Reporte por clase",
        "",
        "```",
        report_text.rstrip(),
        "```",
        "",
        "## Resumen",
        "",
        "```json",
        json.dumps(resumen, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    out = out_dir / f"{task}_{tag}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    (out_dir / f"{task}_{tag}.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ner", "cls"], required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input", default=None,
                    help="JSONL a evaluar. Por defecto, el test split sintetico.")
    ap.add_argument("--tag", default="sintetico",
                    help="Etiqueta del conjunto, para nombrar el reporte.")
    args = ap.parse_args()

    root = cfg["_root"]
    default_input = f"data/annotated/{args.task}_test.jsonl"
    input_rel = args.input or default_input

    if args.task == "ner":
        report, resumen = eval_ner(root, args.model_dir, cfg["ner"]["max_length"], input_rel)
    else:
        report, resumen = eval_cls(root, args.model_dir, cfg["classifier"]["max_length"], input_rel)

    out = save_report(root, args.task, args.tag, args.model_dir, input_rel, report, resumen)
    print(f"[OK] reporte -> {out}")


if __name__ == "__main__":
    main()
