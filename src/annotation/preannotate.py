"""Anotacion SEMIAUTOMATICA con LLM local (Ollama) sobre descripciones reales.

Implementa el patron de la propuesta: el propio LLM actua como PRE-ANOTADOR y
luego queda la revision manual. Produce, para cada aviso:
  - NER  : etiquetas BIO alineadas a tokens (via matching de substrings)
  - CLS  : clases de senal del vendedor (multilabel)

Salida: data/annotated/preannotated.jsonl con {id, tokens, ner_tags, text, signals}
Ese archivo se revisa manualmente y luego se pasa a prepare_dataset.py.

Uso:
  python -m src.annotation.preannotate --input data/raw/zonaprop_caba.jsonl --limit 1200
"""
from __future__ import annotations
import argparse
from pathlib import Path

from tqdm import tqdm

from src.utils.config import load_config
from src.utils.io import read_jsonl, write_jsonl
from src.utils.text import word_tokenize, tag_bio
from src.annotation.label_schema import ENTITY_TYPES, SIGNAL_CLASSES
from src.annotation.llm_client import get_llm, parse_json

NER_PROMPT = """Sos un anotador experto de avisos inmobiliarios en espanol (CABA).
Extrae de la DESCRIPCION las menciones EXACTAS (copiadas textualmente) de estos tipos:
- AMENITY: comodidades (pileta, cochera, parrilla, balcon, sum, gimnasio, laundry, baulera, terraza...).
- ESTADO: estado de la propiedad (a estrenar, excelente estado, a reciclar, refaccionado...).
- ANTIGUEDAD: antiguedad (ej. "10 anios", "a estrenar", "20 anos de antiguedad").
- ORIENTACION: orientacion o disposicion (frente, contrafrente, norte, sur, este, oeste).
- EXPENSAS: menciones de expensas (ej. "$85.000", "expensas bajas").

Devolve SOLO un JSON con esta forma exacta (listas de strings copiados tal cual del texto):
{{"AMENITY": [], "ESTADO": [], "ANTIGUEDAD": [], "ORIENTACION": [], "EXPENSAS": []}}

DESCRIPCION:
{desc}
"""

CLS_PROMPT = """Sos un analista de avisos inmobiliarios en espanol (CABA).
Indica que SENALES DEL VENDEDOR estan presentes en la DESCRIPCION:
- DUENO_DIRECTO: vende el dueno / trato directo / sin inmobiliaria.
- OPORTUNIDAD: oportunidad / permuta / escucho ofertas.
- URGENCIA: venta urgente / necesita vender rapido.
- REFACCION: requiere refaccion / a reciclar / para poner a punto.

Devolve SOLO un JSON: {{"signals": ["..."]}} con las que apliquen (puede ser vacio).

DESCRIPCION:
{desc}
"""


def annotate_one(llm, desc: str) -> dict:
    tokens = word_tokenize(desc)

    # --- NER ---
    ner_raw = parse_json(llm.invoke(NER_PROMPT.format(desc=desc)).content)
    phrases = {t: [s for s in ner_raw.get(t, []) if isinstance(s, str)] for t in ENTITY_TYPES}
    ner_tags = tag_bio(tokens, phrases)

    # --- CLS ---
    cls_raw = parse_json(llm.invoke(CLS_PROMPT.format(desc=desc)).content)
    signals = [s for s in cls_raw.get("signals", []) if s in SIGNAL_CLASSES]

    return {"tokens": tokens, "ner_tags": ner_tags, "text": desc, "signals": sorted(set(signals))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--input", default="data/raw/zonaprop_caba.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["_root"])
    limit = args.limit or cfg["annotation"]["n_to_annotate"]

    llm = get_llm(cfg)
    out = []
    recs = list(read_jsonl(root / args.input))[:limit]
    for r in tqdm(recs, desc="Pre-anotando (LLM local)"):
        desc = r.get("description") or r.get("text") or ""
        if not desc.strip():
            continue
        ann = annotate_one(llm, desc)
        ann["id"] = r.get("id", "")
        out.append(ann)

    out_path = root / "data/annotated/preannotated.jsonl"
    write_jsonl(out, out_path)
    print(f"[OK] {len(out)} avisos pre-anotados -> {out_path}")
    print("     Revisa manualmente y luego: python -m src.annotation.prepare_dataset --input data/annotated/preannotated.jsonl")


if __name__ == "__main__":
    main()
