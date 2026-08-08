"""Revision manual del subconjunto pre-anotado por el LLM.

La propuesta plantea anotacion SEMIAUTOMATICA: el LLM pre-anota y despues hay
revision humana. Este script hace practicable esa revision y, sobre todo,
produce un numero que hay que reportar: la **tasa de acuerdo LLM vs humano**,
que dice cuanto se puede confiar en las etiquetas.

Muestra cada aviso con las entidades y senales que propuso el LLM, y se marca
si estan bien o mal. Revisar una muestra alcanza: con 50-80 avisos la tasa de
acuerdo ya tiene un error muestral chico.

  # revisar una muestra de 60 avisos
  python -m src.annotation.review --limit 60

  # exportar los conjuntos de evaluacion, sin revisar
  python -m src.annotation.review --solo-exportar

Salidas:
  data/annotated/reviewed.jsonl        avisos revisados (con marca de correccion)
  data/annotated/real_ner.jsonl        conjunto de evaluacion externa — NER
  data/annotated/real_cls.jsonl        conjunto de evaluacion externa — clasificacion
  reports/annotation_agreement.json    tasa de acuerdo LLM vs humano
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.utils.config import load_config
from src.utils.io import read_jsonl, write_jsonl
from src.utils.text import group_entities


def mostrar(rec: dict, i: int, total: int) -> None:
    print("\n" + "=" * 78)
    print(f"  Aviso {i + 1} de {total}    (id: {rec.get('id', '?')})")
    print("=" * 78)
    texto = rec.get("text", "")
    print(texto[:700] + ("..." if len(texto) > 700 else ""))

    ents = group_entities(list(zip(rec["tokens"], rec["ner_tags"])))
    print("\n  ENTIDADES propuestas por el LLM:")
    if ents:
        for e in ents:
            print(f"    {e['type']:14s} -> {e['text']}")
    else:
        print("    (ninguna)")

    print("\n  SENALES propuestas por el LLM:")
    print(f"    {', '.join(rec.get('signals', [])) or '(ninguna)'}")


def preguntar() -> str:
    """Devuelve 'ok', 'ner', 'cls', 'ambas' o 'saltar'."""
    print("\n  ¿Estan bien las etiquetas?")
    print("    [Enter] si    |    n = NER mal    |    c = senales mal")
    print("    a = ambas mal |    s = saltar     |    q = terminar")
    r = input("  > ").strip().lower()
    return {"": "ok", "n": "ner", "c": "cls", "a": "ambas", "s": "saltar", "q": "q"}.get(r, "ok")


def exportar_evaluacion(recs: list[dict], root: Path) -> None:
    """Deja el conjunto real en el formato que consume evaluate.py."""
    ner = [{"id": r.get("id", ""), "tokens": r["tokens"], "ner_tags": r["ner_tags"]}
           for r in recs]
    cls = [{"id": r.get("id", ""), "text": r["text"], "signals": r.get("signals", [])}
           for r in recs]
    write_jsonl(ner, root / "data/annotated/real_ner.jsonl")
    write_jsonl(cls, root / "data/annotated/real_cls.jsonl")
    print(f"[OK] conjunto de evaluacion real -> data/annotated/real_{{ner,cls}}.jsonl "
          f"({len(recs)} avisos)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/annotated/preannotated.jsonl")
    ap.add_argument("--limit", type=int, default=60,
                    help="cuantos avisos revisar a mano (muestra aleatoria)")
    ap.add_argument("--solo-exportar", action="store_true",
                    help="no revisa: solo genera los conjuntos de evaluacion")
    args = ap.parse_args()

    cfg = load_config()
    root = Path(cfg["_root"])
    src = root / args.input
    if not src.exists():
        print(f"[ERROR] no existe {src}.")
        print("        Corre antes:  python -m src.annotation.preannotate --input data/raw/zonaprop_caba.jsonl")
        return

    recs = list(read_jsonl(src))
    print(f"Cargados {len(recs)} avisos pre-anotados desde {args.input}")
    exportar_evaluacion(recs, root)
    if args.solo_exportar:
        return

    # Muestra aleatoria reproducible
    rng = random.Random(cfg["project"]["seed"])
    muestra = rng.sample(recs, min(args.limit, len(recs)))

    conteo = {"ok": 0, "ner": 0, "cls": 0, "ambas": 0, "saltar": 0}
    revisados = []
    for i, rec in enumerate(muestra):
        mostrar(rec, i, len(muestra))
        veredicto = preguntar()
        if veredicto == "q":
            print("\n  Revision interrumpida por el usuario.")
            break
        conteo[veredicto] += 1
        revisados.append({**rec, "revision": veredicto})

    n = sum(v for k, v in conteo.items() if k != "saltar")
    if n == 0:
        print("\nNo se reviso ningun aviso.")
        return

    # Acuerdo = el humano no corrigio nada de esa tarea
    acuerdo_ner = (conteo["ok"] + conteo["cls"]) / n
    acuerdo_cls = (conteo["ok"] + conteo["ner"]) / n
    acuerdo_total = conteo["ok"] / n

    resumen = {
        "avisos_revisados": n,
        "acuerdo_ner": round(acuerdo_ner, 4),
        "acuerdo_clasificacion": round(acuerdo_cls, 4),
        "acuerdo_ambas_tareas": round(acuerdo_total, 4),
        "detalle": conteo,
        "nota": ("Acuerdo = proporcion de avisos donde el revisor humano no corrigio "
                 "la propuesta del LLM para esa tarea."),
    }

    write_jsonl(revisados, root / "data/annotated/reviewed.jsonl")
    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotation_agreement.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"  Revisados            : {n}")
    print(f"  Acuerdo NER          : {acuerdo_ner:.1%}")
    print(f"  Acuerdo clasificacion: {acuerdo_cls:.1%}")
    print(f"  Acuerdo en ambas     : {acuerdo_total:.1%}")
    print("=" * 78)
    print(f"[OK] reporte -> reports/annotation_agreement.json")


if __name__ == "__main__":
    main()
