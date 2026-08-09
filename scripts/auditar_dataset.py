"""Audita el dataset primario contra lo que promete la propuesta aprobada.

La propuesta dice, textual:

    "Por cada aviso se capturan campos estructurados (precio, superficie,
     ambientes, antiguedad, ubicacion) y el texto libre de la descripcion."

Este script verifica si eso efectivamente se cumple, campo por campo, y mide
una cosa mas: cuando el campo estructurado viene VACIO, ¿la descripcion lo
menciona igual? Esa es la contribucion concreta de la capa de NLP al dataset
tabular, y conviene medirla en vez de suponerla.

  python scripts/auditar_dataset.py

Sale: reports/auditoria_dataset.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl          # noqa: E402

# (etiqueta en la propuesta, campo en el esquema)
PROMETIDOS = [
    ("precio", "price_amount"),
    ("superficie", "surface_total_m2"),
    ("ambientes", "rooms"),
    ("antiguedad", "age_years"),
    ("ubicacion", "neighborhood"),
    ("texto libre", "description"),
]

EXTRA = [("moneda", "price_currency"), ("expensas", "expenses_amount"),
         ("dormitorios", "bedrooms"), ("banos", "bathrooms"),
         ("direccion", "address"), ("url", "url")]

# Formas en que un aviso menciona la antiguedad en prosa
PATRONES_ANTIGUEDAD = [
    r"\b\d{1,3}\s*a[nñ]os?\b",
    r"\bantig[uü]edad\b",
    r"\ba\s+estrenar\b",
    r"\ben\s+pozo\b",
    r"\bconstrucci[oó]n\s+\d{4}\b",
]


def lleno(rec: dict, campo: str) -> bool:
    return rec.get(campo) not in (None, "", [])


def completitud(recs: list[dict]) -> dict:
    n = len(recs)
    return {campo: {"etiqueta": etiqueta,
                    "completos": sum(1 for r in recs if lleno(r, campo)),
                    "total": n,
                    "pct": round(sum(1 for r in recs if lleno(r, campo)) / n, 4)}
            for etiqueta, campo in PROMETIDOS}


def rescate_por_texto(recs: list[dict]) -> dict:
    """Cuantos avisos sin `age_years` mencionan igual la antiguedad en el texto."""
    faltan = [r for r in recs if not lleno(r, "age_years")]
    rescatables = sum(
        1 for r in faltan
        if any(re.search(p, (r.get("description") or "").lower()) for p in PATRONES_ANTIGUEDAD))
    n = len(recs)
    return {
        "avisos_sin_campo": len(faltan),
        "mencionados_en_el_texto": rescatables,
        "pct_de_los_faltantes": round(rescatables / len(faltan), 4) if faltan else None,
        "cobertura_solo_campo": round((n - len(faltan)) / n, 4),
        "cobertura_sumando_texto": round((n - len(faltan) + rescatables) / n, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/real_caba.jsonl")
    args = ap.parse_args()

    ruta = ROOT / args.input
    if not ruta.exists():
        print(f"[ERROR] no existe {ruta}. Corre antes el scrape.")
        return
    recs = list(read_jsonl(ruta))

    por_fuente = {}
    for src in sorted({r.get("source", "?") for r in recs}):
        sub = [r for r in recs if r.get("source") == src]
        por_fuente[src] = completitud(sub)

    reporte = {
        "avisos": len(recs),
        "campos_prometidos_por_la_propuesta": completitud(recs),
        "campos_adicionales": {
            campo: round(sum(1 for r in recs if lleno(r, campo)) / len(recs), 4)
            for _, campo in EXTRA},
        "por_fuente": por_fuente,
        "aporte_del_texto_a_la_antiguedad": rescate_por_texto(recs),
    }

    out = ROOT / "reports" / "auditoria_dataset.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Dataset primario: {len(recs)} avisos\n")
    print("  Campos que promete la propuesta:")
    for etiqueta, campo in PROMETIDOS:
        d = reporte["campos_prometidos_por_la_propuesta"][campo]
        estado = "OK" if d["pct"] == 1 else ("PARCIAL" if d["pct"] else "FALTA")
        print(f"    {etiqueta:12s} {d['completos']:3d}/{d['total']:3d}  {d['pct']:5.0%}  {estado}")

    r = reporte["aporte_del_texto_a_la_antiguedad"]
    print(f"\n  Antiguedad — aporte de la capa de NLP:")
    print(f"    sin el campo estructurado           : {r['avisos_sin_campo']}")
    print(f"    de esos, mencionada en la descripcion: {r['mencionados_en_el_texto']} "
          f"({r['pct_de_los_faltantes']:.0%})")
    print(f"    cobertura {r['cobertura_solo_campo']:.0%} -> {r['cobertura_sumando_texto']:.0%} "
          f"sumando el texto")
    print(f"\n[OK] reporte -> {out}")


if __name__ == "__main__":
    main()
