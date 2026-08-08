"""Revision manual del subconjunto pre-anotado por el LLM.

La propuesta plantea anotacion SEMIAUTOMATICA: el LLM pre-anota y despues hay
revision humana. Este script hace practicable esa revision y produce dos cosas:

  1. **Etiquetas gold humanas.** El revisor corrige las entidades y las senales,
     no solo las marca como buenas o malas. El resultado reemplaza al conjunto
     de evaluacion generado por el LLM, asi el F1 sobre datos reales se mide
     contra verdad de referencia y no contra otro modelo.

  2. **Tasa de acuerdo LLM vs humano.** Cuanto se puede confiar en el
     pre-anotador, que es un dato a reportar por si solo.

El progreso se guarda despues de CADA aviso: se puede cortar con 'q' y retomar
mas tarde sin perder lo hecho.

  python -m src.annotation.review --limit 60      # revisar
  python -m src.annotation.review --solo-exportar # exportar sin revisar

Salidas:
  data/annotated/reviewed.jsonl        avisos revisados, con etiquetas corregidas
  data/annotated/real_ner.jsonl        conjunto de evaluacion (gold humano)
  data/annotated/real_cls.jsonl        idem, para clasificacion
  reports/annotation_agreement.json    acuerdo LLM vs humano
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.utils.config import load_config
from src.utils.io import read_jsonl, write_jsonl
from src.utils.text import group_entities, tag_bio
from src.annotation.label_schema import ENTITY_TYPES, SIGNAL_CLASSES

SEP = "=" * 78


# --------------------------------------------------------------------------
# Presentacion
# --------------------------------------------------------------------------

def contexto(texto: str, frase: str, ancho: int = 34) -> str:
    """Devuelve la frase con un poco de texto alrededor, para ubicarla rapido."""
    pos = texto.lower().find(frase.lower())
    if pos < 0:
        return "(no se encuentra en el texto)"
    ini = max(0, pos - ancho)
    fin = min(len(texto), pos + len(frase) + ancho)
    izq = ("..." if ini > 0 else "") + texto[ini:pos]
    der = texto[pos + len(frase):fin] + ("..." if fin < len(texto) else "")
    return f"{izq}[{texto[pos:pos + len(frase)]}]{der}".replace("\n", " ")


def mostrar(rec: dict, i: int, total: int, ents: list[dict],
            originales: dict | None = None) -> None:
    print("\n" + SEP)
    print(f"  Aviso {i + 1} de {total}    (id: {rec.get('id', '?')})")
    print(SEP)
    # Se muestra el texto anotado COMPLETO. Truncarlo seria peligroso: habria
    # entidades etiquetadas en la parte no visible, y el revisor podria
    # borrarlas creyendo que estan inventadas.
    texto = rec.get("text", "")
    print(texto)
    print(f"\n  {'─' * 70}")
    print(f"  FIN DE LA PARTE ANOTADA ({len(texto)} caracteres).")

    # El resto del aviso se muestra solo como contexto: no se etiqueta porque
    # BETO, con max_length=192 sub-tokens, ni siquiera llega a leer hasta aca
    # (lee hasta el caracter ~870 en promedio).
    completo = (originales or {}).get(rec.get("id", ""), "")
    if completo and len(completo) > len(texto):
        resto = completo[len(texto):]
        print(f"  Sigue {len(resto)} caracteres mas, SOLO COMO CONTEXTO —")
        print(f"  no se etiquetan porque el modelo no llega a leerlos:")
        print(f"  {'─' * 70}")
        print(f"  {resto}")
    print(f"  {'─' * 70}")

    print("\n  ENTIDADES propuestas por el LLM (con su contexto):")
    if ents:
        for n, e in enumerate(ents, 1):
            print(f"    {n:2d}. {e['type']:12s} -> {e['text']}")
            print(f"        {contexto(texto, e['text'])}")
    else:
        print("     (ninguna)")

    print(f"\n  SENALES propuestas: {', '.join(rec.get('signals', [])) or '(ninguna)'}")


# --------------------------------------------------------------------------
# Correccion de entidades
# --------------------------------------------------------------------------

def pedir_correccion_ner(ents: list[dict]) -> tuple[list[dict], bool]:
    """Devuelve (entidades corregidas, hubo_cambios)."""
    print("\n  [NER] Numeros a BORRAR, separados por espacio. Enter = ninguno.")
    borrar = input("       borrar > ").strip()
    idx_borrar = set()
    if borrar:
        for t in borrar.split():
            if t.isdigit() and 1 <= int(t) <= len(ents):
                idx_borrar.add(int(t) - 1)

    quedan = [e for i, e in enumerate(ents) if i not in idx_borrar]

    print(f"  [NER] AGREGAR entidades. Formato: TIPO texto; TIPO texto")
    print(f"        Tipos: {' '.join(ENTITY_TYPES)}")
    print("        Enter = ninguna.")
    agregar = input("       agregar > ").strip()
    nuevas = []
    if agregar:
        for parte in agregar.split(";"):
            parte = parte.strip()
            if not parte:
                continue
            cabeza, _, texto = parte.partition(" ")
            tipo = cabeza.strip().upper()
            texto = texto.strip()
            if tipo in ENTITY_TYPES and texto:
                nuevas.append({"type": tipo, "text": texto})
            else:
                print(f"        (ignorado: '{parte}' — tipo desconocido o sin texto)")

    return quedan + nuevas, bool(idx_borrar or nuevas)


def pedir_correccion_cls(signals: list[str]) -> tuple[list[str], bool]:
    """Devuelve (senales corregidas, hubo_cambios)."""
    opciones = "  ".join(f"{n}={c}" for n, c in enumerate(SIGNAL_CLASSES, 1))
    print(f"\n  [SENALES] {opciones}")
    print("            Enter = estan bien | numeros = conjunto correcto | 0 = ninguna")
    r = input("       senales > ").strip()
    if not r:
        return signals, False
    if r == "0":
        return [], sorted(signals) != []
    elegidas = sorted({SIGNAL_CLASSES[int(t) - 1] for t in r.split()
                       if t.isdigit() and 1 <= int(t) <= len(SIGNAL_CLASSES)})
    return elegidas, elegidas != sorted(signals)


# --------------------------------------------------------------------------
# Persistencia
# --------------------------------------------------------------------------

def exportar_evaluacion(recs: list[dict], root: Path, etiqueta: str) -> None:
    ner = [{"id": r.get("id", ""), "tokens": r["tokens"], "ner_tags": r["ner_tags"]}
           for r in recs]
    cls = [{"id": r.get("id", ""), "text": r["text"], "signals": r.get("signals", [])}
           for r in recs]
    write_jsonl(ner, root / "data/annotated/real_ner.jsonl")
    write_jsonl(cls, root / "data/annotated/real_cls.jsonl")
    print(f"[OK] conjunto de evaluacion ({etiqueta}) -> data/annotated/real_{{ner,cls}}.jsonl "
          f"({len(recs)} avisos)")


def guardar_progreso(revisados: list[dict], conteo: dict, root: Path) -> dict:
    """Guarda despues de cada aviso, para poder cortar sin perder trabajo."""
    write_jsonl(revisados, root / "data/annotated/reviewed.jsonl")
    n = len(revisados)
    resumen = {
        "avisos_revisados": n,
        "acuerdo_ner": round(conteo["ner_ok"] / n, 4) if n else None,
        "acuerdo_clasificacion": round(conteo["cls_ok"] / n, 4) if n else None,
        "acuerdo_ambas_tareas": round(conteo["ambas_ok"] / n, 4) if n else None,
        "entidades_borradas": conteo["borradas"],
        "entidades_agregadas": conteo["agregadas"],
        "nota": ("Acuerdo = proporcion de avisos donde el revisor humano no tuvo que corregir "
                 "la propuesta del LLM para esa tarea. Las etiquetas de reviewed.jsonl son "
                 "GOLD HUMANO y reemplazan a las del LLM en el conjunto de evaluacion."),
    }
    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotation_agreement.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumen


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/annotated/preannotated.jsonl")
    ap.add_argument("--limit", type=int, default=60,
                    help="cuantos avisos revisar (muestra aleatoria reproducible)")
    ap.add_argument("--solo-exportar", action="store_true",
                    help="exporta las etiquetas del LLM sin revisar nada")
    args = ap.parse_args()

    cfg = load_config()
    root = Path(cfg["_root"])
    src = root / args.input
    if not src.exists():
        print(f"[ERROR] no existe {src}.")
        print("        Corre antes:  python -m src.annotation.preannotate "
              "--input data/raw/real_caba.jsonl")
        return

    recs = list(read_jsonl(src))
    print(f"Cargados {len(recs)} avisos pre-anotados desde {args.input}")

    # Texto original completo, para mostrarlo como contexto durante la revision.
    originales: dict[str, str] = {}
    crudo = root / "data/raw/real_caba.jsonl"
    if crudo.exists():
        originales = {r.get("id", ""): r.get("description", "") for r in read_jsonl(crudo)}

    if args.solo_exportar:
        exportar_evaluacion(recs, root, "etiquetas del LLM, sin revisar")
        return

    # Retomar una revision cortada a la mitad
    previos_path = root / "data/annotated/reviewed.jsonl"
    revisados = list(read_jsonl(previos_path)) if previos_path.exists() else []
    ya_vistos = {r.get("id") for r in revisados}
    if revisados:
        print(f"Retomando: ya habia {len(revisados)} avisos revisados.")

    rng = random.Random(cfg["project"]["seed"])
    muestra = [r for r in rng.sample(recs, min(args.limit, len(recs)))
               if r.get("id") not in ya_vistos]

    if not muestra:
        print("No queda nada por revisar con ese --limit.")
        exportar_evaluacion(revisados, root, "GOLD HUMANO")
        return

    print(f"\nQuedan {len(muestra)} avisos por revisar.")
    print("Se guarda despues de cada uno: podes cortar con 'q' y retomar despues.\n")

    conteo = {"ner_ok": 0, "cls_ok": 0, "ambas_ok": 0, "borradas": 0, "agregadas": 0}
    # Recontar lo ya revisado para que el acuerdo sea sobre el total
    for r in revisados:
        rev = r.get("_revision", {})
        conteo["ner_ok"] += int(not rev.get("cambio_ner", False))
        conteo["cls_ok"] += int(not rev.get("cambio_cls", False))
        conteo["ambas_ok"] += int(not rev.get("cambio_ner", False)
                                  and not rev.get("cambio_cls", False))

    for i, rec in enumerate(muestra):
        ents = group_entities(list(zip(rec["tokens"], rec["ner_tags"])))
        mostrar(rec, len(revisados), len(revisados) + len(muestra) - i, ents, originales)

        try:
            ents_ok, cambio_ner = pedir_correccion_ner(ents)
            signals_ok, cambio_cls = pedir_correccion_cls(rec.get("signals", []))
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Revision interrumpida. El progreso quedo guardado.")
            break

        # Re-etiquetar BIO a partir de las entidades corregidas
        frases = {t: [e["text"] for e in ents_ok if e["type"] == t] for t in ENTITY_TYPES}
        nuevos_tags = tag_bio(rec["tokens"], frases)

        # tag_bio solo puede etiquetar frases que aparezcan TEXTUALMENTE en el
        # aviso. Si el revisor escribio algo que no esta (una tilde de mas, una
        # palabra distinta), la entidad se perderia en silencio y el gold
        # quedaria mal sin que nadie se entere. Por eso se avisa.
        quedaron = {(e["type"], e["text"].lower())
                    for e in group_entities(list(zip(rec["tokens"], nuevos_tags)))}
        perdidas = [e for e in ents_ok
                    if (e["type"], e["text"].lower()) not in quedaron]
        if perdidas:
            print("\n  AVISO: estas entidades no quedaron marcadas:")
            for e in perdidas:
                print(f"           {e['type']} -> {e['text']}")
            print("         Puede ser por dos motivos:")
            print("          a) el texto no aparece TAL CUAL en el aviso (revisa tildes);")
            print("          b) otra entidad mas larga se superpone y gana — por ejemplo, si")
            print("             agregas 'al frente' se come al 'frente' que ya estaba.")
            print("         Si es el caso (b), esta todo bien: quedo la version mas completa.")

        conteo["borradas"] += max(0, len(ents) - len([e for e in ents_ok if e in ents]))
        conteo["agregadas"] += len([e for e in ents_ok if e not in ents])
        conteo["ner_ok"] += int(not cambio_ner)
        conteo["cls_ok"] += int(not cambio_cls)
        conteo["ambas_ok"] += int(not cambio_ner and not cambio_cls)

        revisados.append({**rec,
                          "ner_tags": nuevos_tags,
                          "signals": signals_ok,
                          "_revision": {"cambio_ner": cambio_ner, "cambio_cls": cambio_cls}})
        resumen = guardar_progreso(revisados, conteo, root)
        print(f"  guardado ({len(revisados)} revisados, "
              f"acuerdo NER {resumen['acuerdo_ner']:.0%})")

    resumen = guardar_progreso(revisados, conteo, root)
    exportar_evaluacion(revisados, root, "GOLD HUMANO")

    print("\n" + SEP)
    print(f"  Revisados             : {resumen['avisos_revisados']}")
    print(f"  Acuerdo NER           : {resumen['acuerdo_ner']:.1%}")
    print(f"  Acuerdo clasificacion : {resumen['acuerdo_clasificacion']:.1%}")
    print(f"  Acuerdo en ambas      : {resumen['acuerdo_ambas_tareas']:.1%}")
    print(SEP)
    print("\nAhora volve a evaluar contra el gold humano:")
    print("  python -m src.models.evaluate --task ner --model_dir models/ner-beto \\")
    print("      --input data/annotated/real_ner.jsonl --tag real")
    print("  python -m src.models.evaluate --task cls --model_dir models/cls-beto \\")
    print("      --input data/annotated/real_cls.jsonl --tag real")


if __name__ == "__main__":
    main()
