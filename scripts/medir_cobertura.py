"""¿Hace falta NLP si el portal ya publica los atributos tabulados?

LA PREGUNTA
-----------
Argenprop expone en cada pagina de detalle una ficha estructurada con amenities,
ambientes y servicios. Si esa ficha estuviera siempre completa, hacer NER sobre
la descripcion seria reinventar algo que ya existe.

Pero la ficha la completa **el publicador**, y no todos la completan. Este script
mide cuanto se pierde por confiar solo en ella:

  - Cuantos avisos directamente NO tienen ficha de amenities.
  - Cuantos amenities aparecen en la DESCRIPCION pero NO estan tildados.
  - Cuantos estan tildados pero no se mencionan en el texto.

El primer y el segundo numero son los que justifican (o no) la capa de NLP.

  python scripts/medir_cobertura.py --n 30

Sale: reports/cobertura_tabulada.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import argenprop as ap          # noqa: E402
from src.utils.config import load_config        # noqa: E402
from src.utils.io import read_jsonl             # noqa: E402

# Amenities canonicos y las formas en que aparecen escritos. Se comparan las dos
# fuentes con el MISMO vocabulario, para que la comparacion sea justa.
AMENITIES = {
    "pileta": ["pileta", "piscina"],
    "cochera": ["cochera", "garage", "garaje"],
    "parrilla": ["parrilla"],
    "gimnasio": ["gimnasio", "gym"],
    "sum": ["sum", "salon de fiestas", "salón de fiestas", "salon de usos"],
    "laundry": ["laundry", "lavadero"],
    "sauna": ["sauna"],
    "solarium": ["solarium", "solárium"],
    "baulera": ["baulera", "bauleras"],
    "balcon": ["balcon", "balcón"],
    "ascensor": ["ascensor"],
    "terraza": ["terraza"],
    "jacuzzi": ["jacuzzi", "hidromasaje"],
    "quincho": ["quincho"],
    "seguridad": ["seguridad", "vigilancia"],
}


def detectar(texto: str) -> set[str]:
    """Que amenities canonicos aparecen en un texto."""
    bajo = texto.lower()
    return {canon for canon, formas in AMENITIES.items()
            if any(f in bajo for f in formas)}


def main():
    apar = argparse.ArgumentParser()
    apar.add_argument("--n", type=int, default=30, help="cuantos avisos muestrear")
    apar.add_argument("--min-delay", type=float, default=30.0)
    apar.add_argument("--max-delay", type=float, default=50.0)
    args = apar.parse_args()

    cfg = load_config()
    avisos = [r for r in read_jsonl(ROOT / cfg["argenprop"]["out_path"]) if r.get("url")]
    rng = random.Random(cfg["project"]["seed"])
    muestra = rng.sample(avisos, min(args.n, len(avisos)))
    print(f"Midiendo cobertura sobre {len(muestra)} avisos de {len(avisos)}.\n")

    filas, fallidos = [], 0
    for i, aviso in enumerate(muestra, 1):
        try:
            html = ap.fetch_con_backoff(aviso["url"])
        except Exception as e:
            print(f"  [{i}/{len(muestra)}] ERROR: {str(e)[:70]}")
            fallidos += 1
            continue
        if not html.strip():
            print(f"  [{i}/{len(muestra)}] sin respuesta (el sitio nos esta limitando)")
            fallidos += 1
            if fallidos >= 5:
                print("\nDemasiadas respuestas vacias seguidas: se corta.")
                break
            continue

        ficha = ap.parse_detail_features(html)
        en_ficha = detectar(" ".join(ficha["tabulados"]))
        en_texto = detectar(aviso.get("description", ""))

        filas.append({
            "url": aviso["url"],
            "tiene_ficha": bool(ficha["tabulados"]),
            "en_ficha": sorted(en_ficha),
            "en_texto": sorted(en_texto),
            "solo_en_texto": sorted(en_texto - en_ficha),
            "solo_en_ficha": sorted(en_ficha - en_texto),
        })
        print(f"  [{i}/{len(muestra)}] ficha={len(en_ficha):2d} texto={len(en_texto):2d} "
              f"solo_en_texto={len(en_texto - en_ficha):2d}")

        if i < len(muestra):
            time.sleep(rng.uniform(args.min_delay, args.max_delay))

    if not filas:
        print("\nNo se pudo medir nada: el sitio no respondio. Reintentar mas tarde.")
        return

    n = len(filas)
    sin_ficha = sum(1 for f in filas if not f["tiene_ficha"])
    total_solo_texto = sum(len(f["solo_en_texto"]) for f in filas)
    total_solo_ficha = sum(len(f["solo_en_ficha"]) for f in filas)
    total_ficha = sum(len(f["en_ficha"]) for f in filas)
    total_texto = sum(len(f["en_texto"]) for f in filas)
    con_perdida = sum(1 for f in filas if f["solo_en_texto"])

    resumen = {
        "avisos_medidos": n,
        "avisos_sin_ficha_de_amenities": sin_ficha,
        "pct_sin_ficha": round(sin_ficha / n, 4),
        "amenities_en_ficha": total_ficha,
        "amenities_en_descripcion": total_texto,
        "amenities_SOLO_en_descripcion": total_solo_texto,
        "amenities_SOLO_en_ficha": total_solo_ficha,
        "avisos_con_al_menos_un_amenity_solo_en_texto": con_perdida,
        "pct_avisos_con_perdida": round(con_perdida / n, 4),
        "nota": ("'Solo en descripcion' = amenities que se perderian si uno confiara unicamente "
                 "en la ficha tabulada del portal. Es la medida de cuanto aporta la capa de NLP. "
                 "'Solo en ficha' es el caso inverso: el publicador lo tildo pero no lo escribio."),
    }

    out = ROOT / "reports" / "cobertura_tabulada.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"resumen": resumen, "detalle": filas},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"  Avisos medidos                          : {n}")
    print(f"  Sin ficha de amenities                  : {sin_ficha} ({sin_ficha/n:.0%})")
    print(f"  Amenities tildados en la ficha          : {total_ficha}")
    print(f"  Amenities mencionados en la descripcion : {total_texto}")
    print(f"  SOLO en la descripcion (se perderian)   : {total_solo_texto}")
    print(f"  SOLO en la ficha (no estan en el texto) : {total_solo_ficha}")
    print(f"  Avisos donde el texto aporta algo nuevo : {con_perdida} ({con_perdida/n:.0%})")
    print("=" * 72)
    print(f"[OK] reporte -> {out}")


if __name__ == "__main__":
    main()
