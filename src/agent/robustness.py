"""Medicion de la ROBUSTEZ del parser ante cambios de layout.

La propuesta (seccion 3.3) compromete evaluar la "robustez ante variaciones de
pagina". El problema que motiva el proyecto es justamente que los scrapers por
reglas fijas se rompen cuando el portal cambia el HTML.

La idea de la medicion: tomar un HTML que el parser SI sabe leer y aplicarle
degradaciones que imitan cambios reales de ZonaProp, midiendo que campos
sobreviven. `parser.py` esta escrito en dos niveles — selectores `data-qa`
primero, regex sobre texto plano despues — y esto cuantifica cuanto aporta el
segundo nivel.

Se puede correr sobre los fixtures reales (`data/fixtures/`) o, si no hay,
sobre una muestra sintetica incluida aca (asi el corrector puede reproducirlo
sin haber scrapeado nada).

  python -m src.agent.robustness
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.agent.parser import parse_detail

# Campos estructurados que nos interesa recuperar de cada aviso.
FIELDS = ["price_amount", "rooms", "bedrooms", "bathrooms",
          "age_years", "expenses_amount", "neighborhood", "description"]

# HTML de muestra con el layout "sano" (selectores data-qa presentes).
SAMPLE_HTML = """<html><head><title>Depto en Palermo</title></head><body>
  <span data-qa="POSTING_CARD_PRICE">USD 185.000</span>
  <div data-qa="longDescription">Hermoso 3 ambientes en Palermo, 75 m2 totales.
  2 dormitorios, 1 bano. Antiguedad 15 anios. Cuenta con pileta y cochera.
  Expensas $85.000. Dueno directo.</div></body></html>"""

SAMPLE_URL = "https://www.zonaprop.com.ar/propiedades/depto-palermo-123.html"


# --- Degradaciones: cada una imita un cambio de layout plausible -------------

def degrade_drop_data_qa(html: str) -> str:
    """El portal renombra/elimina los atributos data-qa (el cambio mas comun).

    Se conserva el contenido de texto: es lo que deben rescatar los fallbacks.
    """
    return re.sub(r'\sdata-qa="[^"]*"', "", html)


def degrade_rename_to_classes(html: str) -> str:
    """Migran de data-qa a clases CSS con nombres distintos."""
    html = html.replace('data-qa="longDescription"', 'class="section-description"')
    html = html.replace('data-qa="POSTING_CARD_PRICE"', 'class="price-tag"')
    return html


def degrade_strip_all_attributes(html: str) -> str:
    """Peor caso: se pierden TODOS los ganchos de markup, queda solo el texto."""
    return re.sub(r'\s(?:data-qa|class|id)="[^"]*"', "", html)


DEGRADATIONS = {
    "original": lambda h: h,
    "sin_data_qa": degrade_drop_data_qa,
    "renombrado_a_clases": degrade_rename_to_classes,
    "sin_ningun_atributo": degrade_strip_all_attributes,
}


# --- Evaluacion --------------------------------------------------------------

def _recovered(rec: dict) -> dict:
    """Que campos vinieron con un valor utilizable."""
    out = {}
    for f in FIELDS:
        v = rec.get(f)
        out[f] = bool(v) if f != "description" else bool(v and len(v.strip()) > 20)
    return out


def evaluate_html(html: str, url: str = SAMPLE_URL) -> dict:
    """Corre el parser sobre cada variante degradada del mismo HTML."""
    baseline = _recovered(parse_detail(html, url))
    # Solo tiene sentido medir la perdida sobre campos que el layout sano SI daba.
    relevant = [f for f, ok in baseline.items() if ok]

    results = {}
    for name, fn in DEGRADATIONS.items():
        rec = _recovered(parse_detail(fn(html), url))
        kept = [f for f in relevant if rec[f]]
        results[name] = {
            "campos_recuperados": kept,
            "campos_perdidos": [f for f in relevant if not rec[f]],
            "tasa_retencion": round(len(kept) / len(relevant), 4) if relevant else None,
        }
    return {"campos_base": relevant, "variantes": results}


def evaluate_fixtures(root: str | Path) -> dict:
    """Evalua sobre los fixtures reales si existen; si no, sobre la muestra."""
    fixtures = sorted((Path(root) / "data" / "fixtures").glob("detail_*.html"))
    if fixtures:
        per_file = {}
        for fx in fixtures:
            per_file[fx.name] = evaluate_html(fx.read_text(encoding="utf-8", errors="ignore"))
        fuente = f"fixtures reales ({len(fixtures)} avisos)"
    else:
        per_file = {"muestra_sintetica": evaluate_html(SAMPLE_HTML)}
        fuente = "muestra sintetica (no hay fixtures reales)"

    # Promedio de retencion por variante, a traves de los archivos evaluados.
    resumen = {}
    for variante in DEGRADATIONS:
        tasas = [v["variantes"][variante]["tasa_retencion"] for v in per_file.values()
                 if v["variantes"][variante]["tasa_retencion"] is not None]
        resumen[variante] = round(sum(tasas) / len(tasas), 4) if tasas else None
    return {"fuente": fuente, "retencion_promedio": resumen, "detalle": per_file}


def main():
    from src.utils.config import load_config
    root = load_config()["_root"]
    rep = evaluate_fixtures(root)

    print(f"Robustez del parser — fuente: {rep['fuente']}\n")
    for variante, tasa in rep["retencion_promedio"].items():
        print(f"  {variante:24s} retencion de campos: {tasa}")

    out_dir = Path(root) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "parser_robustness.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] reporte -> {out}")


if __name__ == "__main__":
    main()
