"""Generador de avisos SINTETICOS de departamentos en venta en CABA.

Objetivo: permitir correr TODO el pipeline de NLP (anotacion, fine-tuning y
evaluacion de NER + clasificacion) SIN depender del scrape real de ZonaProp.

Cada aviso generado trae, ademas de los campos estructurados y la descripcion
en texto libre:
  - tokens        : lista de palabras (tokenizacion por palabra)
  - ner_tags      : etiqueta BIO alineada a cada token  (GOLD para NER)
  - signals       : clases de senal del vendedor presentes (GOLD multilabel)

Como controlamos la generacion, las etiquetas son "gold" por construccion,
lo que permite entrenar y medir F1 real sin anotacion manual. En datos reales
esas etiquetas provienen de la anotacion semiautomatica (ver src/annotation).
"""
from __future__ import annotations
import argparse
import hashlib
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from src.utils.config import load_config, set_seed
from src.utils.io import write_jsonl
from src.annotation.label_schema import ENTITY_TYPES, SIGNAL_CLASSES
from src.utils.text import word_tokenize, detokenize

# --------------------------------------------------------------------------
# Vocabularios (espanol rioplatense, dominio inmobiliario CABA)
# --------------------------------------------------------------------------
NEIGHBORHOODS = [
    "Palermo", "Belgrano", "Caballito", "Recoleta", "Villa Urquiza", "Núñez",
    "Almagro", "Villa Crespo", "Flores", "Colegiales", "Saavedra", "Barracas",
    "San Telmo", "Puerto Madero", "Villa del Parque", "Boedo", "Chacarita",
    "Villa Devoto", "Parque Patricios", "Balvanera", "Retiro", "Coghlan",
]

# Cada amenity es una lista de tokens (puede ser multi-palabra).
# Con tildes, igual que los avisos reales: BETO es un modelo *cased* y su
# vocabulario distingue "balcón" de "balcon".
AMENITIES = [
    ["pileta"], ["cochera"], ["parrilla"], ["balcón"], ["balcón", "aterrazado"],
    ["SUM"], ["gimnasio"], ["laundry"], ["solárium"], ["baulera"], ["terraza"],
    ["jacuzzi"], ["sauna"], ["seguridad", "24", "horas"], ["cochera", "cubierta"],
    ["pileta", "climatizada"], ["sala", "de", "juegos"], ["espacio", "guardacoches"],
    ["quincho"], ["parrilla", "propia"], ["balcón", "corrido"], ["cochera", "fija"],
]
ESTADOS = [
    ["a", "estrenar"], ["excelente", "estado"], ["muy", "buen", "estado"],
    ["refaccionado", "a", "nuevo"], ["a", "reciclar"], ["para", "refaccionar"],
    ["impecable"], ["en", "perfecto", "estado"], ["a", "refaccionar"],
    ["recientemente", "reciclado"],
]
ORIENTACIONES = [
    ["al", "frente"], ["contrafrente"], ["orientación", "norte"],
    ["al", "norte"], ["al", "sur"], ["luminoso", "al", "este"],
    ["orientación", "noroeste"], ["al", "contrafrente"], ["vista", "abierta", "al", "oeste"],
]

# --------------------------------------------------------------------------
# DISTRACTORES: sustantivos que NO son entidades.
#
# Sin esto el modelo aprende "sustantivo despues de 'Cuenta con' = AMENITY" en
# vez del vocabulario real de amenities, y sobre texto real termina etiquetando
# cosas como "universidades" o "ventilacion". Estas oraciones le ensenan a
# predecir "O".
#
# INVARIANTE: ningun token de aca puede aparecer en AMENITIES / ESTADOS /
# ORIENTACIONES, o las etiquetas quedarian contradictorias entre avisos.
# Lo verifica test_generador_sin_colisiones() en tests/test_pipeline.py.
# --------------------------------------------------------------------------
RELLENO = [
    "Ingreso por un amplio living comedor con muy buena ventilación natural.",
    "Cocina integrada con mesada de granito y bajo mesada.",
    "Dormitorio principal con placard empotrado y vestidor.",
    "El edificio cuenta con ascensor y palier de distribución.",
    "A pocas cuadras del subte, colectivos y avenidas principales.",
    "Zona con universidades, colegios, comercios y supermercados.",
    "Muy próximo a plazas, hospitales y centros comerciales.",
    "Lavadero independiente con conexión para lavarropas.",
    "Toilette de recepción y baño completo con bañera.",
    "Excelente luminosidad durante todo el día.",
    "Consultar disponibilidad para visitas y coordinación de horarios.",
    "Apto crédito hipotecario. Se aceptan consultas por WhatsApp.",
    "Piso de parquet en dormitorios y cerámica en áreas húmedas.",
    "Calefacción por radiadores y agua caliente central.",
    "Unidad al día con las cuotas y libre deuda.",
]

# --------------------------------------------------------------------------
# Utilidades de construccion token->BIO
# --------------------------------------------------------------------------
class Builder:
    """Acumula tokens y sus etiquetas BIO mientras se arma la descripcion."""
    def __init__(self):
        self.tokens: list[str] = []
        self.tags: list[str] = []

    def add_plain(self, text: str):
        for tok in word_tokenize(text):
            self.tokens.append(tok)
            self.tags.append("O")

    def add_entity(self, toks: list[str], label: str):
        for i, tok in enumerate(toks):
            self.tokens.append(tok)
            self.tags.append(("B-" if i == 0 else "I-") + label)

    def text(self) -> str:
        return detokenize(self.tokens)




# --------------------------------------------------------------------------
# Generacion de un aviso
# --------------------------------------------------------------------------
def make_listing(rng: random.Random, idx: int) -> dict:
    barrio = rng.choice(NEIGHBORHOODS)
    rooms = rng.choice([1, 2, 2, 3, 3, 4])
    covered = round(rng.uniform(28, 140), 1)
    total = round(covered + rng.uniform(0, 20), 1)
    age = rng.choice([0, 0, 5, 10, 15, 20, 30, 40, 50])
    currency = "USD"
    price = int(rng.uniform(70000, 420000) / 1000) * 1000
    expenses = rng.choice([45000, 60000, 85000, 120000, 150000, 0])

    b = Builder()
    signals: set[str] = set()

    # Cada bloque se arma como una funcion y despues se BARAJA el orden. Si el
    # orden fuera siempre el mismo, el modelo aprenderia la posicion en vez del
    # contenido, y no transferiria a avisos reales.
    def bloque_estado():
        estado = rng.choice(ESTADOS)
        b.add_plain(rng.choice(["Propiedad", "Unidad", "Departamento", "Se vende"]))
        b.add_entity(estado, "ESTADO")
        b.add_plain(".")
        if " ".join(estado) in ("a reciclar", "para refaccionar", "a refaccionar"):
            signals.add("REFACCION")

    def bloque_antiguedad():
        if age == 0:
            b.add_plain(rng.choice(["Unidad", "Propiedad"]))
            b.add_entity(["a", "estrenar"], "ANTIGUEDAD")
            b.add_plain(".")
        else:
            b.add_plain(rng.choice(["Antigüedad", "Edificio de", "Con una antigüedad de"]))
            b.add_entity([str(age), "años"], "ANTIGUEDAD")
            b.add_plain(".")

    def bloque_orientacion():
        b.add_plain(rng.choice(["Departamento", "Unidad ubicada", "La propiedad da"]))
        b.add_entity(rng.choice(ORIENTACIONES), "ORIENTACION")
        b.add_plain(".")

    def bloque_amenities():
        chosen = rng.sample(AMENITIES, rng.randint(1, 4))
        b.add_plain(rng.choice(["Cuenta con", "Amenities:", "El edificio ofrece",
                                "Dispone de", "Incluye"]))
        for i, am in enumerate(chosen):
            b.add_entity(am, "AMENITY")
            if i < len(chosen) - 1:
                b.add_plain(",")
        b.add_plain(".")

    def bloque_expensas():
        if expenses and rng.random() < 0.8:
            b.add_plain(rng.choice(["Expensas", "Expensas aproximadas de", "Gastos comunes"]))
            b.add_entity([f"${expenses:,}".replace(",", ".")], "EXPENSAS")
            b.add_plain("por mes.")
        elif rng.random() < 0.5:
            b.add_plain("Con")
            b.add_entity(["expensas", "bajas"], "EXPENSAS")
            b.add_plain(".")

    def bloque_relleno():
        """Oraciones SIN entidades: le ensenan al modelo a predecir 'O'."""
        for frase in rng.sample(RELLENO, rng.randint(2, 5)):
            b.add_plain(frase)

    # Apertura (siempre primero, como en los avisos reales)
    b.add_plain(rng.choice([
        f"Venta de departamento de {rooms} ambientes en {barrio}.",
        f"Excelente departamento de {rooms} ambientes en el corazón de {barrio}.",
        f"Se vende {rooms} ambientes en {barrio}, Capital Federal.",
    ]))

    bloques = [bloque_relleno]
    if rng.random() < 0.9:
        bloques.append(bloque_estado)
    if rng.random() < 0.8:
        bloques.append(bloque_antiguedad)
    if rng.random() < 0.75:
        bloques.append(bloque_orientacion)
    bloques.append(bloque_amenities)
    bloques.append(bloque_expensas)
    if rng.random() < 0.6:
        bloques.append(bloque_relleno)   # mas relleno: los avisos reales son largos

    rng.shuffle(bloques)
    for bl in bloques:
        bl()

    # Senales del vendedor (clasificacion multilabel)
    if rng.random() < 0.35:
        b.add_plain(rng.choice(["Dueño directo.", "Trato directo, sin inmobiliaria.",
                                "Vende dueño.", "Sin comisión inmobiliaria."]))
        signals.add("DUENO_DIRECTO")
    if rng.random() < 0.30:
        b.add_plain(rng.choice(["Oportunidad única.", "Escucho ofertas.",
                                "Apto permuta por menor valor.", "Excelente oportunidad de inversión."]))
        signals.add("OPORTUNIDAD")
    if rng.random() < 0.20:
        b.add_plain(rng.choice(["Venta urgente.", "Necesito vender ya.", "Urge la venta.",
                                "Se vende con urgencia por mudanza."]))
        signals.add("URGENCIA")

    # Cierre
    b.add_plain(f"Superficie cubierta {covered} m2. Excelente ubicación en {barrio}.")

    text = b.text()
    listing_id = hashlib.md5(f"{idx}-{text}".encode()).hexdigest()[:12]

    return {
        "id": listing_id,
        "url": f"https://www.zonaprop.com.ar/propiedades/synthetic-{idx}.html",
        "source": "synthetic",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "price_amount": float(price),
        "price_currency": currency,
        "expenses_amount": float(expenses) if expenses else None,
        "surface_total_m2": total,
        "surface_covered_m2": covered,
        "rooms": rooms,
        "age_years": age,
        "neighborhood": barrio,
        "title": f"Departamento {rooms} amb en {barrio}",
        "description": text,
        # --- GOLD para NLP ---
        "tokens": b.tokens,
        "ner_tags": b.tags,
        "signals": sorted(signals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=None, help="Cantidad de avisos (override config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    rng = random.Random(cfg["project"]["seed"])

    n = args.n or cfg["synthetic"]["n_listings"]
    out_dir = Path(cfg["_root"]) / cfg["synthetic"]["out_dir"]
    records = [make_listing(rng, i) for i in range(n)]

    out_path = out_dir / "listings.jsonl"
    write_jsonl(records, out_path)

    # Chequeo rapido de balance de clases
    from collections import Counter
    cnt = Counter()
    for r in records:
        for s in r["signals"]:
            cnt[s] += 1
    print(f"[OK] {n} avisos sinteticos -> {out_path}")
    print(f"     Senales (multilabel): {dict(cnt)}")
    ent = Counter(t.split('-')[1] for r in records for t in r['ner_tags'] if t != 'O')
    print(f"     Entidades NER (tokens B/I): {dict(ent)}")


if __name__ == "__main__":
    main()
