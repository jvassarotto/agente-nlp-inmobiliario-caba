"""Esquemas de etiquetado para las dos tareas de NLP entrenables.

1) NER (token classification): reconoce atributos latentes dentro de la
   descripcion en texto libre (amenities, estado, antiguedad, orientacion,
   expensas).
2) Clasificacion (sequence classification, MULTILABEL): senales del
   vendedor / oportunidad a nivel de aviso.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# NER — tipos de entidad (esquema BIO)
# ---------------------------------------------------------------------------
ENTITY_TYPES = [
    "AMENITY",       # pileta, cochera, parrilla, balcon, sum, gimnasio, laundry...
    "ESTADO",        # a estrenar, refaccionado, a reciclar, excelente estado...
    "ANTIGUEDAD",    # "15 anios", "20 anos de antiguedad", "a estrenar"
    "ORIENTACION",   # frente, contrafrente, norte, sur, este, oeste...
    "EXPENSAS",      # "expensas $85.000", "expensas bajas"
]

# Etiquetas BIO derivadas: O + B-/I- por tipo
BIO_LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {lab: i for i, lab in enumerate(BIO_LABELS)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# Clasificacion — clases de senal del vendedor / oportunidad (multilabel)
# ---------------------------------------------------------------------------
SIGNAL_CLASSES = [
    "DUENO_DIRECTO",   # "dueno directo", "trato directo", "sin inmobiliaria"
    "OPORTUNIDAD",     # "oportunidad", "permuta", "escucho ofertas", "apto credito oportunidad"
    "URGENCIA",        # "venta urgente", "necesita vender", "recibo ofertas ya"
    "REFACCION",       # "a reciclar", "para refaccionar", "a poner a punto"
]
CLASS2ID = {c: i for i, c in enumerate(SIGNAL_CLASSES)}
ID2CLASS = {i: c for c, i in CLASS2ID.items()}
