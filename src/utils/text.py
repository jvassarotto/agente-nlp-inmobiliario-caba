"""Tokenizacion por palabra y utilidades de etiquetado BIO por matching.

Tokenizador unico para TODO el proyecto (generador sintetico + anotacion LLM),
asi los tokens son consistentes entre paths.

Reglas:
  - Numeros/montos con separadores se mantienen enteros: "$85.000", "30.8", "24".
  - Palabras (con acentos) sin puntuacion pegada: "cochera", "anios", "m2".
  - Puntuacion de oracion como token propio: "." "," ";" ":"
"""
from __future__ import annotations
import re

# Orden importa: primero montos/numeros, luego palabras, luego puntuacion.
_TOKEN_RE = re.compile(r"\$?\d[\d.,]*\d|\$?\d|\w+|[.,;:]", re.UNICODE)


def word_tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text or "")


def detokenize(tokens: list[str]) -> str:
    """Reconstruye texto legible pegando la puntuacion al token previo."""
    out: list[str] = []
    for t in tokens:
        if re.fullmatch(r"[.,;:]", t) and out:
            out[-1] = out[-1] + t
        else:
            out.append(t)
    return " ".join(out)


def tag_bio(tokens: list[str], phrases_by_label: dict[str, list[str]]) -> list[str]:
    """Asigna BIO a `tokens` buscando cada frase (substring) como sublista.

    Estrategia greedy: frases mas largas primero, no pisa tokens ya etiquetados.
    Case-insensitive.
    """
    tags = ["O"] * len(tokens)
    items: list[tuple[str, list[str]]] = []
    for label, phrases in phrases_by_label.items():
        for ph in phrases:
            pt = word_tokenize(ph)
            if pt:
                items.append((label, pt))
    items.sort(key=lambda x: len(x[1]), reverse=True)

    low = [t.lower() for t in tokens]
    for label, pt in items:
        L = len(pt)
        pt_low = [t.lower() for t in pt]
        i = 0
        while i <= len(tokens) - L:
            if low[i:i + L] == pt_low and all(tags[i + k] == "O" for k in range(L)):
                tags[i] = "B-" + label
                for k in range(1, L):
                    tags[i + k] = "I-" + label
                i += L
            else:
                i += 1
    return tags


def group_entities(pairs: list[tuple[str, str]]) -> list[dict]:
    """Agrupa una secuencia [(palabra, etiqueta_BIO)] en spans de entidad."""
    ents, cur, cur_lab = [], [], None
    for w, lab in pairs:
        if lab.startswith("B-"):
            if cur:
                ents.append({"type": cur_lab, "text": " ".join(cur)})
            cur, cur_lab = [w], lab[2:]
        elif lab.startswith("I-") and cur_lab == lab[2:]:
            cur.append(w)
        else:
            if cur:
                ents.append({"type": cur_lab, "text": " ".join(cur)})
            cur, cur_lab = [], None
    if cur:
        ents.append({"type": cur_lab, "text": " ".join(cur)})
    return ents
