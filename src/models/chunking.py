"""Procesamiento de avisos LARGOS con ventanas deslizantes.

EL PROBLEMA
-----------
BETO tiene un limite ARQUITECTONICO de 512 sub-tokens: son las posiciones que
aprendio durante su pre-entrenamiento, no un parametro que se pueda subir. Los
avisos reales llegan a 1.173 sub-tokens, asi que ni con el maximo se leerian
enteros: con max_length=512 entrarian completos 86 de 105.

LA SOLUCION
-----------
Partir el aviso en ventanas que si entren, pasar cada una por el modelo y unir
las predicciones. Asi el largo del aviso deja de ser un limite.

Dos detalles que hacen que funcione bien:

1. **Las ventanas se solapan.** Si se cortara en seco, una entidad que cae justo
   en el borde ("balcon | aterrazado") se partiria al medio y ninguna ventana la
   veria completa.

2. **Al unir, gana la ventana donde la palabra esta mas al centro.** Una palabra
   al borde de una ventana tiene contexto de un solo lado; la misma palabra, en
   la ventana siguiente, puede estar en el medio y con contexto completo. Esa
   prediccion es la que vale.
"""
from __future__ import annotations


def armar_ventanas(palabras: list[str], tokenizer, max_length: int,
                   solape: int = 24) -> list[tuple[int, int]]:
    """Divide una lista de palabras en ventanas [inicio, fin) que entren en el modelo.

    Devuelve indices sobre `palabras`. El presupuesto descuenta los tokens
    especiales ([CLS] y [SEP]) que agrega el tokenizador.
    """
    if not palabras:
        return []

    presupuesto = max_length - 2          # [CLS] y [SEP]
    # Sub-tokens que consume cada palabra. Se calcula una sola vez.
    costos = [max(1, len(tokenizer.tokenize(p))) for p in palabras]

    ventanas: list[tuple[int, int]] = []
    inicio = 0
    n = len(palabras)
    while inicio < n:
        total, fin = 0, inicio
        while fin < n and total + costos[fin] <= presupuesto:
            total += costos[fin]
            fin += 1
        if fin == inicio:      # una sola palabra ya excede el presupuesto
            fin = inicio + 1
        ventanas.append((inicio, fin))
        if fin >= n:
            break
        # La proxima ventana arranca antes del final de esta, para solaparse.
        inicio = max(fin - solape, inicio + 1)
    return ventanas


def predecir_por_ventanas(palabras: list[str], tokenizer, model, max_length: int,
                          id2label: dict, device: str = "cpu",
                          solape: int = 24) -> list[str]:
    """Etiqueta BIO para CADA palabra, sin importar el largo del aviso."""
    import torch

    ventanas = armar_ventanas(palabras, tokenizer, max_length, solape)
    # Por palabra: (centralidad de la mejor prediccion, etiqueta)
    mejor: list[tuple[int, str]] = [(-1, "O")] * len(palabras)

    for ini, fin in ventanas:
        trozo = palabras[ini:fin]
        enc = tokenizer(trozo, is_split_into_words=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        word_ids = enc.word_ids(0)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            pred_ids = model(**enc).logits[0].argmax(-1).tolist()

        prev = None
        for pos, wid in enumerate(word_ids):
            if wid is None or wid == prev:
                prev = wid
                continue
            prev = wid
            idx_global = ini + wid
            if idx_global >= len(palabras):
                continue
            # Que tan lejos esta la palabra de los bordes de esta ventana
            centralidad = min(wid, (fin - ini) - 1 - wid)
            if centralidad > mejor[idx_global][0]:
                mejor[idx_global] = (centralidad, id2label[pred_ids[pos]])

    return [etq for _, etq in mejor]


def clasificar_por_ventanas(texto: str, tokenizer, model, max_length: int,
                            device: str = "cpu") -> "list[float]":
    """Probabilidad de cada clase sobre un aviso de cualquier largo.

    Se toma el MAXIMO entre ventanas: si la senal aparece en alguna parte del
    aviso, el aviso la tiene. Promediar la diluiria — una mencion de "dueno
    directo" en un aviso largo quedaria ahogada por el resto del texto.
    """
    import numpy as np
    import torch

    palabras = texto.split()
    ventanas = armar_ventanas(palabras, tokenizer, max_length) or [(0, len(palabras))]
    probs_por_ventana = []

    for ini, fin in ventanas:
        trozo = " ".join(palabras[ini:fin])
        enc = tokenizer(trozo, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits[0].float().cpu().numpy()
        probs_por_ventana.append(1 / (1 + np.exp(-logits)))

    return np.max(np.vstack(probs_por_ventana), axis=0).tolist()
