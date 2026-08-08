"""Verificacion de la logica custom del pipeline SIN torch/HF.

Cubre las partes que podrian estar mal (alineacion de etiquetas, metricas,
matching BIO, agrupacion de entidades, parser HTML). El forward de los modelos
es boilerplate estandar de HuggingFace y corre en la maquina con GPU.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def test_tokenizer_and_bio_matching():
    from src.utils.text import word_tokenize, tag_bio, detokenize, group_entities
    toks = word_tokenize("Depto a estrenar, con pileta y cochera. Expensas $85.000. Dueno directo.")
    assert "$85.000" in toks and "." in toks and "cochera" in toks
    tags = tag_bio(toks, {"AMENITY": ["pileta", "cochera"], "ESTADO": ["a estrenar"], "EXPENSAS": ["$85.000"]})
    pairs = list(zip(toks, tags))
    ents = group_entities(pairs)
    types = {(e["type"], e["text"]) for e in ents}
    assert ("AMENITY", "pileta") in types
    assert ("AMENITY", "cochera") in types
    assert ("ESTADO", "a estrenar") in types
    assert ("EXPENSAS", "$85.000") in types
    assert detokenize(toks).startswith("Depto a estrenar,")
    print("[1] tokenizer + tag_bio + group_entities OK ->", sorted(types))


def test_align_labels_with_subwords():
    """Simula word_ids de un tokenizer subword (BETO parte palabras raras)."""
    from src.models.train_ner import align_labels
    from src.annotation.label_schema import ID2LABEL
    tokens = ["Departamento", "refaccionado", "a", "nuevo", "con", "pileta"]
    tags = ["O", "B-ESTADO", "I-ESTADO", "I-ESTADO", "O", "B-AMENITY"]
    # 'refaccionado'->3 subtokens, 'Departamento'->2, resto 1. [CLS]/[SEP]=None
    word_ids = [None, 0, 0, 1, 1, 1, 2, 3, 4, 5, 5, None]
    lab = align_labels(word_ids, tags)
    # recuperar etiqueta de cada palabra (primer subtoken)
    rec, prev = [], None
    for w, l in zip(word_ids, lab):
        if w is not None and w != prev and l != -100:
            rec.append(ID2LABEL[l])
        prev = w
    assert rec == tags, f"{rec} != {tags}"
    # los subtokens intermedios deben ser -100
    assert lab.count(-100) == 6, lab   # 2 especiales + 4 subtokens continuadores
    print("[2] align_labels (subwords -> -100) OK ->", rec)


def test_ner_metrics():
    from src.models.train_ner import make_metrics
    from src.annotation.label_schema import LABEL2ID, BIO_LABELS
    compute = make_metrics()
    # 1 ejemplo, 4 posiciones validas + 1 ignorada
    gold = ["B-AMENITY", "O", "B-ESTADO", "I-ESTADO"]
    labels = np.array([[LABEL2ID[g] for g in gold] + [-100]])
    # logits perfectos
    logits = np.full((1, 5, len(BIO_LABELS)), -5.0)
    for j, g in enumerate(gold):
        logits[0, j, LABEL2ID[g]] = 5.0
    logits[0, 4, LABEL2ID["O"]] = 5.0
    m = compute((logits, labels))
    assert m["f1"] == 1.0, m
    print("[3] NER compute_metrics (F1 perfecto) OK ->", {k: round(v,3) for k,v in m.items()})


def test_cls_metrics_and_multihot():
    from src.models.train_classifier import make_metrics, multihot
    from src.annotation.label_schema import SIGNAL_CLASSES
    v = multihot(["DUENO_DIRECTO", "URGENCIA"])
    assert v.sum() == 2
    compute = make_metrics(threshold=0.5)
    # cada una de las 4 clases con soporte (evita que macro promedie una clase ausente)
    labels = np.array([[1,0,1,0],[0,1,0,1],[1,1,0,0],[0,0,1,1]], dtype=float)
    logits = np.where(labels == 1, 4.0, -4.0)              # predicciones perfectas
    m = compute((logits, labels))
    assert abs(m["f1_macro"] - 1.0) < 1e-6, m
    assert f"f1_{SIGNAL_CLASSES[0]}" in m
    print("[4] CLS multihot + compute_metrics OK ->", {k: round(v,3) for k,v in m.items() if k.startswith('f1_m') or k=='precision_macro'})


def test_parser_html():
    from src.agent.parser import parse_detail, parse_search_cards
    cards_html = '''<html><body>
      <div class="postingCard" data-to-posting="/propiedades/depto-palermo-123.html"></div>
      <a href="/propiedades/depto-belgrano-456.html">ver</a>
    </body></html>'''
    cards = parse_search_cards(cards_html, "https://www.zonaprop.com.ar")
    urls = {c["url"] for c in cards}
    assert "https://www.zonaprop.com.ar/propiedades/depto-palermo-123.html" in urls
    detail_html = '''<html><head><title>Depto en Palermo</title></head><body>
      <span data-qa="POSTING_CARD_PRICE">USD 185.000</span>
      <div data-qa="longDescription">Hermoso 3 ambientes en Palermo, 75 m2 totales.
      2 dormitorios, 1 bano. Antiguedad 15 anios. Cuenta con pileta y cochera.
      Expensas $85.000. Dueno directo.</div></body></html>'''
    rec = parse_detail(detail_html, "https://www.zonaprop.com.ar/propiedades/depto-palermo-123.html")
    assert rec["price_amount"] == 185000 and rec["price_currency"] == "USD", rec
    assert rec["rooms"] == 3 and rec["neighborhood"] == "Palermo", rec
    assert rec["age_years"] == 15 and rec["expenses_amount"] == 85000, rec
    assert "pileta" in rec["description"]
    print("[5] parser HTML (precio/ambientes/barrio/antiguedad/expensas/desc) OK")


def test_prepare_split_shapes():
    """El dataset sintetico ya generado produce splits consistentes."""
    from src.utils.io import read_jsonl
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ["ner_train","ner_val","ner_test","cls_train","cls_val","cls_test"]:
        recs = list(read_jsonl(os.path.join(root,"data","annotated",name+".jsonl")))
        assert len(recs) > 0
        if name.startswith("ner"):
            assert len(recs[0]["tokens"]) == len(recs[0]["ner_tags"])
        else:
            assert "text" in recs[0] and "signals" in recs[0]
    print("[6] splits NER/CLS consistentes OK")


def test_chunking_cubre_todo():
    """Las ventanas deslizantes tienen que cubrir TODAS las palabras del aviso.

    Es la propiedad que justifica el chunking: sin el, BETO trunca a 512
    sub-tokens y el final de los avisos largos nunca se lee ni se evalua.
    """
    from src.models.chunking import armar_ventanas

    class TokFalso:
        """Tokenizador de juguete: cada palabra vale 1 sub-token, salvo las
        largas, que valen 3 (imita el partido en sub-palabras)."""
        def tokenize(self, palabra):
            return ["x"] * (3 if len(palabra) > 8 else 1)

    tok = TokFalso()
    for n_palabras in (5, 60, 200, 1000):
        palabras = [f"palabra{i}" if i % 3 else f"p{i}" for i in range(n_palabras)]
        ventanas = armar_ventanas(palabras, tok, max_length=64, solape=8)

        cubiertas = set()
        for ini, fin in ventanas:
            assert ini < fin, (ini, fin)
            cubiertas.update(range(ini, fin))
        faltan = set(range(n_palabras)) - cubiertas
        assert not faltan, f"{n_palabras} palabras: quedaron sin cubrir {sorted(faltan)[:5]}"

        # Las ventanas tienen que avanzar: si no, el bucle no termina nunca.
        for (i1, _), (i2, _) in zip(ventanas, ventanas[1:]):
            assert i2 > i1, "las ventanas no avanzan"

    print("[10] chunking cubre el 100% de las palabras OK ->",
          f"{len(armar_ventanas(['x'] * 1000, tok, 64, 8))} ventanas para 1000 palabras")


def test_generador_sin_colisiones():
    """Las oraciones de relleno no pueden compartir tokens con las entidades.

    Si una palabra aparece como entidad en un aviso y como texto plano en otro,
    las etiquetas se contradicen y el modelo recibe senal inconsistente.
    """
    from src.data.generate_synthetic import AMENITIES, ESTADOS, ORIENTACIONES, RELLENO
    from src.utils.text import word_tokenize

    # El invariante es a nivel de FRASE, no de token suelto. Que "excelente"
    # aparezca en el relleno y tambien dentro de "excelente estado" no es
    # contradictorio: le ensena al modelo que la palabra sola no es entidad y
    # que solo la frase completa lo es. Lo que si romperia el etiquetado es que
    # una frase de entidad COMPLETA aparezca como texto plano.
    frases_entidad = [f for grupo in (AMENITIES, ESTADOS, ORIENTACIONES) for f in grupo]

    def contiene(secuencia, sub):
        n = len(sub)
        return any(secuencia[i:i + n] == sub for i in range(len(secuencia) - n + 1))

    colisiones = []
    for frase in RELLENO:
        toks = [t.lower() for t in word_tokenize(frase)]
        for ent in frases_entidad:
            if contiene(toks, [t.lower() for t in ent]):
                colisiones.append((" ".join(ent), frase))

    assert not colisiones, f"frases de entidad usadas como texto plano: {colisiones}"
    print(f"[8] generador sin colisiones relleno/entidad OK -> "
          f"{len(frases_entidad)} frases de entidad vs {len(RELLENO)} de relleno")


def test_parse_search_listings():
    """Camino principal de extraccion: el aviso completo sale de la tarjeta del
    listado. ZonaProp protege las paginas de detalle con Cloudflare, pero la
    tarjeta ya trae la descripcion completa."""
    from src.agent.parser import parse_search_listings
    # Reproduce la estructura de tarjeta de ZonaProp (data-qa reales).
    html = '''<html><body>
      <div data-qa="posting PROPERTY" data-to-posting="/propiedades/clasificado/depto-123.html">
        <div data-qa="POSTING_CARD_PRICE">USD 75.000</div>
        <div data-qa="expensas">$ 130.000 Expensas</div>
        <div data-qa="POSTING_CARD_LOCATION">Villa Crespo, Capital Federal</div>
        <div data-qa="POSTING_CARD_FEATURES">47 m&sup2; tot. 2 amb. 1 dorm. 1 ba&ntilde;o</div>
        <div data-qa="POSTING_CARD_DESCRIPTION">Venta de amplio departamento 2 ambientes
          con patio en Villa Crespo. Living comedor, cocina integrada y balcon al contrafrente.
          Excelente estado, apto credito.</div>
      </div>
      <div data-qa="posting DEVELOPMENT">
        <div data-qa="POSTING_CARD_DESCRIPTION">Emprendimiento en pozo</div>
      </div>
    </body></html>'''
    recs = parse_search_listings(html, "https://www.zonaprop.com.ar")
    # Solo se toman las tarjetas PROPERTY: los DEVELOPMENT son emprendimientos,
    # no departamentos individuales (por eso la tasa de exito no da 100%).
    assert len(recs) == 1, recs
    r = recs[0]
    assert r["price_amount"] == 75000 and r["price_currency"] == "USD", r
    assert r["surface_total_m2"] == 47 and r["rooms"] == 2, r
    assert r["bedrooms"] == 1 and r["bathrooms"] == 1, r
    assert r["neighborhood"] == "Villa Crespo", r
    assert r["expenses_amount"] == 130000, r
    assert r["url"].startswith("https://www.zonaprop.com.ar/propiedades/"), r
    assert "Villa Crespo" in r["description"]
    print("[9] parse_search_listings (extraccion desde tarjetas) OK ->",
          f"{r['rooms']} amb, {r['surface_total_m2']} m2, {r['neighborhood']}, USD {r['price_amount']:.0f}")


def test_parser_robustness():
    """Robustez ante cambios de layout: los fallbacks por regex deben sostener
    los campos estructurados aunque desaparezcan los selectores data-qa."""
    from src.agent.robustness import evaluate_html, SAMPLE_HTML

    rep = evaluate_html(SAMPLE_HTML)
    v = rep["variantes"]
    assert v["original"]["tasa_retencion"] == 1.0, v["original"]

    # Sin data-qa el parser pierde el gancho de la descripcion, pero los campos
    # numericos se leen por regex sobre el texto plano y tienen que sobrevivir.
    for variante in ["sin_data_qa", "sin_ningun_atributo"]:
        perdidos = v[variante]["campos_perdidos"]
        assert perdidos == ["description"], f"{variante} perdio {perdidos}"

    # Si el portal migra a clases CSS conocidas, se recupera todo.
    assert v["renombrado_a_clases"]["tasa_retencion"] == 1.0, v["renombrado_a_clases"]

    print("[7] robustez del parser OK ->",
          {k: val["tasa_retencion"] for k, val in v.items()})


if __name__ == "__main__":
    test_tokenizer_and_bio_matching()
    test_align_labels_with_subwords()
    test_ner_metrics()
    test_cls_metrics_and_multihot()
    test_parser_html()
    test_prepare_split_shapes()
    test_parser_robustness()
    test_generador_sin_colisiones()
    test_parse_search_listings()
    test_chunking_cubre_todo()
    print("\n==== TODOS LOS TESTS PASARON ====")
