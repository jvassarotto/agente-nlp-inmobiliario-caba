"""Parser HTML -> campos estructurados + texto libre para avisos de ZonaProp.

Disenado para ser RESILIENTE a cambios de layout (uno de los problemas que
motiva el proyecto): intenta primero selectores `data-qa` estables y cae a
extraccion por regex sobre el texto plano cuando esos selectores fallan.

NOTA: los selectores de ZonaProp cambian con el tiempo y el portal tiene
proteccion anti-bot. Revisar/ajustar SELECTORS si baja la tasa de extraccion.
"""
from __future__ import annotations
import hashlib
import re
from datetime import datetime, timezone
from typing import Optional

from bs4 import BeautifulSoup

# Barrios de CABA para normalizar ubicacion desde el texto de direccion
CABA_NEIGHBORHOODS = [
    "Palermo", "Belgrano", "Caballito", "Recoleta", "Villa Urquiza", "Nunez",
    "Almagro", "Villa Crespo", "Flores", "Colegiales", "Saavedra", "Barracas",
    "San Telmo", "Puerto Madero", "Villa del Parque", "Boedo", "Chacarita",
    "Villa Devoto", "Parque Patricios", "Balvanera", "Retiro", "Coghlan",
    "Nunez", "Villa Luro", "Monserrat", "San Nicolas", "Constitucion",
]


def _num(text: str) -> Optional[float]:
    if not text:
        return None
    t = text.replace(".", "").replace(",", ".")
    m = re.search(r"\d+(?:\.\d+)?", t)
    return float(m.group(0)) if m else None


def parse_price(text: str) -> tuple[Optional[float], Optional[str]]:
    if not text:
        return None, None
    cur = "USD" if re.search(r"\bUSD|u\$s|US\$", text, re.I) else ("ARS" if "$" in text else None)
    return _num(text), cur


def parse_search_cards(html: str, base_url: str) -> list[dict]:
    """Extrae URLs (y datos rapidos) de los avisos en una pagina de listado."""
    soup = BeautifulSoup(html, "lxml")
    cards = []
    # Selector estable historicamente usado por ZonaProp
    for node in soup.select('[data-qa="posting PROPERTY"], div.postingCard, [data-to-posting]'):
        href = node.get("data-to-posting") or ""
        if not href:
            a = node.find("a", href=True)
            href = a["href"] if a else ""
        if href and href.startswith("/"):
            href = base_url.rstrip("/") + href
        if href:
            cards.append({"url": href})
    # Fallback: cualquier link a /propiedades/...
    if not cards:
        for a in soup.find_all("a", href=True):
            if "/propiedades/" in a["href"]:
                href = a["href"]
                if href.startswith("/"):
                    href = base_url.rstrip("/") + href
                cards.append({"url": href})
    # Dedup preservando orden
    seen, out = set(), []
    for c in cards:
        if c["url"] not in seen:
            seen.add(c["url"])
            out.append(c)
    return out


def parse_card(node, base_url: str = "") -> dict:
    """Extrae un aviso COMPLETO desde su tarjeta en la pagina de listado.

    ZonaProp protege las paginas de detalle con un challenge de Cloudflare, pero
    las tarjetas del listado ya traen todo lo necesario — incluida la descripcion
    completa (1.500-3.500 caracteres), que es el insumo de la capa de NLP.

    Extraer desde el listado ademas es mas cortes: una sola request devuelve
    25-30 avisos en lugar de uno.
    """
    def qa(name):
        el = node.select_one(f'[data-qa="{name}"]')
        return el.get_text(" ", strip=True) if el else ""

    href = node.get("data-to-posting") or ""
    if not href:
        a = node.find("a", href=True)
        href = a["href"] if a else ""
    if href.startswith("/"):
        href = base_url.rstrip("/") + href

    price, cur = parse_price(qa("POSTING_CARD_PRICE"))
    desc = qa("POSTING_CARD_DESCRIPTION")
    ubicacion = qa("POSTING_CARD_LOCATION")

    # "47 m2 tot. 2 amb. 1 dorm. 1 bano" -> campos numericos
    feats = qa("POSTING_CARD_FEATURES")

    def rx(pattern, texto=feats):
        m = re.search(pattern, texto, re.I)
        return m.group(1) if m else None

    surface = _num(rx(r"([\d\.,]+)\s*m²?\s*tot"))
    rooms = rx(r"(\d+)\s*amb")
    bedrooms = rx(r"(\d+)\s*dorm")
    bathrooms = rx(r"(\d+)\s*ba(?:n|ñ)o")

    expensas_txt = qa("expensas")
    barrio = ubicacion.split(",")[0].strip() if ubicacion else None
    if not barrio:
        barrio = next((n for n in CABA_NEIGHBORHOODS if n.lower() in desc.lower()), None)

    listing_id = hashlib.md5((href or desc[:64]).encode()).hexdigest()[:12]
    return {
        "id": listing_id,
        "url": href,
        "source": "zonaprop",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "price_amount": price,
        "price_currency": cur,
        "expenses_amount": _num(expensas_txt) if expensas_txt else None,
        "surface_total_m2": surface,
        "rooms": int(rooms) if rooms else None,
        "bedrooms": int(bedrooms) if bedrooms else None,
        "bathrooms": int(bathrooms) if bathrooms else None,
        "age_years": None,          # no aparece en la tarjeta
        "neighborhood": barrio,
        "title": ubicacion or None,
        "description": desc,
    }


def parse_search_listings(html: str, base_url: str) -> list[dict]:
    """Devuelve los avisos completos de una pagina de listado."""
    soup = BeautifulSoup(html, "lxml")
    nodes = soup.select('[data-qa="posting PROPERTY"]')
    if not nodes:
        # Fallback si cambia el data-qa de la tarjeta
        nodes = soup.select("div.postingCard, [data-to-posting]")
    out, seen = [], set()
    for n in nodes:
        rec = parse_card(n, base_url)
        if rec["description"] and rec["id"] not in seen:
            seen.add(rec["id"])
            out.append(rec)
    return out


def parse_detail(html: str, url: str = "") -> dict:
    """Extrae campos estructurados + descripcion de la pagina de detalle."""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    def qa(name):
        el = soup.select_one(f'[data-qa="{name}"]')
        return el.get_text(" ", strip=True) if el else ""

    price_txt = qa("POSTING_CARD_PRICE") or qa("adPrice") or ""
    price, cur = parse_price(price_txt or text[:120])

    expenses_txt = qa("expensas") or ""
    if not expenses_txt:
        m = re.search(r"expensas[^\d]{0,15}\$?\s*([\d\.]+)", text, re.I)
        expenses_txt = m.group(1) if m else ""

    # Descripcion (texto libre) — nucleo NLP
    desc = qa("longDescription") or qa("description")
    if not desc:
        el = soup.select_one("#longDescription, .section-description, [class*='description']")
        desc = el.get_text(" ", strip=True) if el else ""

    # Atributos rapidos via regex sobre el texto
    def rx(pattern):
        m = re.search(pattern, text, re.I)
        return m.group(1) if m else None

    surface_total = _num(rx(r"([\d\.]+)\s*m2?\s*(?:tot|total)?"))
    rooms = rx(r"(\d+)\s*ambiente")
    bedrooms = rx(r"(\d+)\s*dormitor")
    bathrooms = rx(r"(\d+)\s*ba(?:n|ñ)o")
    age = rx(r"antig[uü]edad[^\d]{0,10}(\d+)")

    neighborhood = next((n for n in CABA_NEIGHBORHOODS if n.lower() in text.lower()), None)

    listing_id = hashlib.md5((url or desc[:64]).encode()).hexdigest()[:12]
    return {
        "id": listing_id,
        "url": url,
        "source": "zonaprop",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "price_amount": price,
        "price_currency": cur,
        "expenses_amount": _num(expenses_txt) if expenses_txt else None,
        "surface_total_m2": surface_total,
        "rooms": int(rooms) if rooms else None,
        "bedrooms": int(bedrooms) if bedrooms else None,
        "bathrooms": int(bathrooms) if bathrooms else None,
        "age_years": int(age) if age else None,
        "neighborhood": neighborhood,
        "title": (soup.title.get_text(strip=True) if soup.title else None),
        "description": desc,
    }
