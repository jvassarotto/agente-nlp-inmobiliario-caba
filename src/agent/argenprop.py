"""Fuente alternativa de avisos: Argenprop.

POR QUE EXISTE ESTE MODULO
--------------------------
ZonaProp resulto inviable para construir un dataset de tamanio util: protege el
sitio con Cloudflare y, medido empiricamente, **desafia el navegador automatizado
a partir de la segunda request de cada sesion** (ver README, seccion sobre el
anti-bot). Se descarto explicitamente evadir esa proteccion.

Argenprop, en cambio, **permite el acceso** en su robots.txt:

    Allow: /*?pagina-1$
    ...
    Allow: /*?pagina-10$
    Disallow: /*?pagina-

Es decir: las paginas 1 a 10 de cada busqueda estan expresamente habilitadas, y
de la 11 en adelante no. Ese limite se respeta **en el codigo** (MAX_PAGINA), no
solo en la documentacion.

Ventajas sobre las tarjetas de ZonaProp:
  - No hace falta navegador: responde a una request HTTP comun.
  - La tarjeta incluye **antiguedad**, que ZonaProp no daba.
  - 20 avisos por pagina.
"""
from __future__ import annotations

import hashlib
import re
import time
import urllib.request
from datetime import datetime, timezone

from bs4 import BeautifulSoup

BASE_URL = "https://www.argenprop.com"

# Limite impuesto por robots.txt (Allow hasta pagina-10, Disallow de ahi en mas).
# NO subir este numero: seria acceder a rutas que el sitio marca como no
# permitidas para agentes automatizados.
MAX_PAGINA = 10

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def fetch(url: str, timeout: int = 30) -> str:
    """Descarga una pagina. No hace falta navegador: es HTML servido de una."""
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept-Language": "es-AR,es;q=0.9",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_con_backoff(url: str, intentos: int = 4, espera_inicial: float = 30.0) -> str:
    """Descarga respetando el rate-limit del sitio.

    Argenprop responde **HTTP 202 con cuerpo vacio** cuando se le pide demasiado
    rapido: acepta la request pero no devuelve nada. No es un bloqueo duro, es
    una senial de "vas muy rapido".

    La respuesta correcta ante eso es **esperar mas**, no insistir ni disfrazar
    al cliente. Por eso el backoff es exponencial y arranca alto.
    """
    espera = espera_inicial
    for intento in range(1, intentos + 1):
        html = fetch(url)
        if html.strip():
            return html
        if intento < intentos:
            print(f"      (respuesta vacia: el sitio pide bajar el ritmo; "
                  f"esperando {espera:.0f}s antes de reintentar)")
            time.sleep(espera)
            espera *= 2
    return ""


def build_url(path: str, pagina: int = 1) -> str:
    """Arma la URL de una busqueda, respetando el tope de robots.txt."""
    if pagina > MAX_PAGINA:
        raise ValueError(
            f"pagina {pagina} excede el maximo permitido por robots.txt ({MAX_PAGINA})")
    url = BASE_URL.rstrip("/") + path
    return url if pagina <= 1 else f"{url}?pagina-{pagina}"


def _num(text: str):
    """Convierte '2.400.000' o '300' a float. Devuelve None si no hay numero."""
    if not text:
        return None
    m = re.search(r"\d[\d.]*", text.replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(".", ""))
    except ValueError:
        return None


def parse_card(node) -> dict:
    """Extrae un aviso completo de una tarjeta `.listing__item`."""
    def sel(css):
        el = node.select_one(css)
        return el.get_text(" ", strip=True) if el else ""

    a = node.find("a", href=True)
    href = a["href"] if a else ""
    url = BASE_URL + href if href.startswith("/") else href

    precio_txt = sel(".card__price")
    moneda = sel(".card__currency") or ("USD" if "USD" in precio_txt else None)
    # El precio viene pegado a la moneda: "USD 2.400.000"
    precio = _num(precio_txt.replace(moneda or "", ""))

    # "300 m² cubie. 3 dorm. 17 años"
    feats = sel(".card__main-features")

    def rx(patron):
        m = re.search(patron, feats, re.I)
        return m.group(1) if m else None

    superficie = _num(rx(r"([\d.]+)\s*m²"))
    dormitorios = rx(r"(\d+)\s*dorm")
    banios = rx(r"(\d+)\s*ba(?:n|ñ)o")
    antiguedad = rx(r"(\d+)\s*a(?:n|ñ)os?")

    # "Departamento en Venta en Palermo Chico, Palermo" -> barrio
    titulo_primario = sel(".card__title--primary")
    barrio = None
    if titulo_primario:
        m = re.search(r"\ben\s+([^,]+?)(?:,|$)", titulo_primario.split(" en Venta en ")[-1])
        barrio = (m.group(1) if m else titulo_primario.split(",")[-1]).strip()

    # Los ambientes no estan en las features pero si en el slug de la URL
    m_amb = re.search(r"(\d+)-ambientes", href)
    ambientes = m_amb.group(1) if m_amb else None

    # La descripcion en texto libre: titular + cuerpo
    titular = sel(".card__title")
    cuerpo = sel(".card__info")
    descripcion = " ".join(p for p in (titular, cuerpo) if p).strip()

    return {
        "id": hashlib.md5((url or descripcion[:64]).encode()).hexdigest()[:12],
        "url": url,
        "source": "argenprop",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "price_amount": precio,
        "price_currency": moneda,
        "expenses_amount": _num(sel(".card__expenses")),
        "surface_total_m2": superficie,
        "rooms": int(ambientes) if ambientes else None,
        "bedrooms": int(dormitorios) if dormitorios else None,
        "bathrooms": int(banios) if banios else None,
        "age_years": int(antiguedad) if antiguedad else None,
        "neighborhood": barrio,
        "address": sel(".card__address") or None,
        "title": titulo_primario or None,
        "description": descripcion,
    }


def parse_detail_features(html: str) -> dict:
    """Extrae la ficha ESTRUCTURADA de la pagina de detalle.

    Argenprop publica en el detalle unas listas `ul.property-features` con los
    amenities, ambientes y servicios que el publicador tildo. Sirve para
    responder una pregunta central del proyecto: **¿hace falta NLP sobre la
    descripcion, si el portal ya da los atributos tabulados?**

    Devuelve:
      - `tabulados`: todos los items de esas listas, normalizados a minusculas.
      - `clave_valor`: los pares tipo "Cant. Dormitorios: 3".
    """
    soup = BeautifulSoup(html, "lxml")
    tabulados: list[str] = []
    clave_valor: dict[str, str] = {}

    for ul in soup.select("ul.property-features"):
        for li in ul.find_all("li"):
            txt = li.get_text(" ", strip=True)
            if not txt:
                continue
            if ":" in txt:
                k, _, v = txt.partition(":")
                clave_valor[k.strip()] = v.strip()
            else:
                tabulados.append(txt.strip().lower())

    return {"tabulados": sorted(set(tabulados)), "clave_valor": clave_valor}


def parse_listings_con_conteo(html: str) -> tuple[int, list[dict]]:
    """Devuelve (tarjetas detectadas, avisos extraidos con exito).

    Los dos numeros hacen falta para la metrica del agente: la tasa de exito es
    extraidos / detectados. Una tarjeta detectada pero sin descripcion usable
    cuenta como fallo de extraccion.
    """
    soup = BeautifulSoup(html, "lxml")
    nodes = soup.select(".listing__item")
    if not nodes:
        nodes = soup.select(".card")   # fallback si cambia el contenedor
    out = []
    for n in nodes:
        rec = parse_card(n)
        if rec["description"] and rec["url"]:
            out.append(rec)
    return len(nodes), out


def parse_listings(html: str) -> list[dict]:
    """Devuelve los avisos de una pagina de listado."""
    return parse_listings_con_conteo(html)[1]
