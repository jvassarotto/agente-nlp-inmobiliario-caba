"""Herramientas de navegador (Playwright) expuestas como tools de LangChain.

El agente ReAct usa estas tools para sortear el contenido dinamico y la
proteccion anti-bot de ZonaProp. La sesion de navegador es un singleton a
nivel de modulo para reutilizar cookies entre pasos.

Requiere:  playwright install chromium   (una sola vez)
"""
from __future__ import annotations
import random
import time
from typing import Optional

from langchain_core.tools import tool

from src.utils.config import load_config
from src.agent.parser import parse_search_cards, parse_detail, parse_search_listings
from src.agent import metrics

_FULL_CFG = load_config()
_CFG = _FULL_CFG["scrape"]
_ROOT = _FULL_CFG["_root"]
_STATE: dict = {"pw": None, "browser": None, "page": None}

# Buffer donde el parser deja los avisos extraidos (lo consume run_scrape)
EXTRACTED: list[dict] = []
SEEN_URLS: set[str] = set()


def _polite_sleep():
    time.sleep(random.uniform(_CFG["min_delay_s"], _CFG["max_delay_s"]))


def _ensure_browser():
    if _STATE["page"] is not None:
        return _STATE["page"]
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=_CFG["headless"])
    ctx = browser.new_context(
        user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        locale="es-AR",
        viewport={"width": 1366, "height": 768},
    )
    page = ctx.new_page()
    _STATE.update(pw=pw, browser=browser, page=page)
    return page


def close_browser():
    if _STATE["browser"]:
        _STATE["browser"].close()
    if _STATE["pw"]:
        _STATE["pw"].stop()
    _STATE.update(pw=None, browser=None, page=None)


def _ir_a(url: str, etiqueta, descripcion: str) -> str:
    """Navega a una URL de listado y deja el HTML listo para extraer."""
    page = _ensure_browser()
    try:
        page.goto(url, timeout=_CFG["timeout_ms"], wait_until="domcontentloaded")
        _polite_sleep()
        html = page.content()
        metrics.save_fixture(html, url, "search", _ROOT)
        cards = parse_search_cards(html, _CFG["base_url"])
        _STATE["_last_cards"] = cards
        _STATE["_last_html"] = html
        _STATE["_last_page_number"] = etiqueta
        if not cards:
            # Sintoma tipico del challenge de Cloudflare: la pagina responde
            # 200 pero con una interstitial, sin ninguna tarjeta de aviso.
            metrics.record_error(f"sin_avisos_{etiqueta}", f"posible bloqueo anti-bot en {url}")
        return f"{descripcion}: {len(cards)} avisos detectados."
    except Exception as e:
        metrics.record_error(f"goto_{etiqueta}", e)
        return f"ERROR navegando {descripcion}: {e}. Sugerencia: reintentar o esperar."


@tool
def goto_search_page(page_number: int) -> str:
    """Navega a la pagina N del listado de departamentos en venta en CABA.
    Devuelve cuantos avisos se detectaron en esa pagina."""
    base = _CFG["base_url"].rstrip("/")
    path = _CFG["listing_path"]
    url = base + (path if page_number <= 1
                  else path.replace(".html", f"-pagina-{page_number}.html"))
    return _ir_a(url, page_number, f"Pagina {page_number}")


@tool
def goto_barrio_search(barrio_path: str) -> str:
    """Navega a la busqueda de departamentos en venta de un barrio de CABA.

    `barrio_path` es una ruta como "/departamentos-venta-palermo.html".
    Util porque la primera pagina de cada busqueda responde bien, mientras que
    las URLs paginadas quedan bloqueadas por el anti-bot del portal.
    """
    url = _CFG["base_url"].rstrip("/") + barrio_path
    etiqueta = barrio_path.strip("/").replace(".html", "")
    return _ir_a(url, etiqueta, f"Busqueda {etiqueta}")


@tool
def extract_current_page() -> str:
    """Abre cada aviso detectado en la ultima pagina de listado, extrae sus
    campos estructurados + descripcion, y los guarda. Devuelve cuantos se
    extrajeron con exito."""
    cards = _STATE.get("_last_cards", [])

    # --- Camino por defecto: extraer desde las tarjetas del listado ---------
    # ZonaProp protege las paginas de detalle con un challenge de Cloudflare,
    # pero las tarjetas del listado ya traen la descripcion completa. Ademas de
    # esquivar el bloqueo, es mas cortes: una request devuelve 25-30 avisos.
    if _CFG.get("extract_from", "cards") == "cards":
        html = _STATE.get("_last_html", "")
        recs = parse_search_listings(html, _CFG["base_url"]) if html else []
        ok = 0
        for rec in recs:
            if rec["url"] in SEEN_URLS:
                continue
            EXTRACTED.append(rec)
            SEEN_URLS.add(rec["url"])
            ok += 1
        detectados = len(cards) or len(recs)
        metrics.record_page(_STATE.get("_last_page_number", -1), detectados, ok, [])
        return (f"Extraidos {ok}/{detectados} avisos de la pagina. "
                f"Total acumulado: {len(EXTRACTED)}.")

    # --- Camino alternativo: visitar cada pagina de detalle -----------------
    page = _ensure_browser()
    ok = 0
    failed: list[str] = []
    for c in cards:
        url = c["url"]
        if url in SEEN_URLS:
            continue
        try:
            page.goto(url, timeout=_CFG["timeout_ms"], wait_until="domcontentloaded")
            _polite_sleep()
            html = page.content()
            metrics.save_fixture(html, url, "detail", _ROOT)
            rec = parse_detail(html, url)
            if rec.get("description"):
                EXTRACTED.append(rec)
                SEEN_URLS.add(url)
                ok += 1
            else:
                # Se navego bien pero el parser no encontro descripcion utilizable:
                # es justamente el modo de falla que motiva el parser resiliente.
                failed.append(url)
        except Exception as e:
            failed.append(url)
            metrics.record_error("extract_detail", e)
    metrics.record_page(_STATE.get("_last_page_number", -1), len(cards), ok, failed)
    return f"Extraidos {ok}/{len(cards)} avisos de la pagina. Total acumulado: {len(EXTRACTED)}."


@tool
def has_next_page() -> str:
    """Indica si existe pagina siguiente en el listado."""
    page = _ensure_browser()
    try:
        nxt = page.query_selector('[data-qa="PAGING_NEXT"], a[title="Siguiente"]')
        return "SI, hay pagina siguiente." if nxt else "NO, es la ultima pagina."
    except Exception as e:
        return f"No se pudo determinar: {e}"


TOOLS = [goto_search_page, extract_current_page, has_next_page]
