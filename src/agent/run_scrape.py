"""Entrypoint del scraping. Construye el dataset primario de ZonaProp.

Dos modos:
  --mode agent          -> el agente ReAct (LLM local) orquesta la navegacion.
  --mode deterministic  -> bucle deterministico (mas robusto para corridas largas).

Uso tipico (primera corrida real):
  playwright install chromium
  python -m src.agent.run_scrape --mode deterministic --max-pages 400

Salida: data/raw/zonaprop_caba.jsonl (un aviso por linea, esquema Listing).
"""
from __future__ import annotations
import argparse
from pathlib import Path

from src.utils.config import load_config
from src.utils.io import write_jsonl, read_jsonl
from src.agent import browser_tools as bt
from src.agent import metrics


def run_deterministic(cfg, max_pages, max_listings):
    page = 1
    while page <= max_pages and len(bt.EXTRACTED) < max_listings:
        msg = bt.goto_search_page.invoke({"page_number": page})
        print(f"[p{page}] {msg}")
        if "ERROR" in msg:
            # un reintento simple
            msg = bt.goto_search_page.invoke({"page_number": page})
            if "ERROR" in msg:
                print(f"[p{page}] se omite tras reintento fallido")
                page += 1
                continue
        print(f"[p{page}] {bt.extract_current_page.invoke({})}")
        if "NO" in bt.has_next_page.invoke({}):
            print("Fin del listado.")
            break
        page += 1


def run_argenprop(cfg, max_listings):
    """Recorre Argenprop: barrios x paginas 1..10 (tope de su robots.txt)."""
    import random
    import time

    from src.agent import argenprop as ap

    ac = cfg["argenprop"]
    tope = min(int(ac.get("max_pagina", ap.MAX_PAGINA)), ap.MAX_PAGINA)
    # Se parte de lo ya scrapeado: cada corrida ACUMULA en vez de pisar. Como el
    # sitio limita el ritmo, el dataset se junta a lo largo de varias corridas.
    salida = Path(cfg["_root"]) / ac["out_path"]
    if salida.exists():
        previos = list(read_jsonl(salida))
        bt.EXTRACTED.extend(previos)
        print(f"Retomando desde {len(previos)} avisos ya scrapeados.")

    vistos: set[str] = {r["url"] for r in bt.EXTRACTED if r.get("url")}
    total_detectados = 0
    barrios_vacios = 0      # cortacircuitos: si el sitio nos corto, no insistir

    for barrio in ac["barrios"]:
        if len(bt.EXTRACTED) >= max_listings:
            break
        if barrios_vacios >= 3:
            print("\nTres barrios seguidos sin resultados: el sitio nos esta limitando.")
            print("Se corta la corrida. Reintentar mas tarde o subir los delays.")
            break
        antes = len(bt.EXTRACTED)
        for pagina in range(1, tope + 1):
            if len(bt.EXTRACTED) >= max_listings:
                break
            url = ap.build_url(barrio, pagina)
            try:
                detectadas, recs = ap.parse_listings_con_conteo(ap.fetch_con_backoff(url))
            except Exception as e:
                metrics.record_error(f"argenprop_{barrio}_p{pagina}", e)
                print(f"  [{barrio} p{pagina}] ERROR: {str(e)[:90]}")
                break   # si falla una pagina, se pasa al proximo barrio

            nuevos = 0
            for r in recs:
                if r["url"] not in vistos:
                    vistos.add(r["url"])
                    bt.EXTRACTED.append(r)
                    nuevos += 1
            total_detectados += detectadas
            # La tasa de exito mide si el PARSER logro extraer cada tarjeta
            # detectada. Que un aviso ya estuviera de una corrida previa NO es
            # un fallo de extraccion: `nuevos` se informa aparte.
            metrics.record_page(f"{barrio.split('/')[-1]}-p{pagina}", detectadas, len(recs), [])
            print(f"  [{barrio.split('/')[-1]:20s} p{pagina:2d}] {detectadas:2d} detectadas, "
                  f"{len(recs):2d} extraidas, {nuevos:2d} nuevas | total {len(bt.EXTRACTED)}")

            if not recs:
                break   # sin resultados: el barrio se agoto (o nos cortaron)
            time.sleep(random.uniform(ac["min_delay_s"], ac["max_delay_s"]))

        barrios_vacios = 0 if len(bt.EXTRACTED) > antes else barrios_vacios + 1

    print(f"\nDetectados {total_detectados}, unicos {len(bt.EXTRACTED)}")


def build_grid(sc) -> list[str]:
    """Arma las rutas de busqueda combinando barrio x ambientes.

    Cada celda es una busqueda distinta, no una pagina siguiente de la misma:
    por eso no dispara el bloqueo que si aparece al paginar.
    """
    barrios = sc.get("barrios") or []
    ambientes = sc.get("ambientes") or [""]
    return [f"/departamentos-venta-{b}{a}.html" for b in barrios for a in ambientes]


def run_grid(cfg, max_listings):
    """Recorre la grilla de busquedas, una request por celda."""
    sc = cfg["scrape"]
    rutas = build_grid(sc)
    print(f"Grilla: {len(rutas)} busquedas "
          f"({len(sc.get('barrios', []))} barrios x {len(sc.get('ambientes', ['']))} filtros de ambientes)")

    bloqueadas = 0
    for i, ruta in enumerate(rutas, 1):
        if len(bt.EXTRACTED) >= max_listings:
            print(f"Alcanzado el limite de {max_listings} avisos.")
            break
        msg = bt.goto_barrio_search.invoke({"barrio_path": ruta})
        print(f"[{i}/{len(rutas)}] {msg}")
        if "ERROR" in msg or ": 0 avisos" in msg:
            # Sin avisos = celda vacia o interstitial del anti-bot. Se sigue con
            # la proxima busqueda en lugar de insistir contra la misma URL.
            bloqueadas += 1
            if bloqueadas >= 8:
                print("Demasiadas busquedas sin resultados seguidas: se corta la corrida.")
                break
            continue
        bloqueadas = 0
        print(f"        {bt.extract_current_page.invoke({})}")


def run_agent(cfg, max_pages):
    from src.agent.react_agent import build_agent
    agent = build_agent()
    task = (f"Recorre hasta {max_pages} paginas del listado de departamentos en venta "
            f"en CABA y extrae todos los avisos. Empeza por la pagina 1.")
    for step in agent.stream({"messages": [("user", task)]}, stream_mode="values",
                             config={"recursion_limit": cfg["agent"]["max_react_steps"] * max_pages}):
        last = step["messages"][-1]
        if hasattr(last, "content") and last.content:
            print(last.content[:300])


def main():
    cfg = load_config()
    sc = cfg["scrape"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["agent", "deterministic", "grid", "argenprop"],
                    default="deterministic")
    ap.add_argument("--max-pages", type=int, default=sc["max_pages"])
    ap.add_argument("--max-listings", type=int, default=sc["max_listings"])
    ap.add_argument("--out", default=sc["out_path"])
    args = ap.parse_args()

    metrics.set_mode(args.mode)
    try:
        if args.mode == "agent":
            run_agent(cfg, args.max_pages)
        elif args.mode == "argenprop":
            if args.out == sc["out_path"]:      # sin --out explicito
                args.out = cfg["argenprop"]["out_path"]
            run_argenprop(cfg, args.max_listings)
        elif args.mode == "grid":
            run_grid(cfg, args.max_listings)
        else:
            run_deterministic(cfg, args.max_pages, args.max_listings)
    finally:
        n = write_jsonl(bt.EXTRACTED, args.out)
        bt.close_browser()
        # El reporte se guarda SIEMPRE, incluso si la corrida se corto por
        # bloqueo anti-bot: una corrida fallida tambien es un dato a reportar.
        report = metrics.save_report(cfg["_root"], out_name=f"agent_metrics_{args.mode}.json")
        s = metrics.summary()
        print(f"[OK] {n} avisos guardados -> {args.out}")
        print(f"[METRICAS] detectados={s['listings_detected']} extraidos={s['listings_extracted']} "
              f"tasa_exito={s['extraction_success_rate']} -> {report}")


if __name__ == "__main__":
    main()
