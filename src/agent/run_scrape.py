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

from src.utils.config import load_config
from src.utils.io import write_jsonl
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
    ap.add_argument("--mode", choices=["agent", "deterministic"], default="deterministic")
    ap.add_argument("--max-pages", type=int, default=sc["max_pages"])
    ap.add_argument("--max-listings", type=int, default=sc["max_listings"])
    ap.add_argument("--out", default=sc["out_path"])
    args = ap.parse_args()

    metrics.set_mode(args.mode)
    try:
        if args.mode == "agent":
            run_agent(cfg, args.max_pages)
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
