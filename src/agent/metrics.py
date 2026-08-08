"""Instrumentacion de la capa de agente.

La propuesta aprobada compromete dos metricas para la Capa 1 (seccion 3.3):
  - **Tasa de exito de extraccion**: de los avisos que el agente DETECTA en una
    pagina de listado, cuantos termina extrayendo con su descripcion utilizable.
  - **Robustez ante variaciones de pagina**: cuanto sostiene el parser cuando el
    portal cambia el layout (se mide aparte, en tests/test_pipeline.py, corriendo
    el parser sobre fixtures degradados).

Este modulo acumula la primera en memoria durante la corrida y la vuelca a
`reports/agent_metrics.json` al terminar.

Ademas guarda **fixtures** (HTML crudo de algunas paginas) en `data/fixtures/`.
Sirven para dos cosas: que el corrector pueda reproducir el parseo sin tocar
ZonaProp, y alimentar el test de robustez.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

# Acumulador de la corrida. Lo consume save_report() al final.
_RUN: dict = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "mode": None,
    "pages": [],       # una entrada por pagina de listado visitada
    "errors": [],      # errores de navegacion / parseo
    "fixtures": [],    # archivos HTML guardados
}

_FIXTURE_COUNT = {"search": 0, "detail": 0}


def set_mode(mode: str) -> None:
    """Registra si la corrida fue 'agent' (ReAct) o 'deterministic'."""
    _RUN["mode"] = mode


def record_page(page_number: int, detected: int, extracted: int, failed_urls: list[str]) -> None:
    """Registra el resultado de procesar una pagina del listado."""
    _RUN["pages"].append({
        "page": page_number,
        "detected": detected,
        "extracted": extracted,
        "failed": len(failed_urls),
        "success_rate": round(extracted / detected, 4) if detected else None,
        "failed_urls": failed_urls[:10],   # muestra acotada, para diagnostico
    })


def record_error(stage: str, detail: str) -> None:
    """Registra un error recuperable (navegacion, timeout, parseo)."""
    _RUN["errors"].append({
        "stage": stage,
        "detail": str(detail)[:300],
        "at": datetime.now(timezone.utc).isoformat(),
    })


def _safe_name(url: str, prefix: str, idx: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", url.split("//")[-1])[:60].strip("_")
    return f"{prefix}_{idx:02d}_{slug}.html"


def save_fixture(html: str, url: str, kind: str, root: Path, max_per_kind: int = 3) -> None:
    """Guarda HTML crudo en data/fixtures/ (hasta `max_per_kind` por tipo).

    `kind` es 'search' (pagina de listado) o 'detail' (pagina de aviso).
    """
    if _FIXTURE_COUNT.get(kind, 0) >= max_per_kind:
        return
    out_dir = Path(root) / "data" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = _FIXTURE_COUNT[kind]
    name = _safe_name(url, kind, idx)
    (out_dir / name).write_text(html, encoding="utf-8", errors="ignore")
    _FIXTURE_COUNT[kind] = idx + 1
    _RUN["fixtures"].append({"kind": kind, "file": name, "url": url})


def summary() -> dict:
    """Agrega los totales de la corrida."""
    detected = sum(p["detected"] for p in _RUN["pages"])
    extracted = sum(p["extracted"] for p in _RUN["pages"])
    rates = [p["success_rate"] for p in _RUN["pages"] if p["success_rate"] is not None]
    return {
        "pages_visited": len(_RUN["pages"]),
        "listings_detected": detected,
        "listings_extracted": extracted,
        # Metrica principal: avisos extraidos / avisos detectados
        "extraction_success_rate": round(extracted / detected, 4) if detected else None,
        # Peor pagina: util para mostrar variabilidad entre paginas
        "worst_page_success_rate": round(min(rates), 4) if rates else None,
        "errors": len(_RUN["errors"]),
    }


def save_report(root: Path, out_name: str = "agent_metrics.json") -> Path:
    """Vuelca metricas + detalle por pagina a reports/agent_metrics.json."""
    out_dir = Path(root) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **{k: _RUN[k] for k in ("started_at", "mode")},
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary(),
        "pages": _RUN["pages"],
        "errors": _RUN["errors"],
        "fixtures": _RUN["fixtures"],
    }
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
