"""Agente ReAct (LangGraph create_react_agent) con LLM local (Ollama).

El LLM actua como POLICY: razona sobre el estado de la pagina y decide acciones
(navegar, paginar, extraer, reintentar), usando las tools de browser_tools.py.
Coincide con el patron visto en la materia (LangChain/LangGraph create_agent).
"""
from __future__ import annotations

from src.annotation.llm_client import get_llm
from src.agent.browser_tools import TOOLS

SYSTEM_PROMPT = """Sos un agente que recolecta avisos de departamentos en venta en CABA desde ZonaProp.
Tu objetivo es recorrer el listado pagina por pagina y extraer los avisos.

Herramientas disponibles:
- goto_search_page(page_number): navega a una pagina del listado.
- extract_current_page(): extrae y guarda los avisos de la pagina actual.
- has_next_page(): indica si hay pagina siguiente.

Estrategia:
1. Empeza en la pagina 1 con goto_search_page.
2. Llama extract_current_page para guardar los avisos.
3. Usa has_next_page; si hay siguiente, avanza; si no, termina.
4. Si una accion devuelve ERROR, reintenta esa pagina una vez antes de continuar.
5. No repitas paginas ya procesadas. Se conciso en tu razonamiento.

Cuando termines o alcances el limite de paginas, responde 'LISTO'."""


def build_agent():
    """Construye el agente ReAct. Requiere Ollama corriendo."""
    from langgraph.prebuilt import create_react_agent
    llm = get_llm()
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)
