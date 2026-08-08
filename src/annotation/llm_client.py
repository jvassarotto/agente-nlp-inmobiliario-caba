"""Cliente LLM local (Ollama) para pre-anotacion. 100% gratis, sin API keys.

Requiere Ollama corriendo:  https://ollama.com  ->  `ollama serve`
y el modelo descargado:      `ollama pull llama3.2:3b-instruct-q4_K_M`
"""
from __future__ import annotations
import json
import re
from typing import Any

from src.utils.config import load_config


def get_llm(cfg: dict | None = None):
    """Devuelve un ChatOllama configurado desde configs/config.yaml."""
    from langchain_ollama import ChatOllama
    cfg = cfg or load_config()
    a = cfg["agent"]
    return ChatOllama(
        model=a["model"],
        base_url=a["base_url"],
        temperature=a.get("temperature", 0.0),
        num_ctx=a.get("num_ctx", 8192),
        format="json",   # fuerza salida JSON
    )


def parse_json(text: str) -> dict[str, Any]:
    """Extrae el primer objeto JSON de la respuesta del LLM de forma robusta."""
    if isinstance(text, dict):
        return text
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}
