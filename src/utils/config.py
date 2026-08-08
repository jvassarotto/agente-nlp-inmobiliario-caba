"""Carga de configuracion central (configs/config.yaml) y semillas."""
from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Lee el YAML de configuracion y devuelve un dict."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve(cfg: dict[str, Any], rel: str) -> Path:
    """Convierte una ruta relativa del config en ruta absoluta bajo el repo."""
    return (ROOT / rel).resolve()


def set_seed(seed: int = 42) -> None:
    """Fija semillas para reproducibilidad (random, numpy, torch si esta)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
