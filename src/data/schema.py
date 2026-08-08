"""Esquema del aviso inmobiliario (campos estructurados + texto libre).

Se usa tanto para lo que devuelve el agente al scrapear ZonaProp como para
el dataset sintetico. Pydantic valida y normaliza los tipos.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Listing(BaseModel):
    # Identificacion / origen
    id: str = Field(..., description="ID del aviso o hash de la URL")
    url: Optional[str] = None
    source: str = "zonaprop"
    scraped_at: Optional[str] = None

    # Campos estructurados (tabulares)
    price_amount: Optional[float] = Field(None, description="Monto de precio")
    price_currency: Optional[str] = Field(None, description="USD / ARS")
    expenses_amount: Optional[float] = Field(None, description="Expensas en ARS")
    surface_total_m2: Optional[float] = None
    surface_covered_m2: Optional[float] = None
    rooms: Optional[int] = Field(None, description="Ambientes")
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    age_years: Optional[int] = Field(None, description="Antiguedad en anios (0 = a estrenar)")
    neighborhood: Optional[str] = Field(None, description="Barrio de CABA")
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Texto libre (nucleo del enriquecimiento NLP)
    title: Optional[str] = None
    description: str = Field(..., description="Descripcion en texto libre del aviso")

    class Config:
        extra = "allow"
