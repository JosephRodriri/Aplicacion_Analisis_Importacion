"""Clasificadores de productos y categorías."""
import json
from pathlib import Path

from src.domain.constants import PRODUCT_OTRO


def load_product_rules(config_path: Path) -> dict[str, list[str]]:
    """Carga reglas de clasificación desde JSON.

    Estructura esperada: {nombre_producto: [keyword1, keyword2, ...]}
    Las keywords se comparan en lowercase contra la descripción.
    """
    if not config_path.is_file():
        return {}
    with open(config_path, encoding="utf-8") as fh:
        return json.load(fh)


def classify_product(description: str, rules: dict[str, list[str]]) -> str:
    """Clasifica una descripción de partida arancelaria en un producto.

    Devuelve el nombre del producto si alguna keyword coincide, o 'Otro'.
    """
    if not description:
        return PRODUCT_OTRO
    desc_lower = str(description).lower()
    for product_name, keywords in rules.items():
        if any(kw in desc_lower for kw in keywords):
            return product_name
    return PRODUCT_OTRO
